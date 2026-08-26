#!/usr/bin/env python3
"""CPU-only finalizer for one fused-pointwise SM120 smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load_exact_source(name: str, path: Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"exact finalizer source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


control_manifest = load_exact_source(
    "_pypto_fused_pointwise_sm120_control_manifest",
    ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
)
control_manifest.reject_control_bytecode_cache(ROOT)
contract = load_exact_source(
    "_pypto_fused_pointwise_sm120_contract",
    ROOT / "tools/_pypto_fused_pointwise_sm120_contract.py",
)


class FinalizeError(RuntimeError):
    """The provisional fused-pointwise smoke cannot be promoted."""


def require_no_site_finalizer() -> None:
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        raise FinalizeError("fused-pointwise SM120 finalizer requires Python -E -B -S")


# This child only deserializes immutable evidence through public ``pypto.ir``
# and compiler data types.  It invokes no compiler producer or NVIDIA runtime
# entry point, constructs no executable, and calls no CUDA Runtime/Driver API.
# Importing public PyPTO may transitively import Torch; the only Torch call is
# the state-only assertion that CUDA was never initialized.
REPLAY_AUDIT_PROGRAM = r"""
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

workspace = Path(sys.argv[1]).resolve(strict=True)
replay = Path(sys.argv[2]).resolve(strict=True)
site = workspace / "envs/pypto-nvidia/lib/python3.14/site-packages"
sys.path.insert(0, str(site))

def load(name, path):
    resolved = path.resolve(strict=True)
    assert resolved == path and not path.is_symlink() and path.is_file()
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module

control = load(
    "replay_fused_control",
    workspace / "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
)
control.reject_control_bytecode_cache(workspace)
contract = load(
    "replay_fused_contract",
    workspace / "tools/_pypto_fused_pointwise_sm120_contract.py",
)
runner = load("replay_frontend_runner", workspace / contract.RUNNER_RELATIVE_PATH)
runner.validate_pypto_python_source(workspace)
pypto = runner.bootstrap_exact_pypto(
    workspace,
    (workspace / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True).parent,
)
from pypto import compiler

request_bytes = (replay / "compile-request.msgpack").read_bytes()
request = compiler.CompileRequest.deserialize(request_bytes)
assert request.serialize() == request_bytes
target = request.target_info
target_info = {
    "device_ordinal": target.device_ordinal,
    "device_name": target.device_name,
    "device_uuid": target.device_uuid,
    "pci_device_id": target.pci_device_id,
    "traits": runner.target_traits_document(target.traits),
    "cuda_toolkit_version": target.cuda_toolkit_version,
    "cuda_driver_version": target.cuda_driver_version,
    "tensor_ir_revision": target.tensor_ir_revision,
    "cuda_tile_revision": target.cuda_tile_revision,
    "supported_compute_dtypes": runner.supported_dtype_names(
        pypto, target.supported_compute_dtypes
    ),
}
hir_records = []
artifact_records = []
for case in contract.CASE_SPECS:
    hir_bytes = (replay / f"{case.name}.hir.msgpack").read_bytes()
    assert len(hir_bytes) == case.expected_hir_bytes
    assert runner.sha256_bytes(hir_bytes) == case.expected_hir_sha256
    program = pypto.ir.deserialize(hir_bytes)
    assert isinstance(program, pypto.ir.Program)
    assert bytes(pypto.ir.serialize(program)) == hir_bytes
    function = program.get_function("fused_main")
    assert function is not None
    assert list(function.param_directions) == [pypto.ir.ParamDirection.In] * case.input_count
    source_bytes = (replay / f"{case.name}.source.mlir").read_bytes()
    assert len(source_bytes) == case.expected_source_ir_bytes
    assert runner.sha256_bytes(source_bytes) == case.expected_source_ir_digest
    assert source_bytes == runner.canonical_tensor_ir_source(case)
    spec_bytes = (replay / f"{case.name}.build-spec.msgpack").read_bytes()
    spec = compiler.KernelBuildSpec.deserialize(spec_bytes)
    assert spec.serialize() == spec_bytes
    artifact_bytes = (replay / f"{case.name}.artifact.msgpack").read_bytes()
    artifact = compiler.Artifact.deserialize(artifact_bytes, request, spec)
    assert artifact.serialize() == artifact_bytes
    runner.validate_compiled_artifact(
        compiler, spec, artifact, request, case, source_bytes
    )
    cubin_bytes = (replay / f"{case.name}.cubin").read_bytes()
    assert cubin_bytes == bytes(artifact.device_code)
    kernel = artifact.kernel_abi
    descriptors = list(kernel.argument_layout.operand_descriptors)
    expected_strides = list(case.strides) if len(case.shape) > 1 else []
    assert not artifact.fallback_used
    assert artifact.actual_target.compute_capability == 120
    assert len(artifact.device_code) == case.expected_device_code_bytes
    assert artifact.device_code_sha256 == case.expected_device_code_sha256
    assert spec.source_ir_digest == case.expected_source_ir_digest
    assert spec.static_specialization_digest == case.expected_static_specialization_digest
    assert spec.symbolic_specialization_digest == case.expected_symbolic_specialization_digest
    assert spec.argument_abi_digest == case.expected_argument_abi_digest
    assert spec.result_abi_digest == case.expected_result_abi_digest
    assert spec.mutation_abi_digest == case.expected_mutation_abi_digest
    assert spec.callable_abi_digest == case.expected_callable_abi_digest
    assert spec.callable_abi_digest == kernel.identity_digest
    assert artifact.identities.kernel_build_spec_digest == spec.identity_digest
    assert artifact.identities.source_ir_digest == spec.source_ir_digest
    assert artifact.identities.argument_abi_digest == spec.argument_abi_digest
    assert artifact.identities.result_abi_digest == spec.result_abi_digest
    assert artifact.identities.mutation_abi_digest == spec.mutation_abi_digest
    assert kernel.runtime_kernel_name == "tensor_ir_rtk"
    assert kernel.entry_function_name == "pypto_fused_pointwise_v2"
    assert kernel.argument_packing_policy == compiler.ArtifactArgumentPackingPolicy.PointerOnly
    assert kernel.grid_abi.policy == compiler.ArtifactGridPolicy.Static
    assert tuple(kernel.grid_abi.static_dimensions) == case.expected_grid
    assert tuple(kernel.grid_abi.tile_sizes) == case.tile_sizes
    assert kernel.grid_abi.shape_operand_index == 0
    assert kernel.argument_layout.input_operand_count == case.input_count
    assert kernel.argument_layout.total_kernel_argument_count == case.expected_kernel_arguments
    assert not kernel.argument_layout.uniform_signature
    assert len(descriptors) == case.expected_kernel_arguments
    assert kernel.workspace_abi.size_bytes == 0
    for descriptor in descriptors:
        assert descriptor.kind == compiler.ArtifactOperandKind.Tensor
        assert list(descriptor.shape) == list(case.shape)
        assert list(descriptor.strides) == expected_strides
        assert descriptor.dynamic_size_count == 0
        assert descriptor.dynamic_stride_count == 0
        assert descriptor.explicit_strides is (len(case.shape) > 1)
    hir_records.append({
        "case": case.name,
        "bytes": len(hir_bytes),
        "sha256": runner.sha256_bytes(hir_bytes),
        "canonical_reserialization_equal": True,
        "parameter_directions": ["In"] * case.input_count,
        "input_count": case.input_count,
        "assignment_count": case.assignment_count,
        "operator_sequence": list(case.operator_sequence),
    })
    artifact_records.append({
        "case": case.name,
        "build_spec_identity_digest": spec.identity_digest,
        "source_ir_digest": spec.source_ir_digest,
        "source_ir_bytes": len(source_bytes),
        "callable_abi_digest": spec.callable_abi_digest,
        "static_specialization_digest": spec.static_specialization_digest,
        "symbolic_specialization_digest": spec.symbolic_specialization_digest,
        "argument_abi_digest": spec.argument_abi_digest,
        "result_abi_digest": spec.result_abi_digest,
        "mutation_abi_digest": spec.mutation_abi_digest,
        "artifact_identity_digest": artifact.identity_digest,
        "cache_key_digest": artifact.cache_key_digest,
        "loader_compatibility_digest": artifact.loader_compatibility_digest,
        "device_code_bytes": len(artifact.device_code),
        "device_code_sha256": artifact.device_code_sha256,
        "kernel_abi_identity_digest": kernel.identity_digest,
        "entry_function_name": kernel.entry_function_name,
        "fallback_used": artifact.fallback_used,
        "input_operand_count": case.input_count,
        "assignment_count": case.assignment_count,
        "operator_sequence": list(case.operator_sequence),
    })
assert {"triton", "sglang", "flashinfer"}.isdisjoint(sys.modules)
if "torch" in sys.modules:
    import torch
    assert not torch.cuda.is_initialized()
print(json.dumps({
    "compile_request": {
        "byte_identity_digest": request.byte_compile_identity_digest,
        "loader_compatibility_input_digest": request.loader_compatibility_input_digest,
        "device_autotune_identity_digest": request.device_autotune_identity_digest,
    },
    "target_info": target_info,
    "hir_programs": hir_records,
    "artifacts": artifact_records,
}, sort_keys=True, separators=(",", ":")))
"""


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise FinalizeError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalizeError(f"{description} is not a lowercase SHA-256")
    return value


def require_int(value: object, description: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalizeError(f"{description} is not an integer")
    if positive and value <= 0:
        raise FinalizeError(f"{description} is not positive")
    return value


def require_exact_keys(
    value: object, expected: set[str], description: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise FinalizeError(
            f"{description} key set differs: expected={sorted(expected)}, actual={actual}"
        )
    return value


def validate_child_gate_schema(value: object) -> dict[str, object]:
    return require_exact_keys(
        value,
        {
            "static_identity",
            "gpu",
            "free_memory_mib",
            "protected_heavy_pids",
            "protected_runtime_pids",
            "unreadable_protected_maps",
            "nvidia_compute_pids",
            "control_manifest",
        },
        "child pre-CUDA gate",
    )


def validate_target_traits_schema(value: object) -> dict[str, object]:
    return require_exact_keys(
        value,
        {
            "compute_capability",
            "multiprocessor_count",
            "warp_size",
            "max_threads_per_block",
            "max_threads_per_multiprocessor",
            "max_blocks_per_multiprocessor",
            "max_block_dim_x",
            "max_block_dim_y",
            "max_block_dim_z",
            "max_grid_dim_x",
            "max_grid_dim_y",
            "max_grid_dim_z",
            "l1_cache_line_bytes",
            "default_shared_memory_per_cta_bytes",
            "max_shared_memory_per_cta_bytes",
            "shared_memory_per_multiprocessor_bytes",
            "registers_per_cta",
            "max_registers_per_thread",
            "registers_per_multiprocessor",
            "l2_cache_size_bytes",
            "total_global_memory_bytes",
        },
        "runtime target traits",
    )


def require_workspace(workspace: Path) -> Path:
    resolved = workspace.resolve(strict=True)
    if workspace.absolute() != resolved or resolved != ROOT:
        raise FinalizeError(f"workspace must be the exact project root: {ROOT}")
    return resolved


def require_regular(path: Path, workspace: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FinalizeError(f"{description} is missing: {path}") from error
    if path.absolute() != resolved or (
        resolved != workspace and workspace not in resolved.parents
    ):
        raise FinalizeError(f"{description} is not canonical and workspace-owned")
    identity = resolved.stat()
    if path.is_symlink() or not stat.S_ISREG(identity.st_mode):
        raise FinalizeError(f"{description} must be a regular non-symlink file")
    return resolved


def load_canonical(
    path: Path, workspace: Path, description: str
) -> tuple[dict[str, object], bytes]:
    resolved = require_regular(path, workspace, description)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizeError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise FinalizeError(f"{description} is not canonical JSON")
    return value, raw


def publish_no_replace(path: Path, value: object) -> str:
    if path.exists() or path.is_symlink():
        raise FinalizeError(f"final evidence already exists: {path}")
    parent = path.parent.resolve(strict=True)
    if parent != ROOT and ROOT not in parent.parents:
        raise FinalizeError("final evidence parent escapes the workspace")
    encoded = canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FinalizeError(f"final evidence already exists: {path}") from error
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(encoded)


def git_identity(repository: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout

    return {
        "head": git("rev-parse", "HEAD").strip(),
        "tree": git("rev-parse", "HEAD^{tree}").strip(),
        "clean": not git("status", "--porcelain=v1"),
    }


def validate_exact_file(
    path: Path, workspace: Path, size: int, digest: str, description: str
) -> dict[str, object]:
    resolved = require_regular(path, workspace, description)
    if resolved.stat().st_size != size or sha256_file(resolved) != digest:
        raise FinalizeError(f"{description} differs from the fixed contract")
    return {
        "path": resolved.relative_to(workspace).as_posix(),
        "bytes": size,
        "sha256": digest,
    }


def validate_scope(provisional: dict[str, object]) -> None:
    scope = require_exact_keys(
        provisional.get("scope"),
        {
            "frontend_family",
            "fixed_fixture_set",
            "fixed_fixture_correctness",
            "general_operator_correctness",
            "legacy_cp44_unchanged",
            "model_forward",
            "strict_coverage_result",
            "performance_result",
            "cuda_graph_result",
        },
        "scope",
    )
    if scope != {
        "frontend_family": "FusedPointwiseV2",
        "fixed_fixture_set": "full-nine-case-numerical-v1",
        "fixed_fixture_correctness": True,
        "general_operator_correctness": False,
        "legacy_cp44_unchanged": True,
        "model_forward": False,
        "strict_coverage_result": False,
        "performance_result": False,
        "cuda_graph_result": False,
    }:
        raise FinalizeError("scope exceeds the fused-pointwise correctness claim")
    forbidden_key_fragments = (
        "latency",
        "throughput",
        "tokens_per",
        "bandwidth",
        "flops",
        "duration_ns",
        "cuda_event",
        "benchmark",
        "speedup",
    )
    forbidden_exact_keys = {
        "model_correct",
        "qwen_correct",
        "coverage_percent",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                if lowered in forbidden_exact_keys or any(
                    fragment in lowered for fragment in forbidden_key_fragments
                ):
                    raise FinalizeError(f"unaccepted claim field is forbidden: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(provisional)


def validate_provisional_schema(provisional: dict[str, object]) -> None:
    require_exact_keys(
        provisional,
        {
            "schema_version",
            "smoke",
            "acceptance",
            "scope",
            "inputs",
            "run_context",
            "runtime",
        },
        "provisional",
    )
    inputs = require_exact_keys(
        provisional.get("inputs"),
        {
            "integrity",
            "pypto",
            "tensor_ir_head",
            "cuda_tile_head",
            "llvm_head",
            "replay_files",
            "control_manifest",
        },
        "provisional inputs",
    )
    integrity = require_exact_keys(
        inputs.get("integrity"),
        {
            "anchor_generator",
            "compile_anchors",
            "contract",
            "runner",
            "controller",
            "environment_lock",
            "versions_lock",
            "workspace_lock",
            "pypto_dso",
            "cuda_runtime",
        },
        "provisional integrity",
    )
    for name, record in integrity.items():
        require_exact_keys(record, {"path", "bytes", "sha256"}, f"integrity {name}")
    require_exact_keys(inputs.get("pypto"), {"head", "tree", "clean"}, "PyPTO identity")
    run = require_exact_keys(
        provisional.get("run_context"),
        {
            "run_id",
            "mode",
            "pid",
            "pgid",
            "start_ticks",
            "preflight",
            "gate",
            "start_barrier_sha256",
            "protected_zero_nvidia_policy",
        },
        "run context",
    )
    require_exact_keys(run.get("preflight"), {"path", "sha256"}, "preflight anchor")
    require_exact_keys(run.get("gate"), {"path", "sha256", "document"}, "gate anchor")
    runtime = require_exact_keys(
        provisional.get("runtime"),
        {
            "torch",
            "child_pre_cuda_gate",
            "libcudart_paths",
            "observation",
            "compile_request",
            "hir_programs",
            "artifacts",
            "executions",
            "case_order",
            "compile_invocations_per_case",
            "repetitions_per_case",
            "module_lifetimes",
            "explicit_packet_releases",
            "explicit_unloads",
            "non_default_current_stream",
            "distinct_nondefault_reference_stream",
            "reference_compute_outside_candidate_coverage",
            "external_reference_synchronizations",
            "external_synchronization",
            "fallback_used",
            "forbidden_provider_imports",
        },
        "runtime",
    )
    require_exact_keys(
        runtime.get("torch"),
        {"version", "git_version", "cuda", "hip", "module_path"},
        "runtime Torch",
    )
    validate_child_gate_schema(runtime.get("child_pre_cuda_gate"))
    observation = require_exact_keys(
        runtime.get("observation"),
        {
            "device_ordinal",
            "device_name",
            "device_uuid",
            "pci_device_id",
            "traits",
            "cuda_toolkit_version",
            "cuda_driver_version",
            "tensor_ir_revision",
            "cuda_tile_revision",
            "supported_compute_dtypes",
            "cuda_driver_release_provenance",
            "cuda_driver_api_version",
            "cuda_runtime_api_version",
            "cuda_runtime_library_path",
            "context_address",
            "context_id",
        },
        "runtime observation",
    )
    validate_target_traits_schema(observation.get("traits"))
    require_exact_keys(
        runtime.get("compile_request"),
        {
            "byte_identity_digest",
            "loader_compatibility_input_digest",
            "device_autotune_identity_digest",
        },
        "compile request",
    )


def validate_runtime_identity(
    provisional: dict[str, object],
    workspace: Path,
    preflight: dict[str, object],
    gate: dict[str, object],
    control_identity: dict[str, object],
) -> None:
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    torch = runtime["torch"]
    assert isinstance(torch, dict)
    expected_torch_path = (
        workspace / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
    ).resolve(strict=True)
    if (
        torch.get("version") != contract.EXPECTED_TORCH_VERSION
        or torch.get("git_version") != contract.EXPECTED_TORCH_GIT
        or torch.get("cuda") != contract.EXPECTED_TORCH_CUDA
        or torch.get("hip") is not None
        or Path(str(torch.get("module_path", ""))) != expected_torch_path
    ):
        raise FinalizeError("runtime Torch identity differs")
    expected_runtime = str(
        (workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
    )
    if runtime.get("libcudart_paths") != [expected_runtime]:
        raise FinalizeError("runtime libcudart provider differs")
    observation = require_exact_keys(
        runtime.get("observation"),
        {
            "device_ordinal",
            "device_name",
            "device_uuid",
            "pci_device_id",
            "traits",
            "cuda_toolkit_version",
            "cuda_driver_version",
            "tensor_ir_revision",
            "cuda_tile_revision",
            "supported_compute_dtypes",
            "cuda_driver_release_provenance",
            "cuda_driver_api_version",
            "cuda_runtime_api_version",
            "cuda_runtime_library_path",
            "context_address",
            "context_id",
        },
        "runtime observation",
    )
    traits = validate_target_traits_schema(observation.get("traits"))
    if (
        observation.get("device_ordinal") != 0
        or observation.get("device_name") != contract.EXPECTED_DEVICE_NAME
        or not isinstance(observation.get("device_uuid"), str)
        or re.fullmatch(
            r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(observation.get("device_uuid")),
        )
        is None
        or not isinstance(observation.get("pci_device_id"), str)
        or not observation.get("pci_device_id")
        or traits.get("compute_capability") != 120
        or traits.get("multiprocessor_count") != contract.EXPECTED_SM_COUNT
        or observation.get("cuda_toolkit_version")
        != contract.EXPECTED_CUDA_TOOLKIT_VERSION
        or observation.get("cuda_driver_version") != contract.EXPECTED_DRIVER_RELEASE
        or observation.get("tensor_ir_revision") != contract.TENSOR_IR_HEAD
        or observation.get("cuda_tile_revision") != contract.CUDA_TILE_HEAD
        or observation.get("supported_compute_dtypes")
        != list(contract.EXPECTED_SUPPORTED_COMPUTE_DTYPES)
        or observation.get("cuda_driver_release_provenance")
        != contract.EXPECTED_DRIVER_RELEASE
        or observation.get("cuda_runtime_library_path") != expected_runtime
        or require_int(
            observation.get("cuda_driver_api_version"), "driver API", positive=True
        )
        < contract.MINIMUM_CUDA_DRIVER_API_VERSION
        or require_int(
            observation.get("cuda_runtime_api_version"), "runtime API", positive=True
        )
        < contract.MINIMUM_CUDA_RUNTIME_API_VERSION
    ):
        raise FinalizeError("live PyPTO runtime observation differs")
    for name, value in traits.items():
        require_int(value, f"target trait {name}", positive=True)
    require_int(observation.get("context_address"), "context address", positive=True)
    require_int(observation.get("context_id"), "context ID", positive=True)
    compile_request = runtime.get("compile_request")
    assert isinstance(compile_request, dict)
    for name, value in compile_request.items():
        require_sha(value, f"CompileRequest {name}")
    child_gate = validate_child_gate_schema(runtime.get("child_pre_cuda_gate"))
    if (
        child_gate.get("control_manifest") != control_identity
        or child_gate.get("static_identity") != gate.get("static_identity")
        or child_gate.get("nvidia_compute_pids") != []
        or child_gate.get("protected_runtime_pids") != []
        or child_gate.get("unreadable_protected_maps") != []
        or require_int(
            child_gate.get("free_memory_mib"), "child gate free memory", positive=True
        )
        < 4096
    ):
        raise FinalizeError("child pre-CUDA gate differs")
    child_gpu = require_exact_keys(
        child_gate.get("gpu"),
        {"name", "compute_capability", "memory_mib", "used_mib", "driver"},
        "child pre-CUDA GPU identity",
    )
    if (
        child_gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or child_gpu.get("compute_capability") != "12.0"
        or child_gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        raise FinalizeError("child pre-CUDA GPU identity differs")
    if preflight.get("torch") != gate.get("static_identity"):
        raise FinalizeError("preflight/gate static identity differs")


def expected_guard_sha256(dtype: str, value: float) -> str:
    words = _encode_words([value] * contract.GUARD_ELEMENTS, dtype)
    code = "I" if dtype == "float32" else "H"
    return sha256_bytes(struct.pack(f"<{len(words)}{code}", *words))


def validate_frontend_results(provisional: dict[str, object]) -> None:
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    cases = list(contract.CASE_SPECS)
    expected_order = list(contract.CASE_ORDER)
    lifetime_count = sum(case.repetitions for case in cases)
    if (
        runtime.get("case_order") != expected_order
        or runtime.get("compile_invocations_per_case") != 1
        or runtime.get("repetitions_per_case") != 2
        or runtime.get("module_lifetimes") != lifetime_count
        or runtime.get("explicit_packet_releases") != lifetime_count
        or runtime.get("explicit_unloads") != lifetime_count
        or runtime.get("non_default_current_stream") is not True
        or runtime.get("distinct_nondefault_reference_stream") is not True
        or runtime.get("reference_compute_outside_candidate_coverage") is not True
        or runtime.get("external_reference_synchronizations") != lifetime_count
        or runtime.get("external_synchronization") is not True
        or runtime.get("fallback_used") is not False
        or runtime.get("forbidden_provider_imports") != []
    ):
        raise FinalizeError("runtime aggregate contract differs")
    hir_programs = runtime.get("hir_programs")
    artifacts = runtime.get("artifacts")
    if (
        not isinstance(hir_programs, list)
        or len(hir_programs) != len(cases)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(cases)
    ):
        raise FinalizeError("frontend HIR/Artifact set is incomplete")
    artifact_identities: dict[str, str] = {}
    for hir_record, artifact_record, case in zip(
        hir_programs, artifacts, cases, strict=True
    ):
        hir_record = require_exact_keys(
            hir_record,
            {
                "case",
                "bytes",
                "sha256",
                "serialized_once",
                "deserialized_before_compile",
                "canonical_reserialization_equal",
                "structural_equal",
                "parameter_directions",
                "input_count",
                "assignment_count",
                "operator_sequence",
            },
            f"HIR {case.name}",
        )
        if (
            hir_record.get("case") != case.name
            or require_int(
                hir_record.get("bytes"), f"HIR {case.name} bytes", positive=True
            )
            != case.expected_hir_bytes
            or hir_record.get("sha256") != case.expected_hir_sha256
            or hir_record.get("serialized_once") is not True
            or hir_record.get("deserialized_before_compile") is not True
            or hir_record.get("canonical_reserialization_equal") is not True
            or hir_record.get("structural_equal") is not True
            or hir_record.get("parameter_directions") != ["In"] * case.input_count
            or hir_record.get("input_count") != case.input_count
            or hir_record.get("assignment_count") != case.assignment_count
            or hir_record.get("operator_sequence") != list(case.operator_sequence)
        ):
            raise FinalizeError(f"HIR {case.name} round-trip evidence differs")
        require_sha(hir_record.get("sha256"), f"HIR {case.name}")
        artifact_record = require_exact_keys(
            artifact_record,
            {
                "case",
                "compile_api",
                "compiler_invocations",
                "build_spec_identity_digest",
                "source_ir_digest",
                "source_ir_bytes",
                "callable_abi_digest",
                "static_specialization_digest",
                "symbolic_specialization_digest",
                "argument_abi_digest",
                "result_abi_digest",
                "mutation_abi_digest",
                "artifact_identity_digest",
                "cache_key_digest",
                "loader_compatibility_digest",
                "device_code_bytes",
                "device_code_sha256",
                "kernel_abi_identity_digest",
                "entry_function_name",
                "fallback_used",
                "expected_grid",
                "expected_kernel_arguments",
                "expected_device_code_bytes",
                "expected_device_code_sha256",
                "hir_sha256",
                "hir_bytes",
                "hir_roundtrip_exact",
                "input_operand_count",
                "assignment_count",
                "operator_sequence",
            },
            f"Artifact {case.name}",
        )
        if (
            artifact_record.get("case") != case.name
            or artifact_record.get("compile_api")
            != "pypto.compiler.compile_structured_strict"
            or artifact_record.get("compiler_invocations") != 1
            or artifact_record.get("build_spec_identity_digest")
            != case.expected_build_spec_identity_digest
            or artifact_record.get("artifact_identity_digest")
            != case.expected_artifact_identity_digest
            or artifact_record.get("source_ir_digest") != case.expected_source_ir_digest
            or artifact_record.get("source_ir_bytes") != case.expected_source_ir_bytes
            or artifact_record.get("static_specialization_digest")
            != case.expected_static_specialization_digest
            or artifact_record.get("symbolic_specialization_digest")
            != case.expected_symbolic_specialization_digest
            or artifact_record.get("argument_abi_digest")
            != case.expected_argument_abi_digest
            or artifact_record.get("result_abi_digest")
            != case.expected_result_abi_digest
            or artifact_record.get("mutation_abi_digest")
            != case.expected_mutation_abi_digest
            or artifact_record.get("callable_abi_digest")
            != case.expected_callable_abi_digest
            or artifact_record.get("callable_abi_digest")
            != artifact_record.get("kernel_abi_identity_digest")
            or artifact_record.get("entry_function_name") != "pypto_fused_pointwise_v2"
            or artifact_record.get("fallback_used") is not False
            or artifact_record.get("expected_grid") != list(case.expected_grid)
            or artifact_record.get("expected_kernel_arguments")
            != case.expected_kernel_arguments
            or artifact_record.get("device_code_bytes")
            != case.expected_device_code_bytes
            or artifact_record.get("device_code_sha256")
            != case.expected_device_code_sha256
            or artifact_record.get("expected_device_code_bytes")
            != case.expected_device_code_bytes
            or artifact_record.get("expected_device_code_sha256")
            != case.expected_device_code_sha256
            or artifact_record.get("hir_sha256") != hir_record.get("sha256")
            or artifact_record.get("hir_bytes") != hir_record.get("bytes")
            or artifact_record.get("hir_roundtrip_exact") is not True
            or artifact_record.get("input_operand_count") != case.input_count
            or artifact_record.get("assignment_count") != case.assignment_count
            or artifact_record.get("operator_sequence") != list(case.operator_sequence)
        ):
            raise FinalizeError(f"Artifact {case.name} frontend join differs")
        for field in (
            "build_spec_identity_digest",
            "source_ir_digest",
            "callable_abi_digest",
            "static_specialization_digest",
            "symbolic_specialization_digest",
            "argument_abi_digest",
            "result_abi_digest",
            "mutation_abi_digest",
            "artifact_identity_digest",
            "cache_key_digest",
            "loader_compatibility_digest",
            "device_code_sha256",
        ):
            require_sha(artifact_record.get(field), f"Artifact {case.name} {field}")
        require_int(
            artifact_record.get("device_code_bytes"),
            f"Artifact {case.name} device code",
            positive=True,
        )
        artifact_identities[case.name] = str(
            artifact_record["artifact_identity_digest"]
        )

    observation = runtime["observation"]
    assert isinstance(observation, dict)
    context = observation["context_address"]
    context_id = observation["context_id"]
    executions = runtime.get("executions")
    expected_pairs = [
        (case, repetition) for case in cases for repetition in range(case.repetitions)
    ]
    if not isinstance(executions, list) or len(executions) != len(expected_pairs):
        raise FinalizeError("execution lifetime set is incomplete")
    for execution, (case, repetition) in zip(executions, expected_pairs, strict=True):
        execution = require_exact_keys(
            execution,
            {
                "case",
                "repetition",
                "lifetime_ordinal",
                "fresh_executable",
                "artifact_identity_digest",
                "dtype",
                "shape",
                "strides",
                "grid",
                "kernel_argument_count",
                "raw_current_stream",
                "raw_reference_stream",
                "non_default_stream",
                "distinct_nondefault_reference_stream",
                "reference_stream_synchronized_before_candidate",
                "reference_stream_policy",
                "candidate_stream_policy",
                "reference_compute_boundary",
                "capture_free_before",
                "capture_free_at_launch",
                "external_stream_synchronized",
                "expected_logical_bytes_sha256",
                "actual_logical_bytes_sha256",
                "input_hashes",
                "guard_hashes",
                "guard_elements",
                "input_unchanged",
                "guards_unchanged",
                "comparison",
                "comparison_passed",
                "packet_released_after_synchronization",
                "explicit_unload",
                "terminal_state",
                "bound_context_before_unload",
                "bound_context_id_before_unload",
                "bound_context_after_unload",
                "bound_context_id_after_unload",
            },
            f"execution {case.name}/{repetition}",
        )
        if (
            execution.get("case") != case.name
            or execution.get("repetition") != repetition
            or execution.get("lifetime_ordinal") != repetition
            or execution.get("fresh_executable") is not True
            or execution.get("artifact_identity_digest")
            != artifact_identities[case.name]
            or execution.get("dtype") != case.dtype
            or execution.get("shape") != list(case.shape)
            or execution.get("strides") != list(case.strides)
            or execution.get("grid") != list(case.expected_grid)
            or execution.get("kernel_argument_count") != case.expected_kernel_arguments
            or execution.get("non_default_stream") is not True
            or execution.get("distinct_nondefault_reference_stream") is not True
            or execution.get("reference_stream_synchronized_before_candidate")
            is not True
            or execution.get("reference_stream_policy")
            != contract.REFERENCE_STREAM_POLICY
            or execution.get("candidate_stream_policy")
            != contract.CANDIDATE_STREAM_POLICY
            or execution.get("reference_compute_boundary")
            != contract.REFERENCE_COMPUTE_BOUNDARY
            or execution.get("capture_free_before") is not True
            or execution.get("capture_free_at_launch") is not True
            or execution.get("external_stream_synchronized") is not True
            or execution.get("input_unchanged") is not True
            or execution.get("guard_elements") != contract.GUARD_ELEMENTS
            or execution.get("guards_unchanged") is not True
            or execution.get("comparison_passed") is not True
            or execution.get("packet_released_after_synchronization") is not True
            or execution.get("explicit_unload") is not True
            or execution.get("terminal_state") != "Unloaded"
            or execution.get("bound_context_before_unload") != context
            or execution.get("bound_context_id_before_unload") != context_id
            or execution.get("bound_context_after_unload") != 0
            or execution.get("bound_context_id_after_unload") != 0
        ):
            raise FinalizeError(
                f"execution {case.name}/{repetition} does not prove correctness"
            )
        stream = require_int(
            execution.get("raw_current_stream"),
            f"execution {case.name}/{repetition} stream",
            positive=True,
        )
        if stream in {1, 2}:
            raise FinalizeError(
                f"execution {case.name}/{repetition} used a default stream"
            )
        require_sha(
            execution.get("actual_logical_bytes_sha256"),
            f"execution {case.name}/{repetition} output",
        )
        require_sha(
            execution.get("expected_logical_bytes_sha256"),
            f"execution {case.name}/{repetition} reference",
        )
        reference_stream = require_int(
            execution.get("raw_reference_stream"),
            f"execution {case.name}/{repetition} reference stream",
            positive=True,
        )
        if reference_stream in {1, 2, stream}:
            raise FinalizeError(
                f"execution {case.name}/{repetition} reference stream differs"
            )
        input_hashes = execution.get("input_hashes")
        if not isinstance(input_hashes, list) or len(input_hashes) != case.input_count:
            raise FinalizeError(
                f"execution {case.name}/{repetition} input hashes differ"
            )
        for ordinal, record in enumerate(input_hashes):
            record = require_exact_keys(
                record,
                {"ordinal", "before_sha256", "after_sha256", "unchanged"},
                f"execution {case.name}/{repetition} input {ordinal}",
            )
            if (
                record.get("ordinal") != ordinal
                or record.get("before_sha256") != record.get("after_sha256")
                or record.get("unchanged") is not True
            ):
                raise FinalizeError(
                    f"execution {case.name}/{repetition} input {ordinal} mutated"
                )
            require_sha(record.get("before_sha256"), "individual input hash")
        guards = execution.get("guard_hashes")
        if not isinstance(guards, list) or len(guards) != case.input_count + 1:
            raise FinalizeError(f"execution {case.name}/{repetition} guards differ")
        for ordinal, guard in enumerate(guards):
            guard = require_exact_keys(
                guard,
                {
                    "allocation",
                    "prefix_before_sha256",
                    "prefix_after_sha256",
                    "suffix_before_sha256",
                    "suffix_after_sha256",
                    "unchanged",
                },
                f"execution {case.name}/{repetition} guard",
            )
            if ordinal < case.input_count:
                expected_allocation = f"input{ordinal}"
                expected_prefix = expected_guard_sha256(
                    case.dtype, contract.INPUT_GUARD_PREFIX_BASE + ordinal
                )
                expected_suffix = expected_guard_sha256(
                    case.dtype, contract.INPUT_GUARD_SUFFIX_BASE + ordinal
                )
            else:
                expected_allocation = "output"
                expected_prefix = expected_guard_sha256(
                    case.dtype, contract.OUTPUT_GUARD_PREFIX
                )
                expected_suffix = expected_guard_sha256(
                    case.dtype, contract.OUTPUT_GUARD_SUFFIX
                )
            if (
                guard.get("allocation") != expected_allocation
                or guard.get("prefix_before_sha256") != guard.get("prefix_after_sha256")
                or guard.get("prefix_before_sha256") != expected_prefix
                or guard.get("suffix_before_sha256") != guard.get("suffix_after_sha256")
                or guard.get("suffix_before_sha256") != expected_suffix
                or guard.get("unchanged") is not True
            ):
                raise FinalizeError(f"execution {case.name}/{repetition} guard mutated")
            for field in (
                "prefix_before_sha256",
                "prefix_after_sha256",
                "suffix_before_sha256",
                "suffix_after_sha256",
            ):
                require_sha(guard.get(field), f"execution guard {field}")
        comparison = require_exact_keys(
            execution.get("comparison"),
            {
                "policy",
                "max_ulp_limit",
                "rtol",
                "atol",
                "observed_max_ulp",
                "observed_max_relative_error",
                "observed_max_absolute_error",
                "special_classification_and_sign_passed",
                "negative_zero_fma_discriminator_passed",
                "no_subnormals",
            },
            f"execution {case.name}/{repetition} comparison",
        )
        if (
            comparison.get("policy") != case.comparison
            or comparison.get("max_ulp_limit") != case.max_ulp
            or comparison.get("rtol") != case.rtol
            or comparison.get("atol") != case.atol
            or comparison.get("special_classification_and_sign_passed") is not True
            or comparison.get("negative_zero_fma_discriminator_passed")
            != (case.family == "arithmetic")
            or comparison.get("no_subnormals") is not True
        ):
            raise FinalizeError(f"execution {case.name}/{repetition} policy differs")
        observed_ulp = require_int(
            comparison.get("observed_max_ulp"), "observed ULP distance"
        )
        observed_errors = (
            comparison.get("observed_max_relative_error"),
            comparison.get("observed_max_absolute_error"),
        )
        if (
            observed_ulp < 0
            or observed_ulp > case.max_ulp
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in observed_errors
            )
        ):
            raise FinalizeError(
                f"execution {case.name}/{repetition} error metric differs"
            )


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(value)))[0]


def _bf16_word(value: float) -> int:
    bits = _f32_word(value)
    exponent = bits & 0x7F800000
    fraction = bits & 0x007FFFFF
    if exponent == 0x7F800000 and fraction:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _word_value(word: int, dtype: str) -> float:
    bits = word if dtype == "float32" else word << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _quantize(value: float, dtype: str) -> float:
    word = _f32_word(value) if dtype == "float32" else _bf16_word(value)
    return _word_value(word, dtype)


def _encode_words(values: list[float], dtype: str) -> list[int]:
    return [
        _f32_word(value) if dtype == "float32" else _bf16_word(value)
        for value in values
    ]


def _decode_words(raw: bytes, dtype: str) -> list[int]:
    width, code = (4, "I") if dtype == "float32" else (2, "H")
    if not raw or len(raw) % width:
        raise FinalizeError("numerical replay tensor byte width differs")
    return list(struct.unpack(f"<{len(raw) // width}{code}", raw))


def pack_numerical_words(words: list[int], dtype: str) -> bytes:
    code = "I" if dtype == "float32" else "H"
    return struct.pack(f"<{len(words)}{code}", *words)


def _classify(word: int, dtype: str) -> tuple[str, int]:
    fraction_bits = 23 if dtype == "float32" else 7
    sign = word >> (fraction_bits + 8)
    exponent = (word >> fraction_bits) & 0xFF
    fraction = word & ((1 << fraction_bits) - 1)
    if exponent == 0xFF:
        return ("nan" if fraction else "inf", sign)
    if exponent == 0:
        return ("zero" if fraction == 0 else "subnormal", sign)
    return ("finite", sign)


def _ordered(word: int, dtype: str) -> int:
    bits = 32 if dtype == "float32" else 16
    sign = 1 << (bits - 1)
    return (~word & ((1 << bits) - 1)) if word & sign else word | sign


def _binary(left: float, right: float, dtype: str, operation: str) -> float:
    left32, right32 = _f32(left), _f32(right)
    if operation == "add":
        result = left32 + right32
    elif operation == "sub":
        result = left32 - right32
    elif operation == "mul":
        result = left32 * right32
    else:
        raise FinalizeError(f"unsupported CPU replay operation: {operation}")
    return _quantize(_f32(result), dtype)


def _input_words(case: object, repetition: int, ordinal: int) -> list[int]:
    count = 1
    for dimension in case.shape:
        count *= dimension
    values: list[float] = []
    for index in range(count):
        if case.family == "arithmetic":
            power = 23 if case.dtype == "float32" else 7
            if index == 0:
                value = (
                    1.0 + 2.0**-power,
                    -(1.0 + 2.0 ** (1 - power)),
                    1.0,
                    0.0,
                )[ordinal]
            elif ordinal == 0:
                value = (((17 * index + 3 * repetition) % 31) - 15) / 16
            elif ordinal == 1:
                value = (((11 * index + 5 * repetition) % 29) - 14) / 16
            elif ordinal == 2:
                value = 1 + ((((7 * index + repetition) % 5) - 2) / 8)
            else:
                value = (((13 * index + 3 * repetition) % 17) - 8) / 32
        elif case.family == "exp":
            value = (((13 * index + 5) % 65) - 32) / 8
        elif case.family == "recip":
            value = (-1.0 if index % 2 else 1.0) * ((1 + ((7 * index + 3) % 63)) / 8)
            if index < 4:
                value = (0.125, -0.5, 2.0, -8.0)[index]
        elif case.family == "rsqrt":
            cycle = (
                1 / 64,
                1 / 32,
                1 / 16,
                1 / 8,
                1 / 4,
                1 / 2,
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
            )
            value = cycle[index % len(cycle)]
        elif case.family == "maximum-boundary":
            value = (((17 * index + 13 * ordinal + 7 * repetition) % 127) - 63) / 256
        else:
            raise FinalizeError(f"unsupported CPU input family: {case.family}")
        values.append(float(value))
    if repetition == 1 and case.family == "exp":
        values[:5] = [-math.inf, math.inf, math.nan, -0.0, 0.0]
    elif repetition == 1 and case.family == "recip":
        values[:5] = [0.0, -0.0, math.inf, -math.inf, math.nan]
    elif repetition == 1 and case.family == "rsqrt":
        values[:7] = [-1.0, -math.inf, math.nan, -0.0, 0.0, math.inf, 4.0]
    return _encode_words([_quantize(value, case.dtype) for value in values], case.dtype)


def _reciprocal(value: float) -> float:
    if math.isnan(value):
        return math.nan
    if value == 0.0:
        return math.copysign(math.inf, value)
    if math.isinf(value):
        return math.copysign(0.0, value)
    return 1.0 / value


def _rsqrt(value: float) -> float:
    if math.isnan(value) or value < 0.0:
        return math.nan
    if value == 0.0:
        return math.copysign(math.inf, value)
    if math.isinf(value):
        return 0.0
    return 1.0 / math.sqrt(value)


def _cpu_reference_words(case: object, inputs: list[list[int]]) -> list[int]:
    values = [[_word_value(word, case.dtype) for word in source] for source in inputs]
    if case.family == "arithmetic":
        result = [_binary(left, left, case.dtype, "mul") for left in values[0]]
        result = [
            _binary(left, right, case.dtype, "add")
            for left, right in zip(result, values[1], strict=True)
        ]
        scale, offset, subtract = [
            _quantize(value, case.dtype) for value in case.scalar_literals
        ]
        for operation, scalar in (("mul", scale), ("add", offset), ("sub", subtract)):
            result = [_binary(value, scalar, case.dtype, operation) for value in result]
        result = [
            _binary(left, right, case.dtype, "mul")
            for left, right in zip(result, values[2], strict=True)
        ]
        result = [
            _binary(left, right, case.dtype, "sub")
            for left, right in zip(result, values[3], strict=True)
        ]
        words = _encode_words(result, case.dtype)
        sign = 0x80000000 if case.dtype == "float32" else 0x8000
        return [word ^ sign for word in words]
    if case.family == "exp":
        results = [
            math.exp(value) if not (math.isinf(value) and value < 0) else 0.0
            for value in values[0]
        ]
        return _encode_words(results, case.dtype)
    if case.family == "recip":
        return _encode_words([_reciprocal(value) for value in values[0]], case.dtype)
    if case.family == "rsqrt":
        return _encode_words([_rsqrt(value) for value in values[0]], case.dtype)
    if case.family == "maximum-boundary":
        result = list(values[0])
        for source in values[1:]:
            result = [
                _binary(left, right, case.dtype, "add")
                for left, right in zip(result, source, strict=True)
            ]
        words = _encode_words(result, case.dtype)
        sign = 0x80000000
        for _ in range(49):
            words = [word ^ sign for word in words]
        return words
    raise FinalizeError(f"unsupported CPU reference family: {case.family}")


def _compare_words(
    case: object, actual: list[int], reference: list[int]
) -> dict[str, object]:
    if len(actual) != len(reference):
        raise FinalizeError("numerical replay cardinality differs")
    max_ulp = 0
    max_relative = 0.0
    max_absolute = 0.0
    for index, (actual_word, reference_word) in enumerate(
        zip(actual, reference, strict=True)
    ):
        actual_class = _classify(actual_word, case.dtype)
        reference_class = _classify(reference_word, case.dtype)
        if "subnormal" in {actual_class[0], reference_class[0]}:
            raise FinalizeError("numerical replay contains a subnormal")
        if reference_class[0] != "finite":
            if reference_class[0] == "nan":
                if actual_class[0] != "nan":
                    raise FinalizeError(f"NaN classification differs at {index}")
            elif actual_class != reference_class:
                raise FinalizeError(f"special sign/classification differs at {index}")
            continue
        if actual_class[0] != "finite":
            raise FinalizeError(f"finite classification differs at {index}")
        if case.comparison.startswith("exact"):
            if actual_word != reference_word:
                raise FinalizeError(f"exact numerical replay differs at {index}")
            continue
        ulp = abs(
            _ordered(actual_word, case.dtype) - _ordered(reference_word, case.dtype)
        )
        actual_value = _word_value(actual_word, case.dtype)
        reference_value = _word_value(reference_word, case.dtype)
        absolute = abs(actual_value - reference_value)
        relative = absolute / abs(reference_value)
        if (
            ulp > case.max_ulp
            or absolute > case.rtol * abs(reference_value) + case.atol
        ):
            raise FinalizeError(f"numerical replay tolerance differs at {index}")
        max_ulp = max(max_ulp, ulp)
        max_relative = max(max_relative, relative)
        max_absolute = max(max_absolute, absolute)
    return {
        "observed_max_ulp": max_ulp,
        "observed_max_relative_error": max_relative,
        "observed_max_absolute_error": max_absolute,
    }


def audit_numerical_replay(
    provisional: dict[str, object], workspace: Path, run_id: str
) -> list[dict[str, object]]:
    replay = contract.replay_directory(workspace, run_id)
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    executions = runtime["executions"]
    assert isinstance(executions, list)
    output: list[dict[str, object]] = []
    execution_index = 0
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            execution = executions[execution_index]
            execution_index += 1
            assert isinstance(execution, dict)
            inputs: list[list[int]] = []
            for ordinal in range(case.input_count):
                raw = (
                    replay / f"{case.name}.r{repetition}.input{ordinal}.bin"
                ).read_bytes()
                words = _decode_words(raw, case.dtype)
                if words != _input_words(case, repetition, ordinal):
                    raise FinalizeError(
                        f"CPU input reconstruction differs: {case.name}/{repetition}/{ordinal}"
                    )
                hashes = execution["input_hashes"]
                assert isinstance(hashes, list) and isinstance(hashes[ordinal], dict)
                if sha256_bytes(raw) != hashes[ordinal].get("before_sha256"):
                    raise FinalizeError("raw input/execution hash join differs")
                inputs.append(words)
            reference_raw = (
                replay / f"{case.name}.r{repetition}.reference.bin"
            ).read_bytes()
            actual_raw = (replay / f"{case.name}.r{repetition}.actual.bin").read_bytes()
            reference_words = _decode_words(reference_raw, case.dtype)
            actual_words = _decode_words(actual_raw, case.dtype)
            if sha256_bytes(reference_raw) != execution.get(
                "expected_logical_bytes_sha256"
            ) or sha256_bytes(actual_raw) != execution.get(
                "actual_logical_bytes_sha256"
            ):
                raise FinalizeError("raw output/execution hash join differs")
            cpu_words = _cpu_reference_words(case, inputs)
            torch_vs_cpu = _compare_words(case, reference_words, cpu_words)
            candidate_vs_torch = _compare_words(case, actual_words, reference_words)
            candidate_vs_cpu = _compare_words(case, actual_words, cpu_words)
            comparison = execution["comparison"]
            assert isinstance(comparison, dict)
            if any(
                comparison.get(name) != value
                for name, value in candidate_vs_torch.items()
            ):
                raise FinalizeError("recorded/reconstructed numerical metrics differ")
            cpu_raw = pack_numerical_words(cpu_words, case.dtype)
            output.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "independent_cpu_input_reconstruction": True,
                    "independent_cpu_reference_reconstruction": True,
                    "actual_sha256": sha256_bytes(actual_raw),
                    "candidate_vs_cpu": candidate_vs_cpu,
                    "candidate_vs_torch": candidate_vs_torch,
                    "cpu_reference_sha256": sha256_bytes(cpu_raw),
                    "torch_reference_sha256": sha256_bytes(reference_raw),
                    "torch_vs_cpu": torch_vs_cpu,
                }
            )
    return output


def validate_replay(
    provisional: dict[str, object], workspace: Path, run_id: str
) -> list[dict[str, object]]:
    inputs = provisional["inputs"]
    assert isinstance(inputs, dict)
    records = inputs.get("replay_files")
    expected_names = ["compile-request.msgpack"]
    for case in contract.CASE_SPECS:
        expected_names.extend(
            [
                f"{case.name}.hir.msgpack",
                f"{case.name}.source.mlir",
                f"{case.name}.build-spec.msgpack",
                f"{case.name}.artifact.msgpack",
                f"{case.name}.cubin",
            ]
        )
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            expected_names.extend(
                f"{case.name}.r{repetition}.input{ordinal}.bin"
                for ordinal in range(case.input_count)
            )
            expected_names.extend(
                [
                    f"{case.name}.r{repetition}.reference.bin",
                    f"{case.name}.r{repetition}.actual.bin",
                ]
            )
    if not isinstance(records, list) or len(records) != len(expected_names):
        raise FinalizeError("replay file set is incomplete")
    replay = contract.replay_directory(workspace, run_id)
    if (
        replay.is_symlink()
        or not replay.is_dir()
        or replay.absolute() != replay.resolve(strict=True)
    ):
        raise FinalizeError("replay directory is not canonical")
    actual_names = sorted(path.name for path in replay.iterdir())
    expected_directory_names = sorted([*expected_names, contract.PROVISIONAL_NAME])
    if actual_names != expected_directory_names:
        raise FinalizeError("replay directory has a missing or extra file")
    normalized: list[dict[str, object]] = []
    for record, name in zip(records, expected_names, strict=True):
        record = require_exact_keys(
            record, {"path", "bytes", "sha256"}, f"replay {name}"
        )
        path = replay / name
        if record.get("path") != path.relative_to(workspace).as_posix():
            raise FinalizeError(f"replay path differs: {name}")
        resolved = require_regular(path, workspace, f"replay {name}")
        size = require_int(record.get("bytes"), f"replay {name} size", positive=True)
        digest = require_sha(record.get("sha256"), f"replay {name}")
        if (
            resolved.stat().st_size != size
            or stat.S_IMODE(resolved.stat().st_mode) != 0o444
            or sha256_file(resolved) != digest
        ):
            raise FinalizeError(f"replay bytes differ: {name}")
        normalized.append(dict(record))
    return normalized


def audit_replay_semantics(
    provisional: dict[str, object], workspace: Path, run_id: str
) -> dict[str, object]:
    python = (workspace / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True)
    replay = contract.replay_directory(workspace, run_id)
    command = [
        str(python),
        "-I",
        "-B",
        "-S",
        "-c",
        REPLAY_AUDIT_PROGRAM,
        str(workspace),
        str(replay),
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise FinalizeError("CPU-only replay audit failed: " + completed.stderr[-2048:])
    try:
        audited = json.loads(completed.stdout, object_pairs_hook=duplicate_key_guard)
    except json.JSONDecodeError as error:
        raise FinalizeError("CPU-only replay audit output is not JSON") from error
    audited = require_exact_keys(
        audited,
        {"compile_request", "target_info", "hir_programs", "artifacts"},
        "CPU-only replay audit",
    )
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    observation = runtime["observation"]
    assert isinstance(observation, dict)
    target_fields = (
        "device_ordinal",
        "device_name",
        "device_uuid",
        "pci_device_id",
        "traits",
        "cuda_toolkit_version",
        "cuda_driver_version",
        "tensor_ir_revision",
        "cuda_tile_revision",
        "supported_compute_dtypes",
    )
    expected_target = {name: observation[name] for name in target_fields}
    if audited.get("compile_request") != runtime.get("compile_request"):
        raise FinalizeError("replayed CompileRequest differs")
    if audited.get("target_info") != expected_target:
        raise FinalizeError("replayed TargetInfo differs")
    expected_hir = [
        {
            "case": record["case"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "canonical_reserialization_equal": True,
            "parameter_directions": ["In"] * case.input_count,
            "input_count": case.input_count,
            "assignment_count": case.assignment_count,
            "operator_sequence": list(case.operator_sequence),
        }
        for record, case in zip(
            runtime["hir_programs"], contract.CASE_SPECS, strict=True
        )
    ]
    if audited.get("hir_programs") != expected_hir:
        raise FinalizeError("replayed HIR differs")
    artifact_fields = (
        "case",
        "build_spec_identity_digest",
        "source_ir_digest",
        "source_ir_bytes",
        "callable_abi_digest",
        "static_specialization_digest",
        "symbolic_specialization_digest",
        "argument_abi_digest",
        "result_abi_digest",
        "mutation_abi_digest",
        "artifact_identity_digest",
        "cache_key_digest",
        "loader_compatibility_digest",
        "device_code_bytes",
        "device_code_sha256",
        "kernel_abi_identity_digest",
        "entry_function_name",
        "fallback_used",
        "input_operand_count",
        "assignment_count",
        "operator_sequence",
    )
    expected_artifacts = [
        {name: record[name] for name in artifact_fields}
        for record in runtime["artifacts"]
    ]
    if audited.get("artifacts") != expected_artifacts:
        raise FinalizeError("replayed Artifact semantics differ")
    return {
        "command_sha256": sha256_bytes("\0".join(command).encode()),
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        **audited,
    }


def validate_integrity(
    provisional: dict[str, object],
    workspace: Path,
    control_identity: dict[str, object],
) -> None:
    inputs = provisional["inputs"]
    assert isinstance(inputs, dict)
    if (
        inputs.get("control_manifest") != control_identity
        or inputs.get("pypto")
        != {"head": contract.PYPTO_HEAD, "tree": contract.PYPTO_TREE, "clean": True}
        or inputs.get("tensor_ir_head") != contract.TENSOR_IR_HEAD
        or inputs.get("cuda_tile_head") != contract.CUDA_TILE_HEAD
        or inputs.get("llvm_head") != contract.LLVM_HEAD
    ):
        raise FinalizeError("provisional source/control identity differs")
    expected_paths = {
        "anchor_generator": workspace / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
        "compile_anchors": workspace / contract.COMPILE_ANCHORS_RELATIVE_PATH,
        "contract": workspace / "tools/_pypto_fused_pointwise_sm120_contract.py",
        "runner": workspace / contract.RUNNER_RELATIVE_PATH,
        "controller": workspace / "tools/run_pypto_fused_pointwise_sm120_isolated.py",
        "environment_lock": workspace / "ENVIRONMENT.lock",
        "versions_lock": workspace / "VERSIONS.lock",
        "workspace_lock": workspace / "WORKSPACE.lock",
        "pypto_dso": workspace / contract.PYPTO_DSO_RELATIVE_PATH,
        "cuda_runtime": workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
    }
    integrity = inputs["integrity"]
    assert isinstance(integrity, dict)
    for name, path in expected_paths.items():
        resolved = require_regular(path, workspace, f"integrity {name}")
        expected = {
            "path": resolved.relative_to(workspace).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        if integrity.get(name) != expected:
            raise FinalizeError(f"provisional integrity differs: {name}")


def validate_audit(
    value: object,
    description: str,
    *,
    authorized: bool,
    require_zero_owned: bool,
) -> None:
    value = require_exact_keys(
        value,
        {
            "owned_nvidia_compute_pids",
            "external_nvidia_compute_pids",
            "protected_nvidia_compute_pids",
            "protected_nvidia_runtime_mapping_pids",
            "unreadable_protected_maps",
            "protected_heavy_pids",
            "protected_cpu_lane_authorized",
            "free_memory_mib",
            "gpu",
        },
        description,
    )
    if (
        value.get("external_nvidia_compute_pids") != []
        or value.get("protected_nvidia_compute_pids") != []
        or value.get("protected_nvidia_runtime_mapping_pids") != []
        or value.get("unreadable_protected_maps") != []
        or value.get("protected_cpu_lane_authorized") is not authorized
    ):
        raise FinalizeError(f"{description} does not prove NVIDIA isolation")
    owned = value.get("owned_nvidia_compute_pids")
    if not isinstance(owned, list) or (require_zero_owned and owned):
        raise FinalizeError(f"{description} owned compute set differs")
    if (
        require_int(
            value.get("free_memory_mib"),
            f"{description} free GPU memory",
            positive=True,
        )
        < 4096
    ):
        raise FinalizeError(f"{description} is below the GPU memory floor")
    gpu = require_exact_keys(
        value.get("gpu"),
        {"name", "compute_capability", "memory_mib", "used_mib", "driver"},
        f"{description} GPU identity",
    )
    if (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != "12.0"
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        raise FinalizeError(f"{description} GPU identity differs")


def validate_preflight(
    value: dict[str, object], process: dict[str, object], workspace: Path
) -> None:
    require_exact_keys(
        value,
        {
            "coexistence_policy_version",
            "cwd",
            "failures",
            "gpu",
            "gpu_smoke_free_memory_floor_mib",
            "gpu_smoke_policy_version",
            "mem_available_kib",
            "memory_floor_kib",
            "mode",
            "nvidia_compute_audit_ok",
            "nvidia_compute_pids",
            "ok",
            "policy",
            "policy_version",
            "protected_activity_waiver_applied",
            "protected_cpu_only_coexistence_requested",
            "protected_gpu_smoke_waiver_applied",
            "protected_heavy_processes",
            "protected_nvidia_compute_pids",
            "protected_nvidia_runtime_mapping_pids",
            "protected_processes",
            "protected_zero_nvidia_gpu_smoke_requested",
            "torch",
            "unreadable_protected_maps",
            "workspace",
            "workspace_processes",
        },
        "preflight",
    )
    protected = value.get("protected_processes")
    protected_heavy = value.get("protected_heavy_processes")
    requested = value.get("protected_zero_nvidia_gpu_smoke_requested")
    waiver = value.get("protected_gpu_smoke_waiver_applied")
    if not isinstance(protected, list) or not isinstance(protected_heavy, list):
        raise FinalizeError("preflight protected process inventory is malformed")
    if requested is True:
        expected_waiver = bool(protected)
    elif requested is False:
        expected_waiver = False
        if protected_heavy:
            raise FinalizeError("exclusive GPU smoke has a protected heavy process")
    else:
        raise FinalizeError("preflight GPU-smoke authorization is malformed")
    if (
        value.get("policy_version") != 3
        or value.get("gpu_smoke_policy_version") != contract.GPU_SMOKE_POLICY_VERSION
        or value.get("mode") != "gpu-smoke"
        or value.get("workspace") != str(workspace)
        or value.get("cwd") != str(workspace)
        or value.get("ok") is not True
        or value.get("nvidia_compute_audit_ok") is not True
        or waiver is not expected_waiver
        or value.get("protected_nvidia_compute_pids") != []
        or value.get("protected_nvidia_runtime_mapping_pids") != []
        or value.get("unreadable_protected_maps") != []
        or value.get("nvidia_compute_pids") != []
        or value.get("failures") != []
    ):
        raise FinalizeError("preflight does not prove zero-NVIDIA coexistence")
    gpu = require_exact_keys(
        value.get("gpu"),
        {"name", "compute_capability", "memory_mib", "used_mib", "driver"},
        "preflight GPU identity",
    )
    try:
        free_memory_mib = int(str(gpu["memory_mib"])) - int(str(gpu["used_mib"]))
    except ValueError as error:
        raise FinalizeError("preflight GPU memory identity is malformed") from error
    if (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != "12.0"
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
        or free_memory_mib < 4096
        or value.get("gpu_smoke_free_memory_floor_mib") != 4096
        or value.get("memory_floor_kib") not in {24 * 1024 * 1024, 32 * 1024 * 1024}
    ):
        raise FinalizeError("preflight GPU identity or resource floor differs")
    torch = require_exact_keys(
        value.get("torch"),
        {
            "source",
            "environment_lock_sha256",
            "version",
            "git_version",
            "cuda",
            "hip",
            "python_executable",
            "libcudart_path",
            "libcudart_size",
            "libcudart_sha256",
            "libcudart_record_owned",
            "nvidia_runtime_mappings",
            "cuda_initialized",
            "forbidden_dsos",
        },
        "preflight static Torch identity",
    )
    if (
        torch.get("version") != contract.EXPECTED_TORCH_VERSION
        or torch.get("git_version") != contract.EXPECTED_TORCH_GIT
        or torch.get("cuda") != contract.EXPECTED_TORCH_CUDA
        or torch.get("hip") is not None
        or torch.get("cuda_initialized") is not False
        or torch.get("nvidia_runtime_mappings") != []
        or torch.get("environment_lock_sha256") != contract.ENVIRONMENT_LOCK_SHA256
        or torch.get("python_executable")
        != str((workspace / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True))
        or torch.get("libcudart_path")
        != str((workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))
        or torch.get("libcudart_sha256") != contract.CUDA_RUNTIME_SHA256
        or torch.get("libcudart_size") != contract.CUDA_RUNTIME_SIZE
        or torch.get("libcudart_record_owned") is not True
    ):
        raise FinalizeError("preflight static Torch/libcudart identity differs")
    gpu_smoke = process.get("gpu_smoke")
    if not isinstance(gpu_smoke, dict) or (
        gpu_smoke.get("policy_version") != contract.GPU_SMOKE_POLICY_VERSION
        or gpu_smoke.get("requested") is not requested
        or gpu_smoke.get("waiver_applied") is not expected_waiver
        or gpu_smoke.get("authorization")
        != (contract.GPU_SMOKE_AUTHORIZATION if requested else None)
    ):
        raise FinalizeError("process GPU-smoke authorization is malformed")


def validate_process_schema(process: dict[str, object], workspace: Path) -> None:
    require_exact_keys(
        process,
        {
            "schema",
            "run_id",
            "workspace",
            "environment",
            "environment_access_lock",
            "framework_profile",
            "framework_launch",
            "mode",
            "coexistence",
            "gpu_smoke",
            "preflight",
            "resource_policy",
            "command",
            "pid",
            "pgid",
            "start_ticks",
            "started_at",
            "status",
            "gpu_smoke_pre_release_audit",
            "gpu_smoke_last_audit",
            "gpu_smoke_post_exit_audit",
            "return_code",
            "finished_at",
        },
        "process metadata",
    )
    environment_lock = require_exact_keys(
        process.get("environment_access_lock"),
        {"path", "mode", "device", "inode"},
        "environment access lock",
    )
    if (
        process.get("environment") != str(workspace / "envs/pypto-nvidia")
        or environment_lock.get("path")
        != str(workspace / "runs/environment-pypto-nvidia.lock")
        or environment_lock.get("mode") != "shared"
    ):
        raise FinalizeError("process environment or shared lock differs")
    require_int(
        environment_lock.get("device"), "environment lock device", positive=True
    )
    require_int(environment_lock.get("inode"), "environment lock inode", positive=True)
    resource = require_exact_keys(
        process.get("resource_policy"),
        {"timeout_seconds", "minimum_free_disk_bytes", "owned_run_pause_memory_kib"},
        "process resource policy",
    )
    if (
        resource.get("timeout_seconds") != contract.GPU_SMOKE_TIMEOUT_SECONDS
        or resource.get("minimum_free_disk_bytes")
        != contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
        or resource.get("owned_run_pause_memory_kib") != 16 * 1024 * 1024
    ):
        raise FinalizeError("process resource policy differs")
    if (
        not isinstance(process.get("started_at"), str)
        or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", str(process["started_at"])) is None
        or not isinstance(process.get("finished_at"), str)
        or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", str(process["finished_at"])) is None
    ):
        raise FinalizeError("process start or finish timestamp is malformed")
    gpu_smoke = require_exact_keys(
        process.get("gpu_smoke"),
        {
            "policy_version",
            "requested",
            "waiver_applied",
            "authorization",
            "start_barrier_path",
            "gate_path",
            "memory_floor_kib",
            "gpu_free_memory_floor_mib",
            "protected_heavy_processes",
            "protected_nvidia_compute_pids",
            "protected_nvidia_runtime_mapping_pids",
            "unreadable_protected_maps",
            "gate_sha256",
            "start_barrier_sha256",
            "release_authorized_at",
        },
        "process GPU-smoke policy",
    )
    expected_memory_floor = (
        24 * 1024 * 1024 if gpu_smoke.get("requested") is True else 32 * 1024 * 1024
    )
    if (
        gpu_smoke.get("policy_version") != contract.GPU_SMOKE_POLICY_VERSION
        or gpu_smoke.get("memory_floor_kib") != expected_memory_floor
        or gpu_smoke.get("gpu_free_memory_floor_mib") != 4096
        or gpu_smoke.get("protected_nvidia_compute_pids") != []
        or gpu_smoke.get("protected_nvidia_runtime_mapping_pids") != []
        or gpu_smoke.get("unreadable_protected_maps") != []
    ):
        raise FinalizeError("process GPU-smoke safety policy differs")
    coexistence = require_exact_keys(
        process.get("coexistence"),
        {
            "policy_version",
            "requested",
            "waiver_applied",
            "memory_floor_kib",
            "protected_heavy_processes",
            "protected_nvidia_compute_pids",
        },
        "CPU-only coexistence metadata",
    )
    if (
        coexistence.get("requested") is not False
        or coexistence.get("waiver_applied") is not False
    ):
        raise FinalizeError("GPU smoke cannot use the CPU-only coexistence policy")


def validate_run_documents(
    *,
    workspace: Path,
    run_id: str,
    process: dict[str, object],
    preflight: dict[str, object],
    preflight_raw: bytes,
    gate: dict[str, object],
    gate_raw: bytes,
    barrier: dict[str, object],
    barrier_raw: bytes,
    provisional: dict[str, object],
    control_identity: dict[str, object],
) -> None:
    expected_process_keys = {
        "schema",
        "run_id",
        "workspace",
        "environment",
        "environment_access_lock",
        "framework_profile",
        "framework_launch",
        "mode",
        "coexistence",
        "gpu_smoke",
        "preflight",
        "resource_policy",
        "command",
        "pid",
        "pgid",
        "start_ticks",
        "started_at",
        "status",
        "gpu_smoke_pre_release_audit",
        "gpu_smoke_last_audit",
        "gpu_smoke_post_exit_audit",
        "return_code",
        "finished_at",
    }
    require_exact_keys(process, expected_process_keys, "process metadata")
    validate_process_schema(process, workspace)
    if (
        process.get("schema") != 3
        or process.get("run_id") != run_id
        or process.get("workspace") != str(workspace)
        or process.get("environment") != str(workspace / "envs/pypto-nvidia")
        or process.get("framework_profile") != "pypto"
        or process.get("framework_launch") is not False
        or process.get("mode") != "gpu-smoke"
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("command") != contract.fixed_child_command(workspace)
    ):
        raise FinalizeError("completed process metadata differs")
    resource = process.get("resource_policy")
    if not isinstance(resource, dict) or (
        resource.get("timeout_seconds") != contract.GPU_SMOKE_TIMEOUT_SECONDS
        or resource.get("minimum_free_disk_bytes")
        != contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
    ):
        raise FinalizeError("process resource policy differs")
    preflight_anchor = process.get("preflight")
    preflight_path = workspace / "runs" / run_id / "preflight.json"
    if preflight_anchor != {
        "path": str(preflight_path),
        "sha256": sha256_bytes(preflight_raw),
    }:
        raise FinalizeError("process/preflight digest join differs")
    validate_preflight(preflight, process, workspace)
    requested = preflight.get("protected_zero_nvidia_gpu_smoke_requested")
    if requested not in {True, False} or (
        preflight.get("mode") != "gpu-smoke"
        or preflight.get("ok") is not True
        or preflight.get("failures") != []
        or preflight.get("nvidia_compute_audit_ok") is not True
        or preflight.get("nvidia_compute_pids") != []
        or preflight.get("protected_nvidia_compute_pids") != []
        or preflight.get("protected_nvidia_runtime_mapping_pids") != []
        or preflight.get("unreadable_protected_maps") != []
    ):
        raise FinalizeError("preflight does not prove zero-NVIDIA isolation")
    validate_audit(
        process.get("gpu_smoke_pre_release_audit"),
        "pre-release audit",
        authorized=bool(requested),
        require_zero_owned=True,
    )
    validate_audit(
        process.get("gpu_smoke_last_audit"),
        "periodic audit",
        authorized=bool(requested),
        require_zero_owned=False,
    )
    validate_audit(
        process.get("gpu_smoke_post_exit_audit"),
        "post-exit audit",
        authorized=bool(requested),
        require_zero_owned=True,
    )
    gate_path = workspace / "runs" / run_id / "gpu-smoke-gate.json"
    barrier_path = workspace / "runs" / run_id / "gpu-smoke-start-barrier.json"
    gpu_smoke = process.get("gpu_smoke")
    if not isinstance(gpu_smoke, dict) or (
        gpu_smoke.get("gate_path") != str(gate_path)
        or gpu_smoke.get("start_barrier_path") != str(barrier_path)
        or gpu_smoke.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
        or gpu_smoke.get("requested") is not requested
    ):
        raise FinalizeError("process gate/barrier policy differs")
    identity = {
        "schema": contract.GPU_SMOKE_POLICY_VERSION,
        "run_id": run_id,
        "pid": process.get("pid"),
        "pgid": process.get("pgid"),
        "start_ticks": process.get("start_ticks"),
    }
    require_exact_keys(
        barrier,
        {"schema", "run_id", "pid", "pgid", "start_ticks", "gate_path", "gate_sha256"},
        "start barrier",
    )
    require_exact_keys(
        gate,
        {
            "schema",
            "run_id",
            "pid",
            "pgid",
            "start_ticks",
            "command",
            "preflight",
            "static_identity",
            "control_manifest",
            "runtime_isolation",
        },
        "pre-release gate",
    )
    if any(barrier.get(name) != value for name, value in identity.items()) or (
        barrier.get("gate_path") != str(gate_path)
        or barrier.get("gate_sha256") != sha256_bytes(gate_raw)
    ):
        raise FinalizeError("start barrier identity differs")
    if any(gate.get(name) != value for name, value in identity.items()) or (
        gate.get("command") != contract.fixed_child_command(workspace)
        or gate.get("preflight") != preflight_anchor
        or gate.get("static_identity") != preflight.get("torch")
        or gate.get("control_manifest") != control_identity
        or gate.get("runtime_isolation") != process.get("gpu_smoke_pre_release_audit")
    ):
        raise FinalizeError("pre-release gate identity differs")
    run = provisional.get("run_context")
    if not isinstance(run, dict) or (
        any(
            run.get(name) != value
            for name, value in identity.items()
            if name != "schema"
        )
        or run.get("mode") != "gpu-smoke"
        or run.get("protected_zero_nvidia_policy") is not requested
        or run.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
        or run.get("preflight")
        != {
            "path": preflight_path.relative_to(workspace).as_posix(),
            "sha256": sha256_bytes(preflight_raw),
        }
        or run.get("gate")
        != {
            "path": str(gate_path),
            "sha256": sha256_bytes(gate_raw),
            "document": gate,
        }
    ):
        raise FinalizeError("provisional run context differs")


def finalize(
    *, workspace: Path, run_id: str, expected_provisional_sha256: str
) -> tuple[dict[str, object], Path, str]:
    workspace = require_workspace(workspace)
    require_no_site_finalizer()
    control_identity = control_manifest.validate_control_manifest(workspace)
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise FinalizeError("run ID is malformed")
    require_sha(expected_provisional_sha256, "external provisional anchor")
    run_directory = workspace / "runs" / run_id
    process_path = run_directory / "process.json"
    preflight_path = run_directory / "preflight.json"
    gate_path = run_directory / "gpu-smoke-gate.json"
    barrier_path = run_directory / "gpu-smoke-start-barrier.json"
    provisional_path = contract.provisional_path(workspace, run_id)
    process, process_raw = load_canonical(process_path, workspace, "process metadata")
    preflight, preflight_raw = load_canonical(preflight_path, workspace, "preflight")
    gate, gate_raw = load_canonical(gate_path, workspace, "GPU-smoke gate")
    barrier, barrier_raw = load_canonical(barrier_path, workspace, "start barrier")
    provisional, provisional_raw = load_canonical(
        provisional_path, workspace, "provisional smoke"
    )
    if sha256_bytes(provisional_raw) != expected_provisional_sha256:
        raise FinalizeError("external provisional SHA-256 anchor differs")
    validate_provisional_schema(provisional)
    if (
        provisional.get("schema_version") != contract.SMOKE_SCHEMA_VERSION
        or provisional.get("smoke") != contract.SMOKE_NAME
        or provisional.get("acceptance")
        != "gpu-execution-complete-awaiting-run-finalization"
    ):
        raise FinalizeError("provisional top-level identity differs")
    validate_scope(provisional)
    validate_run_documents(
        workspace=workspace,
        run_id=run_id,
        process=process,
        preflight=preflight,
        preflight_raw=preflight_raw,
        gate=gate,
        gate_raw=gate_raw,
        barrier=barrier,
        barrier_raw=barrier_raw,
        provisional=provisional,
        control_identity=control_identity,
    )
    validate_integrity(provisional, workspace, control_identity)
    validate_runtime_identity(provisional, workspace, preflight, gate, control_identity)
    validate_frontend_results(provisional)
    replay_files = validate_replay(provisional, workspace, run_id)
    numerical_replay = audit_numerical_replay(provisional, workspace, run_id)
    replay_semantics = audit_replay_semantics(provisional, workspace, run_id)
    exact_inputs = {
        "anchor_generator": validate_exact_file(
            workspace / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
            workspace,
            contract.ANCHOR_GENERATOR_SIZE,
            contract.ANCHOR_GENERATOR_SHA256,
            "fused-pointwise anchor generator",
        ),
        "compile_anchors": validate_exact_file(
            workspace / contract.COMPILE_ANCHORS_RELATIVE_PATH,
            workspace,
            contract.COMPILE_ANCHORS_SIZE,
            contract.COMPILE_ANCHORS_SHA256,
            "fused-pointwise compile anchors",
        ),
        "runner": validate_exact_file(
            workspace / contract.RUNNER_RELATIVE_PATH,
            workspace,
            contract.RUNNER_SIZE,
            contract.RUNNER_SHA256,
            "fused-pointwise smoke runner",
        ),
        "pypto_dso": validate_exact_file(
            workspace / contract.PYPTO_DSO_RELATIVE_PATH,
            workspace,
            contract.PYPTO_DSO_SIZE,
            contract.PYPTO_DSO_SHA256,
            "PyPTO product DSO",
        ),
        "cuda_runtime": validate_exact_file(
            workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
            workspace,
            contract.CUDA_RUNTIME_SIZE,
            contract.CUDA_RUNTIME_SHA256,
            "CUDA Runtime provider",
        ),
        "python": validate_exact_file(
            workspace / contract.PYTHON_REAL_RELATIVE_PATH,
            workspace,
            contract.PYTHON_SIZE,
            contract.PYTHON_SHA256,
            "selected Python",
        ),
    }
    if sha256_file(workspace / "ENVIRONMENT.lock") != contract.ENVIRONMENT_LOCK_SHA256:
        raise FinalizeError("ENVIRONMENT.lock differs from the fixed contract")
    pypto_identity = git_identity(workspace / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise FinalizeError("PyPTO source identity differs at finalization")
    report = {
        "schema_version": contract.SMOKE_SCHEMA_VERSION,
        "smoke": contract.SMOKE_NAME,
        "status": "accepted-real-sm120-fused-pointwise-nine-case-correctness-gate",
        "scope": provisional["scope"],
        "not_claimed": [
            "general FusedPointwiseV2 correctness",
            "individual transcendental accuracy outside the frozen cases",
            "other chains, shapes, ranks, scalars, or special-value domains",
            "subnormal or high_precision behavior",
            "Cubin determinism across different builds or toolchains",
            "reduction, matmul, or memory lowering",
            "performance",
            "CUDA Graph correctness",
            "TorchInductor or SGLang integration",
            "Qwen3.5 correctness or strict coverage",
            "any extension or reinterpretation of accepted CP44",
        ],
        "run": {
            "run_id": run_id,
            "process_path": process_path.relative_to(workspace).as_posix(),
            "process_sha256": sha256_bytes(process_raw),
            "preflight_path": preflight_path.relative_to(workspace).as_posix(),
            "preflight_sha256": sha256_bytes(preflight_raw),
            "gate_path": gate_path.relative_to(workspace).as_posix(),
            "gate_sha256": sha256_bytes(gate_raw),
            "start_barrier_path": barrier_path.relative_to(workspace).as_posix(),
            "start_barrier_sha256": sha256_bytes(barrier_raw),
            "provisional_path": provisional_path.relative_to(workspace).as_posix(),
            "provisional_sha256": expected_provisional_sha256,
            "command": contract.fixed_child_command(workspace),
            "zero_nvidia_interference": True,
        },
        "inputs": {
            "exact_files": exact_inputs,
            "control_manifest": control_identity,
            "environment_lock_sha256": contract.ENVIRONMENT_LOCK_SHA256,
            "pypto": pypto_identity,
            "tensor_ir_head": contract.TENSOR_IR_HEAD,
            "cuda_tile_head": contract.CUDA_TILE_HEAD,
            "llvm_head": contract.LLVM_HEAD,
            "replay_files": replay_files,
            "numerical_replay": numerical_replay,
            "replay_semantics": replay_semantics,
        },
        "result": provisional["runtime"],
        "finalizer": {
            "path": Path(__file__)
            .resolve(strict=True)
            .relative_to(workspace)
            .as_posix(),
            "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "cpu_only_deserialization": True,
            "source_audit_compiler_entrypoints_absent": True,
            "torch_cuda_initialized": False,
        },
    }
    output_parent = workspace / contract.FINAL_REPORT_DIRECTORY
    output_parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    output = contract.final_report_path(workspace, run_id)
    digest = publish_no_replace(output, report)
    return report, output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-provisional-sha256", required=True)
    args = parser.parse_args()
    _report, path, digest = finalize(
        workspace=args.workspace,
        run_id=args.run_id,
        expected_provisional_sha256=args.expected_provisional_sha256,
    )
    print(
        json.dumps(
            {"status": "accepted", "path": str(path), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
