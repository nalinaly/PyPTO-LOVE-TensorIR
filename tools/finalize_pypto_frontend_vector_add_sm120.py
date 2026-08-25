#!/usr/bin/env python3
"""CPU-only finalizer for one frontend vector-add SM120 smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import _pypto_frontend_sm120_control_manifest as control_manifest
import _pypto_frontend_vector_add_sm120_contract as contract


ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalizeError(RuntimeError):
    """The provisional frontend smoke cannot be promoted."""


def require_no_site_finalizer() -> None:
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        raise FinalizeError("frontend SM120 finalizer requires Python -E -B -S")


# This child only deserializes immutable evidence through public ``pypto.ir``
# and compiler data types.  It invokes no compiler producer or NVIDIA runtime
# entry point, constructs no executable, and calls no CUDA Runtime/Driver API.
# Importing public PyPTO may transitively import Torch; the only Torch call is
# the state-only assertion that CUDA was never initialized.
REPLAY_AUDIT_PROGRAM = r"""
import importlib.util
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve(strict=True)
replay = Path(sys.argv[2]).resolve(strict=True)
site = workspace / "envs/pypto-nvidia/lib/python3.14/site-packages"
sys.path.insert(0, str(site))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

contract = load(
    "replay_frontend_contract",
    workspace / "tools/_pypto_frontend_vector_add_sm120_contract.py",
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
    function = program.get_function("add_main")
    assert function is not None
    assert list(function.param_directions) == [
        pypto.ir.ParamDirection.In,
        pypto.ir.ParamDirection.In,
    ]
    spec_bytes = (replay / f"{case.name}.build-spec.msgpack").read_bytes()
    spec = compiler.KernelBuildSpec.deserialize(spec_bytes)
    assert spec.serialize() == spec_bytes
    artifact_bytes = (replay / f"{case.name}.artifact.msgpack").read_bytes()
    artifact = compiler.Artifact.deserialize(artifact_bytes, request, spec)
    assert artifact.serialize() == artifact_bytes
    runner.validate_compiled_artifact(compiler, spec, artifact, request, case)
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
    assert kernel.entry_function_name == "pypto_vector_add"
    assert kernel.argument_packing_policy == compiler.ArtifactArgumentPackingPolicy.PointerOnly
    assert kernel.grid_abi.policy == compiler.ArtifactGridPolicy.Static
    assert tuple(kernel.grid_abi.static_dimensions) == case.expected_grid
    assert tuple(kernel.grid_abi.tile_sizes) == case.tile_sizes
    assert kernel.argument_layout.input_operand_count == 2
    assert kernel.argument_layout.total_kernel_argument_count == case.expected_kernel_arguments
    assert not kernel.argument_layout.uniform_signature
    assert len(descriptors) == 3
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
        "parameter_directions": ["In", "In"],
    })
    artifact_records.append({
        "case": case.name,
        "build_spec_identity_digest": spec.identity_digest,
        "source_ir_digest": spec.source_ir_digest,
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
            "provider",
            "frontend_hir",
            "runtime_object",
            "operator_correctness",
            "model_forward",
            "strict_coverage_result",
            "performance_result",
            "cuda_graph_result",
        },
        "scope",
    )
    if scope != {
        "provider": "pypto.tensorir",
        "frontend_hir": True,
        "runtime_object": "NvidiaExecutable",
        "operator_correctness": True,
        "model_forward": False,
        "strict_coverage_result": False,
        "performance_result": False,
        "cuda_graph_result": False,
    }:
        raise FinalizeError("scope exceeds the frontend vector-add correctness claim")
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


def validate_frontend_results(provisional: dict[str, object]) -> None:
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    cases = list(contract.CASE_SPECS)
    expected_order = [case.name for case in cases]
    lifetime_count = sum(case.repetitions for case in cases)
    if (
        runtime.get("case_order") != expected_order
        or runtime.get("compile_invocations_per_case") != 1
        or runtime.get("repetitions_per_case") != 2
        or runtime.get("module_lifetimes") != lifetime_count
        or runtime.get("explicit_packet_releases") != lifetime_count
        or runtime.get("explicit_unloads") != lifetime_count
        or runtime.get("non_default_current_stream") is not True
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
            or hir_record.get("parameter_directions") != ["In", "In"]
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
            },
            f"Artifact {case.name}",
        )
        if (
            artifact_record.get("case") != case.name
            or artifact_record.get("compile_api")
            != "pypto.compiler.compile_structured_strict"
            or artifact_record.get("compiler_invocations") != 1
            or artifact_record.get("source_ir_digest") != case.expected_source_ir_digest
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
            or artifact_record.get("entry_function_name") != "pypto_vector_add"
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
                "non_default_stream",
                "capture_free_before",
                "capture_free_at_launch",
                "external_stream_synchronized",
                "expected_logical_bytes_sha256",
                "actual_logical_bytes_sha256",
                "input_bytes_sha256",
                "input_unchanged",
                "torch_equal",
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
            or execution.get("capture_free_before") is not True
            or execution.get("capture_free_at_launch") is not True
            or execution.get("external_stream_synchronized") is not True
            or execution.get("input_unchanged") is not True
            or execution.get("torch_equal") is not True
            or execution.get("packet_released_after_synchronization") is not True
            or execution.get("explicit_unload") is not True
            or execution.get("terminal_state") != "Unloaded"
            or execution.get("bound_context_before_unload") != context
            or execution.get("bound_context_id_before_unload") != context_id
            or execution.get("bound_context_after_unload") != 0
            or execution.get("bound_context_id_after_unload") != 0
            or execution.get("expected_logical_bytes_sha256")
            != execution.get("actual_logical_bytes_sha256")
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
            execution.get("input_bytes_sha256"),
            f"execution {case.name}/{repetition} inputs",
        )


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
                f"{case.name}.build-spec.msgpack",
                f"{case.name}.artifact.msgpack",
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
            "parameter_directions": ["In", "In"],
        }
        for record in runtime["hir_programs"]
    ]
    if audited.get("hir_programs") != expected_hir:
        raise FinalizeError("replayed HIR differs")
    artifact_fields = (
        "case",
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
        "device_code_bytes",
        "device_code_sha256",
        "kernel_abi_identity_digest",
        "entry_function_name",
        "fallback_used",
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
        "contract": workspace / "tools/_pypto_frontend_vector_add_sm120_contract.py",
        "runner": workspace / contract.RUNNER_RELATIVE_PATH,
        "controller": workspace / "tools/run_pypto_frontend_sm120_isolated.py",
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
    replay_semantics = audit_replay_semantics(provisional, workspace, run_id)
    exact_inputs = {
        "runner": validate_exact_file(
            workspace / contract.RUNNER_RELATIVE_PATH,
            workspace,
            contract.RUNNER_SIZE,
            contract.RUNNER_SHA256,
            "frontend smoke runner",
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
        "status": "accepted-real-sm120-frontend-vector-add-correctness-smoke",
        "scope": provisional["scope"],
        "not_claimed": [
            "performance",
            "CUDA Graph correctness",
            "operator families beyond the two vector-add fixtures",
            "TorchInductor or SGLang integration",
            "Qwen3.5 correctness or strict coverage",
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
