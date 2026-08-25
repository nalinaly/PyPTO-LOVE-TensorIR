#!/usr/bin/env python3
"""Finalize one correctness-only PyPTO NvidiaExecutable SM120 run."""

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

import _pypto_nvidia_executable_sm120_contract as contract
import _pypto_nvidia_sm120_control_manifest as control_manifest


ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalizeError(RuntimeError):
    """The provisional smoke cannot be promoted."""


def require_no_site_finalizer() -> None:
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        raise FinalizeError("SM120 smoke finalizer requires Python -E -B -S")


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

contract = load("replay_contract", workspace / "tools/_pypto_nvidia_executable_sm120_contract.py")
runner = load("replay_runner", workspace / contract.RUNNER_RELATIVE_PATH)
runner.validate_pypto_python_source(workspace)
pypto = runner.bootstrap_exact_pypto(
    workspace,
    (workspace / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True).parent,
)
from pypto import compiler

request = compiler.CompileRequest.deserialize(
    (replay / "compile-request.msgpack").read_bytes()
)
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
contracts = runner.make_case_contracts(compiler)
records = []
for case in contract.CASE_SPECS:
    source, abi, tiles = contracts[case.name]
    spec = compiler.KernelBuildSpec.deserialize(
        (replay / f"{case.name}.build-spec.msgpack").read_bytes()
    )
    expected_spec = runner.build_spec(
        compiler, request, source, abi, tiles, contract.PIPELINE_REVISION
    )
    assert spec.identity_digest == expected_spec.identity_digest
    artifact = compiler.Artifact.deserialize(
        (replay / f"{case.name}.artifact.msgpack").read_bytes(),
        request,
        spec,
    )
    assert artifact.kernel_abi.identity_digest == abi.identity_digest
    assert not artifact.fallback_used
    assert len(artifact.device_code) == case.expected_device_code_bytes
    assert artifact.device_code_sha256 == case.expected_device_code_sha256
    records.append({
        "case": case.name,
        "source_sha256": compiler.Artifact.compute_source_ir_digest(source),
        "build_spec_identity_digest": spec.identity_digest,
        "artifact_identity_digest": artifact.identity_digest,
        "cache_key_digest": artifact.cache_key_digest,
        "loader_compatibility_digest": artifact.loader_compatibility_digest,
        "device_code_bytes": len(artifact.device_code),
        "device_code_sha256": artifact.device_code_sha256,
        "kernel_abi_identity_digest": artifact.kernel_abi.identity_digest,
        "entry_function_name": artifact.kernel_abi.entry_function_name,
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
    "cases": records,
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


def require_workspace(workspace: Path) -> Path:
    lexical = workspace.absolute()
    resolved = workspace.resolve(strict=True)
    if lexical != resolved or resolved != ROOT:
        raise FinalizeError(f"workspace must be the exact project root: {ROOT}")
    return resolved


def require_regular(path: Path, workspace: Path, description: str) -> Path:
    lexical = path.absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FinalizeError(f"{description} is missing: {path}") from error
    if lexical != resolved or (
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"head": head, "tree": tree, "clean": not dirty}


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


def validate_preflight(value: dict[str, object], process: dict[str, object]) -> None:
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
        or free_memory_mib < 4 * 1024
        or value.get("gpu_smoke_free_memory_floor_mib") != 4 * 1024
        or value.get("memory_floor_kib") not in {24 * 1024 * 1024, 32 * 1024 * 1024}
    ):
        raise FinalizeError("preflight GPU identity or resource floor differs")
    torch = value.get("torch")
    require_exact_keys(
        torch,
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
    if not isinstance(torch, dict) or (
        torch.get("version") != contract.EXPECTED_TORCH_VERSION
        or torch.get("git_version") != contract.EXPECTED_TORCH_GIT
        or torch.get("cuda") != contract.EXPECTED_TORCH_CUDA
        or torch.get("hip") is not None
        or torch.get("cuda_initialized") is not False
        or torch.get("nvidia_runtime_mappings") != []
        or torch.get("environment_lock_sha256") != contract.ENVIRONMENT_LOCK_SHA256
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
        raise FinalizeError(f"{description} has invalid owned compute PIDs")
    free_memory = require_int(
        value.get("free_memory_mib"), f"{description} free memory", positive=True
    )
    if free_memory < 4 * 1024:
        raise FinalizeError(f"{description} is below the GPU-memory floor")
    gpu = value.get("gpu")
    if not isinstance(gpu, dict) or (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != "12.0"
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        raise FinalizeError(f"{description} GPU identity differs")


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
            "environment_lock",
            "versions_lock",
            "workspace_lock",
            "pypto_dso",
            "cuda_runtime",
        },
        "provisional integrity",
    )
    for name, record in integrity.items():
        require_exact_keys(
            record, {"path", "bytes", "sha256"}, f"integrity record {name}"
        )
    require_exact_keys(inputs.get("pypto"), {"head", "tree", "clean"}, "PyPTO identity")
    run_context = require_exact_keys(
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
        "provisional run context",
    )
    require_exact_keys(
        run_context.get("preflight"), {"path", "sha256"}, "run-context preflight"
    )
    require_exact_keys(
        run_context.get("gate"), {"path", "sha256", "document"}, "run-context gate"
    )
    runtime = require_exact_keys(
        provisional.get("runtime"),
        {
            "torch",
            "child_pre_cuda_gate",
            "libcudart_paths",
            "observation",
            "compile_request",
            "artifacts",
            "executions",
            "case_order",
            "repetitions_per_case",
            "module_lifetimes",
            "explicit_unloads",
            "non_default_current_stream",
            "external_synchronization",
            "fallback_used",
            "forbidden_provider_imports",
        },
        "provisional runtime",
    )
    require_exact_keys(
        runtime.get("torch"),
        {"version", "git_version", "cuda", "hip", "module_path"},
        "runtime Torch",
    )
    require_exact_keys(
        runtime.get("child_pre_cuda_gate"),
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
    require_exact_keys(
        observation.get("traits"),
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
    require_exact_keys(
        runtime.get("compile_request"),
        {
            "byte_identity_digest",
            "loader_compatibility_input_digest",
            "device_autotune_identity_digest",
        },
        "runtime CompileRequest",
    )


def validate_scope(provisional: dict[str, object]) -> None:
    expected = {
        "provider": "pypto.tensorir",
        "runtime_object": "NvidiaExecutable",
        "operator_correctness": True,
        "model_forward": False,
        "strict_coverage_result": False,
        "performance_result": False,
        "cuda_graph_result": False,
    }
    if provisional.get("scope") != expected:
        raise FinalizeError("provisional scope is not correctness-only")

    forbidden_key_fragments = (
        "latency",
        "throughput",
        "tokens_per",
        "bandwidth",
        "flops",
        "duration_ns",
        "cuda_event",
        "benchmark",
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                if any(fragment in lowered for fragment in forbidden_key_fragments):
                    raise FinalizeError(f"performance-like field is forbidden: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(provisional)


def validate_provisional_integrity(
    provisional: dict[str, object],
    workspace: Path,
    control_identity: dict[str, object],
) -> None:
    inputs = provisional["inputs"]
    assert isinstance(inputs, dict)
    if (
        inputs.get("tensor_ir_head") != contract.TENSOR_IR_HEAD
        or inputs.get("cuda_tile_head") != contract.CUDA_TILE_HEAD
        or inputs.get("llvm_head") != contract.LLVM_HEAD
        or inputs.get("control_manifest") != control_identity
        or inputs.get("pypto")
        != {
            "head": contract.PYPTO_HEAD,
            "tree": contract.PYPTO_TREE,
            "clean": True,
        }
    ):
        raise FinalizeError("provisional source identities differ")
    expected_paths = {
        "contract": workspace / "tools/_pypto_nvidia_executable_sm120_contract.py",
        "runner": workspace / contract.RUNNER_RELATIVE_PATH,
        "environment_lock": workspace / "ENVIRONMENT.lock",
        "versions_lock": workspace / "VERSIONS.lock",
        "workspace_lock": workspace / "WORKSPACE.lock",
        "pypto_dso": workspace / contract.PYPTO_DSO_RELATIVE_PATH,
        "cuda_runtime": workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
    }
    integrity = inputs["integrity"]
    assert isinstance(integrity, dict)
    for name, path in expected_paths.items():
        record = integrity[name]
        assert isinstance(record, dict)
        resolved = require_regular(path, workspace, f"provisional {name}")
        if record != {
            "path": resolved.relative_to(workspace).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }:
            raise FinalizeError(f"provisional integrity differs for {name}")


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
        or torch.get("module_path") != str(expected_torch_path)
        or runtime.get("libcudart_paths")
        != [str((workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))]
    ):
        raise FinalizeError("live Torch or libcudart runtime identity differs")
    observation = runtime["observation"]
    assert isinstance(observation, dict)
    traits = observation["traits"]
    assert isinstance(traits, dict)
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
        or observation.get("supported_compute_dtypes") != ["BF16", "FP32"]
        or observation.get("cuda_driver_release_provenance")
        != contract.EXPECTED_DRIVER_RELEASE
        or require_int(
            observation.get("cuda_driver_api_version"),
            "observed Driver API",
            positive=True,
        )
        < contract.MINIMUM_CUDA_DRIVER_API_VERSION
        or require_int(
            observation.get("cuda_runtime_api_version"),
            "observed Runtime API",
            positive=True,
        )
        < contract.MINIMUM_CUDA_RUNTIME_API_VERSION
        or observation.get("cuda_runtime_library_path")
        != str((workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))
    ):
        raise FinalizeError("live PyPTO runtime observation differs")
    for name, value in traits.items():
        require_int(value, f"target trait {name}", positive=True)
    compile_request = runtime["compile_request"]
    assert isinstance(compile_request, dict)
    for name, value in compile_request.items():
        require_sha(value, f"CompileRequest {name}")
    child_gate = runtime["child_pre_cuda_gate"]
    assert isinstance(child_gate, dict)
    parent_static = gate.get("static_identity")
    if (
        child_gate.get("static_identity") != parent_static
        or parent_static != preflight.get("torch")
        or child_gate.get("control_manifest") != control_identity
        or child_gate.get("protected_runtime_pids") != []
        or child_gate.get("unreadable_protected_maps") != []
        or child_gate.get("nvidia_compute_pids") != []
        or require_int(
            child_gate.get("free_memory_mib"),
            "child gate free memory",
            positive=True,
        )
        < 4 * 1024
    ):
        raise FinalizeError("child pre-CUDA gate differs from parent evidence")
    child_gpu = child_gate.get("gpu")
    if not isinstance(child_gpu, dict) or (
        child_gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or child_gpu.get("compute_capability") != "12.0"
        or child_gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        raise FinalizeError("child pre-CUDA GPU identity differs")


def validate_replay(
    provisional: dict[str, object], workspace: Path, run_id: str
) -> list[dict[str, object]]:
    inputs = provisional.get("inputs")
    if not isinstance(inputs, dict):
        raise FinalizeError("provisional inputs are missing")
    replay_files = inputs.get("replay_files")
    if not isinstance(replay_files, list) or len(replay_files) != 7:
        raise FinalizeError("provisional replay file set is incomplete")
    expected_names = [
        "compile-request.msgpack",
        "static.build-spec.msgpack",
        "static.artifact.msgpack",
        "dynamic.build-spec.msgpack",
        "dynamic.artifact.msgpack",
        "scalar.build-spec.msgpack",
        "scalar.artifact.msgpack",
    ]
    replay_directory = contract.replay_directory(workspace, run_id)
    actual_names = sorted(path.name for path in replay_directory.iterdir())
    if actual_names != sorted([*expected_names, contract.PROVISIONAL_NAME]):
        raise FinalizeError("replay directory has a missing or extra file")
    normalized: list[dict[str, object]] = []
    for record, name in zip(replay_files, expected_names, strict=True):
        if not isinstance(record, dict):
            raise FinalizeError("replay identity is malformed")
        path = replay_directory / name
        resolved = require_regular(path, workspace, f"replay file {name}")
        mode = stat.S_IMODE(resolved.stat().st_mode)
        size = require_int(record.get("bytes"), f"replay file {name} bytes")
        digest = require_sha(record.get("sha256"), f"replay file {name} digest")
        expected_path = resolved.relative_to(workspace).as_posix()
        if (
            record.get("path") != expected_path
            or resolved.stat().st_size != size
            or sha256_file(resolved) != digest
            or mode != 0o444
        ):
            raise FinalizeError(f"replay file {name} identity differs")
        normalized.append(dict(record))
    return normalized


def audit_replay_semantics(
    provisional: dict[str, object], workspace: Path, run_id: str
) -> dict[str, object]:
    python = workspace / "envs/pypto-nvidia/bin/python"
    command = [
        str(python),
        "-I",
        "-B",
        "-S",
        "-c",
        REPLAY_AUDIT_PROGRAM,
        str(workspace),
        str(contract.replay_directory(workspace, run_id)),
    ]
    environment = {
        "PATH": "/usr/local/cuda-13.3/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": (
            f"{workspace}/envs/pypto-nvidia/lib:"
            "/usr/lib/wsl/lib:/usr/local/cuda-13.3/lib64"
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or result.stderr:
        raise FinalizeError(
            "exact-DSO replay audit failed: "
            f"rc={result.returncode}, stderr={result.stderr[:1000]!r}"
        )
    try:
        audited = json.loads(result.stdout, object_pairs_hook=duplicate_key_guard)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FinalizeError("exact-DSO replay audit output is invalid") from error
    audited = require_exact_keys(
        audited,
        {"compile_request", "target_info", "cases"},
        "semantic replay audit",
    )
    runtime = provisional["runtime"]
    assert isinstance(runtime, dict)
    if audited.get("compile_request") != runtime.get("compile_request"):
        raise FinalizeError("replay CompileRequest differs from runtime evidence")
    observation = runtime.get("observation")
    assert isinstance(observation, dict)
    observed_target = {
        name: observation[name]
        for name in (
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
    }
    if audited.get("target_info") != observed_target:
        raise FinalizeError("replay TargetInfo differs from runtime observation")
    cases = audited.get("cases")
    artifacts = runtime.get("artifacts")
    if (
        not isinstance(cases, list)
        or not isinstance(artifacts, list)
        or len(cases) != 3
    ):
        raise FinalizeError("semantic replay case set is malformed")
    for replay_case, artifact, case in zip(
        cases, artifacts, contract.CASE_SPECS, strict=True
    ):
        replay_case = require_exact_keys(
            replay_case,
            {
                "case",
                "source_sha256",
                "build_spec_identity_digest",
                "artifact_identity_digest",
                "cache_key_digest",
                "loader_compatibility_digest",
                "device_code_bytes",
                "device_code_sha256",
                "kernel_abi_identity_digest",
                "entry_function_name",
                "fallback_used",
            },
            f"semantic replay {case.name}",
        )
        assert isinstance(artifact, dict)
        for field in (
            "case",
            "source_sha256",
            "build_spec_identity_digest",
            "artifact_identity_digest",
            "cache_key_digest",
            "loader_compatibility_digest",
            "device_code_bytes",
            "device_code_sha256",
            "kernel_abi_identity_digest",
            "entry_function_name",
            "fallback_used",
        ):
            if replay_case.get(field) != artifact.get(field):
                raise FinalizeError(
                    f"semantic replay {case.name} field differs: {field}"
                )
        if replay_case.get("kernel_abi_identity_digest") is None:
            raise FinalizeError(f"semantic replay {case.name} lacks kernel ABI")
    return {
        "command_sha256": sha256_bytes("\0".join(command).encode("utf-8")),
        "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
        "compile_request": audited["compile_request"],
        "target_info": audited["target_info"],
        "cases": audited["cases"],
    }


def validate_executions(provisional: dict[str, object]) -> None:
    runtime = provisional.get("runtime")
    if not isinstance(runtime, dict):
        raise FinalizeError("provisional runtime is missing")
    if (
        runtime.get("case_order") != [case.name for case in contract.CASE_SPECS]
        or runtime.get("repetitions_per_case") != 2
        or runtime.get("module_lifetimes") != 6
        or runtime.get("explicit_unloads") != 6
        or runtime.get("non_default_current_stream") is not True
        or runtime.get("external_synchronization") is not True
        or runtime.get("fallback_used") is not False
        or runtime.get("forbidden_provider_imports") != []
    ):
        raise FinalizeError("provisional runtime summary is incomplete")
    observation = runtime.get("observation")
    assert isinstance(observation, dict)
    observation_context = require_int(
        observation.get("context_address"), "observation context", positive=True
    )
    observation_context_id = require_int(
        observation.get("context_id"), "observation context ID", positive=True
    )
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise FinalizeError("provisional Artifact set is incomplete")
    artifact_identities: dict[str, str] = {}
    for artifact, case in zip(artifacts, contract.CASE_SPECS, strict=True):
        artifact = require_exact_keys(
            artifact,
            {
                "case",
                "source_sha256",
                "build_spec_identity_digest",
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
            },
            f"{case.name} Artifact",
        )
        if (
            artifact.get("case") != case.name
            or artifact.get("fallback_used") is not False
            or artifact.get("expected_grid") != list(case.expected_grid)
            or artifact.get("expected_kernel_arguments")
            != case.expected_kernel_arguments
            or artifact.get("expected_device_code_bytes")
            != case.expected_device_code_bytes
            or artifact.get("expected_device_code_sha256")
            != case.expected_device_code_sha256
            or artifact.get("device_code_bytes") != case.expected_device_code_bytes
            or artifact.get("device_code_sha256") != case.expected_device_code_sha256
        ):
            raise FinalizeError(f"{case.name} Artifact evidence differs")
        artifact_identities[case.name] = require_sha(
            artifact.get("artifact_identity_digest"),
            f"{case.name} Artifact identity",
        )
        for field in (
            "source_sha256",
            "build_spec_identity_digest",
            "artifact_identity_digest",
            "cache_key_digest",
            "loader_compatibility_digest",
            "device_code_sha256",
            "kernel_abi_identity_digest",
        ):
            require_sha(artifact.get(field), f"{case.name} {field}")
        require_int(
            artifact.get("device_code_bytes"),
            f"{case.name} device code bytes",
            positive=True,
        )
    executions = runtime.get("executions")
    expected_pairs = [
        (case, repetition)
        for case in contract.CASE_SPECS
        for repetition in range(case.repetitions)
    ]
    if not isinstance(executions, list) or len(executions) != len(expected_pairs):
        raise FinalizeError("provisional execution set is incomplete")
    for execution, (case, repetition) in zip(executions, expected_pairs, strict=True):
        execution = require_exact_keys(
            execution,
            {
                "case",
                "repetition",
                "artifact_identity_digest",
                "dtype",
                "shape",
                "strides",
                "grid",
                "kernel_argument_count",
                "raw_current_stream",
                "non_default_stream",
                "external_stream_synchronized",
                "expected_logical_bytes_sha256",
                "actual_logical_bytes_sha256",
                "input_bytes_sha256",
                "input_unchanged",
                "torch_equal",
                "padding_unchanged",
                "packet_released_after_synchronization",
                "explicit_unload",
                "terminal_state",
                "bound_context_before_unload",
                "bound_context_id_before_unload",
                "bound_context_after_unload",
            },
            f"execution {case.name}/{repetition}",
        )
        if (
            execution.get("case") != case.name
            or execution.get("repetition") != repetition
            or execution.get("dtype") != case.dtype
            or execution.get("shape") != list(case.shape)
            or execution.get("strides") != list(case.strides)
            or execution.get("grid") != list(case.expected_grid)
            or execution.get("kernel_argument_count") != case.expected_kernel_arguments
            or execution.get("non_default_stream") is not True
            or execution.get("external_stream_synchronized") is not True
            or execution.get("torch_equal") is not True
            or execution.get("padding_unchanged") is not True
            or execution.get("input_unchanged") is not True
            or execution.get("packet_released_after_synchronization") is not True
            or execution.get("explicit_unload") is not True
            or execution.get("terminal_state") != "Unloaded"
            or execution.get("bound_context_after_unload") != 0
        ):
            raise FinalizeError(
                f"execution {case.name}/{repetition} does not prove correctness"
            )
        if execution.get("expected_logical_bytes_sha256") != execution.get(
            "actual_logical_bytes_sha256"
        ):
            raise FinalizeError(f"execution {case.name}/{repetition} bytes differ")
        raw_stream = require_int(
            execution.get("raw_current_stream"),
            f"execution {case.name}/{repetition} stream",
            positive=True,
        )
        if raw_stream in {1, 2}:
            raise FinalizeError(
                f"execution {case.name}/{repetition} used a default stream"
            )
        if (
            execution.get("bound_context_before_unload") != observation_context
            or execution.get("bound_context_id_before_unload") != observation_context_id
        ):
            raise FinalizeError(
                f"execution {case.name}/{repetition} context join differs"
            )
        if execution.get("artifact_identity_digest") != artifact_identities[case.name]:
            raise FinalizeError(
                f"execution {case.name}/{repetition} Artifact join differs"
            )
        require_sha(
            execution.get("actual_logical_bytes_sha256"),
            f"execution {case.name}/{repetition} output",
        )
        require_sha(
            execution.get("input_bytes_sha256"),
            f"execution {case.name}/{repetition} input",
        )


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
        {
            "timeout_seconds",
            "minimum_free_disk_bytes",
            "owned_run_pause_memory_kib",
        },
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
        or gpu_smoke.get("gpu_free_memory_floor_mib") != 4 * 1024
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
    process, process_raw = load_canonical(
        process_path, workspace, "run process metadata"
    )
    preflight, preflight_raw = load_canonical(
        preflight_path, workspace, "run preflight"
    )
    gate, gate_raw = load_canonical(gate_path, workspace, "GPU-smoke gate")
    barrier, barrier_raw = load_canonical(
        barrier_path, workspace, "GPU-smoke start barrier"
    )
    provisional, provisional_raw = load_canonical(
        provisional_path, workspace, "provisional smoke"
    )
    if sha256_bytes(provisional_raw) != expected_provisional_sha256:
        raise FinalizeError("external provisional SHA-256 anchor differs")
    validate_provisional_schema(provisional)
    validate_process_schema(process, workspace)
    if (
        process.get("schema") != 3
        or process.get("run_id") != run_id
        or process.get("workspace") != str(workspace)
        or process.get("mode") != "gpu-smoke"
        or process.get("framework_launch") is not False
        or process.get("framework_profile") != "pypto"
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("command") != contract.fixed_child_command(workspace)
        or "gpu_smoke_abort" in process
        or process.get("surviving_group_pids")
    ):
        raise FinalizeError("completed process metadata is not an accepted smoke run")
    preflight_anchor = process.get("preflight")
    if not isinstance(preflight_anchor, dict) or (
        preflight_anchor.get("path") != str(preflight_path)
        or preflight_anchor.get("sha256") != sha256_bytes(preflight_raw)
    ):
        raise FinalizeError("process/preflight digest join failed")
    validate_preflight(preflight, process)
    protected_authorized = (
        preflight.get("protected_zero_nvidia_gpu_smoke_requested") is True
    )
    validate_audit(
        process.get("gpu_smoke_pre_release_audit"),
        "pre-release audit",
        authorized=protected_authorized,
        require_zero_owned=True,
    )
    validate_audit(
        process.get("gpu_smoke_last_audit"),
        "periodic audit",
        authorized=protected_authorized,
        require_zero_owned=False,
    )
    validate_audit(
        process.get("gpu_smoke_post_exit_audit"),
        "post-exit audit",
        authorized=protected_authorized,
        require_zero_owned=True,
    )
    gpu_smoke = process["gpu_smoke"]
    assert isinstance(gpu_smoke, dict)
    if (
        gpu_smoke.get("gate_path") != str(gate_path)
        or gpu_smoke.get("start_barrier_path") != str(barrier_path)
        or gpu_smoke.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
    ):
        raise FinalizeError("process/gate/barrier digest join failed")
    require_exact_keys(
        barrier,
        {
            "schema",
            "run_id",
            "pid",
            "pgid",
            "start_ticks",
            "gate_path",
            "gate_sha256",
        },
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
    if (
        barrier.get("schema") != contract.GPU_SMOKE_POLICY_VERSION
        or barrier.get("run_id") != run_id
        or barrier.get("pid") != process.get("pid")
        or barrier.get("pgid") != process.get("pgid")
        or barrier.get("start_ticks") != process.get("start_ticks")
        or barrier.get("gate_path") != str(gate_path)
        or barrier.get("gate_sha256") != sha256_bytes(gate_raw)
    ):
        raise FinalizeError("start barrier identity differs")
    if (
        gate.get("schema") != contract.GPU_SMOKE_POLICY_VERSION
        or gate.get("run_id") != run_id
        or gate.get("pid") != process.get("pid")
        or gate.get("pgid") != process.get("pgid")
        or gate.get("start_ticks") != process.get("start_ticks")
        or gate.get("command") != contract.fixed_child_command(workspace)
        or gate.get("preflight") != process.get("preflight")
        or gate.get("static_identity") != preflight.get("torch")
        or gate.get("control_manifest") != control_identity
        or gate.get("runtime_isolation") != process.get("gpu_smoke_pre_release_audit")
    ):
        raise FinalizeError("pre-release gate identity differs")
    if (
        provisional.get("schema_version") != contract.SMOKE_SCHEMA_VERSION
        or provisional.get("smoke") != contract.SMOKE_NAME
        or provisional.get("acceptance")
        != "gpu-execution-complete-awaiting-run-finalization"
    ):
        raise FinalizeError("provisional top-level identity differs")
    run_context = provisional.get("run_context")
    if not isinstance(run_context, dict) or (
        run_context.get("run_id") != run_id
        or run_context.get("mode") != "gpu-smoke"
        or run_context.get("pid") != process.get("pid")
        or run_context.get("pgid") != process.get("pgid")
        or run_context.get("start_ticks") != process.get("start_ticks")
        or run_context.get("protected_zero_nvidia_policy") is not protected_authorized
        or run_context.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
        or run_context.get("preflight")
        != {
            "path": preflight_path.relative_to(workspace).as_posix(),
            "sha256": sha256_bytes(preflight_raw),
        }
        or run_context.get("gate")
        != {
            "path": str(gate_path),
            "sha256": sha256_bytes(gate_raw),
            "document": gate,
        }
    ):
        raise FinalizeError("provisional run context differs")
    validate_provisional_integrity(provisional, workspace, control_identity)
    validate_runtime_identity(provisional, workspace, preflight, gate, control_identity)
    validate_scope(provisional)
    replay_files = validate_replay(provisional, workspace, run_id)
    replay_semantics = audit_replay_semantics(provisional, workspace, run_id)
    validate_executions(provisional)
    exact_inputs = {
        "runner": validate_exact_file(
            workspace / contract.RUNNER_RELATIVE_PATH,
            workspace,
            contract.RUNNER_SIZE,
            contract.RUNNER_SHA256,
            "smoke runner",
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
    environment_lock = require_regular(
        workspace / "ENVIRONMENT.lock", workspace, "ENVIRONMENT.lock"
    )
    if sha256_file(environment_lock) != contract.ENVIRONMENT_LOCK_SHA256:
        raise FinalizeError("ENVIRONMENT.lock differs from the fixed contract")
    pypto = git_identity(workspace / "projects" / "pypto")
    if pypto != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise FinalizeError("PyPTO source identity differs at finalization")
    report = {
        "schema_version": contract.SMOKE_SCHEMA_VERSION,
        "smoke": contract.SMOKE_NAME,
        "status": "accepted-real-sm120-correctness-smoke",
        "scope": provisional["scope"],
        "not_claimed": [
            "performance",
            "CUDA Graph correctness",
            "frontend HIR lowering",
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
            "pypto": pypto,
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
