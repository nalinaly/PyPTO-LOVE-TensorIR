#!/usr/bin/env python3
"""Direct v2 child for the fixed fused-pointwise SM120 numerical gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
BASE_RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120.py")
BASE_RUNNER_SIZE = 66_999
BASE_RUNNER_SHA256 = "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15"


class SmokeV2Error(RuntimeError):
    """The v2 admission or fixed numerical transaction is invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact(
    name: str,
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise SmokeV2Error(f"exact v2 child source is noncanonical: {path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if expected_size is not None and len(raw) != expected_size:
        raise SmokeV2Error(f"exact v2 child source size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SmokeV2Error(f"exact v2 child source hash differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = digest
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = load_exact(
    "_pypto_fused_pointwise_sm120_runner_v1_base",
    ROOT / BASE_RUNNER_RELATIVE_PATH,
    expected_size=BASE_RUNNER_SIZE,
    expected_sha256=BASE_RUNNER_SHA256,
)


def workspace_from_environment() -> tuple[Path, str]:
    workspace = Path(os.environ.get("PYPTO_WORKSPACE_ROOT", ""))
    if (
        not workspace.is_absolute()
        or workspace.resolve(strict=True) != workspace
        or workspace != ROOT
    ):
        raise SmokeV2Error("PYPTO_WORKSPACE_ROOT is not the canonical project root")
    run_id = os.environ.get("PYPTO_RUN_ID", "")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise SmokeV2Error("PYPTO_RUN_ID is malformed")
    if os.environ.get("PYPTO_RUN_MODE") != "gpu-smoke":
        raise SmokeV2Error("v2 child requires gpu-smoke mode")
    if (
        os.environ.get("PYPTO_ALLOW_FALLBACK") != "0"
        or os.environ.get("PYPTO_STRICT_COVERAGE") != "1"
    ):
        raise SmokeV2Error("v2 child requires strict coverage with fallback disabled")
    if os.environ.get("PYTHONPATH", ""):
        raise SmokeV2Error("v2 child requires an empty PYTHONPATH")
    if os.environ.get("SGLANG_PLUGINS") != "__pypto_exact_nvidia_smoke_no_plugins__":
        raise SmokeV2Error("v2 child requires the no-plugin policy")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise SmokeV2Error("v2 child requires Python -I -B -S")
    if {"torch", "pypto", "triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeV2Error("GPU/framework modules loaded before the v2 barrier")
    return workspace, run_id


def wait_for_start_barrier(workspace: Path, run_id: str) -> dict[str, object]:
    path = workspace / "runs" / run_id / "gpu-smoke-start-barrier.json"
    if Path(os.environ.get("PYPTO_GPU_SMOKE_START_BARRIER", "")) != path:
        raise SmokeV2Error("v2 start-barrier path differs")
    deadline = time.monotonic() + 60.0
    while not path.exists():
        if time.monotonic() >= deadline:
            raise SmokeV2Error("timed out before v2 gate release")
        time.sleep(0.05)
    barrier, _ = base.load_canonical_json(path, "v2 start barrier")
    identity = {
        "schema": 2,
        "run_id": run_id,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "start_ticks": base._process_start_ticks(os.getpid()),
    }
    if any(barrier.get(name) != value for name, value in identity.items()):
        raise SmokeV2Error("v2 barrier process identity differs")
    gate_path = workspace / "runs" / run_id / "gpu-smoke-gate.json"
    if Path(str(barrier.get("gate_path", ""))) != gate_path:
        raise SmokeV2Error("v2 gate path differs")
    gate, gate_raw = base.load_canonical_json(gate_path, "v2 GPU-smoke gate")
    if sha256_bytes(gate_raw) != barrier.get("gate_sha256"):
        raise SmokeV2Error("v2 gate/barrier digest join failed")
    if any(gate.get(name) != value for name, value in identity.items()):
        raise SmokeV2Error("v2 gate process identity differs")
    return {"barrier": barrier, "gate": gate}


def load_contract_and_child_gate(
    workspace: Path, parent_gate: dict[str, object]
) -> tuple[Any, dict[str, object]]:
    control = load_exact(
        "_pypto_fused_pointwise_sm120_control_manifest_v2",
        workspace / "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py",
    )
    control.reject_control_bytecode_cache(workspace)
    contract = load_exact(
        "_pypto_fused_pointwise_sm120_contract_v2",
        workspace / "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
    )
    control_identity = control.validate_control_manifest(workspace)
    if parent_gate.get("control_manifest") != control_identity:
        raise SmokeV2Error("parent/child v2 control identity differs")
    runner = Path(__file__).resolve(strict=True)
    if (
        runner != workspace / contract.RUNNER_RELATIVE_PATH
        or runner.stat().st_size != contract.RUNNER_SIZE
        or sha256_file(runner) != contract.RUNNER_SHA256
        or contract.fixed_child_command(workspace) != sys.orig_argv
    ):
        raise SmokeV2Error("live v2 direct-child identity differs")
    preflight = load_exact(
        "preflight_gpu_smoke_v2_child",
        workspace / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
    )
    static_identity = preflight.static_torch_identity()
    if static_identity.get("static_identity_error"):
        raise SmokeV2Error(str(static_identity["static_identity_error"]))
    requested = os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
    floor = (
        contract.PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
        if requested
        else contract.EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
    )
    available = preflight.mem_available_kib()
    if available < floor:
        raise SmokeV2Error("child admission host-memory floor failed")
    gpu = preflight.nvidia_identity()
    if gpu.get("compute_capability") != "12.0":
        raise SmokeV2Error("child admission target is not SM120")
    free_memory_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    if free_memory_mib < contract.GPU_FREE_MEMORY_FLOOR_MIB:
        raise SmokeV2Error("child admission GPU-memory floor failed")
    compute_pids = preflight.nvidia_compute_pids()
    if compute_pids:
        raise SmokeV2Error(f"child admission found NVIDIA compute PIDs: {compute_pids}")
    _all, protected, _workspace = preflight.process_table()
    protected_runtime, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    if protected_runtime or unreadable:
        raise SmokeV2Error(
            "child admission cannot prove protected NVIDIA isolation: "
            f"runtime={protected_runtime}, unreadable={unreadable}"
        )
    protected_heavy = [
        item for item in protected if preflight.is_heavy_command(item.command)
    ]
    if protected_heavy and not requested:
        raise SmokeV2Error("protected CPU lane exists without v2 authorization")
    if (
        requested
        and os.environ.get("PYPTO_GPU_SMOKE_AUTHORIZATION")
        != contract.GPU_SMOKE_AUTHORIZATION
    ):
        raise SmokeV2Error("v2 GPU-smoke authorization differs")
    return contract, {
        "static_identity": static_identity,
        "gpu": gpu,
        "free_memory_mib": free_memory_mib,
        "mem_available_kib": available,
        "host_memory_floor_kib": floor,
        "admission_policy": preflight.policy_document(),
        "protected_heavy_pids": [item.pid for item in protected_heavy],
        "protected_runtime_pids": protected_runtime,
        "unreadable_protected_maps": unreadable,
        "nvidia_compute_pids": sorted(compute_pids),
        "control_manifest": control_identity,
        "base_runner": {
            "path": BASE_RUNNER_RELATIVE_PATH.as_posix(),
            "bytes": BASE_RUNNER_SIZE,
            "sha256": BASE_RUNNER_SHA256,
        },
    }


def integrity_record(path: Path, workspace: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(workspace).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def run_smoke() -> tuple[dict[str, object], Path, str]:
    workspace, run_id = workspace_from_environment()
    barrier_evidence = wait_for_start_barrier(workspace, run_id)
    contract, child_gate = load_contract_and_child_gate(
        workspace, barrier_evidence["gate"]
    )
    base.validate_pypto_python_source(workspace)
    site = workspace / "envs/pypto-nvidia/lib/python3.14/site-packages"
    if site.is_symlink() or not site.is_dir() or site.resolve(strict=True) != site:
        raise SmokeV2Error("selected site-packages path is noncanonical")
    sys.path.insert(0, str(site))
    import torch

    if (
        str(torch.__version__) != contract.EXPECTED_TORCH_VERSION
        or str(torch.version.git_version) != contract.EXPECTED_TORCH_GIT
        or torch.version.cuda != contract.EXPECTED_TORCH_CUDA
        or torch.version.hip is not None
    ):
        raise SmokeV2Error("live Torch identity differs")
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeV2Error("forbidden provider imported before CUDA initialization")
    torch.cuda.set_device(0)
    torch.cuda.init()
    if (
        torch.cuda.get_device_name(0) != contract.EXPECTED_DEVICE_NAME
        or tuple(torch.cuda.get_device_capability(0))
        != contract.EXPECTED_COMPUTE_CAPABILITY
    ):
        raise SmokeV2Error("live CUDA target differs")
    runtime_paths = base.mapped_library_paths("libcudart.so")
    expected_runtime = str(
        (workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
    )
    if runtime_paths != [expected_runtime]:
        raise SmokeV2Error("live libcudart provider set differs")
    maps_lower = Path("/proc/self/maps").read_text(errors="replace").lower()
    if any(
        marker in maps_lower for marker in ("libamdhip64", "libhsa-runtime64", "gemsim")
    ):
        raise SmokeV2Error("forbidden runtime mapping after Torch import")
    dso = (workspace / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True)
    if (
        dso.stat().st_size != contract.PYPTO_DSO_SIZE
        or sha256_file(dso) != contract.PYPTO_DSO_SHA256
    ):
        raise SmokeV2Error("exact PyPTO DSO differs")
    pypto_module = base.bootstrap_exact_pypto(workspace, dso.parent)
    from pypto import compiler
    from pypto.runtime import nvidia as runtime

    ir = pypto_module.ir
    info = compiler.get_nvidia_backend_build_info()
    if (
        not info.compiled
        or not info.compiler_factory_available
        or info.pypto_revision != contract.PYPTO_HEAD
        or info.tensor_ir_revision != contract.TENSOR_IR_HEAD
        or info.cuda_tile_revision != contract.CUDA_TILE_HEAD
        or info.llvm_revision != contract.LLVM_HEAD
    ):
        raise SmokeV2Error("exact PyPTO build identity differs")
    observation = runtime.observe_current_nvidia_runtime(
        contract.EXPECTED_DRIVER_RELEASE, expected_runtime
    )
    target = observation.target_info
    if (
        target.device_name != contract.EXPECTED_DEVICE_NAME
        or target.traits.compute_capability != 120
        or target.traits.multiprocessor_count != contract.EXPECTED_SM_COUNT
        or observation.cuda_runtime_library_path != expected_runtime
        or observation.cuda_driver_api_version
        < contract.MINIMUM_CUDA_DRIVER_API_VERSION
        or observation.cuda_runtime_api_version
        < contract.MINIMUM_CUDA_RUNTIME_API_VERSION
    ):
        raise SmokeV2Error("PyPTO runtime observation differs")
    if torch.cuda.is_current_stream_capturing():
        raise SmokeV2Error("compilation cannot begin during CUDA Graph capture")
    request = compiler.CompileRequest(target, base.toolchain_identity(compiler, info))
    replay = contract.replay_directory(workspace, run_id)
    replay.mkdir(mode=0o700, parents=False, exist_ok=False)
    replay_files: list[dict[str, object]] = []

    def replay_file(name: str, payload: bytes) -> None:
        path = replay / name
        digest = base.publish_no_replace(path, payload)
        replay_files.append(
            {
                "path": path.relative_to(workspace).as_posix(),
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    replay_file("compile-request.msgpack", request.serialize())
    if tuple(case.name for case in contract.CASE_SPECS) != contract.CASE_ORDER:
        raise SmokeV2Error("case order differs")
    artifacts: dict[str, Any] = {}
    artifact_records: list[dict[str, object]] = []
    hir_records: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        original = base.make_program(pypto_module, ir, case)
        hir_bytes = bytes(pypto_module.ir.serialize(original))
        restored = pypto_module.ir.deserialize(hir_bytes)
        if (
            len(hir_bytes) != case.expected_hir_bytes
            or sha256_bytes(hir_bytes) != case.expected_hir_sha256
            or not isinstance(restored, ir.Program)
            or not ir.structural_equal(original, restored, enable_auto_mapping=True)
            or bytes(pypto_module.ir.serialize(restored)) != hir_bytes
        ):
            raise SmokeV2Error(f"{case.name} HIR round-trip differs")
        restored_function = restored.get_function("fused_main")
        if (
            restored_function is None
            or list(restored_function.param_directions)
            != [ir.ParamDirection.In] * case.input_count
        ):
            raise SmokeV2Error(f"{case.name} input directions differ")
        replay_file(f"{case.name}.hir.msgpack", hir_bytes)
        source_ir = base.canonical_tensor_ir_source(case)
        if (
            len(source_ir) != case.expected_source_ir_bytes
            or sha256_bytes(source_ir) != case.expected_source_ir_digest
        ):
            raise SmokeV2Error(f"{case.name} source anchor differs")
        replay_file(f"{case.name}.source.mlir", source_ir)
        result = compiler.compile_structured_strict(
            restored, request, base.schedule(compiler, case.tile_sizes)
        )
        validated = base.validate_structured_result(
            compiler, result, request, case, source_ir
        )
        build_spec = validated["build_spec"]
        artifact = validated["artifact"]
        record = validated["record"]
        assert isinstance(record, dict)
        record.update(
            {
                "hir_sha256": sha256_bytes(hir_bytes),
                "hir_bytes": len(hir_bytes),
                "hir_roundtrip_exact": True,
            }
        )
        artifacts[case.name] = artifact
        replay_file(f"{case.name}.build-spec.msgpack", build_spec.serialize())
        replay_file(f"{case.name}.artifact.msgpack", artifact.serialize())
        replay_file(f"{case.name}.cubin", bytes(artifact.device_code))
        artifact_records.append(record)
        hir_records.append(
            {
                "case": case.name,
                "bytes": len(hir_bytes),
                "sha256": sha256_bytes(hir_bytes),
                "serialized_once": True,
                "deserialized_before_compile": True,
                "canonical_reserialization_equal": True,
                "structural_equal": True,
                "parameter_directions": ["In"] * case.input_count,
                "input_count": case.input_count,
                "assignment_count": case.assignment_count,
                "operator_sequence": list(case.operator_sequence),
            }
        )
    candidate_stream = torch.cuda.Stream(device=0)
    reference_stream = torch.cuda.Stream(device=0)
    if int(candidate_stream.cuda_stream) == int(reference_stream.cuda_stream):
        raise SmokeV2Error("candidate and reference streams are not distinct")
    executions: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            executions.append(
                base.execute_case(
                    torch,
                    runtime,
                    artifacts[case.name],
                    request,
                    observation,
                    candidate_stream,
                    reference_stream,
                    case,
                    repetition,
                    replay_file,
                )
            )
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeV2Error("forbidden provider imported during v2 smoke")
    integrity_paths = {
        "anchor_generator": workspace / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
        "compile_anchors": workspace / contract.COMPILE_ANCHORS_RELATIVE_PATH,
        "base_runner": workspace / BASE_RUNNER_RELATIVE_PATH,
        "contract": workspace / "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
        "runner": Path(__file__).resolve(strict=True),
        "controller": workspace / contract.CONTROLLER_RELATIVE_PATH,
        "preflight": workspace / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
        "control_validator": workspace / contract.CONTROL_VALIDATOR_RELATIVE_PATH,
        "environment_lock": workspace / "ENVIRONMENT.lock",
        "versions_lock": workspace / "VERSIONS.lock",
        "workspace_lock": workspace / "WORKSPACE.lock",
        "pypto_dso": workspace / contract.PYPTO_DSO_RELATIVE_PATH,
        "cuda_runtime": workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
    }
    integrity = {
        name: integrity_record(path, workspace)
        for name, path in integrity_paths.items()
    }
    pypto_identity = base.git_identity(workspace / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise SmokeV2Error("PyPTO source changed during smoke")
    initial_path = Path(os.environ["PYPTO_INITIAL_PREFLIGHT_REPORT_PATH"])
    initial_sha256 = os.environ["PYPTO_INITIAL_PREFLIGHT_REPORT_SHA256"]
    preflight_path = Path(os.environ["PYPTO_PREFLIGHT_REPORT_PATH"])
    preflight_sha256 = os.environ["PYPTO_PREFLIGHT_REPORT_SHA256"]
    if (
        sha256_file(initial_path) != initial_sha256
        or sha256_file(preflight_path) != preflight_sha256
    ):
        raise SmokeV2Error("v2 preflight sidecar changed during smoke")
    barrier = barrier_evidence["barrier"]
    gate_document = barrier_evidence["gate"]
    provisional = {
        "schema_version": contract.SMOKE_SCHEMA_VERSION,
        "smoke": contract.SMOKE_NAME,
        "acceptance": "gpu-execution-complete-awaiting-run-finalization",
        "scope": {
            "frontend_family": "FusedPointwiseV2",
            "fixed_fixture_set": "full-nine-case-numerical-v1",
            "fixed_fixture_correctness": True,
            "general_operator_correctness": False,
            "legacy_cp44_unchanged": True,
            "model_forward": False,
            "strict_coverage_result": False,
            "performance_result": False,
            "cuda_graph_result": False,
        },
        "inputs": {
            "integrity": integrity,
            "pypto": pypto_identity,
            "tensor_ir_head": contract.TENSOR_IR_HEAD,
            "cuda_tile_head": contract.CUDA_TILE_HEAD,
            "llvm_head": contract.LLVM_HEAD,
            "replay_files": replay_files,
            "control_manifest": child_gate["control_manifest"],
        },
        "run_context": {
            "run_id": run_id,
            "mode": "gpu-smoke",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "start_ticks": base._process_start_ticks(os.getpid()),
            "initial_preflight": {
                "path": initial_path.relative_to(workspace).as_posix(),
                "sha256": initial_sha256,
            },
            "preflight": {
                "path": preflight_path.relative_to(workspace).as_posix(),
                "sha256": preflight_sha256,
            },
            "gate": {
                "path": str(barrier["gate_path"]),
                "sha256": str(barrier["gate_sha256"]),
                "document": gate_document,
            },
            "start_barrier_sha256": sha256_file(
                Path(os.environ["PYPTO_GPU_SMOKE_START_BARRIER"])
            ),
            "protected_zero_nvidia_policy": (
                os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
            ),
            "admission_policy": child_gate["admission_policy"],
        },
        "runtime": {
            "torch": {
                "version": str(torch.__version__),
                "git_version": str(torch.version.git_version),
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "module_path": str(Path(torch.__file__).resolve(strict=True)),
            },
            "child_pre_cuda_gate": child_gate,
            "libcudart_paths": runtime_paths,
            "observation": {
                "device_ordinal": target.device_ordinal,
                "device_name": target.device_name,
                "device_uuid": target.device_uuid,
                "pci_device_id": target.pci_device_id,
                "traits": base.target_traits_document(target.traits),
                "cuda_toolkit_version": target.cuda_toolkit_version,
                "cuda_driver_version": target.cuda_driver_version,
                "tensor_ir_revision": target.tensor_ir_revision,
                "cuda_tile_revision": target.cuda_tile_revision,
                "supported_compute_dtypes": base.supported_dtype_names(
                    pypto_module, target.supported_compute_dtypes
                ),
                "cuda_driver_release_provenance": target.cuda_driver_version,
                "cuda_driver_api_version": observation.cuda_driver_api_version,
                "cuda_runtime_api_version": observation.cuda_runtime_api_version,
                "cuda_runtime_library_path": observation.cuda_runtime_library_path,
                "context_address": observation.context_address,
                "context_id": observation.context_id,
            },
            "compile_request": {
                "byte_identity_digest": request.byte_compile_identity_digest,
                "loader_compatibility_input_digest": request.loader_compatibility_input_digest,
                "device_autotune_identity_digest": request.device_autotune_identity_digest,
            },
            "hir_programs": hir_records,
            "artifacts": artifact_records,
            "executions": executions,
            "case_order": list(contract.CASE_ORDER),
            "compile_invocations_per_case": 1,
            "repetitions_per_case": 2,
            "module_lifetimes": len(executions),
            "explicit_packet_releases": len(executions),
            "explicit_unloads": len(executions),
            "non_default_current_stream": True,
            "distinct_nondefault_reference_stream": True,
            "reference_compute_outside_candidate_coverage": True,
            "external_reference_synchronizations": len(executions),
            "external_synchronization": True,
            "fallback_used": False,
            "forbidden_provider_imports": [],
        },
    }
    path = contract.provisional_path(workspace, run_id)
    payload = base.canonical_json(provisional)
    digest = base.publish_no_replace(path, payload)
    return provisional, path, digest


def main() -> int:
    document, path, digest = run_smoke()
    print(
        json.dumps(
            {"acceptance": document["acceptance"], "path": str(path), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


def __getattr__(name: str) -> Any:
    return getattr(base, name)


if __name__ == "__main__":
    raise SystemExit(main())
