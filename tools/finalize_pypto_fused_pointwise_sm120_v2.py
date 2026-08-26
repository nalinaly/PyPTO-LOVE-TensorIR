#!/usr/bin/env python3
"""CPU-only finalizer for the fused-pointwise SM120 v2 admission family."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
BASE_FINALIZER_RELATIVE_PATH = Path("tools/finalize_pypto_fused_pointwise_sm120.py")
BASE_FINALIZER_SIZE = 95_465
BASE_FINALIZER_SHA256 = (
    "c1724d138a6385d293ba5e79dcbf3208ebb0bac1f0dd734af738dddda5d26a37"
)
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalizeV2Error(RuntimeError):
    """The v2 provisional transaction cannot be promoted."""


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
        raise FinalizeV2Error(f"exact v2 finalizer source is noncanonical: {path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if expected_size is not None and len(raw) != expected_size:
        raise FinalizeV2Error(f"exact v2 finalizer source size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise FinalizeV2Error(f"exact v2 finalizer source hash differs: {path}")
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
    "_pypto_fused_pointwise_sm120_finalizer_v1_base",
    ROOT / BASE_FINALIZER_RELATIVE_PATH,
    expected_size=BASE_FINALIZER_SIZE,
    expected_sha256=BASE_FINALIZER_SHA256,
)
contract = load_exact(
    "_pypto_fused_pointwise_sm120_contract_v2",
    ROOT / "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
)
control = load_exact(
    "_pypto_fused_pointwise_sm120_control_manifest_v2",
    ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py",
)
control.reject_control_bytecode_cache(ROOT)
preflight_adapter = load_exact(
    "preflight_gpu_smoke_v2_finalizer",
    ROOT / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
)


def require_exact_keys(
    value: object, expected: set[str], description: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise FinalizeV2Error(
            f"{description} key set differs: expected={sorted(expected)}, actual={actual}"
        )
    return value


def require_int(value: object, description: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (positive and value <= 0)
    ):
        raise FinalizeV2Error(f"{description} is not a valid integer")
    return value


def require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalizeV2Error(f"{description} is not a lowercase SHA-256")
    return value


def load_canonical(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise FinalizeV2Error(f"{description} is not a regular canonical file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=base.duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizeV2Error(f"{description} is not JSON") from error
    if not isinstance(value, dict) or base.canonical_json(value) != raw:
        raise FinalizeV2Error(f"{description} is not canonical JSON")
    return value, raw


def expected_floor(requested: bool) -> int:
    return (
        contract.PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
        if requested
        else contract.EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
    )


def validate_preflight(value: dict[str, object], *, description: str) -> bool:
    required = {
        "coexistence_policy_version",
        "cwd",
        "failures",
        "gpu",
        "gpu_smoke_admission_policy",
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
    }
    require_exact_keys(value, required, description)
    requested = value.get("protected_zero_nvidia_gpu_smoke_requested")
    if requested not in {True, False}:
        raise FinalizeV2Error(f"{description} authorization is malformed")
    floor = expected_floor(bool(requested))
    protected = value.get("protected_processes")
    expected_waiver = bool(protected) if requested is True else False
    if (
        value.get("policy_version") != 3
        or value.get("gpu_smoke_policy_version") != 2
        or value.get("mode") != "gpu-smoke"
        or value.get("workspace") != str(ROOT)
        or value.get("cwd") != str(ROOT)
        or value.get("ok") is not True
        or value.get("failures") != []
        or value.get("nvidia_compute_audit_ok") is not True
        or value.get("nvidia_compute_pids") != []
        or value.get("protected_nvidia_compute_pids") != []
        or value.get("protected_nvidia_runtime_mapping_pids") != []
        or value.get("unreadable_protected_maps") != []
        or value.get("protected_gpu_smoke_waiver_applied") is not expected_waiver
        or value.get("memory_floor_kib") != floor
        or require_int(
            value.get("mem_available_kib"), f"{description} MemAvailable", positive=True
        )
        < floor
        or value.get("gpu_smoke_free_memory_floor_mib")
        != contract.GPU_FREE_MEMORY_FLOOR_MIB
        or value.get("gpu_smoke_admission_policy")
        != preflight_adapter.policy_document()
    ):
        raise FinalizeV2Error(f"{description} v2 admission differs")
    gpu = require_exact_keys(
        value.get("gpu"),
        {"name", "compute_capability", "memory_mib", "used_mib", "driver"},
        f"{description} GPU",
    )
    free = int(str(gpu["memory_mib"])) - int(str(gpu["used_mib"]))
    if (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != "12.0"
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
        or free < contract.GPU_FREE_MEMORY_FLOOR_MIB
    ):
        raise FinalizeV2Error(f"{description} GPU identity differs")
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
        f"{description} static Torch identity",
    )
    if (
        torch.get("environment_lock_sha256") != contract.ENVIRONMENT_LOCK_SHA256
        or torch.get("version") != contract.EXPECTED_TORCH_VERSION
        or torch.get("git_version") != contract.EXPECTED_TORCH_GIT
        or torch.get("cuda") != contract.EXPECTED_TORCH_CUDA
        or torch.get("hip") is not None
        or torch.get("python_executable")
        != str((ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True))
        or torch.get("libcudart_path")
        != str((ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))
        or torch.get("libcudart_size") != contract.CUDA_RUNTIME_SIZE
        or torch.get("libcudart_sha256") != contract.CUDA_RUNTIME_SHA256
        or torch.get("libcudart_record_owned") is not True
        or torch.get("nvidia_runtime_mappings") != []
        or torch.get("cuda_initialized") is not False
        or torch.get("forbidden_dsos") != []
    ):
        raise FinalizeV2Error(f"{description} static Torch identity differs")
    return bool(requested)


def validate_child_gate(
    value: object, *, requested: bool, control_identity: dict[str, object]
) -> None:
    gate = require_exact_keys(
        value,
        {
            "static_identity",
            "gpu",
            "free_memory_mib",
            "mem_available_kib",
            "host_memory_floor_kib",
            "admission_policy",
            "protected_heavy_pids",
            "protected_runtime_pids",
            "unreadable_protected_maps",
            "nvidia_compute_pids",
            "control_manifest",
            "base_runner",
        },
        "child pre-CUDA gate",
    )
    floor = expected_floor(requested)
    if (
        gate.get("control_manifest") != control_identity
        or gate.get("nvidia_compute_pids") != []
        or gate.get("protected_runtime_pids") != []
        or gate.get("unreadable_protected_maps") != []
        or gate.get("host_memory_floor_kib") != floor
        or require_int(
            gate.get("mem_available_kib"), "child MemAvailable", positive=True
        )
        < floor
        or require_int(
            gate.get("free_memory_mib"), "child GPU free memory", positive=True
        )
        < contract.GPU_FREE_MEMORY_FLOOR_MIB
        or gate.get("admission_policy") != preflight_adapter.policy_document()
        or gate.get("base_runner")
        != {
            "path": contract.BASE_RUNNER_RELATIVE_PATH.as_posix(),
            "bytes": contract.BASE_RUNNER_SIZE,
            "sha256": contract.BASE_RUNNER_SHA256,
        }
    ):
        raise FinalizeV2Error("child pre-CUDA admission differs")
    gpu = require_exact_keys(
        gate.get("gpu"),
        {"name", "compute_capability", "memory_mib", "used_mib", "driver"},
        "child pre-CUDA GPU",
    )
    if (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != "12.0"
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        raise FinalizeV2Error("child pre-CUDA GPU identity differs")


def validate_audit(
    value: object, *, description: str, authorized: bool, zero_owned: bool
) -> None:
    try:
        base.validate_audit(
            value,
            description,
            authorized=authorized,
            require_zero_owned=zero_owned,
        )
    except base.FinalizeError as error:
        raise FinalizeV2Error(str(error)) from error


def expected_replay_names() -> list[str]:
    names = ["compile-request.msgpack"]
    for case in contract.CASE_SPECS:
        names.extend(
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
            names.extend(
                f"{case.name}.r{repetition}.input{ordinal}.bin"
                for ordinal in range(case.input_count)
            )
            names.extend(
                [
                    f"{case.name}.r{repetition}.reference.bin",
                    f"{case.name}.r{repetition}.actual.bin",
                ]
            )
    return names


def validate_replay(
    provisional: dict[str, object], run_id: str
) -> list[dict[str, object]]:
    records = provisional["inputs"]["replay_files"]
    names = expected_replay_names()
    if not isinstance(records, list) or len(records) != len(names):
        raise FinalizeV2Error("replay file set is incomplete")
    replay = contract.replay_directory(ROOT, run_id)
    if (
        replay.is_symlink()
        or not replay.is_dir()
        or replay.resolve(strict=True) != replay
    ):
        raise FinalizeV2Error("replay directory is noncanonical")
    if sorted(path.name for path in replay.iterdir()) != sorted(
        [*names, contract.PROVISIONAL_NAME]
    ):
        raise FinalizeV2Error("replay directory has a missing or extra file")
    normalized: list[dict[str, object]] = []
    for record, name in zip(records, names, strict=True):
        record = require_exact_keys(
            record, {"path", "bytes", "sha256"}, f"replay {name}"
        )
        path = replay / name
        if record.get("path") != path.relative_to(ROOT).as_posix():
            raise FinalizeV2Error(f"replay path differs: {name}")
        size = require_int(record.get("bytes"), f"replay {name} size", positive=True)
        digest = require_sha(record.get("sha256"), f"replay {name}")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or sha256_file(path) != digest
        ):
            raise FinalizeV2Error(f"replay bytes differ: {name}")
        normalized.append(dict(record))
    return normalized


def audit_numerical_replay(
    provisional: dict[str, object], run_id: str
) -> list[dict[str, object]]:
    replay = contract.replay_directory(ROOT, run_id)
    executions = provisional["runtime"]["executions"]
    output: list[dict[str, object]] = []
    index = 0
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            execution = executions[index]
            index += 1
            inputs: list[list[int]] = []
            for ordinal in range(case.input_count):
                raw = (
                    replay / f"{case.name}.r{repetition}.input{ordinal}.bin"
                ).read_bytes()
                words = base._decode_words(raw, case.dtype)
                if words != base._input_words(case, repetition, ordinal):
                    raise FinalizeV2Error(
                        "independent CPU input reconstruction differs"
                    )
                if sha256_bytes(raw) != execution["input_hashes"][ordinal].get(
                    "before_sha256"
                ):
                    raise FinalizeV2Error("raw input hash join differs")
                inputs.append(words)
            reference_raw = (
                replay / f"{case.name}.r{repetition}.reference.bin"
            ).read_bytes()
            actual_raw = (replay / f"{case.name}.r{repetition}.actual.bin").read_bytes()
            reference = base._decode_words(reference_raw, case.dtype)
            actual = base._decode_words(actual_raw, case.dtype)
            cpu = base._cpu_reference_words(case, inputs)
            torch_cpu = base._compare_words(case, reference, cpu)
            candidate_torch = base._compare_words(case, actual, reference)
            candidate_cpu = base._compare_words(case, actual, cpu)
            if execution["comparison"] != candidate_torch:
                raise FinalizeV2Error("recorded/reconstructed numerical metrics differ")
            output.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "independent_cpu_input_reconstruction": True,
                    "independent_cpu_reference_reconstruction": True,
                    "actual_sha256": sha256_bytes(actual_raw),
                    "candidate_vs_cpu": candidate_cpu,
                    "candidate_vs_torch": candidate_torch,
                    "cpu_reference_sha256": sha256_bytes(
                        base.pack_numerical_words(cpu, case.dtype)
                    ),
                    "torch_reference_sha256": sha256_bytes(reference_raw),
                    "torch_vs_cpu": torch_cpu,
                }
            )
    return output


def audit_replay_semantics(
    provisional: dict[str, object], run_id: str
) -> dict[str, object]:
    replay = contract.replay_directory(ROOT, run_id)
    python = (ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True)
    command = [
        str(python),
        "-I",
        "-B",
        "-S",
        "-c",
        base.REPLAY_AUDIT_PROGRAM,
        str(ROOT),
        str(replay),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise FinalizeV2Error("CPU-only DSO replay failed: " + completed.stderr[-2048:])
    audited = json.loads(completed.stdout, object_pairs_hook=base.duplicate_key_guard)
    require_exact_keys(
        audited,
        {"compile_request", "target_info", "hir_programs", "artifacts"},
        "CPU-only DSO replay",
    )
    runtime = provisional["runtime"]
    observation = runtime["observation"]
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
    if audited.get("compile_request") != runtime.get("compile_request") or audited.get(
        "target_info"
    ) != {name: observation[name] for name in target_fields}:
        raise FinalizeV2Error("replayed request/target differs")
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
        raise FinalizeV2Error("replayed HIR differs")
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
        raise FinalizeV2Error("replayed Artifact semantics differ")
    return {
        "command_sha256": sha256_bytes("\0".join(command).encode()),
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        **audited,
    }


def validate_provisional(
    provisional: dict[str, object], control_identity: dict[str, object]
) -> None:
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
    if (
        provisional.get("schema_version") != 2
        or provisional.get("smoke") != contract.SMOKE_NAME
        or provisional.get("acceptance")
        != "gpu-execution-complete-awaiting-run-finalization"
        or inputs.get("control_manifest") != control_identity
    ):
        raise FinalizeV2Error("provisional identity differs")
    require_exact_keys(
        provisional.get("run_context"),
        {
            "run_id",
            "mode",
            "pid",
            "pgid",
            "start_ticks",
            "initial_preflight",
            "preflight",
            "gate",
            "start_barrier_sha256",
            "protected_zero_nvidia_policy",
            "admission_policy",
        },
        "provisional run context",
    )
    require_exact_keys(
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
        "provisional runtime",
    )
    try:
        base.validate_scope(provisional)
        base.validate_frontend_results(provisional)
    except base.FinalizeError as error:
        raise FinalizeV2Error(str(error)) from error


def validate_run_documents(
    *,
    run_id: str,
    process: dict[str, object],
    initial: dict[str, object],
    initial_raw: bytes,
    preflight: dict[str, object],
    preflight_raw: bytes,
    gate: dict[str, object],
    gate_raw: bytes,
    barrier: dict[str, object],
    barrier_raw: bytes,
    provisional: dict[str, object],
    control_identity: dict[str, object],
) -> None:
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
            "initial_preflight",
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
    initial_requested = validate_preflight(initial, description="initial preflight")
    requested = validate_preflight(preflight, description="action preflight")
    if initial_requested is not requested:
        raise FinalizeV2Error("initial/action authorization differs")
    floor = expected_floor(requested)
    resource = process.get("resource_policy")
    gpu_smoke = process.get("gpu_smoke")
    if (
        process.get("schema") != 4
        or process.get("run_id") != run_id
        or process.get("workspace") != str(ROOT)
        or process.get("mode") != "gpu-smoke"
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("command") != contract.fixed_child_command(ROOT)
        or not isinstance(resource, dict)
        or resource.get("owned_run_pause_memory_kib")
        != contract.OWNED_RUN_ABORT_MEMORY_FLOOR_KIB
        or resource.get("timeout_seconds") != contract.GPU_SMOKE_TIMEOUT_SECONDS
        or resource.get("minimum_free_disk_bytes")
        != contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
        or not isinstance(gpu_smoke, dict)
        or gpu_smoke.get("policy_version") != 2
        or gpu_smoke.get("requested") is not requested
        or gpu_smoke.get("authorization")
        != (contract.GPU_SMOKE_AUTHORIZATION if requested else None)
        or gpu_smoke.get("memory_floor_kib") != floor
        or gpu_smoke.get("gpu_free_memory_floor_mib")
        != contract.GPU_FREE_MEMORY_FLOOR_MIB
    ):
        raise FinalizeV2Error("process v2 safety policy differs")
    coexistence = process.get("coexistence")
    if (
        not isinstance(coexistence, dict)
        or coexistence.get("requested") is not False
        or coexistence.get("waiver_applied") is not False
    ):
        raise FinalizeV2Error("GPU smoke used the CPU-only coexistence policy")
    initial_path = ROOT / "runs" / run_id / "initial-preflight.json"
    preflight_path = ROOT / "runs" / run_id / "preflight.json"
    if process.get("initial_preflight") != {
        "path": str(initial_path),
        "sha256": sha256_bytes(initial_raw),
    } or process.get("preflight") != {
        "path": str(preflight_path),
        "sha256": sha256_bytes(preflight_raw),
    }:
        raise FinalizeV2Error("process/preflight digest join differs")
    validate_audit(
        process.get("gpu_smoke_pre_release_audit"),
        description="pre-release audit",
        authorized=requested,
        zero_owned=True,
    )
    validate_audit(
        process.get("gpu_smoke_last_audit"),
        description="periodic audit",
        authorized=requested,
        zero_owned=False,
    )
    validate_audit(
        process.get("gpu_smoke_post_exit_audit"),
        description="post-exit audit",
        authorized=requested,
        zero_owned=True,
    )
    gate_path = ROOT / "runs" / run_id / "gpu-smoke-gate.json"
    identity = {
        "schema": 2,
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
            "initial_preflight",
            "preflight",
            "static_identity",
            "control_manifest",
            "runtime_isolation",
            "admission_policy",
        },
        "pre-release gate",
    )
    if (
        any(gate.get(name) != value for name, value in identity.items())
        or any(barrier.get(name) != value for name, value in identity.items())
        or gate.get("command") != contract.fixed_child_command(ROOT)
        or gate.get("control_manifest") != control_identity
        or gate.get("initial_preflight") != process.get("initial_preflight")
        or gate.get("preflight") != process.get("preflight")
        or gate.get("admission_policy") != preflight_adapter.policy_document()
        or gate.get("runtime_isolation") != process.get("gpu_smoke_pre_release_audit")
        or barrier.get("gate_path") != str(gate_path)
        or barrier.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
    ):
        raise FinalizeV2Error("gate/barrier identity differs")
    run = provisional["run_context"]
    if (
        run.get("initial_preflight")
        != {
            "path": initial_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(initial_raw),
        }
        or run.get("preflight")
        != {
            "path": preflight_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(preflight_raw),
        }
        or run.get("gate")
        != {"path": str(gate_path), "sha256": sha256_bytes(gate_raw), "document": gate}
        or run.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
        or run.get("admission_policy") != preflight_adapter.policy_document()
        or run.get("protected_zero_nvidia_policy") is not requested
    ):
        raise FinalizeV2Error("provisional run-context joins differ")
    validate_child_gate(
        provisional["runtime"].get("child_pre_cuda_gate"),
        requested=requested,
        control_identity=control_identity,
    )
    child_gate = provisional["runtime"]["child_pre_cuda_gate"]
    if child_gate.get("static_identity") != gate.get("static_identity"):
        raise FinalizeV2Error("parent/child static identity differs")


def validate_integrity(provisional: dict[str, object]) -> None:
    expected = {
        "anchor_generator": ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
        "compile_anchors": ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH,
        "base_runner": ROOT / contract.BASE_RUNNER_RELATIVE_PATH,
        "contract": ROOT / "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
        "runner": ROOT / contract.RUNNER_RELATIVE_PATH,
        "controller": ROOT / contract.CONTROLLER_RELATIVE_PATH,
        "preflight": ROOT / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
        "control_validator": ROOT / contract.CONTROL_VALIDATOR_RELATIVE_PATH,
        "environment_lock": ROOT / "ENVIRONMENT.lock",
        "versions_lock": ROOT / "VERSIONS.lock",
        "workspace_lock": ROOT / "WORKSPACE.lock",
        "pypto_dso": ROOT / contract.PYPTO_DSO_RELATIVE_PATH,
        "cuda_runtime": ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH,
    }
    integrity = provisional["inputs"].get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != set(expected):
        raise FinalizeV2Error("provisional integrity set differs")
    for name, path in expected.items():
        resolved = path.resolve(strict=True)
        record = {
            "path": resolved.relative_to(ROOT).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        if integrity.get(name) != record:
            raise FinalizeV2Error(f"provisional integrity differs: {name}")


def validate_fixed_inputs() -> dict[str, dict[str, object]]:
    expected = {
        "runner": (
            ROOT / contract.RUNNER_RELATIVE_PATH,
            contract.RUNNER_SIZE,
            contract.RUNNER_SHA256,
        ),
        "base_runner": (
            ROOT / contract.BASE_RUNNER_RELATIVE_PATH,
            contract.BASE_RUNNER_SIZE,
            contract.BASE_RUNNER_SHA256,
        ),
        "compile_anchors": (
            ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH,
            contract.COMPILE_ANCHORS_SIZE,
            contract.COMPILE_ANCHORS_SHA256,
        ),
        "pypto_dso": (
            ROOT / contract.PYPTO_DSO_RELATIVE_PATH,
            contract.PYPTO_DSO_SIZE,
            contract.PYPTO_DSO_SHA256,
        ),
        "cuda_runtime": (
            ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH,
            contract.CUDA_RUNTIME_SIZE,
            contract.CUDA_RUNTIME_SHA256,
        ),
        "python": (
            ROOT / contract.PYTHON_REAL_RELATIVE_PATH,
            contract.PYTHON_SIZE,
            contract.PYTHON_SHA256,
        ),
    }
    output: dict[str, dict[str, object]] = {}
    for name, (path, size, digest) in expected.items():
        resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not path.is_file()
            or resolved != path
            or resolved.stat().st_size != size
            or sha256_file(resolved) != digest
        ):
            raise FinalizeV2Error(f"fixed input differs: {name}")
        output[name] = {
            "path": resolved.relative_to(ROOT).as_posix(),
            "bytes": size,
            "sha256": digest,
        }
    if sha256_file(ROOT / "ENVIRONMENT.lock") != contract.ENVIRONMENT_LOCK_SHA256:
        raise FinalizeV2Error("ENVIRONMENT.lock differs")
    return output


def validate_runtime_identity(
    provisional: dict[str, object],
    preflight: dict[str, object],
    gate: dict[str, object],
) -> None:
    runtime = provisional["runtime"]
    torch = runtime["torch"]
    expected_torch = (
        ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
    ).resolve(strict=True)
    if (
        torch.get("version") != contract.EXPECTED_TORCH_VERSION
        or torch.get("git_version") != contract.EXPECTED_TORCH_GIT
        or torch.get("cuda") != contract.EXPECTED_TORCH_CUDA
        or torch.get("hip") is not None
        or Path(str(torch.get("module_path", ""))) != expected_torch
        or runtime.get("libcudart_paths")
        != [str((ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))]
        or preflight.get("torch") != gate.get("static_identity")
    ):
        raise FinalizeV2Error("runtime Torch/provider identity differs")
    observation = runtime["observation"]
    traits = observation["traits"]
    if (
        observation.get("device_name") != contract.EXPECTED_DEVICE_NAME
        or traits.get("compute_capability") != 120
        or traits.get("multiprocessor_count") != contract.EXPECTED_SM_COUNT
        or observation.get("cuda_toolkit_version")
        != contract.EXPECTED_CUDA_TOOLKIT_VERSION
        or observation.get("cuda_driver_version") != contract.EXPECTED_DRIVER_RELEASE
        or observation.get("tensor_ir_revision") != contract.TENSOR_IR_HEAD
        or observation.get("cuda_tile_revision") != contract.CUDA_TILE_HEAD
        or observation.get("cuda_runtime_library_path")
        != str((ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))
        or require_int(
            observation.get("cuda_driver_api_version"), "driver API", positive=True
        )
        < contract.MINIMUM_CUDA_DRIVER_API_VERSION
        or require_int(
            observation.get("cuda_runtime_api_version"), "runtime API", positive=True
        )
        < contract.MINIMUM_CUDA_RUNTIME_API_VERSION
    ):
        raise FinalizeV2Error("runtime target identity differs")
    for value in runtime["compile_request"].values():
        require_sha(value, "CompileRequest identity")


def finalize(
    *, workspace: Path, run_id: str, expected_provisional_sha256: str
) -> tuple[dict[str, object], Path, str]:
    resolved = workspace.resolve(strict=True)
    if workspace.absolute() != resolved or resolved != ROOT:
        raise FinalizeV2Error("workspace must be the exact root")
    base.require_no_site_finalizer()
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise FinalizeV2Error("run ID is malformed")
    require_sha(expected_provisional_sha256, "external provisional anchor")
    control_identity = control.validate_control_manifest(ROOT)
    run_dir = ROOT / "runs" / run_id
    process_path = run_dir / "process.json"
    initial_path = run_dir / "initial-preflight.json"
    preflight_path = run_dir / "preflight.json"
    gate_path = run_dir / "gpu-smoke-gate.json"
    barrier_path = run_dir / "gpu-smoke-start-barrier.json"
    process, process_raw = load_canonical(process_path, "process metadata")
    initial, initial_raw = load_canonical(initial_path, "initial preflight")
    preflight, preflight_raw = load_canonical(preflight_path, "action preflight")
    gate, gate_raw = load_canonical(gate_path, "GPU-smoke gate")
    barrier, barrier_raw = load_canonical(barrier_path, "start barrier")
    provisional_path = contract.provisional_path(ROOT, run_id)
    provisional, provisional_raw = load_canonical(provisional_path, "provisional")
    for path in (process_path, initial_path, preflight_path, gate_path, barrier_path):
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise FinalizeV2Error(f"run sidecar mode differs: {path.name}")
    if stat.S_IMODE(provisional_path.stat().st_mode) != 0o444:
        raise FinalizeV2Error("provisional mode differs")
    if sha256_bytes(provisional_raw) != expected_provisional_sha256:
        raise FinalizeV2Error("external provisional SHA-256 differs")
    validate_provisional(provisional, control_identity)
    validate_run_documents(
        run_id=run_id,
        process=process,
        initial=initial,
        initial_raw=initial_raw,
        preflight=preflight,
        preflight_raw=preflight_raw,
        gate=gate,
        gate_raw=gate_raw,
        barrier=barrier,
        barrier_raw=barrier_raw,
        provisional=provisional,
        control_identity=control_identity,
    )
    validate_integrity(provisional)
    validate_runtime_identity(provisional, preflight, gate)
    exact_files = validate_fixed_inputs()
    replay_files = validate_replay(provisional, run_id)
    numerical_replay = audit_numerical_replay(provisional, run_id)
    replay_semantics = audit_replay_semantics(provisional, run_id)
    pypto_identity = base.git_identity(ROOT / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise FinalizeV2Error("PyPTO identity differs at finalization")
    report = {
        "schema_version": 2,
        "smoke": contract.SMOKE_NAME,
        "status": "accepted-real-sm120-fused-pointwise-nine-case-correctness-gate-v2",
        "scope": provisional["scope"],
        "not_claimed": [
            "general FusedPointwiseV2 correctness",
            "other chains shapes ranks scalars subnormals or high_precision behavior",
            "Cubin determinism across different builds or toolchains",
            "reduction matmul memory lowering performance CUDA Graph framework model or strict coverage",
            "any extension or reinterpretation of accepted CP44 or CP46 v1",
        ],
        "admission_policy": preflight_adapter.policy_document(),
        "run": {
            "run_id": run_id,
            "process_sha256": sha256_bytes(process_raw),
            "initial_preflight_sha256": sha256_bytes(initial_raw),
            "preflight_sha256": sha256_bytes(preflight_raw),
            "gate_sha256": sha256_bytes(gate_raw),
            "start_barrier_sha256": sha256_bytes(barrier_raw),
            "provisional_sha256": expected_provisional_sha256,
            "command": contract.fixed_child_command(ROOT),
            "zero_nvidia_interference": True,
        },
        "inputs": {
            "control_manifest": control_identity,
            "pypto": pypto_identity,
            "replay_files": replay_files,
            "numerical_replay": numerical_replay,
            "replay_semantics": replay_semantics,
            "exact_files": exact_files,
            "base_v1_runner": {
                "path": contract.BASE_RUNNER_RELATIVE_PATH.as_posix(),
                "bytes": contract.BASE_RUNNER_SIZE,
                "sha256": contract.BASE_RUNNER_SHA256,
            },
            "compile_anchors": {
                "path": contract.COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
                "bytes": contract.COMPILE_ANCHORS_SIZE,
                "sha256": contract.COMPILE_ANCHORS_SHA256,
            },
        },
        "result": provisional["runtime"],
        "finalizer": {
            "path": Path(__file__).resolve(strict=True).relative_to(ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "base_v1_finalizer_sha256": BASE_FINALIZER_SHA256,
            "cpu_only_deserialization": True,
            "torch_cuda_initialized": False,
        },
    }
    output_parent = ROOT / contract.FINAL_REPORT_DIRECTORY
    output_parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    output = contract.final_report_path(ROOT, run_id)
    try:
        digest = base.publish_no_replace(output, report)
    except base.FinalizeError as error:
        raise FinalizeV2Error(str(error)) from error
    return report, output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-provisional-sha256", required=True)
    args = parser.parse_args()
    report, output, digest = finalize(
        workspace=args.workspace,
        run_id=args.run_id,
        expected_provisional_sha256=args.expected_provisional_sha256,
    )
    print(
        json.dumps(
            {"path": str(output), "sha256": digest, "status": report["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
