#!/usr/bin/env python3
"""Owned-process controller for additive CPU-only coexistence policy v2."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import pathlib
import secrets
import signal
import shutil
import subprocess
import sys
import time
import traceback
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ControllerError(RuntimeError):
    """The CPU-v2 child cannot be admitted or managed safely."""


def load_source(name: str, path: pathlib.Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControllerError(f"CPU-v2 source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = load_source(
    "_pypto_cpu_coexistence_v2_contract_for_controller",
    ROOT / "tools/_pypto_cpu_coexistence_v2_contract.py",
)
preflight = load_source(
    "preflight_cpu_coexistence_v2_controller",
    ROOT / contract.PREFLIGHT_RELATIVE_PATH,
)
control = load_source(
    "_pypto_cpu_coexistence_v2_control_for_controller",
    ROOT / contract.CONTROL_VALIDATOR_RELATIVE_PATH,
)
stop_run = contract.load_exact(
    "stop_run",
    ROOT / contract.BASE_STOP_RELATIVE_PATH,
    contract.BASE_STOP_SIZE,
    contract.BASE_STOP_SHA256,
)
base_nvidia_contract = contract.load_exact(
    "_pypto_nvidia_executable_sm120_contract",
    ROOT / contract.BASE_NVIDIA_CONTRACT_RELATIVE_PATH,
    contract.BASE_NVIDIA_CONTRACT_SIZE,
    contract.BASE_NVIDIA_CONTRACT_SHA256,
)
base_nvidia_control = contract.load_exact(
    "_pypto_nvidia_sm120_control_manifest",
    ROOT / contract.BASE_NVIDIA_CONTROL_RELATIVE_PATH,
    contract.BASE_NVIDIA_CONTROL_SIZE,
    contract.BASE_NVIDIA_CONTROL_SHA256,
)
base_preflight = contract.load_exact(
    "preflight",
    ROOT / contract.BASE_PREFLIGHT_RELATIVE_PATH,
    contract.BASE_PREFLIGHT_SIZE,
    contract.BASE_PREFLIGHT_SHA256,
)
base = contract.load_exact(
    "_pypto_cpu_v2_base_isolation",
    ROOT / contract.BASE_ISOLATION_RELATIVE_PATH,
    contract.BASE_ISOLATION_SIZE,
    contract.BASE_ISOLATION_SHA256,
)
if (
    base.preflight_tool is not base_preflight
    or base.stop_run is not stop_run
    or base.nvidia_smoke_contract is not base_nvidia_contract
    or base.nvidia_smoke_control is not base_nvidia_control
):
    raise ControllerError("CPU-v2 exact base dependency identity differs")


START_GATE_PROGRAM = r"""
import hashlib,json,os,sys,time
from pathlib import Path
gate=Path(sys.argv[1]); run_id=sys.argv[2]; expected_digest=sys.argv[3]; timeout=float(sys.argv[4]); command=sys.argv[5:]
if timeout <= 0: raise SystemExit(75)
deadline=time.monotonic()+timeout
while not gate.exists():
    if time.monotonic() >= deadline: raise SystemExit(75)
    time.sleep(0.05)
if gate.is_symlink() or not gate.is_file() or gate.resolve(strict=True) != gate: raise SystemExit(75)
if gate.stat().st_mode & 0o777 != 0o600: raise SystemExit(75)
try: document=json.loads(gate.read_bytes())
except Exception: raise SystemExit(75)
if set(document) != {"schema_version","kind","run_id","pid","pgid","start_ticks","command_sha256"}: raise SystemExit(75)
if document["schema_version"] != 1 or document["kind"] != "pypto-cpu-only-coexistence-v2" or document["run_id"] != run_id: raise SystemExit(75)
if document["pid"] != os.getpid() or document["pgid"] != os.getpgrp(): raise SystemExit(75)
observed_start=int(Path(f"/proc/{os.getpid()}/stat").read_text().rpartition(")")[2].split()[19])
if document["start_ticks"] != observed_start: raise SystemExit(75)
if document["command_sha256"] != expected_digest: raise SystemExit(75)
if hashlib.sha256("\0".join(command).encode()).hexdigest() != expected_digest: raise SystemExit(75)
os.execvpe(command[0],command,os.environ)
"""
if (
    contract.START_GATE_SCHEMA_VERSION != 1
    or contract.START_GATE_TIMEOUT_SECONDS != 60
):
    raise ControllerError("CPU-v2 start-gate program constants differ")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256("\0".join(command).encode()).hexdigest()


def gated_launch_command(
    command: list[str],
    *,
    run_id: str,
    gate_path: pathlib.Path,
    timeout_seconds: float = contract.START_GATE_TIMEOUT_SECONDS,
) -> list[str]:
    python = ROOT / base_nvidia_contract.PYTHON_REAL_RELATIVE_PATH
    if (
        python.is_symlink()
        or not python.is_file()
        or python.resolve(strict=True) != python
        or python.stat().st_size != base_nvidia_contract.PYTHON_SIZE
        or sha256_file(python) != base_nvidia_contract.PYTHON_SHA256
    ):
        raise ControllerError("CPU-v2 exact start-gate interpreter differs")
    return [
        str(python),
        "-I",
        "-B",
        "-S",
        "-c",
        START_GATE_PROGRAM,
        str(gate_path),
        run_id,
        command_sha256(command),
        str(timeout_seconds),
        *command,
    ]


def release_start_gate(
    *,
    path: pathlib.Path,
    metadata_path: pathlib.Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    if metadata.get("metadata_complete") is not True:
        raise ControllerError("CPU-v2 incomplete metadata cannot release start gate")
    gate = metadata.get("start_gate")
    if (
        not isinstance(gate, dict)
        or set(gate)
        != {
            "schema_version",
            "path",
            "timeout_seconds",
            "released",
            "sha256",
            "released_at",
        }
        or gate.get("schema_version") != contract.START_GATE_SCHEMA_VERSION
        or gate.get("path") != str(path)
        or gate.get("timeout_seconds") != contract.START_GATE_TIMEOUT_SECONDS
        or gate.get("released") is not False
        or gate.get("sha256") is not None
        or gate.get("released_at") is not None
        or not isinstance(metadata.get("requested_command_sha256"), str)
        or contract.SHA256_PATTERN.fullmatch(metadata["requested_command_sha256"])
        is None
    ):
        raise ControllerError("CPU-v2 start gate metadata differs")
    if path.exists() or path.is_symlink():
        raise ControllerError("CPU-v2 start gate already exists")
    document = {
        "schema_version": contract.START_GATE_SCHEMA_VERSION,
        "kind": contract.POLICY_KIND,
        "run_id": metadata["run_id"],
        "pid": metadata["pid"],
        "pgid": metadata["pgid"],
        "start_ticks": metadata["start_ticks"],
        "command_sha256": metadata["requested_command_sha256"],
    }
    base.atomic_json(path, document)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or path.stat().st_mode & 0o777 != 0o600
    ):
        raise ControllerError("CPU-v2 start gate publication differs")
    gate.update(
        {
            "released": True,
            "sha256": sha256_file(path),
            "released_at": datetime.datetime.now(datetime.UTC).strftime(
                "%Y%m%dT%H%M%SZ"
            ),
        }
    )
    base.atomic_json(metadata_path, metadata)
    return document


def acquire_shared_environment_lease() -> object | None:
    handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        try:
            return base.acquire_environment_lock("pypto-nvidia", "shared")
        except base.EnvironmentLockBusy as error:
            print(str(error), file=sys.stderr)
            return None
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def run_preflight(*, description: str) -> tuple[int, dict[str, object]]:
    report = preflight.build_report()
    preflight.validate_report(report, description=description)
    return (0 if report.get("ok") is True else 75), report


def cpu_environment(
    *, run_id: str, run_dir: pathlib.Path, lease: object
) -> dict[str, str]:
    prefix = base.ENVIRONMENTS["pypto-nvidia"].resolve(strict=True)
    environment = base.isolated_environment(
        run_id,
        run_dir,
        environment_prefix=prefix,
        framework_profile="pypto",
        protected_cpu_only_coexistence_requested=True,
        protected_zero_nvidia_gpu_smoke_requested=False,
        exact_nvidia_smoke=True,
    )
    environment.update(base.environment_lock_markers(lease))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYTHONPATH": "",
            "SGLANG_PLUGINS": "__pypto_cpu_v2_no_plugins__",
            "PYPTO_FRAMEWORK_PROFILE": "cpu-only-v2",
            "PYPTO_RUN_MODE": contract.MODE,
            "PYPTO_CPU_COEXISTENCE_V2": "1",
            "PYPTO_CPU_COEXISTENCE_POLICY_VERSION": str(contract.SCHEMA_VERSION),
            "PYPTO_CPU_LAUNCH_MEMORY_FLOOR_KIB": str(
                contract.LAUNCH_MEMORY_FLOOR_KIB
            ),
            "PYPTO_CPU_RESUME_MEMORY_FLOOR_KIB": str(
                contract.RESUME_MEMORY_FLOOR_KIB
            ),
            "PYPTO_CPU_PAUSE_MEMORY_FLOOR_KIB": str(
                contract.PAUSE_MEMORY_FLOOR_KIB
            ),
            "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED": "1",
            "PYPTO_PROTECTED_ACTIVITY_WAIVER_APPLIED": "0",
        }
    )
    return environment


def build_metadata(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    run_id: str,
    command: list[str],
    launch_command: list[str],
    start_gate_path: pathlib.Path,
    timestamp: str,
    lease: object,
    initial_path: pathlib.Path,
    initial_sha256: str,
    action_path: pathlib.Path,
    action_sha256: str,
    action_report: dict[str, object],
    timeout_seconds: int,
    minimum_free_disk_bytes: int,
    control_identity: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "kind": contract.POLICY_KIND,
        "metadata_complete": True,
        "run_id": run_id,
        "workspace": str(ROOT),
        "environment": str(base.ENVIRONMENTS["pypto-nvidia"].resolve(strict=True)),
        "environment_access_lock": {
            "path": str(lease.path),
            "mode": lease.mode,
            "device": lease.device,
            "inode": lease.inode,
        },
        "mode": contract.MODE,
        "framework_launch": False,
        "command": command,
        "launch_command": launch_command,
        "requested_command_sha256": command_sha256(command),
        "start_gate": {
            "schema_version": contract.START_GATE_SCHEMA_VERSION,
            "path": str(start_gate_path),
            "timeout_seconds": contract.START_GATE_TIMEOUT_SECONDS,
            "released": False,
            "sha256": None,
            "released_at": None,
        },
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": base.process_start_ticks(process.pid),
        "started_at": timestamp,
        "status": "running",
        "return_code": None,
        "initial_preflight": {
            "path": str(initial_path),
            "sha256": initial_sha256,
        },
        "preflight": {"path": str(action_path), "sha256": action_sha256},
        "coexistence": {
            "policy": copy.deepcopy(action_report["admission_policy"]),
            "requested": True,
            "waiver_applied": bool(
                action_report.get("protected_activity_waiver_applied")
            ),
            "protected_heavy_processes": action_report.get(
                "protected_heavy_processes", []
            ),
            "protected_nvidia_compute_pids": action_report.get(
                "protected_nvidia_compute_pids", []
            ),
            "protected_nvidia_runtime_mapping_pids": action_report.get(
                "protected_nvidia_runtime_mapping_pids", []
            ),
        },
        "resource_policy": {
            "launch_memory_floor_kib": contract.LAUNCH_MEMORY_FLOOR_KIB,
            "resume_memory_floor_kib": contract.RESUME_MEMORY_FLOOR_KIB,
            "pause_memory_floor_kib": contract.PAUSE_MEMORY_FLOOR_KIB,
            "timeout_seconds": timeout_seconds,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
        },
        "control_manifest": control_identity,
        "coexistence_pauses": [],
    }


def audit_runtime_state(metadata: dict[str, object]) -> dict[str, object]:
    compute_pids = preflight.nvidia_compute_pids()
    all_processes, protected, workspace = preflight.process_table()
    protected_ids = {item.pid for item in protected}
    workspace_ids = {item.pid for item in workspace}
    protected_runtime, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    owned_compute_list, _unclassified = base.partition_compute_pids(
        compute_pids, metadata, all_processes
    )
    owned_compute = set(owned_compute_list)
    protected_compute = (compute_pids & protected_ids) - owned_compute
    workspace_compute = (
        compute_pids & workspace_ids
    ) - owned_compute - protected_compute
    external_compute = (
        compute_pids - owned_compute - protected_compute - workspace_compute
    )
    result = {
        "nvidia_compute_pids": sorted(compute_pids),
        "owned_nvidia_compute_pids": sorted(owned_compute),
        "protected_nvidia_compute_pids": sorted(protected_compute),
        "workspace_nvidia_compute_pids": sorted(workspace_compute),
        "external_nvidia_compute_pids": sorted(external_compute),
        "protected_nvidia_runtime_mapping_pids": protected_runtime,
        "unreadable_protected_maps": unreadable,
    }
    return validate_runtime_audit(result)


def validate_runtime_audit(value: object) -> dict[str, object]:
    expected = {
        "nvidia_compute_pids",
        "owned_nvidia_compute_pids",
        "external_nvidia_compute_pids",
        "protected_nvidia_compute_pids",
        "workspace_nvidia_compute_pids",
        "protected_nvidia_runtime_mapping_pids",
        "unreadable_protected_maps",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ControllerError("CPU-v2 runtime audit key set differs")
    for name in expected:
        pids = value[name]
        if (
            not isinstance(pids, list)
            or pids != sorted(set(pids))
            or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                for pid in pids
            )
        ):
            raise ControllerError(f"CPU-v2 runtime audit PID set differs: {name}")
    classes = (
        set(value["owned_nvidia_compute_pids"]),
        set(value["protected_nvidia_compute_pids"]),
        set(value["workspace_nvidia_compute_pids"]),
        set(value["external_nvidia_compute_pids"]),
    )
    if any(
        classes[left] & classes[right]
        for left in range(len(classes))
        for right in range(left + 1, len(classes))
    ) or set(value["nvidia_compute_pids"]) != set().union(*classes):
        raise ControllerError("CPU-v2 NVIDIA PID partition differs")
    return value


def pressure_reason(
    *,
    available_kib: int | None,
    disk_free_bytes: int | None,
    minimum_free_disk_bytes: int,
    audit: dict[str, object] | None,
    observation_errors: dict[str, str],
) -> dict[str, object] | None:
    if audit is not None and audit["owned_nvidia_compute_pids"]:
        return {
            "reason": "owned-nvidia-compute-became-active",
            "pids": audit["owned_nvidia_compute_pids"],
        }
    if observation_errors:
        return {
            "reason": "resource-observation-failed",
            "errors": dict(sorted(observation_errors.items())),
        }
    assert audit is not None and available_kib is not None and disk_free_bytes is not None
    if audit["protected_nvidia_compute_pids"]:
        return {
            "reason": "protected-nvidia-compute-became-active",
            "pids": audit["protected_nvidia_compute_pids"],
        }
    if audit["workspace_nvidia_compute_pids"]:
        return {
            "reason": "unattributed-workspace-nvidia-compute-became-active",
            "pids": audit["workspace_nvidia_compute_pids"],
        }
    if audit["protected_nvidia_runtime_mapping_pids"]:
        return {
            "reason": "protected-nvidia-runtime-became-active",
            "pids": audit["protected_nvidia_runtime_mapping_pids"],
        }
    if audit["unreadable_protected_maps"]:
        return {
            "reason": "protected-maps-became-unreadable",
            "pids": audit["unreadable_protected_maps"],
        }
    if available_kib < contract.PAUSE_MEMORY_FLOOR_KIB:
        return {
            "reason": "host-memory-pause-floor",
            "mem_available_kib": available_kib,
            "floor_kib": contract.PAUSE_MEMORY_FLOOR_KIB,
        }
    if minimum_free_disk_bytes and disk_free_bytes < minimum_free_disk_bytes:
        return {
            "reason": "workspace-disk-floor",
            "free_bytes": disk_free_bytes,
            "floor_bytes": minimum_free_disk_bytes,
        }
    return None


def exact_owned_group_members(metadata: dict[str, object]) -> list[int]:
    pgid = int(metadata["pgid"])
    exact = stop_run.exact_process_group_members(pgid)
    if not exact:
        return []
    owned = stop_run.owned_group_members(metadata)
    if owned != exact:
        raise ControllerError(
            f"CPU-v2 PGID {pgid} exact/owned member sets differ: "
            f"exact={exact}, owned={owned}"
        )
    return exact


def terminate_owned(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
    metadata_path: pathlib.Path | None = None,
) -> int:
    try:
        members = exact_owned_group_members(metadata)
        if not members and process.poll() is not None:
            return process.wait()
        pgid = int(metadata["pgid"])
        stop_run.signal_verified(metadata, signal.SIGTERM)
        if not stop_run.signal_verified_followup(metadata, signal.SIGCONT, pgid):
            return process.wait() if process.poll() is None else int(process.returncode)
    except (OSError, RuntimeError) as error:
        metadata["status"] = "group-ownership-ambiguous"
        metadata["termination_error"] = f"{type(error).__name__}: {error}"
        if metadata_path is not None:
            base.atomic_json(metadata_path, metadata)
        return 75
    deadline = time.monotonic() + 5
    try:
        if process.poll() is None:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass
    while time.monotonic() < deadline:
        try:
            members = exact_owned_group_members(metadata)
        except (OSError, RuntimeError) as error:
            metadata["status"] = "group-ownership-ambiguous"
            metadata["termination_error"] = f"{type(error).__name__}: {error}"
            if metadata_path is not None:
                base.atomic_json(metadata_path, metadata)
            return 75
        if not members:
            break
        time.sleep(0.1)
    try:
        survivors = exact_owned_group_members(metadata)
    except (OSError, RuntimeError) as error:
        metadata["status"] = "group-ownership-ambiguous"
        metadata["termination_error"] = f"{type(error).__name__}: {error}"
        if metadata_path is not None:
            base.atomic_json(metadata_path, metadata)
        return 75
    if survivors:
        try:
            stop_run.signal_verified_followup(metadata, signal.SIGSTOP, pgid)
        except stop_run.GroupRevalidationError as error:
            metadata["status"] = "group-ownership-ambiguous"
            metadata["termination_error"] = f"{type(error).__name__}: {error}"
            if metadata_path is not None:
                base.atomic_json(metadata_path, metadata)
            return 75
        metadata["status"] = "paused"
        metadata["termination_surviving_group_pids"] = survivors
        if metadata_path is not None:
            base.atomic_json(metadata_path, metadata)
        return 75
    if metadata_path is not None:
        base.atomic_json(metadata_path, metadata)
    return process.returncode if process.returncode is not None else process.wait()


def record_ownership_ambiguity(
    metadata: dict[str, object],
    metadata_path: pathlib.Path,
    *,
    operation: str,
    error: BaseException,
) -> tuple[int, bool]:
    metadata["status"] = "group-ownership-ambiguous"
    metadata["ownership_error"] = {
        "operation": operation,
        "error": f"{type(error).__name__}: {error}",
    }
    base.atomic_json(metadata_path, metadata)
    return 75, True


def wait_with_watchdog(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
    *,
    timeout_seconds: int,
    minimum_free_disk_bytes: int,
    metadata_path: pathlib.Path,
) -> tuple[int, bool]:
    deadline = time.monotonic() + timeout_seconds
    paused = False
    pause_count = 0
    pauses = metadata["coexistence_pauses"]
    assert isinstance(pauses, list)
    while True:
        try:
            return process.wait(timeout=contract.POLL_SECONDS), False
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            observation_errors: dict[str, str] = {}
            try:
                available: int | None = preflight.mem_available_kib()
            except Exception as error:
                available = None
                observation_errors["memory"] = f"{type(error).__name__}: {error}"
            try:
                disk_free: int | None = shutil.disk_usage(ROOT).free
            except Exception as error:
                disk_free = None
                observation_errors["disk"] = f"{type(error).__name__}: {error}"
            try:
                audit = audit_runtime_state(metadata)
            except Exception as error:
                audit = None
                observation_errors["nvidia"] = f"{type(error).__name__}: {error}"
            if now >= deadline:
                abort = {
                    "reason": "owned-run-timeout",
                    "timeout_seconds": timeout_seconds,
                }
                metadata["coexistence_abort"] = abort
                code = terminate_owned(process, metadata, metadata_path)
                base.atomic_json(metadata_path, metadata)
                return code, True
            reason = pressure_reason(
                available_kib=available,
                disk_free_bytes=disk_free,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
                audit=audit,
                observation_errors=observation_errors,
            )
            if audit is not None:
                metadata["coexistence_last_audit"] = audit
            if reason is not None and reason["reason"] == "owned-nvidia-compute-became-active":
                metadata["coexistence_abort"] = reason
                code = terminate_owned(process, metadata, metadata_path)
                base.atomic_json(metadata_path, metadata)
                return code, True
            if paused:
                recovery_safe = (
                    reason is None
                    and available is not None
                    and disk_free is not None
                    and available >= contract.RESUME_MEMORY_FLOOR_KIB
                    and (
                        minimum_free_disk_bytes == 0
                        or disk_free >= minimum_free_disk_bytes
                    )
                )
                if not recovery_safe:
                    continue
                try:
                    stop_run.signal_verified(metadata, signal.SIGCONT)
                except ProcessLookupError:
                    return process.wait(), False
                except (OSError, RuntimeError) as error:
                    return record_ownership_ambiguity(
                        metadata,
                        metadata_path,
                        operation="resume",
                        error=error,
                    )
                paused = False
                metadata["status"] = "running"
                pauses[-1]["resumed_at"] = datetime.datetime.now(
                    datetime.UTC
                ).strftime("%Y%m%dT%H%M%SZ")
                base.atomic_json(metadata_path, metadata)
                continue
            if reason is None:
                continue
            try:
                stop_run.signal_verified(metadata, signal.SIGSTOP)
            except ProcessLookupError:
                return process.wait(), False
            except (OSError, RuntimeError) as error:
                return record_ownership_ambiguity(
                    metadata,
                    metadata_path,
                    operation="pause",
                    error=error,
                )
            pause_count += 1
            reason.update(
                {
                    "owned_run_action": "verified-sigstop",
                    "pause_index": pause_count,
                    "paused_at": datetime.datetime.now(datetime.UTC).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                }
            )
            pauses.append(reason)
            metadata["status"] = "paused"
            base.atomic_json(metadata_path, metadata)
            paused = True


def publish_run_id(path: pathlib.Path, run_id: str) -> None:
    parent = path.parent.resolve(strict=True)
    if parent != ROOT and ROOT not in parent.parents:
        raise ControllerError("run-id file parent escapes the workspace")
    if path.exists() or path.is_symlink():
        raise ControllerError("run-id file already exists")
    base.atomic_json(path, {"schema_version": 1, "run_id": run_id})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id-file", type=pathlib.Path, required=True)
    parser.add_argument(
        "--timeout-seconds", type=int, default=contract.DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--minimum-free-disk-gib",
        type=int,
        default=contract.DEFAULT_MINIMUM_FREE_DISK_GIB,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("CPU-v2 controller requires Python -E -B -S")
    if args.timeout_seconds <= 0 or args.minimum_free_disk_gib < 0:
        parser.error("CPU-v2 resource limits are invalid")
    command = contract.validate_command(
        args.command[1:] if args.command[:1] == ["--"] else args.command
    )
    try:
        control_identity = control.validate_control_manifest(ROOT)
    except (
        control.ControlManifestError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "CPU-v2 control manifest refusal: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 75
    lease = acquire_shared_environment_lease()
    if lease is None:
        return 75
    try:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"pypto-cpu-v2-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
        run_dir = ROOT / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        publish_run_id(args.run_id_file, run_id)
        initial_code, initial_report = run_preflight(description="initial CPU-v2 preflight")
        initial_path = run_dir / "initial-preflight.json"
        base.atomic_json(initial_path, initial_report)
        initial_sha = sha256_file(initial_path)
        if initial_code or initial_report.get("ok") is not True:
            return initial_code or 75

        environment = cpu_environment(run_id=run_id, run_dir=run_dir, lease=lease)
        action_code, action_report = run_preflight(
            description="action-boundary CPU-v2 preflight"
        )
        action_path = run_dir / "preflight.json"
        base.atomic_json(action_path, action_report)
        action_sha = sha256_file(action_path)
        if action_code or action_report.get("ok") is not True:
            return action_code or 75
        if (
            initial_report.get("protected_cpu_only_coexistence_requested") is not True
            or action_report.get("protected_cpu_only_coexistence_requested") is not True
            or initial_report.get("admission_policy")
            != action_report.get("admission_policy")
        ):
            raise ControllerError("CPU-v2 initial/action authorization differs")
        minimum_disk = args.minimum_free_disk_gib << 30
        if minimum_disk and shutil.disk_usage(ROOT).free < minimum_disk:
            return 75
        environment.update(
            {
                "PYPTO_INITIAL_PREFLIGHT_REPORT_PATH": str(initial_path),
                "PYPTO_INITIAL_PREFLIGHT_REPORT_SHA256": initial_sha,
                "PYPTO_PREFLIGHT_REPORT_PATH": str(action_path),
                "PYPTO_PREFLIGHT_REPORT_SHA256": action_sha,
                "PYPTO_PROTECTED_ACTIVITY_WAIVER_APPLIED": (
                    "1"
                    if action_report.get("protected_activity_waiver_applied") is True
                    else "0"
                ),
            }
        )
        start_gate_path = run_dir / "start-gate.json"
        launch_command = gated_launch_command(
            command, run_id=run_id, gate_path=start_gate_path
        )

        metadata_path = run_dir / "process.json"
        process: subprocess.Popen[bytes] | None = None
        metadata: dict[str, object] | None = None
        metadata_published = False
        metadata_verified = False
        metadata_complete = False
        startup_cleanup_attempted = False
        child_code: int | None = None
        parent_code: int | None = None

        def ensure_owned_metadata(
            *,
            emergency_on_failure: bool = False,
            startup_error: BaseException | None = None,
        ) -> dict[str, object]:
            nonlocal metadata, metadata_published, metadata_complete
            if metadata is None:
                if process is None:
                    raise ControllerError("CPU-v2 child does not exist")
                try:
                    metadata = build_metadata(
                        process,
                        run_id=run_id,
                        command=command,
                        launch_command=launch_command,
                        start_gate_path=start_gate_path,
                        timestamp=timestamp,
                        lease=lease,
                        initial_path=initial_path,
                        initial_sha256=initial_sha,
                        action_path=action_path,
                        action_sha256=action_sha,
                        action_report=copy.deepcopy(action_report),
                        timeout_seconds=args.timeout_seconds,
                        minimum_free_disk_bytes=minimum_disk,
                        control_identity=control_identity,
                    )
                    metadata_complete = True
                except BaseException as capture_error:
                    if not emergency_on_failure:
                        raise
                    pgid = os.getpgid(process.pid)
                    if pgid != process.pid:
                        raise ControllerError(
                            "CPU-v2 emergency ownership PGID differs"
                        ) from capture_error
                    metadata = {
                        "schema_version": contract.SCHEMA_VERSION,
                        "kind": contract.POLICY_KIND,
                        "run_id": run_id,
                        "workspace": str(ROOT),
                        "command": command,
                        "launch_command": launch_command,
                        "pid": process.pid,
                        "pgid": pgid,
                        "start_ticks": stop_run.process_start_ticks(process.pid),
                        "status": "startup-ownership-recovery",
                        "return_code": None,
                        "metadata_complete": False,
                        "startup_error": (
                            f"{type(startup_error or capture_error).__name__}: "
                            f"{startup_error or capture_error}"
                        ),
                        "capture_error": (
                            f"{type(capture_error).__name__}: {capture_error}"
                        ),
                    }
            if not metadata_published:
                base.atomic_json(metadata_path, metadata)
                metadata_published = True
            return metadata

        def ensure_verified_metadata(
            *,
            emergency_on_failure: bool = False,
            startup_error: BaseException | None = None,
        ) -> dict[str, object]:
            nonlocal metadata_verified
            owned = ensure_owned_metadata(
                emergency_on_failure=emergency_on_failure,
                startup_error=startup_error,
            )
            if not metadata_verified:
                try:
                    stop_run.verify(owned)
                except ProcessLookupError:
                    stop_run.owned_group_members(owned)
                metadata_verified = True
            return owned

        handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_handlers = {item: signal.getsignal(item) for item in handled}
        for item in handled:
            signal.signal(item, base.interrupt_parent)
        startup_failed = False
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
            try:
                def restore_child_mask() -> None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

                process = subprocess.Popen(
                    launch_command,
                    cwd=ROOT,
                    env=environment,
                    start_new_session=True,
                    preexec_fn=restore_child_mask,
                    pass_fds=(lease.descriptor,),
                )
                metadata = ensure_verified_metadata()
                if process.poll() is not None:
                    raise ControllerError("CPU-v2 start-gate child exited before release")
                release_start_gate(
                    path=start_gate_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )
            except BaseException as startup_error:
                startup_failed = True
                startup_cleanup_attempted = True
                parent_code = 75
                if process is not None:
                    try:
                        owned = ensure_verified_metadata(
                            emergency_on_failure=True,
                            startup_error=startup_error,
                        )
                    except BaseException as ownership_error:
                        if metadata is not None:
                            metadata["status"] = "group-ownership-ambiguous"
                            metadata["ownership_error"] = {
                                "operation": "startup",
                                "error": (
                                    f"{type(ownership_error).__name__}: "
                                    f"{ownership_error}"
                                ),
                            }
                            if metadata_published:
                                base.atomic_json(metadata_path, metadata)
                        else:
                            base.atomic_json(
                                run_dir / "ownership-error.json",
                                {
                                    "schema_version": contract.SCHEMA_VERSION,
                                    "kind": contract.POLICY_KIND,
                                    "run_id": run_id,
                                    "pid": process.pid,
                                    "error": (
                                        f"{type(ownership_error).__name__}: "
                                        f"{ownership_error}"
                                    ),
                                    "signal_sent": False,
                                },
                            )
                        try:
                            child_code = process.wait(
                                timeout=contract.START_GATE_TIMEOUT_SECONDS + 5
                            )
                        except subprocess.TimeoutExpired:
                            base.atomic_json(
                                run_dir / "start-gate-timeout.json",
                                {
                                    "schema_version": contract.SCHEMA_VERSION,
                                    "kind": contract.POLICY_KIND,
                                    "run_id": run_id,
                                    "pid": process.pid,
                                    "error": (
                                        "unreleased fixed start-gate child did not "
                                        "self-terminate"
                                    ),
                                    "signal_sent": False,
                                },
                            )
                    else:
                        owned["startup_error"] = (
                            f"{type(startup_error).__name__}: {startup_error}"
                        )
                        owned["status"] = "startup-failed"
                        base.atomic_json(metadata_path, owned)
                        child_code = terminate_owned(process, owned, metadata_path)
                else:
                    base.atomic_json(
                        run_dir / "startup-error.json",
                        {
                            "schema_version": contract.SCHEMA_VERSION,
                            "kind": contract.POLICY_KIND,
                            "run_id": run_id,
                            "error": f"{type(startup_error).__name__}: {startup_error}",
                            "child_created": False,
                        },
                    )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if not startup_failed:
                assert process is not None and metadata is not None
                print(
                    f"PYPTO_CPU_V2_RUN_ID={run_id} PID={process.pid} "
                    f"PGID={metadata['pgid']}",
                    flush=True,
                )
                child_code, aborted = wait_with_watchdog(
                    process,
                    metadata,
                    timeout_seconds=args.timeout_seconds,
                    minimum_free_disk_bytes=minimum_disk,
                    metadata_path=metadata_path,
                )
                parent_code = 75 if aborted else child_code
            if not startup_failed:
                assert process is not None and metadata is not None
                try:
                    survivors = exact_owned_group_members(metadata)
                except (OSError, RuntimeError) as error:
                    metadata["status"] = "group-ownership-ambiguous"
                    metadata["group_exit_error"] = f"{type(error).__name__}: {error}"
                    parent_code = 75
                else:
                    metadata["post_exit_group_members"] = survivors
                    if survivors:
                        parent_code = 75
                        if metadata.get("status") != "group-ownership-ambiguous":
                            metadata["surviving_group_cleanup_code"] = (
                                terminate_owned(process, metadata, metadata_path)
                            )
                    try:
                        metadata["post_cleanup_group_members"] = (
                            exact_owned_group_members(metadata)
                        )
                    except (OSError, RuntimeError) as error:
                        metadata["status"] = "group-ownership-ambiguous"
                        metadata["post_cleanup_group_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                        parent_code = 75
                    else:
                        if metadata["post_cleanup_group_members"]:
                            parent_code = 75
                try:
                    post_exit_audit = audit_runtime_state(metadata)
                except Exception as error:
                    metadata["coexistence_post_exit_audit_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                    parent_code = 75
                else:
                    metadata["coexistence_post_exit_audit"] = post_exit_audit
                    if (
                        post_exit_audit["owned_nvidia_compute_pids"]
                        or post_exit_audit["protected_nvidia_compute_pids"]
                        or post_exit_audit["workspace_nvidia_compute_pids"]
                        or post_exit_audit["protected_nvidia_runtime_mapping_pids"]
                        or post_exit_audit["unreadable_protected_maps"]
                    ):
                        parent_code = 75
        except (KeyboardInterrupt, base.RunInterrupted) as error:
            if process is not None:
                cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
                try:
                    child_code = terminate_owned(
                        process,
                        ensure_verified_metadata(
                            emergency_on_failure=True, startup_error=error
                        ),
                        metadata_path,
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
            parent_code = (
                130 if isinstance(error, KeyboardInterrupt) else 128 + error.signum
            )
        except BaseException:
            if (
                process is not None
                and not startup_cleanup_attempted
            ):
                cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
                try:
                    terminate_owned(
                        process,
                        ensure_verified_metadata(emergency_on_failure=True),
                        metadata_path,
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
            raise
        finally:
            for item, previous in previous_handlers.items():
                signal.signal(item, previous)
            if metadata is not None:
                try:
                    final_members = exact_owned_group_members(metadata)
                except (OSError, RuntimeError) as error:
                    metadata["status"] = "group-ownership-ambiguous"
                    metadata["final_group_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                    parent_code = 75
                else:
                    metadata["final_group_members"] = final_members
                    if (
                        not final_members
                        and process is not None
                        and process.poll() is not None
                        and metadata.get("status")
                        != "group-ownership-ambiguous"
                    ):
                        metadata["status"] = "exited"
                    elif final_members and metadata.get("status") not in {
                        "paused",
                        "group-ownership-ambiguous",
                    }:
                        metadata["status"] = "alive"
                metadata["return_code"] = child_code
                metadata["finished_at"] = datetime.datetime.now(
                    datetime.UTC
                ).strftime("%Y%m%dT%H%M%SZ")
                base.atomic_json(metadata_path, metadata)
        if parent_code is None:
            raise ControllerError("CPU-v2 controller exited without parent status")
        return parent_code
    finally:
        lease.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(75) from None
