#!/usr/bin/env python3
"""Owned-process controller for the fused-pointwise SM120 v2 admission policy."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import sys
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_ISOLATION_RELATIVE_PATH = pathlib.Path("tools/run_isolated.py")
BASE_ISOLATION_SIZE = 76_558
BASE_ISOLATION_SHA256 = (
    "978686ac09743a98233c9616d23b04e57d3a257bd643d5db3b8a71eaac7465c8"
)
BASE_STOP_RELATIVE_PATH = pathlib.Path("tools/stop_run.py")
BASE_STOP_SIZE = 7_885
BASE_STOP_SHA256 = "879a2e3863671531a548c71d788d56298500eab989bd1420d2c7ae01717ddfe4"


class ControllerV2Error(RuntimeError):
    """The v2 controller cannot prove an owned, admitted direct-child run."""


def load_exact(
    name: str,
    path: pathlib.Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControllerV2Error(f"exact v2 controller source is noncanonical: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_size is not None and len(raw) != expected_size:
        raise ControllerV2Error(f"exact v2 controller source size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ControllerV2Error(f"exact v2 controller source hash differs: {path}")
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


control = load_exact(
    "_pypto_fused_pointwise_sm120_control_manifest_v2",
    ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py",
)
control.reject_control_bytecode_cache(ROOT)
contract = load_exact(
    "_pypto_fused_pointwise_sm120_contract_v2",
    ROOT / "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
)
preflight = load_exact(
    "preflight",
    ROOT / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
)
stop_run = load_exact(
    "stop_run",
    ROOT / BASE_STOP_RELATIVE_PATH,
    expected_size=BASE_STOP_SIZE,
    expected_sha256=BASE_STOP_SHA256,
)
sys.modules["_pypto_nvidia_executable_sm120_contract"] = contract
sys.modules["_pypto_nvidia_sm120_control_manifest"] = control
isolation = load_exact(
    "_pypto_gpu_smoke_isolation_v1_base",
    ROOT / BASE_ISOLATION_RELATIVE_PATH,
    expected_size=BASE_ISOLATION_SIZE,
    expected_sha256=BASE_ISOLATION_SHA256,
)
if (
    isolation.preflight_tool is not preflight
    or isolation.stop_run is not stop_run
    or isolation.nvidia_smoke_contract is not contract
    or isolation.nvidia_smoke_control is not control
):
    raise ControllerV2Error("v2 isolation helper dependency injection differs")


def preflight_command(*, allow_protected: bool) -> list[str]:
    command = [
        sys.executable,
        "-E",
        "-B",
        "-S",
        str(ROOT / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH),
        "--mode",
        "gpu-smoke",
        "--json",
    ]
    if allow_protected:
        command.append("--allow-protected-zero-nvidia-gpu-smoke")
    return command


def preflight_environment(environment_prefix: pathlib.Path) -> dict[str, str]:
    return {
        "PATH": (
            f"{environment_prefix}/bin:/usr/local/cuda-13.3/bin:"
            f"{isolation.SYSTEM_EXECUTABLE_PATH}"
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }


def validate_preflight_report(
    report: object, *, allow_protected: bool, description: str
) -> dict[str, object]:
    if not isinstance(report, dict):
        raise ControllerV2Error(f"{description} must be an object")
    expected_floor = (
        contract.PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
        if allow_protected
        else contract.EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
    )
    available = report.get("mem_available_kib")
    if (
        report.get("mode") != "gpu-smoke"
        or report.get("gpu_smoke_policy_version") != contract.GPU_SMOKE_POLICY_VERSION
        or report.get("protected_zero_nvidia_gpu_smoke_requested")
        is not allow_protected
        or report.get("memory_floor_kib") != expected_floor
        or isinstance(available, bool)
        or not isinstance(available, int)
        or available < expected_floor
        or report.get("gpu_smoke_free_memory_floor_mib")
        != contract.GPU_FREE_MEMORY_FLOOR_MIB
        or report.get("gpu_smoke_admission_policy") != preflight.policy_document()
        or report.get("nvidia_compute_audit_ok") is not True
    ):
        raise ControllerV2Error(f"{description} policy identity differs")
    protected = report.get("protected_processes")
    if (
        allow_protected
        and isinstance(protected, list)
        and protected
        and report.get("protected_gpu_smoke_waiver_applied") is not True
    ):
        raise ControllerV2Error(f"{description} protected-lane waiver differs")
    return report


def run_preflight(
    *, allow_protected: bool, environment_prefix: pathlib.Path, description: str
) -> tuple[int, dict[str, object], bytes]:
    completed = subprocess.run(
        preflight_command(allow_protected=allow_protected),
        cwd=ROOT,
        env=preflight_environment(environment_prefix),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ControllerV2Error(f"{description} did not emit valid JSON") from error
    validated = validate_preflight_report(
        report, allow_protected=allow_protected, description=description
    )
    raw = isolation.canonical_json_bytes(validated)
    return completed.returncode, validated, raw


def _write_run_id(path: pathlib.Path, run_id: str) -> None:
    absolute = pathlib.Path(os.path.abspath(os.fspath(path)))
    parent = absolute.parent.resolve(strict=True)
    if ROOT not in parent.parents and parent != ROOT:
        raise ControllerV2Error("--run-id-file must be below the workspace")
    if absolute.exists() or absolute.is_symlink():
        raise ControllerV2Error("--run-id-file must not already exist")
    isolation.atomic_json(absolute, {"run_id": run_id})


def acquire_shared_environment_lease() -> object | None:
    """Acquire the shared environment lease across a blocked-signal boundary."""

    handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        try:
            return isolation.acquire_environment_lock("pypto-nvidia", "shared")
        except isolation.EnvironmentLockBusy as error:
            print(str(error), file=sys.stderr)
            return None
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _gate_and_release(
    *,
    process: subprocess.Popen[bytes],
    metadata: dict[str, object],
    metadata_path: pathlib.Path,
    gate_path: pathlib.Path,
    barrier_path: pathlib.Path,
    control_identity: dict[str, object],
) -> tuple[bool, int | None]:
    snapshot: dict[str, object] | None = None
    violation: dict[str, object] | None = None
    static_identity: dict[str, object] | None = None
    try:
        if (
            gate_path.exists()
            or gate_path.is_symlink()
            or barrier_path.exists()
            or barrier_path.is_symlink()
        ):
            raise ControllerV2Error("GPU-smoke gate or start barrier already exists")
        isolation.validate_exact_nvidia_smoke_inputs()
        static_identity = preflight.static_torch_identity()
        if static_identity.get("static_identity_error"):
            raise ControllerV2Error(str(static_identity["static_identity_error"]))
        if process.poll() is not None:
            raise ControllerV2Error("GPU-smoke child exited before gate release")
        stop_run.verify(metadata)
        snapshot, violation = isolation.audit_gpu_smoke_runtime_state(metadata)
        if (
            violation is None
            and snapshot is not None
            and snapshot.get("owned_nvidia_compute_pids")
        ):
            violation = {
                "reason": "owned-nvidia-compute-before-gate-release",
                "pids": snapshot["owned_nvidia_compute_pids"],
            }
    except Exception as error:
        violation = {
            "reason": "gpu-smoke-v2-pre-release-audit-failed",
            "error": f"{type(error).__name__}: {error}",
        }
    if snapshot is not None:
        metadata["gpu_smoke_pre_release_audit"] = snapshot
    if violation is not None:
        metadata["gpu_smoke_abort"] = violation
        isolation.atomic_json(metadata_path, metadata)
        return False, isolation.terminate_owned_process(
            process, metadata, wait_seconds=5
        )
    assert snapshot is not None and static_identity is not None
    gate = {
        "schema": contract.GPU_SMOKE_POLICY_VERSION,
        "run_id": metadata["run_id"],
        "pid": process.pid,
        "pgid": metadata["pgid"],
        "start_ticks": metadata["start_ticks"],
        "command": metadata["command"],
        "initial_preflight": metadata["initial_preflight"],
        "preflight": metadata["preflight"],
        "static_identity": static_identity,
        "control_manifest": control_identity,
        "runtime_isolation": snapshot,
        "admission_policy": preflight.policy_document(),
    }
    isolation.atomic_json(gate_path, gate)
    gate_sha256 = isolation.sha256_file(gate_path)
    barrier = {
        "schema": contract.GPU_SMOKE_POLICY_VERSION,
        "run_id": metadata["run_id"],
        "pid": process.pid,
        "pgid": metadata["pgid"],
        "start_ticks": metadata["start_ticks"],
        "gate_path": str(gate_path),
        "gate_sha256": gate_sha256,
    }
    barrier_sha256 = hashlib.sha256(isolation.canonical_json_bytes(barrier)).hexdigest()
    gpu_smoke = metadata["gpu_smoke"]
    assert isinstance(gpu_smoke, dict)
    gpu_smoke.update(
        {
            "gate_sha256": gate_sha256,
            "start_barrier_sha256": barrier_sha256,
            "release_authorized_at": datetime.datetime.now(datetime.UTC).strftime(
                "%Y%m%dT%H%M%SZ"
            ),
        }
    )
    isolation.atomic_json(metadata_path, metadata)
    isolation.atomic_json(barrier_path, barrier)
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-protected-zero-nvidia-gpu-smoke", action="store_true")
    parser.add_argument("--run-id-file", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("fused-pointwise v2 controller requires Python -E -B -S")

    control_identity = control.validate_control_manifest(ROOT)
    environment_prefix = isolation.ENVIRONMENTS["pypto-nvidia"].resolve()
    command = contract.fixed_child_command(ROOT)
    isolation.validate_exact_nvidia_smoke_command(command)
    lease = acquire_shared_environment_lease()
    if lease is None:
        return 75
    try:
        isolation.validate_exact_nvidia_smoke_inputs()
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"pypto-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
        run_dir = ROOT / "runs" / run_id
        run_dir.mkdir(parents=True)
        _write_run_id(args.run_id_file, run_id)

        initial_code, initial_report, _ = run_preflight(
            allow_protected=args.allow_protected_zero_nvidia_gpu_smoke,
            environment_prefix=environment_prefix,
            description="initial v2 preflight",
        )
        initial_path = run_dir / "initial-preflight.json"
        isolation.atomic_json(initial_path, initial_report)
        initial_sha256 = isolation.sha256_file(initial_path)
        if initial_code != 0 or initial_report.get("ok") is not True:
            return initial_code if initial_code != 0 else 75

        environment = isolation.isolated_environment(
            run_id,
            run_dir,
            environment_prefix=environment_prefix,
            framework_profile="pypto",
            protected_zero_nvidia_gpu_smoke_requested=(
                args.allow_protected_zero_nvidia_gpu_smoke
            ),
            exact_nvidia_smoke=True,
        )
        environment.update(isolation.environment_lock_markers(lease))
        action_code, action_report, _ = run_preflight(
            allow_protected=args.allow_protected_zero_nvidia_gpu_smoke,
            environment_prefix=environment_prefix,
            description="action-boundary v2 preflight",
        )
        preflight_path = run_dir / "preflight.json"
        isolation.atomic_json(preflight_path, action_report)
        preflight_sha256 = isolation.sha256_file(preflight_path)
        if action_code != 0 or action_report.get("ok") is not True:
            return action_code if action_code != 0 else 75

        barrier_path = run_dir / "gpu-smoke-start-barrier.json"
        gate_path = run_dir / "gpu-smoke-gate.json"
        environment.update(
            {
                "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED": "0",
                "PYPTO_PROTECTED_ACTIVITY_WAIVER_APPLIED": "0",
                "PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED": (
                    "1" if args.allow_protected_zero_nvidia_gpu_smoke else "0"
                ),
                "PYPTO_PROTECTED_GPU_SMOKE_WAIVER_APPLIED": (
                    "1"
                    if action_report.get("protected_gpu_smoke_waiver_applied") is True
                    else "0"
                ),
                "PYPTO_GPU_SMOKE_START_BARRIER": str(barrier_path),
                "PYPTO_PREFLIGHT_REPORT_PATH": str(preflight_path),
                "PYPTO_PREFLIGHT_REPORT_SHA256": preflight_sha256,
                "PYPTO_INITIAL_PREFLIGHT_REPORT_PATH": str(initial_path),
                "PYPTO_INITIAL_PREFLIGHT_REPORT_SHA256": initial_sha256,
                "PYPTO_RUN_MODE": "gpu-smoke",
            }
        )
        minimum_free_disk_bytes = contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
        if shutil.disk_usage(ROOT).free < minimum_free_disk_bytes:
            return 75
        metadata_context = {
            "protected_cpu_only_coexistence_requested": False,
            "protected_activity_waiver_applied": False,
            "protected_zero_nvidia_gpu_smoke_requested": (
                args.allow_protected_zero_nvidia_gpu_smoke
            ),
            "protected_gpu_smoke_waiver_applied": bool(
                action_report.get("protected_gpu_smoke_waiver_applied")
            ),
            "gpu_smoke_start_barrier_path": str(barrier_path),
            "gpu_smoke_gate_path": str(gate_path),
            "preflight_report_sha256": preflight_sha256,
            "preflight_report_path": str(preflight_path),
            "preflight_report": action_report,
            "run_timeout_seconds": contract.GPU_SMOKE_TIMEOUT_SECONDS,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
            "environment_access_lock": {
                "path": str(lease.path),
                "mode": lease.mode,
                "device": lease.device,
                "inode": lease.inode,
            },
        }
        metadata_path = run_dir / "process.json"
        process: subprocess.Popen[bytes] | None = None
        metadata: dict[str, object] | None = None
        child_return_code: int | None = None
        parent_return_code: int | None = None
        handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_handlers = {item: signal.getsignal(item) for item in handled}
        for item in handled:
            signal.signal(item, isolation.interrupt_parent)
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
            try:

                def restore_child_mask() -> None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    start_new_session=True,
                    preexec_fn=restore_child_mask,
                    pass_fds=(lease.descriptor,),
                )
                metadata = isolation.build_run_metadata(
                    process,
                    run_id=run_id,
                    environment_prefix=environment_prefix,
                    framework_profile="pypto",
                    framework_launch=False,
                    mode="gpu-smoke",
                    command=command,
                    timestamp=timestamp,
                    **metadata_context,
                )
                metadata["schema"] = 4
                metadata["initial_preflight"] = {
                    "path": str(initial_path),
                    "sha256": initial_sha256,
                }
                isolation.atomic_json(metadata_path, metadata)
                released, early_code = _gate_and_release(
                    process=process,
                    metadata=metadata,
                    metadata_path=metadata_path,
                    gate_path=gate_path,
                    barrier_path=barrier_path,
                    control_identity=control_identity,
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            print(
                f"PYPTO_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}",
                flush=True,
            )
            if not released:
                child_return_code = early_code
                parent_return_code = 75
            else:
                child_return_code, aborted = isolation.wait_with_gpu_smoke_watchdog(
                    process,
                    metadata,
                    timeout_seconds=contract.GPU_SMOKE_TIMEOUT_SECONDS,
                    minimum_free_disk_bytes=minimum_free_disk_bytes,
                    metadata_path=metadata_path,
                )
                parent_return_code = 75 if aborted else child_return_code
                post_snapshot, post_violation = isolation.audit_gpu_smoke_runtime_state(
                    metadata
                )
                if post_snapshot is not None:
                    metadata["gpu_smoke_post_exit_audit"] = post_snapshot
                if post_violation is not None:
                    metadata["gpu_smoke_abort"] = post_violation
                    parent_return_code = 75
            if process.poll() is not None and metadata.get("status") != "paused":
                survivors = stop_run.owned_group_members(metadata)
                if survivors:
                    metadata["surviving_group_pids"] = survivors
                    metadata["surviving_group_cleanup_code"] = (
                        isolation.terminate_owned_process(
                            process, metadata, wait_seconds=5
                        )
                    )
                    parent_return_code = 75
        except (KeyboardInterrupt, isolation.RunInterrupted) as error:
            if process is not None:
                if metadata is None:
                    metadata = isolation.build_run_metadata(
                        process,
                        run_id=run_id,
                        environment_prefix=environment_prefix,
                        framework_profile="pypto",
                        framework_launch=False,
                        mode="gpu-smoke",
                        command=command,
                        timestamp=timestamp,
                        **metadata_context,
                    )
                if process.poll() is None:
                    child_return_code = isolation.terminate_owned_process(
                        process, metadata
                    )
                else:
                    child_return_code = process.wait()
            parent_return_code = (
                130 if isinstance(error, KeyboardInterrupt) else 128 + error.signum
            )
        except BaseException:
            if process is not None and process.poll() is None:
                if metadata is None:
                    metadata = isolation.build_run_metadata(
                        process,
                        run_id=run_id,
                        environment_prefix=environment_prefix,
                        framework_profile="pypto",
                        framework_launch=False,
                        mode="gpu-smoke",
                        command=command,
                        timestamp=timestamp,
                        **metadata_context,
                    )
                isolation.terminate_owned_process(process, metadata)
            raise
        finally:
            for item, previous in previous_handlers.items():
                signal.signal(item, previous)
            if metadata is not None:
                if (
                    process is not None
                    and process.poll() is not None
                    and not stop_run.process_group_members(int(metadata["pgid"]))
                ):
                    metadata["status"] = "exited"
                elif metadata.get("status") not in {
                    "paused",
                    "group-ownership-ambiguous",
                }:
                    metadata["status"] = "alive"
                metadata["return_code"] = child_return_code
                metadata["finished_at"] = datetime.datetime.now(datetime.UTC).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                isolation.atomic_json(metadata_path, metadata)
        if parent_return_code is None:
            raise ControllerV2Error("v2 run exited without a parent return code")
        return parent_return_code
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
