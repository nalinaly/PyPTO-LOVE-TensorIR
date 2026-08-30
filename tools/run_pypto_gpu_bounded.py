#!/usr/bin/env python3
"""Run bounded release GPU work without the retired 22 GiB gate."""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import tempfile
import time

import preflight
import nvidia_nvml
import run_isolated as isolation
import stop_run


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVIRONMENT = (ROOT / "envs/pypto-nvidia").resolve()
ALLOWED_ENVIRONMENT_PROFILES = {
    "pypto-nvidia": "pypto",
    "pypto-release": "pypto",
    "sglang-baseline": "baseline",
}
PROCESS_SCHEMA_VERSION = 2
HOST_ABORT_KIB = 12 * 1024 * 1024
HOST_EMERGENCY_ABORT_KIB = 11 * 1024 * 1024
HOST_FLOOR_CONSECUTIVE_SAMPLES = 3
GPU_FREE_FLOOR_MIB = 4 * 1024
NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES = 2
POLL_SECONDS = 1

_nvidia_identity_source = "unknown"
_nvidia_compute_source = "unknown"


class BoundedGpuError(RuntimeError):
    """The bounded GPU child cannot be launched safely."""


def mem_available_kib() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise BoundedGpuError("/proc/meminfo has no MemAvailable field")


def host_floor_update(
    available_kib: int, consecutive_below_floor: int
) -> tuple[str | None, int]:
    if available_kib < HOST_EMERGENCY_ABORT_KIB:
        return "host-memory-emergency-floor", consecutive_below_floor + 1
    if available_kib < HOST_ABORT_KIB:
        consecutive_below_floor += 1
        if consecutive_below_floor >= HOST_FLOOR_CONSECUTIVE_SAMPLES:
            return "host-memory-floor", consecutive_below_floor
        return None, consecutive_below_floor
    return None, 0


def nvidia_audit_failure_update(
    consecutive_failures: int,
) -> tuple[str | None, int]:
    consecutive_failures += 1
    reason = (
        "nvidia-telemetry-unavailable"
        if consecutive_failures >= NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES
        else None
    )
    return reason, consecutive_failures


def nvidia_audit_failure_record(
    error: Exception, *, phase: str, sample_index: int
) -> dict[str, object]:
    record: dict[str, object] = {
        "phase": phase,
        "sample_index": sample_index,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if isinstance(error, subprocess.TimeoutExpired):
        command = error.cmd
        record["command"] = (
            list(command) if isinstance(command, (list, tuple)) else command
        )
        record["timeout_seconds"] = error.timeout
    return record


def nvidia_identity() -> dict[str, str]:
    """Use the frozen preflight, with a read-only NVML emergency fallback."""

    global _nvidia_identity_source
    try:
        value = preflight.nvidia_identity()
        _nvidia_identity_source = "nvidia-smi"
        return value
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        try:
            value = nvidia_nvml.query_identity()
            _nvidia_identity_source = "nvml-ctypes"
            return value
        except Exception:
            _nvidia_identity_source = "unavailable"
            raise error


def nvidia_compute_pids() -> set[int]:
    """Use the frozen PID audit, falling back to the same-driver NVML query."""

    global _nvidia_compute_source
    try:
        value = preflight.nvidia_compute_pids()
        _nvidia_compute_source = "nvidia-smi"
        return value
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        try:
            value = nvidia_nvml.query_compute_pids()
            _nvidia_compute_source = "nvml-ctypes"
            return value
        except Exception:
            _nvidia_compute_source = "unavailable"
            raise error


def nvidia_telemetry_sources() -> dict[str, str]:
    """Return the provider used by the most recent identity/PID queries."""

    return {
        "identity": _nvidia_identity_source,
        "compute_pids": _nvidia_compute_source,
    }


def process_stat_full(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> tuple[int, int, int]:
    fields = (proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
    if len(fields) <= 19:
        raise BoundedGpuError(f"malformed /proc/{pid}/stat")
    return int(fields[2]), int(fields[3]), int(fields[19])


def process_stat(pid: int) -> tuple[int, int]:
    pgid, _sid, start_ticks = process_stat_full(pid)
    return pgid, start_ticks


def owned_pgid_rss_kib(pgid: int) -> int:
    total = 0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            observed_pgid, _start_ticks = process_stat(int(entry.name))
            if observed_pgid != pgid:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


def owned_sid_rss_kib(
    sid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> int:
    total = 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            _observed_pgid, observed_sid, _start_ticks = process_stat_full(
                int(entry.name), proc_root
            )
            if observed_sid != sid:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


def process_environment(pid: int) -> dict[str, str]:
    raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def validate_environment_profile(environment: str, profile: str) -> pathlib.Path:
    if ALLOWED_ENVIRONMENT_PROFILES.get(environment) != profile:
        raise BoundedGpuError(
            f"unsupported environment/profile pair: {environment}/{profile}"
        )
    return isolation.ENVIRONMENTS[environment].resolve()


def validate_child(
    raw: list[str], environment: pathlib.Path = ENVIRONMENT
) -> list[str]:
    if not raw or raw[0] != "--":
        raise BoundedGpuError("bounded GPU controller requires `--` before child")
    command = raw[1:]
    expected_python = (environment / "bin/python").resolve(strict=True)
    if (
        len(command) < 3
        or pathlib.Path(command[0]).resolve(strict=True) != expected_python
    ):
        raise BoundedGpuError("child must use selected-prefix Python")
    if command[1] != "-B":
        raise BoundedGpuError("child Python must use -B")
    script = pathlib.Path(command[2]).resolve(strict=True)
    if ROOT not in script.parents or script.suffix != ".py":
        raise BoundedGpuError("child script must be a workspace Python file")
    return [str(expected_python), "-B", str(script), *command[3:]]


def audit(
    run_id: str | None = None,
    owned_pgid: int | None = None,
    owned_sid: int | None = None,
) -> dict[str, object]:
    gpu = nvidia_identity()
    free_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    all_processes, protected, _workspace = preflight.process_table()
    protected_pids = {process.pid for process in protected}
    compute_pids = nvidia_compute_pids()
    protected_compute = sorted(compute_pids & protected_pids)
    protected_mappings, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    owned_compute: list[int] = []
    external_compute: list[int] = []
    for pid in sorted(compute_pids):
        pgid_owned = False
        sid_owned = False
        if owned_pgid is not None or owned_sid is not None:
            try:
                observed_pgid, observed_sid, _start_ticks = process_stat_full(pid)
                pgid_owned = owned_pgid is not None and observed_pgid == owned_pgid
                sid_owned = owned_sid is not None and observed_sid == owned_sid
            except (OSError, ValueError):
                pass
        try:
            environment = process_environment(pid)
        except OSError:
            (owned_compute if pgid_owned or sid_owned else external_compute).append(pid)
            continue
        if pgid_owned or sid_owned or (
            run_id is not None and environment.get("PYPTO_RUN_ID") == run_id
        ):
            owned_compute.append(pid)
        else:
            external_compute.append(pid)
    return {
        "gpu": gpu,
        "nvidia_telemetry_sources": nvidia_telemetry_sources(),
        "gpu_free_mib": free_mib,
        "compute_pids": sorted(compute_pids),
        "owned_compute_pids": owned_compute,
        "external_compute_pids": external_compute,
        "protected_compute_pids": protected_compute,
        "protected_runtime_mapping_pids": protected_mappings,
        "unreadable_protected_maps": unreadable,
        "protected_process_count": len(protected),
        "process_count": len(all_processes),
    }


def audit_ok(report: dict[str, object], *, child_running: bool) -> bool:
    return bool(
        int(report["gpu_free_mib"]) >= GPU_FREE_FLOOR_MIB
        and not report["external_compute_pids"]
        and not report["protected_compute_pids"]
        and not report["protected_runtime_mapping_pids"]
        and not report["unreadable_protected_maps"]
        and (child_running or not report["owned_compute_pids"])
    )


def terminate_owned(
    metadata: dict[str, object], process: subprocess.Popen[bytes]
) -> dict[str, object]:
    primary_error = None
    term_signaled = False
    try:
        stop_run.signal_verified(metadata, signal.SIGTERM)
        term_signaled = True
    except ProcessLookupError:
        pass
    except (OSError, RuntimeError) as error:
        primary_error = f"{type(error).__name__}: {error}"
    cleanup = stop_run.terminate_verified_session_residuals(metadata)
    cleanup["primary_pgid"] = {
        "term_signaled": term_signaled,
        "error": primary_error,
    }
    cleanup["complete"] = bool(cleanup["complete"] and primary_error is None)
    metadata["session_cleanup"] = cleanup
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cleanup["complete"] = False
            cleanup["coordinator_wait_timeout"] = True
    return cleanup


def create_short_tmp_alias(
    run_dir: pathlib.Path,
    *,
    parent_root: pathlib.Path = pathlib.Path("/tmp"),
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Expose the owned run TMPDIR through a short Unix-socket-safe path."""

    target = (run_dir / "tmp").resolve(strict=True)
    parent = pathlib.Path(
        tempfile.mkdtemp(prefix="pypto-ipc-", dir=parent_root)
    ).resolve(strict=True)
    alias = parent / "t"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except BaseException:
        parent.rmdir()
        raise
    if len(str(alias)) >= 64 or alias.resolve(strict=True) != target:
        alias.unlink()
        parent.rmdir()
        raise BoundedGpuError("cannot create a safe short TMPDIR alias")
    return parent, alias, target


def remove_short_tmp_alias(
    parent: pathlib.Path,
    alias: pathlib.Path,
    target: pathlib.Path,
) -> None:
    """Remove only the verified alias and its private empty parent."""

    if not alias.is_symlink() or alias.resolve(strict=True) != target:
        raise BoundedGpuError("short TMPDIR alias identity changed")
    alias.unlink()
    parent.rmdir()


def verify_formal_runtime_identity(
    environment_name: str,
    framework_profile: str,
    environment: dict[str, str],
) -> None:
    if environment_name == "pypto-nvidia":
        return
    prefix = isolation.ENVIRONMENTS[environment_name].resolve()
    python = prefix / "bin" / "python"
    identity_lock = isolation.ENVIRONMENT_LOCKS[environment_name]
    if not identity_lock.is_file():
        raise BoundedGpuError(
            f"formal environment identity lock is missing: {identity_lock}"
        )
    commands = (
        [
            str(python),
            str(ROOT / "tools/environment_identity.py"),
            "--prefix",
            str(prefix),
            "--lock",
            str(identity_lock),
            "--verify",
        ],
        [
            str(python),
            str(ROOT / "tools/audit_python_environment.py"),
            "--prefix",
            str(prefix),
            "--profile",
            framework_profile,
        ],
        [
            str(python),
            str(ROOT / "tools/runtime_identity.py"),
            "--prefix",
            str(prefix),
            "--lock",
            str(identity_lock),
            "--profile",
            framework_profile,
            "--framework",
        ],
    )
    for identity_command in commands:
        subprocess.run(
            identity_command,
            cwd=ROOT,
            env=environment,
            check=True,
        )
    if framework_profile == "pypto":
        subprocess.run(
            [str(python), "-m", "pypto_plugins.bootstrap"],
            cwd=ROOT,
            env=environment,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        choices=tuple(ALLOWED_ENVIRONMENT_PROFILES),
        default="pypto-nvidia",
    )
    parser.add_argument(
        "--framework-profile",
        choices=("pypto", "baseline"),
        default="pypto",
    )
    parser.add_argument("--run-id-file", type=pathlib.Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--minimum-free-disk-gib", type=int, default=64)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not (
        os.sys.flags.ignore_environment
        and os.sys.flags.no_site
        and os.sys.flags.dont_write_bytecode
    ):
        parser.error("controller requires Python -E -B -S")
    environment_prefix = validate_environment_profile(
        args.environment, args.framework_profile
    )
    command = validate_child(args.command, environment_prefix)
    initial_available = mem_available_kib()
    if initial_available < HOST_ABORT_KIB:
        raise BoundedGpuError("MemAvailable is already below 12 GiB abort floor")
    initial_audit = audit()
    if not audit_ok(initial_audit, child_running=False):
        raise BoundedGpuError(
            f"initial NVIDIA coexistence audit failed: {initial_audit}"
        )

    lease = isolation.acquire_environment_lock(args.environment, "shared")
    locked_available = mem_available_kib()
    locked_audit = audit()
    if locked_available < HOST_ABORT_KIB or not audit_ok(
        locked_audit, child_running=False
    ):
        lease.close()
        raise BoundedGpuError(
            "host/GPU coexistence changed while acquiring the environment lock"
        )
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"pypto-gpu-bounded-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    isolation.atomic_json(args.run_id_file, {"schema_version": 1, "run_id": run_id})
    isolation.atomic_json(run_dir / "initial-audit.json", initial_audit)
    isolation.atomic_json(run_dir / "locked-audit.json", locked_audit)
    environment = isolation.isolated_environment(
        run_id,
        run_dir,
        environment_prefix=environment_prefix,
        framework_profile=args.framework_profile,
        protected_zero_nvidia_gpu_smoke_requested=(args.environment == "pypto-nvidia"),
        exact_nvidia_smoke=args.environment == "pypto-nvidia",
    )
    environment.update(isolation.environment_lock_markers(lease))
    environment["PYPTO_RUN_MODE"] = "gpu-bounded"
    try:
        verify_formal_runtime_identity(
            args.environment, args.framework_profile, environment
        )
    except BaseException:
        lease.close()
        raise
    identity_available = mem_available_kib()
    identity_audit = audit()
    isolation.atomic_json(run_dir / "post-identity-audit.json", identity_audit)
    if identity_available < HOST_ABORT_KIB or not audit_ok(
        identity_audit, child_running=False
    ):
        lease.close()
        raise BoundedGpuError(
            "host/GPU coexistence changed during formal identity verification"
        )
    minimum_disk = args.minimum_free_disk_gib << 30
    short_tmp_parent, short_tmp_alias, short_tmp_target = create_short_tmp_alias(
        run_dir
    )
    environment["TMPDIR"] = str(short_tmp_alias)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            start_new_session=True,
            pass_fds=(lease.descriptor,),
        )
    except BaseException:
        remove_short_tmp_alias(
            short_tmp_parent, short_tmp_alias, short_tmp_target
        )
        lease.close()
        raise
    pgid = os.getpgid(process.pid)
    sid = os.getsid(process.pid)
    metadata: dict[str, object] = {
        "schema": PROCESS_SCHEMA_VERSION,
        "mode": "gpu-bounded",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "workspace": str(ROOT),
        "environment": args.environment,
        "framework_profile": args.framework_profile,
        "command": command,
        "pid": process.pid,
        "pgid": pgid,
        "sid": sid,
        "tmpdir": environment["TMPDIR"],
        "start_ticks": isolation.process_start_ticks(process.pid),
        "status": "running",
        "locked_mem_available_kib": locked_available,
        "policy": {
            "schema": 2,
            "kind": "pypto-gpu-resource-policy",
            "launch_admission_floor_kib": None,
            "host_abort_floor_kib": HOST_ABORT_KIB,
            "host_emergency_abort_floor_kib": HOST_EMERGENCY_ABORT_KIB,
            "host_floor_consecutive_samples": HOST_FLOOR_CONSECUTIVE_SAMPLES,
            "gpu_free_floor_mib": GPU_FREE_FLOOR_MIB,
            "nvidia_audit_failure_consecutive_samples": (
                NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES
            ),
            "protected_zero_nvidia_required": True,
            "external_process_signals": False,
            "termination_signal_scope": (
                "verified-pgid-then-verified-session-residuals"
            ),
            "successful_exit_cleanup": "natural-session-empty",
            "rss_accounting_scope": "owned-session-id",
            "formal_identity_verified": args.environment != "pypto-nvidia",
        },
        "environment_access_lock": {
            "path": str(lease.path),
            "mode": lease.mode,
            "device": lease.device,
            "inode": lease.inode,
        },
        "short_tmp_alias": {
            "path": str(short_tmp_alias),
            "target": str(short_tmp_target),
        },
    }
    metadata_path = run_dir / "process.json"
    isolation.atomic_json(metadata_path, metadata)
    print(
        f"PYPTO_GPU_BOUNDED_RUN_ID={run_id} PID={process.pid} "
        f"PGID={metadata['pgid']} SID={metadata['sid']}",
        flush=True,
    )

    deadline = time.monotonic() + args.timeout_seconds
    minimum_available = initial_available
    samples = 0
    maximum_owned_sid_rss = 0
    below_host_floor_samples = 0
    maximum_consecutive_below_host_floor = 0
    consecutive_nvidia_audit_failures = 0
    maximum_consecutive_nvidia_audit_failures = 0
    nvidia_audit_failures: list[dict[str, object]] = []
    abort_reason: str | None = None
    latest_audit = initial_audit
    try:
        while process.poll() is None:
            available = mem_available_kib()
            minimum_available = min(minimum_available, available)
            maximum_owned_sid_rss = max(
                maximum_owned_sid_rss, owned_sid_rss_kib(int(metadata["sid"]))
            )
            samples += 1
            abort_reason, below_host_floor_samples = host_floor_update(
                available, below_host_floor_samples
            )
            maximum_consecutive_below_host_floor = max(
                maximum_consecutive_below_host_floor,
                below_host_floor_samples,
            )
            if abort_reason is not None:
                break
            if shutil.disk_usage(ROOT).free < minimum_disk:
                abort_reason = "workspace-disk-floor"
                break
            if time.monotonic() >= deadline:
                abort_reason = "owned-run-timeout"
                break
            try:
                latest_audit = audit(
                    run_id, int(metadata["pgid"]), int(metadata["sid"])
                )
            except Exception as error:
                nvidia_audit_failures.append(
                    nvidia_audit_failure_record(
                        error, phase="runtime", sample_index=samples
                    )
                )
                (
                    abort_reason,
                    consecutive_nvidia_audit_failures,
                ) = nvidia_audit_failure_update(
                    consecutive_nvidia_audit_failures
                )
                maximum_consecutive_nvidia_audit_failures = max(
                    maximum_consecutive_nvidia_audit_failures,
                    consecutive_nvidia_audit_failures,
                )
                if abort_reason is not None:
                    break
                time.sleep(POLL_SECONDS)
                continue
            consecutive_nvidia_audit_failures = 0
            if not audit_ok(latest_audit, child_running=True):
                abort_reason = "nvidia-coexistence-audit"
                break
            time.sleep(POLL_SECONDS)
        if abort_reason is not None:
            terminate_owned(metadata, process)
            return_code = 75 if process.poll() is None else process.wait()
        else:
            return_code = process.wait()
        if "session_cleanup" not in metadata:
            metadata["session_cleanup"] = (
                stop_run.terminate_verified_session_residuals(
                    metadata, natural_wait_seconds=2.0
                )
            )
    except BaseException:
        terminate_owned(metadata, process)
        raise
    finally:
        lease.close()
        remove_short_tmp_alias(
            short_tmp_parent, short_tmp_alias, short_tmp_target
        )

    post_audit: dict[str, object] | None = None
    for attempt in range(1, NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES + 1):
        try:
            post_audit = audit()
            break
        except Exception as error:
            nvidia_audit_failures.append(
                nvidia_audit_failure_record(
                    error, phase="post-exit", sample_index=attempt
                )
            )
            if attempt < NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES:
                time.sleep(POLL_SECONDS)
    post_audit_available = post_audit is not None
    if post_audit is None:
        post_audit = {
            "status": "unavailable",
            "failure_count": NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES,
        }
    metadata.update(
        {
            "status": (
                "aborted"
                if abort_reason
                else "post-audit-unavailable"
                if not post_audit_available
                else "exited"
                if stop_run.session_cleanup_is_natural(
                    metadata["session_cleanup"]
                )
                else "cleanup-forced"
                if metadata["session_cleanup"]["complete"]
                else "cleanup-incomplete"
            ),
            "return_code": return_code,
            "abort_reason": abort_reason,
            "samples": samples,
            "initial_mem_available_kib": initial_available,
            "minimum_mem_available_kib": minimum_available,
            "maximum_owned_sid_rss_kib": maximum_owned_sid_rss,
            "maximum_consecutive_below_host_floor": (
                maximum_consecutive_below_host_floor
            ),
            "maximum_consecutive_nvidia_audit_failures": (
                maximum_consecutive_nvidia_audit_failures
            ),
            "nvidia_audit_failures": nvidia_audit_failures,
            "latest_runtime_audit": latest_audit,
            "post_audit": post_audit,
        }
    )
    isolation.atomic_json(metadata_path, metadata)
    if (
        abort_reason is not None
        or not stop_run.session_cleanup_is_natural(metadata["session_cleanup"])
        or not post_audit_available
        or not audit_ok(post_audit, child_running=False)
    ):
        return 75
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
