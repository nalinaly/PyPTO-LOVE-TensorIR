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
import time

import preflight
import run_isolated as isolation
import stop_run


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVIRONMENT = (ROOT / "envs/pypto-nvidia").resolve()
ALLOWED_ENVIRONMENT_PROFILES = {
    "pypto-nvidia": "pypto",
    "pypto-release": "pypto",
    "sglang-baseline": "baseline",
}
HOST_ABORT_KIB = 16 * 1024 * 1024
HOST_EMERGENCY_ABORT_KIB = 15 * 1024 * 1024
HOST_FLOOR_CONSECUTIVE_SAMPLES = 3
GPU_FREE_FLOOR_MIB = 4 * 1024
POLL_SECONDS = 1


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


def process_stat(pid: int) -> tuple[int, int]:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    if len(fields) <= 19:
        raise BoundedGpuError(f"malformed /proc/{pid}/stat")
    return int(fields[2]), int(fields[19])


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
    run_id: str | None = None, owned_pgid: int | None = None
) -> dict[str, object]:
    gpu = preflight.nvidia_identity()
    free_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    all_processes, protected, _workspace = preflight.process_table()
    protected_pids = {process.pid for process in protected}
    compute_pids = preflight.nvidia_compute_pids()
    protected_compute = sorted(compute_pids & protected_pids)
    protected_mappings, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    owned_compute: list[int] = []
    external_compute: list[int] = []
    for pid in sorted(compute_pids):
        pgid_owned = False
        if owned_pgid is not None:
            try:
                observed_pgid, _start_ticks = process_stat(pid)
                pgid_owned = observed_pgid == owned_pgid
            except (OSError, ValueError):
                pass
        try:
            environment = process_environment(pid)
        except OSError:
            (owned_compute if pgid_owned else external_compute).append(pid)
            continue
        if pgid_owned or (
            run_id is not None and environment.get("PYPTO_RUN_ID") == run_id
        ):
            owned_compute.append(pid)
        else:
            external_compute.append(pid)
    return {
        "gpu": gpu,
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
) -> None:
    if process.poll() is not None:
        return
    try:
        stop_run.signal_verified(metadata, signal.SIGTERM)
    except (OSError, RuntimeError):
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            stop_run.signal_verified(metadata, signal.SIGKILL)
        except (OSError, RuntimeError):
            pass


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
        raise BoundedGpuError("MemAvailable is already below 16 GiB abort floor")
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
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        start_new_session=True,
        pass_fds=(lease.descriptor,),
    )
    metadata: dict[str, object] = {
        "schema": 1,
        "mode": "gpu-bounded",
        "run_id": run_id,
        "workspace": str(ROOT),
        "environment": args.environment,
        "framework_profile": args.framework_profile,
        "command": command,
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": isolation.process_start_ticks(process.pid),
        "status": "running",
        "locked_mem_available_kib": locked_available,
        "policy": {
            "launch_admission_floor_kib": None,
            "host_abort_floor_kib": HOST_ABORT_KIB,
            "host_emergency_abort_floor_kib": HOST_EMERGENCY_ABORT_KIB,
            "host_floor_consecutive_samples": HOST_FLOOR_CONSECUTIVE_SAMPLES,
            "gpu_free_floor_mib": GPU_FREE_FLOOR_MIB,
            "protected_zero_nvidia_required": True,
            "external_process_signals": False,
            "formal_identity_verified": args.environment != "pypto-nvidia",
        },
        "environment_access_lock": {
            "path": str(lease.path),
            "mode": lease.mode,
            "device": lease.device,
            "inode": lease.inode,
        },
    }
    metadata_path = run_dir / "process.json"
    isolation.atomic_json(metadata_path, metadata)
    print(
        f"PYPTO_GPU_BOUNDED_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}",
        flush=True,
    )

    deadline = time.monotonic() + args.timeout_seconds
    minimum_available = initial_available
    samples = 0
    maximum_owned_rss = 0
    below_host_floor_samples = 0
    maximum_consecutive_below_host_floor = 0
    abort_reason: str | None = None
    latest_audit = initial_audit
    try:
        while process.poll() is None:
            available = mem_available_kib()
            minimum_available = min(minimum_available, available)
            maximum_owned_rss = max(
                maximum_owned_rss, owned_pgid_rss_kib(int(metadata["pgid"]))
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
            latest_audit = audit(run_id, int(metadata["pgid"]))
            if not audit_ok(latest_audit, child_running=True):
                abort_reason = "nvidia-coexistence-audit"
                break
            time.sleep(POLL_SECONDS)
        if abort_reason is not None:
            terminate_owned(metadata, process)
        return_code = process.wait()
    except BaseException:
        terminate_owned(metadata, process)
        raise
    finally:
        lease.close()

    post_audit = audit()
    metadata.update(
        {
            "status": "aborted" if abort_reason else "exited",
            "return_code": return_code,
            "abort_reason": abort_reason,
            "samples": samples,
            "initial_mem_available_kib": initial_available,
            "minimum_mem_available_kib": minimum_available,
            "maximum_owned_pgid_rss_kib": maximum_owned_rss,
            "maximum_consecutive_below_host_floor": (
                maximum_consecutive_below_host_floor
            ),
            "latest_runtime_audit": latest_audit,
            "post_audit": post_audit,
        }
    )
    isolation.atomic_json(metadata_path, metadata)
    if abort_reason is not None or not audit_ok(post_audit, child_running=False):
        return 75
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
