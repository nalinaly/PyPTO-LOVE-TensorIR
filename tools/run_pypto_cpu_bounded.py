#!/usr/bin/env python3
"""Run bounded release CPU builds/tests without the retired 22 GiB gate."""

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

import run_isolated as isolation
import stop_run


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVIRONMENT = (ROOT / "envs/pypto-nvidia").resolve()
ALLOWED_ENVIRONMENT_PROFILES = {
    "pypto-nvidia": "pypto",
    "pypto-release": "pypto",
    "sglang-baseline": "baseline",
}
# CMake/CTest subprocesses can drain after the owned process group is paused.
# Pytest-xdist workers retain imported Torch pages and cannot: stopping all 24
# workers below a pause floor self-locks the run.  Keep these as explicit,
# auditable policies instead of applying one memory action to both workloads.
PROCESS_SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = 2
PAUSE_DRAIN_MODE = "pause-drain"
PYTEST_RESIDENT_MODE = "pytest-resident-workers"
PAUSE_MEMORY_KIB = 12 * 1024 * 1024
RESUME_MEMORY_KIB = 13 * 1024 * 1024
PYTEST_LOW_MEMORY_RECORD_KIB = 12 * 1024 * 1024
PYTEST_EMERGENCY_MEMORY_KIB = 6 * 1024 * 1024
PYTEST_EMERGENCY_CONSECUTIVE_SAMPLES = 3
POLL_SECONDS = 0.2


class BoundedCpuError(RuntimeError):
    """The bounded CPU child cannot be launched safely."""


def mem_available_kib() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise BoundedCpuError("/proc/meminfo has no MemAvailable field")


def process_stat(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> tuple[int, int, int]:
    fields = (proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
    if len(fields) <= 19:
        raise BoundedCpuError(f"malformed /proc/{pid}/stat")
    return int(fields[2]), int(fields[3]), int(fields[19])


def owned_sid_rss_kib(sid: int, proc_root: pathlib.Path = pathlib.Path("/proc")) -> int:
    total = 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            _observed_pgid, observed_sid, _start_ticks = process_stat(
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


def _require_parallel_24(command: list[str], *, cmake: bool) -> None:
    matches = 0
    index = 0
    while index < len(command):
        argument = command[index]
        if cmake and argument == "--parallel":
            if index + 1 >= len(command) or command[index + 1] != "24":
                raise BoundedCpuError("PyPTO build parallelism must be exactly 24")
            matches += 1
            index += 2
            continue
        if cmake and argument.startswith("--parallel="):
            if argument != "--parallel=24":
                raise BoundedCpuError("PyPTO build parallelism must be exactly 24")
            matches += 1
        elif cmake and (argument == "-j" or argument.startswith("-j")):
            raise BoundedCpuError("PyPTO build must not contain a second -j setting")
        elif not cmake and argument == "-j":
            if index + 1 >= len(command) or command[index + 1] != "24":
                raise BoundedCpuError("PyPTO CTest parallelism must be exactly 24")
            matches += 1
            index += 2
            continue
        elif not cmake and argument.startswith("-j"):
            if argument != "-j24":
                raise BoundedCpuError("PyPTO CTest parallelism must be exactly 24")
            matches += 1
        elif not cmake and argument.startswith("--parallel"):
            raise BoundedCpuError("PyPTO CTest must use the pinned -j24 spelling")
        index += 1
    if matches != 1:
        label = "build" if cmake else "CTest"
        raise BoundedCpuError(f"PyPTO {label} requires one exact 24-way setting")


def _workspace_build_directory(raw: str) -> pathlib.Path:
    build_dir = pathlib.Path(raw).resolve(strict=True)
    builds_root = (ROOT / "builds").resolve(strict=True)
    if builds_root not in build_dir.parents:
        raise BoundedCpuError("build/test directory must remain below workspace builds")
    return build_dir


def _ctest_directory(command: list[str]) -> pathlib.Path:
    values: list[str] = []
    for index, argument in enumerate(command):
        if argument == "--test-dir":
            if index + 1 >= len(command):
                raise BoundedCpuError("CTest --test-dir requires a value")
            values.append(command[index + 1])
        elif argument.startswith("--test-dir="):
            values.append(argument.partition("=")[2])
    if len(values) != 1:
        raise BoundedCpuError("CTest requires one explicit --test-dir")
    return _workspace_build_directory(values[0])


def validate_environment_profile(environment: str, profile: str) -> pathlib.Path:
    if ALLOWED_ENVIRONMENT_PROFILES.get(environment) != profile:
        raise BoundedCpuError(
            f"unsupported environment/profile pair: {environment}/{profile}"
        )
    return isolation.ENVIRONMENTS[environment].resolve()


def _require_pytest_parallel_24(arguments: list[str]) -> None:
    matches = 0
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"-n", "--numprocesses"}:
            if index + 1 >= len(arguments) or arguments[index + 1] != "24":
                raise BoundedCpuError("pytest parallelism must be exactly 24")
            matches += 1
            index += 2
            continue
        if argument.startswith("-n") and argument != "-n24":
            raise BoundedCpuError("pytest parallelism must be exactly 24")
        if argument == "-n24":
            matches += 1
        if argument.startswith("--numprocesses="):
            if argument != "--numprocesses=24":
                raise BoundedCpuError("pytest parallelism must be exactly 24")
            matches += 1
        index += 1
    if matches != 1:
        raise BoundedCpuError("pytest requires one exact 24-way setting")


def _validate_release_python(command: list[str], expected_python: pathlib.Path) -> None:
    if pathlib.Path(command[0]).resolve(strict=True) != expected_python:
        raise BoundedCpuError("Python child must use the selected release prefix")
    arguments = command[1:]
    if arguments[:1] == ["-B"]:
        arguments = arguments[1:]
    if arguments[:2] == ["-m", "pytest"]:
        _require_pytest_parallel_24(arguments[2:])
        return
    if not arguments or arguments[0].startswith("-"):
        raise BoundedCpuError("release Python child must be pytest or a script")
    script = pathlib.Path(arguments[0]).resolve(strict=True)
    if ROOT not in script.parents or script.suffix != ".py":
        raise BoundedCpuError("release Python script must remain in the workspace")


def validate_command(
    raw: list[str], environment: pathlib.Path = ENVIRONMENT
) -> list[str]:
    if not raw or raw[0] != "--":
        raise BoundedCpuError("bounded CPU controller requires `--` before command")
    command = raw[1:]
    if len(command) < 2:
        raise BoundedCpuError("bounded CPU command is incomplete")
    executable = pathlib.Path(command[0]).resolve(strict=True)
    expected_cmake = (environment / "bin/cmake").resolve(strict=True)
    expected_ctest = (environment / "bin/ctest").resolve(strict=True)
    expected_python = (environment / "bin/python").resolve(strict=True)
    if executable == expected_cmake:
        if len(command) < 4 or command[1:2] != ["--build"]:
            raise BoundedCpuError("cmake child must be an explicit --build")
        _workspace_build_directory(command[2])
        _require_parallel_24(command[3:], cmake=True)
    elif executable == expected_ctest:
        _ctest_directory(command[1:])
        _require_parallel_24(command[1:], cmake=False)
    elif environment != ENVIRONMENT and executable == expected_python:
        _validate_release_python(command, expected_python)
    else:
        raise BoundedCpuError("child must be selected-prefix cmake or ctest")
    command[0] = str(executable)
    return command


def workload_mode(command: list[str], environment: pathlib.Path = ENVIRONMENT) -> str:
    executable = pathlib.Path(command[0]).resolve(strict=True)
    if executable in {
        (environment / "bin/cmake").resolve(strict=True),
        (environment / "bin/ctest").resolve(strict=True),
    }:
        return PAUSE_DRAIN_MODE
    expected_python = (environment / "bin/python").resolve(strict=True)
    if executable != expected_python:
        raise BoundedCpuError("validated CPU command has no workload policy")
    arguments = command[1:]
    if arguments[:1] == ["-B"]:
        arguments = arguments[1:]
    if arguments[:2] == ["-m", "pytest"]:
        return PYTEST_RESIDENT_MODE
    if not arguments:
        raise BoundedCpuError("validated Python command has no workload policy")
    script = pathlib.Path(arguments[0]).resolve(strict=True)
    if script == (ROOT / "tools/build_release.py").resolve(strict=True):
        if arguments.count("--_worker") != 1:
            raise BoundedCpuError("release build policy requires one --_worker stage")
        worker_index = arguments.index("--_worker")
        if worker_index + 1 >= len(arguments) or arguments[worker_index + 1] not in {
            "wheels",
            "native",
            "ctest",
            "install",
        }:
            raise BoundedCpuError("release build policy has an unknown worker stage")
        if arguments.count("--_jobs") != 1:
            raise BoundedCpuError("release build worker requires one --_jobs 24")
        jobs_index = arguments.index("--_jobs")
        if jobs_index + 1 >= len(arguments) or arguments[jobs_index + 1] != "24":
            raise BoundedCpuError("release build worker requires one --_jobs 24")
        return PAUSE_DRAIN_MODE
    if script == (ROOT / "tools/run_operator_regression.py").resolve(strict=True):
        if arguments[1:].count("--_worker-structure") != 1:
            raise BoundedCpuError(
                "operator CPU worker policy requires --_worker-structure"
            )
        if arguments.count("--_jobs") != 1:
            raise BoundedCpuError("operator CPU worker requires one --_jobs 24")
        jobs_index = arguments.index("--_jobs")
        if jobs_index + 1 >= len(arguments) or arguments[jobs_index + 1] != "24":
            raise BoundedCpuError("operator CPU worker requires one --_jobs 24")
        return PYTEST_RESIDENT_MODE
    raise BoundedCpuError(f"Python CPU workload has no memory policy: {script}")


def memory_policy_record(mode: str) -> dict[str, object]:
    if mode not in {PAUSE_DRAIN_MODE, PYTEST_RESIDENT_MODE}:
        raise BoundedCpuError(f"unknown CPU workload mode: {mode}")
    pause_enabled = mode == PAUSE_DRAIN_MODE
    return {
        "schema": POLICY_SCHEMA_VERSION,
        "kind": "pypto-cpu-memory-policy",
        "workload_mode": mode,
        "launch_admission_floor_kib": None,
        "pause_enabled": pause_enabled,
        "pause_memory_floor_kib": PAUSE_MEMORY_KIB if pause_enabled else None,
        "resume_memory_floor_kib": RESUME_MEMORY_KIB if pause_enabled else None,
        "low_memory_recording_floor_kib": (
            None if pause_enabled else PYTEST_LOW_MEMORY_RECORD_KIB
        ),
        "emergency_abort_memory_floor_kib": (
            None if pause_enabled else PYTEST_EMERGENCY_MEMORY_KIB
        ),
        "emergency_abort_consecutive_samples": (
            None if pause_enabled else PYTEST_EMERGENCY_CONSECUTIVE_SAMPLES
        ),
        "low_memory_action": (
            "pause-owned-pgid" if pause_enabled else "record-and-continue"
        ),
        "emergency_action": (
            None if pause_enabled else "terminate-owned-pgid-after-consecutive-samples"
        ),
        "parallelism": 24,
        "external_process_signals": False,
        "signal_scope": "verified-owned-pgid-only",
        "rss_accounting_scope": "owned-session-id",
    }


def memory_policy_update(
    mode: str,
    available_kib: int,
    *,
    paused: bool,
    consecutive_emergency_samples: int,
) -> tuple[str | None, bool, int]:
    """Return (action, paused, emergency_count) for one host-memory sample."""

    if type(available_kib) is not int or available_kib < 0:
        raise ValueError("MemAvailable sample must be a non-negative integer")
    if (
        type(consecutive_emergency_samples) is not int
        or consecutive_emergency_samples < 0
    ):
        raise ValueError("emergency sample count must be a non-negative integer")
    if mode == PAUSE_DRAIN_MODE:
        if not paused and available_kib < PAUSE_MEMORY_KIB:
            return "pause", True, 0
        if paused and available_kib >= RESUME_MEMORY_KIB:
            return "resume", False, 0
        return None, paused, 0
    if mode != PYTEST_RESIDENT_MODE:
        raise BoundedCpuError(f"unknown CPU workload mode: {mode}")
    if paused:
        raise BoundedCpuError("pytest resident-worker policy must never be paused")
    count = (
        consecutive_emergency_samples + 1
        if available_kib < PYTEST_EMERGENCY_MEMORY_KIB
        else 0
    )
    return (
        "abort" if count >= PYTEST_EMERGENCY_CONSECUTIVE_SAMPLES else None,
        False,
        count,
    )


def low_memory_sample_record(
    mode: str,
    *,
    sample_index: int,
    available_kib: int,
    owned_sid_rss_kib: int,
    consecutive_emergency_samples: int,
) -> dict[str, int] | None:
    if mode != PYTEST_RESIDENT_MODE or available_kib >= PYTEST_LOW_MEMORY_RECORD_KIB:
        return None
    return {
        "sample_index": sample_index,
        "mem_available_kib": available_kib,
        "owned_sid_rss_kib": owned_sid_rss_kib,
        "consecutive_emergency_samples": consecutive_emergency_samples,
    }


def write_run_id(path: pathlib.Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    isolation.atomic_json(path, {"schema_version": 1, "run_id": run_id})


def terminate_owned(
    metadata: dict[str, object], process: subprocess.Popen[bytes]
) -> None:
    if process.poll() is not None:
        return
    try:
        stop_run.signal_verified(metadata, signal.SIGCONT)
    except (OSError, RuntimeError):
        pass
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


def verify_formal_environment_identity(
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
        raise BoundedCpuError(
            f"formal environment identity lock is missing: {identity_lock}"
        )
    for identity_command in (
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
    ):
        subprocess.run(
            identity_command,
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
    parser.add_argument("--timeout-seconds", type=int, default=3600)
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
    command = validate_command(args.command, environment_prefix)
    selected_workload_mode = workload_mode(command, environment_prefix)
    initial_available = mem_available_kib()

    lease = isolation.acquire_environment_lock(args.environment, "shared")
    locked_available = mem_available_kib()
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"pypto-cpu-bounded-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_run_id(args.run_id_file, run_id)
    environment = isolation.isolated_environment(
        run_id,
        run_dir,
        environment_prefix=environment_prefix,
        framework_profile=args.framework_profile,
        protected_cpu_only_coexistence_requested=True,
    )
    environment.update(isolation.environment_lock_markers(lease))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYPTO_RUN_MODE": "cpu-bounded",
        }
    )
    try:
        verify_formal_environment_identity(
            args.environment, args.framework_profile, environment
        )
    except BaseException:
        lease.close()
        raise
    minimum_disk = args.minimum_free_disk_gib << 30
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        start_new_session=True,
        pass_fds=(lease.descriptor,),
    )
    pgid = os.getpgid(process.pid)
    sid = os.getsid(process.pid)
    metadata: dict[str, object] = {
        "schema": PROCESS_SCHEMA_VERSION,
        "mode": "cpu-bounded",
        "run_id": run_id,
        "workspace": str(ROOT),
        "environment": args.environment,
        "framework_profile": args.framework_profile,
        "command": command,
        "pid": process.pid,
        "pgid": pgid,
        "sid": sid,
        "start_ticks": isolation.process_start_ticks(process.pid),
        "status": "running",
        "locked_mem_available_kib": locked_available,
        "policy": {
            **memory_policy_record(selected_workload_mode),
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
        f"PYPTO_CPU_BOUNDED_RUN_ID={run_id} PID={process.pid} "
        f"PGID={metadata['pgid']} SID={metadata['sid']}",
        flush=True,
    )

    deadline = time.monotonic() + args.timeout_seconds
    minimum_available = min(initial_available, locked_available)
    maximum_owned_sid_rss = 0
    samples = 0
    pauses: list[dict[str, object]] = []
    low_memory_samples: list[dict[str, int]] = []
    paused = False
    consecutive_emergency_samples = 0
    maximum_consecutive_emergency_samples = 0
    abort_reason: str | None = None
    try:
        while process.poll() is None:
            available = mem_available_kib()
            minimum_available = min(minimum_available, available)
            owned_sid_rss = owned_sid_rss_kib(int(metadata["sid"]))
            maximum_owned_sid_rss = max(maximum_owned_sid_rss, owned_sid_rss)
            samples += 1
            if shutil.disk_usage(ROOT).free < minimum_disk:
                abort_reason = "workspace-disk-floor"
                break
            if time.monotonic() >= deadline:
                abort_reason = "owned-run-timeout"
                break
            action, next_paused, consecutive_emergency_samples = memory_policy_update(
                selected_workload_mode,
                available,
                paused=paused,
                consecutive_emergency_samples=consecutive_emergency_samples,
            )
            maximum_consecutive_emergency_samples = max(
                maximum_consecutive_emergency_samples,
                consecutive_emergency_samples,
            )
            low_memory_sample = low_memory_sample_record(
                selected_workload_mode,
                sample_index=samples,
                available_kib=available,
                owned_sid_rss_kib=owned_sid_rss,
                consecutive_emergency_samples=consecutive_emergency_samples,
            )
            if low_memory_sample is not None:
                low_memory_samples.append(low_memory_sample)
            if action == "pause":
                stop_run.signal_verified(metadata, signal.SIGSTOP)
                pauses.append({"action": "pause", "mem_available_kib": available})
            elif action == "resume":
                stop_run.signal_verified(metadata, signal.SIGCONT)
                pauses.append({"action": "resume", "mem_available_kib": available})
            elif action == "abort":
                abort_reason = "host-memory-emergency-floor"
                break
            paused = next_paused
            time.sleep(POLL_SECONDS)
        if abort_reason is not None:
            terminate_owned(metadata, process)
        return_code = process.wait()
    except BaseException:
        terminate_owned(metadata, process)
        raise
    finally:
        lease.close()

    metadata.update(
        {
            "status": "aborted" if abort_reason else "exited",
            "return_code": return_code,
            "abort_reason": abort_reason,
            "samples": samples,
            "sample_period_ms": int(POLL_SECONDS * 1000),
            "initial_mem_available_kib": initial_available,
            "minimum_mem_available_kib": minimum_available,
            "maximum_owned_sid_rss_kib": maximum_owned_sid_rss,
            "pauses": pauses,
            "low_memory_samples": low_memory_samples,
            "low_memory_sample_count": len(low_memory_samples),
            "maximum_consecutive_emergency_samples": (
                maximum_consecutive_emergency_samples
            ),
        }
    )
    isolation.atomic_json(metadata_path, metadata)
    return 75 if abort_reason else return_code


if __name__ == "__main__":
    raise SystemExit(main())
