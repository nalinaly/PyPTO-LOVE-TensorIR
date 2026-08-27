#!/usr/bin/env python3
"""Generic policy-2 owned GPU smoke controller for bounded project smokes.

Reuses the pinned policy-2 admission machinery (22 GiB protected-lane /
32 GiB exclusive admission, 16 GiB owned abort, 4 GiB free-GPU floor,
protected zero-NVIDIA audits, verified owned-group signalling) from the
frozen fused-pointwise v2 controller by exact hash. Unlike the frozen
family controllers, the child command is caller-supplied: a single
absolute workspace python script plus arguments. Per D-0018 this lane
carries no per-transaction review ceremony; the safety boundaries are
unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import sys
from types import ModuleType

ROOT = pathlib.Path(__file__).resolve().parents[1]

BASE_POLICY2_RELATIVE_PATH = pathlib.Path(
    "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py"
)
BASE_POLICY2_SIZE = 23_094
BASE_POLICY2_SHA256 = (
    "be93fdac4ac5b5c097ac3899d4b90956afce5af113c06df70eb65f94c10c44c0"
)
BASE_ISOLATION_RELATIVE_PATH = pathlib.Path("tools/run_isolated.py")
BASE_ISOLATION_SIZE = 76_558
BASE_ISOLATION_SHA256 = (
    "978686ac09743a98233c9616d23b04e57d3a257bd643d5db3b8a71eaac7465c8"
)


class ControllerError(RuntimeError):
    """The generic GPU smoke child cannot be admitted safely."""


def load_exact(
    name: str,
    path: pathlib.Path,
    size: int,
    digest: str,
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControllerError(f"exact generic-smoke source is noncanonical: {path}")
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ControllerError(f"exact generic-smoke source differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


base = load_exact(
    "_pypto_generic_smoke_policy2_base",
    ROOT / BASE_POLICY2_RELATIVE_PATH,
    BASE_POLICY2_SIZE,
    BASE_POLICY2_SHA256,
)
isolation = load_exact(
    "_pypto_generic_smoke_isolation",
    ROOT / BASE_ISOLATION_RELATIVE_PATH,
    BASE_ISOLATION_SIZE,
    BASE_ISOLATION_SHA256,
)
preflight = base.preflight
stop_run = isolation.stop_run
contract = base.contract
GPU_SMOKE_TIMEOUT_SECONDS = 1_800
GPU_SMOKE_MINIMUM_FREE_DISK_GIB = 64


def validate_child(command: list[str]) -> list[str]:
    if not command or command[0] != "--":
        raise ControllerError("generic GPU smoke requires `--` before the child command")
    command = command[1:]
    expected_python = str(
        (ROOT / "envs/pypto-nvidia/bin/python").resolve(strict=True)
    )
    actual_python = str(pathlib.Path(command[0]).resolve(strict=True))
    if (
        len(command) < 3
        or actual_python != expected_python
        or command[1] != "-B"
    ):
        raise ControllerError(
            "generic GPU smoke child must be the workspace python with -B"
        )
    script = pathlib.Path(command[2]).resolve(strict=True)
    if ROOT not in script.parents or script.suffix != ".py":
        raise ControllerError(
            "generic GPU smoke child script must be a workspace python file"
        )
    return [command[0], command[1], str(script), *command[3:]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-protected-zero-nvidia-gpu-smoke", action="store_true")
    parser.add_argument("--run-id-file", type=pathlib.Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=GPU_SMOKE_TIMEOUT_SECONDS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("generic GPU smoke controller requires Python -E -B -S")
    command = validate_child(args.command)
    lease = base.acquire_shared_environment_lease()
    if lease is None:
        return 75
    try:
        environment_prefix = isolation.ENVIRONMENTS["pypto-nvidia"].resolve(strict=True)
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"pypto-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
        run_dir = ROOT / "runs" / run_id
        run_dir.mkdir(parents=True)
        base._write_run_id(args.run_id_file, run_id)
        try:
            initial_code, initial_report, _ = base.run_preflight(
                allow_protected=args.allow_protected_zero_nvidia_gpu_smoke,
                environment_prefix=environment_prefix,
                description="initial generic-smoke preflight",
            )
        except base.ControllerV2Error as error:
            print(f"generic-smoke preflight refusal: {error}", file=sys.stderr)
            return 75
        initial_path = run_dir / "initial-preflight.json"
        isolation.atomic_json(initial_path, initial_report)
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
        try:
            action_code, action_report, _ = base.run_preflight(
                allow_protected=args.allow_protected_zero_nvidia_gpu_smoke,
                environment_prefix=environment_prefix,
                description="action-boundary generic-smoke preflight",
            )
        except base.ControllerV2Error as error:
            print(f"generic-smoke preflight refusal: {error}", file=sys.stderr)
            return 75
        preflight_path = run_dir / "preflight.json"
        isolation.atomic_json(preflight_path, action_report)
        if action_code != 0 or action_report.get("ok") is not True:
            return action_code if action_code != 0 else 75
        environment.update(
            {
                "PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED": (
                    "1" if args.allow_protected_zero_nvidia_gpu_smoke else "0"
                ),
                "PYPTO_RUN_MODE": "gpu-smoke",
                "PYPTO_PREFLIGHT_REPORT_PATH": str(preflight_path),
            }
        )
        minimum_free_disk_bytes = GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
        if shutil.disk_usage(ROOT).free < minimum_free_disk_bytes:
            return 75
        metadata = {
            "run_id": run_id,
            "workspace": str(ROOT),
            "command": command,
            "pid": None,
            "pgid": None,
            "status": "running",
            "environment_access_lock": {
                "path": str(lease.path),
                "mode": lease.mode,
                "device": lease.device,
                "inode": lease.inode,
            },
            "gpu_smoke": {
                "requested": args.allow_protected_zero_nvidia_gpu_smoke,
                "waiver_applied": bool(
                    action_report.get("protected_gpu_smoke_waiver_applied")
                ),
            },
            "run_timeout_seconds": args.timeout_seconds,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
        }
        metadata_path = run_dir / "process.json"
        handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_handlers = {item: signal.getsignal(item) for item in handled}
        for item in handled:
            signal.signal(item, isolation.interrupt_parent)
        process = None
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
                metadata["pid"] = process.pid
                metadata["pgid"] = os.getpgid(process.pid)
                metadata["start_ticks"] = isolation.process_start_ticks(process.pid)
                isolation.atomic_json(metadata_path, metadata)
                stop_run.verify(metadata)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            print(
                f"PYPTO_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}",
                flush=True,
            )
            try:
                child_code, aborted = isolation.wait_with_gpu_smoke_watchdog(
                    process,
                    metadata,
                    timeout_seconds=args.timeout_seconds,
                    minimum_free_disk_bytes=minimum_free_disk_bytes,
                    metadata_path=metadata_path,
                )
            except subprocess.TimeoutExpired:
                # The frozen poll occasionally surfaces its internal
                # timeout here; fall back to a blocking wait. The
                # post-exit audits below still verify ownership and
                # external-NVIDIA state.
                child_code = process.wait()
                aborted = False
            except Exception as frozen_abort_race:
                # The frozen abort path can crash in stop_run when a
                # survivor's /proc environ disappears mid-signal. Never
                # mask a live child: only treat as an aborted run when
                # the leader is already dead or the group is empty.
                if process.poll() is None:
                    raise
                print(
                    "generic-smoke frozen abort race (leader exited): "
                    f"{frozen_abort_race!r}",
                    file=sys.stderr,
                )
                child_code = process.wait()
                aborted = True
            except (OSError, RuntimeError) as watchdog_race:
                # The frozen stop primitive can lose a /proc/<pid>/environ
                # race when the leader exits between member enumeration and
                # env verification. Treat an already-dead leader as a normal
                # exit and re-check survivors below; never signal blindly.
                if process.poll() is None:
                    raise
                print(
                    f"generic-smoke watchdog race (leader exited): {watchdog_race}",
                    file=sys.stderr,
                )
                child_code = process.wait()
                aborted = False
            metadata["return_code"] = child_code
            metadata["aborted"] = aborted
            snapshot, violation = isolation.audit_gpu_smoke_runtime_state(metadata)
            metadata["post_exit_audit"] = snapshot
            metadata["post_exit_violation"] = violation
            if violation:
                metadata["status"] = "gpu-smoke-violation"
            survivors = stop_run.exact_process_group_members(metadata["pgid"])
            metadata["post_exit_group_members"] = survivors
            metadata["status"] = (
                "exited" if not survivors else metadata.get("status", "alive")
            )
            isolation.atomic_json(metadata_path, metadata)
            if violation or survivors or child_code != 0:
                return 75
            return 0
        finally:
            for item, previous in previous_handlers.items():
                signal.signal(item, previous)
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
