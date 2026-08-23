#!/usr/bin/env python3
"""Launch one workspace-owned command in an isolated NVIDIA process group."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import secrets
import signal
import subprocess
import sys

import stop_run


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    "pypto-nvidia": ROOT / "envs" / "pypto-nvidia",
    "sglang-baseline-py312": ROOT / "envs" / "sglang-baseline-py312",
}
ENVIRONMENT_LOCKS = {
    "pypto-nvidia": ROOT / "ENVIRONMENT.lock",
    "sglang-baseline-py312": (
        ROOT / "state" / "environments" / "sglang-baseline-py312.lock.json"
    ),
}
PROFILE_ENVIRONMENTS = {
    "pypto": "pypto-nvidia",
    "baseline": "sglang-baseline-py312",
}
PASSTHROUGH_ENV_NAMES = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TZ",
    "USER",
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
}
PYPTO_CONTROL_ENV_NAMES = {
    "PTO_BACKTRACE",
    "PYPTO_ALLOW_FALLBACK",
    "PYPTO_CODEGEN_MAX_WORKERS",
    "PYPTO_COMPILE_PROFILING",
    "PYPTO_EMIT_DEBUG_RUNNER",
    "PYPTO_EMIT_PTO_LOC",
    "PYPTO_INDUCTOR_CUDA_BACKEND",
    "PYPTO_LOG_LEVEL",
    "PYPTO_REBUILD_FROM_PTO",
    "PYPTO_RUNTIME_LOG",
    "PYPTO_RUNTIME_LOG_SYNC",
    "PYPTO_STRICT_COVERAGE",
    "PYPTO_VERIFY_LEVEL",
    "PYPTO_WARNING_LEVEL",
    "TORCHINDUCTOR_FX_COMPILE_MODE",
}
PYPTO_PATH_CONTROL_ENV_NAMES = {
    "PTOAS_ROOT",
    "PTO_ISA_ROOT",
}
SYSTEM_EXECUTABLE_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
)


def process_start_ticks(pid: int) -> int:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def atomic_json(path: pathlib.Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class RunInterrupted(Exception):
    """Raised by the parent signal handler so verified cleanup can run."""

    def __init__(self, signum: int):
        super().__init__(f"run parent received signal {signum}")
        self.signum = signum


def interrupt_parent(signum: int, _frame: object) -> None:
    raise RunInterrupted(signum)


def terminate_owned_process(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
) -> int:
    """Signal only after stop_run revalidates the live process identity."""

    if process.poll() is not None:
        return process.wait()
    try:
        stop_run.signal_verified(metadata, signal.SIGTERM)
    except ProcessLookupError:
        # The child won the poll/verify race and needs only to be reaped.
        return process.wait()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # No escalation: the user explicitly forbids broad/forced cleanup.
        return 75


def build_run_metadata(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    run_id: str,
    environment_prefix: pathlib.Path,
    framework_profile: str,
    framework_launch: bool,
    mode: str,
    command: list[str],
    timestamp: str,
) -> dict[str, object]:
    """Capture the ownership fields required by stop_run.verify."""

    return {
        "schema": 1,
        "run_id": run_id,
        "workspace": str(ROOT),
        "environment": str(environment_prefix),
        "framework_profile": framework_profile,
        "framework_launch": framework_launch,
        "mode": mode,
        "command": command,
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": process_start_ticks(process.pid),
        "started_at": timestamp,
        "status": "running",
    }


def isolated_environment(
    run_id: str,
    run_dir: pathlib.Path,
    *,
    environment_prefix: pathlib.Path,
    framework_profile: str,
) -> dict[str, str]:
    if framework_profile not in PROFILE_ENVIRONMENTS:
        raise ValueError(f"unknown framework profile: {framework_profile!r}")
    expected_prefix = ENVIRONMENTS[PROFILE_ENVIRONMENTS[framework_profile]].resolve()
    if environment_prefix.resolve() != expected_prefix:
        raise ValueError(
            f"framework profile {framework_profile!r} requires environment "
            f"{expected_prefix}, got {environment_prefix.resolve()}"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in PASSTHROUGH_ENV_NAMES
    }
    if framework_profile == "pypto":
        environment.update(
            {
                name: os.environ[name]
                for name in PYPTO_CONTROL_ENV_NAMES
                if name in os.environ
            }
        )
        for name in PYPTO_PATH_CONTROL_ENV_NAMES:
            if name not in os.environ:
                continue
            path = pathlib.Path(os.environ[name]).resolve()
            if not (path == ROOT or ROOT in path.parents):
                raise ValueError(f"{name} must remain below the workspace, got {path}")
            environment[name] = str(path)
    executable_path = [str(pathlib.Path("/usr/local/cuda-13.3/bin"))]
    library_path = ["/usr/lib/wsl/lib", "/usr/local/cuda-13.3/lib64"]
    if environment_prefix.is_dir():
        executable_path.insert(0, str(environment_prefix / "bin"))
        library_path.insert(0, str(environment_prefix / "lib"))
    executable_path.append(SYSTEM_EXECUTABLE_PATH)

    cache_root = ROOT / "caches"
    paths = {
        "TMPDIR": run_dir / "tmp",
        "HF_HOME": cache_root / "huggingface",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor" / run_id,
        "TRITON_CACHE_DIR": cache_root / "triton" / run_id,
        "CUDA_CACHE_PATH": cache_root / "cuda" / run_id,
        "PYPTO_CACHE_DIR": cache_root / "pypto",
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
        "CONDA_PKGS_DIRS": cache_root / "conda-pkgs",
        "CCACHE_DIR": cache_root / "ccache",
        "TRITON_HOME": cache_root / "triton-home",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch-extensions" / run_id,
        "SGLANG_CACHE_DIR": cache_root / "sglang" / run_id,
        "PYPTO_PROG_BUILD_DIR": run_dir / "build-output",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    environment.update({name: str(path) for name, path in paths.items()})
    common_python_paths = (str(ROOT / "upstream" / "sglang" / "python"),)
    if framework_profile == "pypto":
        python_paths = (
            str(ROOT / "projects" / "pypto-framework-plugins" / "src"),
            str(ROOT / "projects" / "pypto-kernels" / "src"),
            str(ROOT / "projects" / "pypto" / "python"),
            *common_python_paths,
        )
        sglang_plugins = "pypto"
    elif framework_profile == "baseline":
        python_paths = common_python_paths
        # A non-empty whitelist with no matching entry point disables every
        # installed general plugin. Empty would mean "load all" in SGLang.
        sglang_plugins = "__pypto_baseline_no_plugins__"
    else:  # guarded by the direct-call invariant above
        raise AssertionError("unreachable framework profile")

    environment.update(
        {
            "PATH": os.pathsep.join(executable_path),
            "LD_LIBRARY_PATH": os.pathsep.join(library_path),
            "CUDA_HOME": "/usr/local/cuda-13.3",
            "CUDA_PATH": "/usr/local/cuda-13.3",
            "CUDACXX": "/usr/local/cuda-13.3/bin/nvcc",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CONDA_PREFIX": str(environment_prefix),
            "CONDA_DEFAULT_ENV": environment_prefix.name,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYPTO_ENV_PREFIX": str(environment_prefix),
            "PYPTO_FRAMEWORK_PROFILE": framework_profile,
            "PYPTO_RUN_ID": run_id,
            "PYPTO_WORKSPACE_ROOT": str(ROOT),
            "PYPTO_SGLANG_SOURCE_ROOT": str(ROOT / "upstream" / "sglang"),
            "SGLANG_PLUGINS": sglang_plugins,
        }
    )
    if framework_profile == "baseline":
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("light", "heavy", "gpu-benchmark"),
        default="light",
    )
    parser.add_argument(
        "--require-framework-plugins",
        action="store_true",
        help="verify installed Torch and SGLang PyPTO entry points before launch",
    )
    parser.add_argument(
        "--environment",
        choices=tuple(ENVIRONMENTS),
        default="pypto-nvidia",
    )
    parser.add_argument(
        "--framework-profile",
        choices=("pypto", "baseline"),
        default="pypto",
    )
    parser.add_argument(
        "--framework-launch",
        action="store_true",
        help="require selected-prefix Python, runtime/SGLang audit, and profile plugin policy",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.require_framework_plugins and args.framework_profile != "pypto":
        parser.error("--require-framework-plugins requires --framework-profile pypto")
    expected_environment = PROFILE_ENVIRONMENTS[args.framework_profile]
    if args.environment != expected_environment:
        parser.error(
            f"--framework-profile {args.framework_profile} requires "
            f"--environment {expected_environment}"
        )

    environment_prefix = ENVIRONMENTS[args.environment].resolve()
    environment_lock = ENVIRONMENT_LOCKS[args.environment].resolve()
    if args.framework_launch:
        expected_python = (environment_prefix / "bin" / "python").resolve()
        command_python = pathlib.Path(command[0]).resolve()
        if command_python != expected_python:
            parser.error(
                "--framework-launch requires the selected environment Python: "
                f"expected {expected_python}, got {command_python}"
            )

    preflight = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "preflight.py"), "--mode", args.mode],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode != 0:
        return preflight.returncode

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"pypto-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    environment = isolated_environment(
        run_id,
        run_dir,
        environment_prefix=environment_prefix,
        framework_profile=args.framework_profile,
    )
    command_requires_identity = (
        args.mode == "gpu-benchmark"
        or args.framework_launch
        or args.require_framework_plugins
    )
    if command_requires_identity:
        python = environment_prefix / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(
                f"project environment is missing; cannot verify runtime: {python}"
            )
        if not environment_lock.is_file():
            raise FileNotFoundError(
                f"environment identity lock is missing: {environment_lock}"
            )
        subprocess.run(
            [
                str(python),
                str(ROOT / "tools" / "environment_identity.py"),
                "--prefix",
                str(environment_prefix),
                "--lock",
                str(environment_lock),
                "--verify",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                str(ROOT / "tools" / "audit_python_environment.py"),
                "--prefix",
                str(environment_prefix),
                "--profile",
                args.framework_profile,
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        runtime_identity_command = [
            str(python),
            str(ROOT / "tools" / "runtime_identity.py"),
            "--prefix",
            str(environment_prefix),
            "--lock",
            str(environment_lock),
            "--profile",
            args.framework_profile,
        ]
        if args.framework_launch:
            runtime_identity_command.append("--framework")
        subprocess.run(
            runtime_identity_command,
            cwd=ROOT,
            env=environment,
            check=True,
        )
    command_requires_plugins = (
        args.require_framework_plugins
        or (args.framework_profile == "pypto" and args.framework_launch)
    )
    if command_requires_plugins:
        python = environment_prefix / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(
                f"project environment is missing; cannot verify plugins: {python}"
            )
        subprocess.run(
            [str(python), "-m", "pypto_plugins.bootstrap"],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    metadata_path = run_dir / "process.json"
    process: subprocess.Popen[bytes] | None = None
    metadata: dict[str, object] | None = None
    parent_return_code: int | None = None
    child_return_code: int | None = None
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, interrupt_parent)
    try:
        # Block termination signals across Popen's return-value/STORE_FAST
        # boundary and metadata publication. A pending signal is delivered as
        # soon as the old mask is restored, when verified cleanup is possible.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled_signals)
        try:
            def restore_child_signal_mask() -> None:
                # The child inherits the parent's temporary block mask across
                # fork. Restore the pre-block mask before exec so verified
                # SIGTERM remains deliverable to the owned process group.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                start_new_session=True,
                preexec_fn=restore_child_signal_mask,
            )
            metadata = build_run_metadata(
                process,
                run_id=run_id,
                environment_prefix=environment_prefix,
                framework_profile=args.framework_profile,
                framework_launch=args.framework_launch,
                mode=args.mode,
                command=command,
                timestamp=timestamp,
            )
            atomic_json(metadata_path, metadata)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        print(
            f"PYPTO_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}",
            flush=True,
        )
        child_return_code = process.wait()
        parent_return_code = child_return_code
    except KeyboardInterrupt:
        if process is not None:
            if process.poll() is None:
                if metadata is None:
                    metadata = build_run_metadata(
                        process,
                        run_id=run_id,
                        environment_prefix=environment_prefix,
                        framework_profile=args.framework_profile,
                        framework_launch=args.framework_launch,
                        mode=args.mode,
                        command=command,
                        timestamp=timestamp,
                    )
                child_return_code = terminate_owned_process(process, metadata)
            else:
                child_return_code = process.wait()
        parent_return_code = 130
    except RunInterrupted as interruption:
        if process is not None:
            if process.poll() is None:
                if metadata is None:
                    metadata = build_run_metadata(
                        process,
                        run_id=run_id,
                        environment_prefix=environment_prefix,
                        framework_profile=args.framework_profile,
                        framework_launch=args.framework_launch,
                        mode=args.mode,
                        command=command,
                        timestamp=timestamp,
                    )
                child_return_code = terminate_owned_process(process, metadata)
            else:
                child_return_code = process.wait()
        parent_return_code = 128 + interruption.signum
    except BaseException:
        if process is not None and process.poll() is None:
            if metadata is None:
                metadata = build_run_metadata(
                    process,
                    run_id=run_id,
                    environment_prefix=environment_prefix,
                    framework_profile=args.framework_profile,
                    framework_launch=args.framework_launch,
                    mode=args.mode,
                    command=command,
                    timestamp=timestamp,
                )
            terminate_owned_process(process, metadata)
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if metadata is not None:
            metadata["status"] = (
                "exited" if process is not None and process.poll() is not None else "alive"
            )
            metadata["return_code"] = child_return_code
            metadata["finished_at"] = datetime.datetime.now(datetime.UTC).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            atomic_json(metadata_path, metadata)
    if parent_return_code is None:
        raise RuntimeError("run exited without a parent return code")
    return parent_return_code


if __name__ == "__main__":
    sys.exit(main())
