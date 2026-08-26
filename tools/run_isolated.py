#!/usr/bin/env python3
"""Launch one workspace-owned command in an isolated NVIDIA process group."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import functools
import hashlib
import json
import os
import pathlib
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import time

import _pypto_nvidia_executable_sm120_contract as nvidia_smoke_contract
import _pypto_nvidia_sm120_control_manifest as nvidia_smoke_control
import preflight as preflight_tool
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
ENVIRONMENT_TRANSACTION_LOCKS = {
    "pypto-nvidia": ROOT / "runs" / "environment-pypto-nvidia.lock",
    "sglang-baseline-py312": (ROOT / "runs" / "environment-sglang-baseline-py312.lock"),
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
COEXISTENCE_ABORT_MEMORY_KIB = 16 * 1024 * 1024
COEXISTENCE_RESUME_MEMORY_KIB = 22 * 1024 * 1024
COEXISTENCE_POLL_SECONDS = 5
GPU_BENCHMARK_POLL_SECONDS = 1
GPU_SMOKE_POLL_SECONDS = 1


class EnvironmentLockBusy(RuntimeError):
    """Raised when a conflicting environment consumer/transaction is active."""


class EnvironmentLockLease:
    """Own one advisory lock and close it deterministically when requested."""

    def __init__(self, descriptor: int, path: pathlib.Path, mode: str):
        self.descriptor = descriptor
        self.path = path
        self.mode = mode
        identity = os.fstat(descriptor)
        self.device = identity.st_dev
        self.inode = identity.st_ino

    def close(self) -> None:
        descriptor = self.descriptor
        if descriptor < 0:
            return
        self.descriptor = -1
        # Do not issue LOCK_UN: pass_fds gives the child the same open-file
        # description.  Closing only this duplicate keeps the flock live while
        # any child/survivor still owns its inherited descriptor.
        os.close(descriptor)

    def __del__(self) -> None:
        # The main() decorator is the deterministic owner.  This is only a
        # last-resort guard for direct helper callers that abandon a lease.
        try:
            self.close()
        except OSError:
            pass


_ACTIVE_ENVIRONMENT_LOCK: EnvironmentLockLease | None = None


def close_registered_environment_lock(function):
    """Close main()'s lease on every return/exception, including SystemExit."""

    @functools.wraps(function)
    def guarded(*args, **kwargs):
        global _ACTIVE_ENVIRONMENT_LOCK
        if _ACTIVE_ENVIRONMENT_LOCK is not None:
            raise RuntimeError("nested run_isolated environment lock registration")
        try:
            return function(*args, **kwargs)
        finally:
            lease = _ACTIVE_ENVIRONMENT_LOCK
            _ACTIVE_ENVIRONMENT_LOCK = None
            if lease is not None:
                lease.close()

    return guarded


def acquire_environment_lock(
    environment_name: str,
    mode: str,
) -> EnvironmentLockLease:
    if mode not in {"shared", "exclusive"}:
        raise ValueError(f"unknown environment lock mode: {mode!r}")
    path = ENVIRONMENT_TRANSACTION_LOCKS[environment_name]
    runs_root = ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    runs_lstat = runs_root.lstat()
    if runs_root.is_symlink() or not stat.S_ISDIR(runs_lstat.st_mode):
        raise RuntimeError(
            f"workspace runs root is not an independent directory: {runs_root}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    if parent != runs_root.resolve(strict=True):
        raise RuntimeError(f"environment lock parent escaped workspace runs: {parent}")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"environment lock is not a regular file: {path}")
        if (
            opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise RuntimeError(f"environment lock ownership/mode is unsafe: {path}")
        if opened.st_dev != named.st_dev or opened.st_ino != named.st_ino:
            raise RuntimeError(f"environment lock pathname changed during open: {path}")
        operation = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise EnvironmentLockBusy(
                f"environment {environment_name} is held by a conflicting "
                f"{mode} consumer/transaction"
            ) from error
        return EnvironmentLockLease(descriptor, path, mode)
    except BaseException:
        os.close(descriptor)
        raise


def environment_lock_markers(lease: EnvironmentLockLease) -> dict[str, str]:
    if lease.descriptor < 0:
        raise RuntimeError("cannot publish a closed environment lock")
    return {
        "PYPTO_ENVIRONMENT_LOCK_FD": str(lease.descriptor),
        "PYPTO_ENVIRONMENT_LOCK_MODE": lease.mode,
        "PYPTO_ENVIRONMENT_LOCK_PATH": str(lease.path),
        "PYPTO_ENVIRONMENT_LOCK_DEV": str(lease.device),
        "PYPTO_ENVIRONMENT_LOCK_INO": str(lease.inode),
        "PYPTO_ENVIRONMENT_LOCK_CONTROLLER_PID": str(os.getpid()),
        "PYPTO_ENVIRONMENT_LOCK_CONTROLLER_START_TICKS": str(
            process_start_ticks(os.getpid())
        ),
    }


def validate_exclusive_environment_command(
    command: list[str],
    environment_prefix: pathlib.Path,
) -> None:
    expected_python = environment_prefix / "bin" / "python"
    command_python = pathlib.Path(os.path.abspath(command[0]))
    if command_python != expected_python:
        raise ValueError(
            "exclusive environment transaction requires selected-prefix Python"
        )
    expected_tool = (ROOT / "tools" / "replace_triton_environment.py").resolve()
    interpreter_tail = command[1:]
    if interpreter_tail[:1] == ["-B"]:
        interpreter_tail = interpreter_tail[1:]
    if (
        not interpreter_tail
        or interpreter_tail[0].startswith("-")
        or not interpreter_tail[0].endswith(".py")
        or pathlib.Path(interpreter_tail[0]).resolve() != expected_tool
    ):
        raise ValueError(
            "exclusive environment transaction is restricted to "
            "a direct replace_triton_environment.py child"
        )
    if any(token.endswith(".py") for token in interpreter_tail[1:]):
        raise ValueError("exclusive replacement command has an extra Python launcher")
    actions = {"--apply", "--recover", "--rollback"} & set(command)
    if len(actions) != 1 or "--plan" in command:
        raise ValueError(
            "exclusive environment transaction requires exactly one mutating "
            "replacement action"
        )


def process_start_ticks(pid: int) -> int:
    return stop_run.process_start_ticks(pid)


def atomic_json(path: pathlib.Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = canonical_json_bytes(value)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_exact_nvidia_smoke_command(command: list[str]) -> None:
    expected = nvidia_smoke_contract.fixed_child_command(ROOT)
    if command != expected:
        raise ValueError(
            "exact PyPTO NVIDIA smoke requires the fixed direct child command: "
            f"expected {expected!r}, got {command!r}"
        )


def _require_exact_regular_file(
    path: pathlib.Path,
    *,
    expected_size: int,
    expected_sha256: str,
    description: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{description} size differs from the fixed smoke contract")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{description} SHA-256 differs from the fixed smoke contract")


def validate_exact_nvidia_smoke_inputs() -> dict[str, object]:
    """Fail before child creation unless every large immutable input is exact."""

    runner = ROOT / nvidia_smoke_contract.RUNNER_RELATIVE_PATH
    _require_exact_regular_file(
        runner,
        expected_size=nvidia_smoke_contract.RUNNER_SIZE,
        expected_sha256=nvidia_smoke_contract.RUNNER_SHA256,
        description="PyPTO NVIDIA smoke runner",
    )
    _require_exact_regular_file(
        ROOT / nvidia_smoke_contract.PYTHON_REAL_RELATIVE_PATH,
        expected_size=nvidia_smoke_contract.PYTHON_SIZE,
        expected_sha256=nvidia_smoke_contract.PYTHON_SHA256,
        description="selected Python interpreter",
    )
    environment_lock = ROOT / "ENVIRONMENT.lock"
    if environment_lock.is_symlink() or not environment_lock.is_file():
        raise ValueError("ENVIRONMENT.lock must be a regular non-symlink file")
    if sha256_file(environment_lock) != (nvidia_smoke_contract.ENVIRONMENT_LOCK_SHA256):
        raise ValueError("ENVIRONMENT.lock differs from the fixed smoke contract")
    _require_exact_regular_file(
        ROOT / nvidia_smoke_contract.PYPTO_DSO_RELATIVE_PATH,
        expected_size=nvidia_smoke_contract.PYPTO_DSO_SIZE,
        expected_sha256=nvidia_smoke_contract.PYPTO_DSO_SHA256,
        description="PyPTO NVIDIA product DSO",
    )
    _require_exact_regular_file(
        ROOT / nvidia_smoke_contract.CUDA_RUNTIME_RELATIVE_PATH,
        expected_size=nvidia_smoke_contract.CUDA_RUNTIME_SIZE,
        expected_sha256=nvidia_smoke_contract.CUDA_RUNTIME_SHA256,
        description="selected Torch CUDA Runtime provider",
    )
    pypto = ROOT / "projects" / "pypto"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pypto,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=pypto,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=pypto,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if (
        head != nvidia_smoke_contract.PYPTO_HEAD
        or tree != nvidia_smoke_contract.PYPTO_TREE
        or dirty
    ):
        raise ValueError("PyPTO source identity differs from the fixed smoke contract")
    ignored_python = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "python/pypto",
        ],
        cwd=pypto,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if ignored_python:
        raise ValueError("PyPTO Python source contains ignored shadow files")
    nested = (
        (
            pypto / "3rdparty" / "nvidia" / "tensor-ir",
            nvidia_smoke_contract.TENSOR_IR_HEAD,
            "TensorIR",
        ),
        (
            pypto / "3rdparty" / "nvidia" / "cuda-tile",
            nvidia_smoke_contract.CUDA_TILE_HEAD,
            "CUDA Tile",
        ),
        (
            pypto / "3rdparty" / "nvidia" / "llvm-project",
            nvidia_smoke_contract.LLVM_HEAD,
            "LLVM",
        ),
    )
    for repository, expected_head, description in nested:
        actual_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        actual_dirty = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if actual_head != expected_head or actual_dirty:
            raise ValueError(
                f"{description} source identity differs from the fixed smoke contract"
            )
    return nvidia_smoke_control.validate_control_manifest(ROOT)


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
    *,
    wait_seconds: int = 30,
) -> int:
    """Signal only after stop_run revalidates the live process identity."""

    pgid = int(metadata["pgid"])
    if process.poll() is not None and not stop_run.process_group_members(pgid):
        return process.wait()
    try:
        stop_run.signal_verified(metadata, signal.SIGTERM)
        # SIGTERM remains pending for a stopped group. SIGCONT is harmless for
        # a running group and closes the STOP-before-metadata race.
        stop_run.signal_verified_followup(metadata, signal.SIGCONT, pgid)
    except ProcessLookupError:
        # The child won the poll/verify race and needs only to be reaped.
        return process.wait() if process.poll() is None else process.returncode
    except stop_run.GroupRevalidationError as error:
        metadata["status"] = "group-ownership-ambiguous"
        metadata["termination_error"] = f"{type(error).__name__}: {error}"
        if error.members is not None:
            metadata["termination_surviving_group_pids"] = error.members
        return 75
    deadline = time.monotonic() + wait_seconds
    try:
        if process.poll() is None:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass
    while stop_run.process_group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    survivors = stop_run.process_group_members(pgid)
    if survivors:
        try:
            stop_run.signal_verified_followup(metadata, signal.SIGSTOP, pgid)
        except stop_run.GroupRevalidationError as error:
            metadata["status"] = "group-ownership-ambiguous"
            metadata["termination_error"] = f"{type(error).__name__}: {error}"
            if error.members is not None:
                metadata["termination_surviving_group_pids"] = error.members
            return 75
        metadata["status"] = "paused"
        metadata["termination_surviving_group_pids"] = survivors
        # No kill escalation: only the verified group was touched.
        return 75
    return process.returncode if process.returncode is not None else process.wait()


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
    protected_cpu_only_coexistence_requested: bool = False,
    protected_activity_waiver_applied: bool = False,
    protected_zero_nvidia_gpu_smoke_requested: bool = False,
    protected_gpu_smoke_waiver_applied: bool = False,
    gpu_smoke_start_barrier_path: str | None = None,
    gpu_smoke_gate_path: str | None = None,
    preflight_report_sha256: str | None = None,
    preflight_report_path: str | None = None,
    preflight_report: dict[str, object] | None = None,
    run_timeout_seconds: int = 0,
    minimum_free_disk_bytes: int = 0,
    environment_access_lock: dict[str, object] | None = None,
) -> dict[str, object]:
    """Capture the ownership fields required by stop_run.verify."""

    return {
        "schema": 3 if mode == "gpu-smoke" else 2,
        "run_id": run_id,
        "workspace": str(ROOT),
        "environment": str(environment_prefix),
        "environment_access_lock": environment_access_lock,
        "framework_profile": framework_profile,
        "framework_launch": framework_launch,
        "mode": mode,
        "coexistence": {
            "policy_version": preflight_tool.COEXISTENCE_POLICY_VERSION,
            "requested": protected_cpu_only_coexistence_requested,
            "waiver_applied": protected_activity_waiver_applied,
            "memory_floor_kib": (
                None
                if preflight_report is None
                else preflight_report.get("memory_floor_kib")
            ),
            "protected_heavy_processes": (
                []
                if preflight_report is None
                else preflight_report.get("protected_heavy_processes", [])
            ),
            "protected_nvidia_compute_pids": (
                []
                if preflight_report is None
                else preflight_report.get("protected_nvidia_compute_pids", [])
            ),
        },
        "gpu_smoke": {
            "policy_version": preflight_tool.GPU_SMOKE_POLICY_VERSION,
            "requested": protected_zero_nvidia_gpu_smoke_requested,
            "waiver_applied": protected_gpu_smoke_waiver_applied,
            "authorization": (
                nvidia_smoke_contract.GPU_SMOKE_AUTHORIZATION
                if protected_zero_nvidia_gpu_smoke_requested
                else None
            ),
            "start_barrier_path": gpu_smoke_start_barrier_path,
            "gate_path": gpu_smoke_gate_path,
            "memory_floor_kib": (
                None
                if preflight_report is None
                else preflight_report.get("memory_floor_kib")
            ),
            "gpu_free_memory_floor_mib": (
                None
                if preflight_report is None
                else preflight_report.get("gpu_smoke_free_memory_floor_mib")
            ),
            "protected_heavy_processes": (
                []
                if preflight_report is None
                else preflight_report.get("protected_heavy_processes", [])
            ),
            "protected_nvidia_compute_pids": (
                []
                if preflight_report is None
                else preflight_report.get("protected_nvidia_compute_pids", [])
            ),
            "protected_nvidia_runtime_mapping_pids": (
                []
                if preflight_report is None
                else preflight_report.get("protected_nvidia_runtime_mapping_pids", [])
            ),
            "unreadable_protected_maps": (
                []
                if preflight_report is None
                else preflight_report.get("unreadable_protected_maps", [])
            ),
        },
        "preflight": {
            "path": preflight_report_path,
            "sha256": preflight_report_sha256,
        },
        "resource_policy": {
            "timeout_seconds": run_timeout_seconds,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
            "owned_run_pause_memory_kib": (
                COEXISTENCE_ABORT_MEMORY_KIB
                if (
                    protected_cpu_only_coexistence_requested
                    or protected_zero_nvidia_gpu_smoke_requested
                    or mode == "gpu-smoke"
                )
                else None
            ),
        },
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
    protected_cpu_only_coexistence_requested: bool = False,
    protected_zero_nvidia_gpu_smoke_requested: bool = False,
    exact_nvidia_smoke: bool = False,
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
    if framework_profile == "pypto" and not exact_nvidia_smoke:
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
    if exact_nvidia_smoke:
        python_paths: tuple[str, ...] = ()
        sglang_plugins = "__pypto_exact_nvidia_smoke_no_plugins__"
    elif framework_profile == "pypto":
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
    if protected_cpu_only_coexistence_requested:
        environment["PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if protected_zero_nvidia_gpu_smoke_requested:
        environment["PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED"] = "1"
        environment["PYPTO_GPU_SMOKE_AUTHORIZATION"] = (
            nvidia_smoke_contract.GPU_SMOKE_AUTHORIZATION
        )
    if exact_nvidia_smoke:
        environment["PYPTO_ALLOW_FALLBACK"] = "0"
        environment["PYPTO_STRICT_COVERAGE"] = "1"
    return environment


def wait_with_coexistence_watchdog(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
    *,
    timeout_seconds: int = 0,
    minimum_free_disk_bytes: int = 0,
    metadata_path: pathlib.Path | None = None,
) -> tuple[int, bool]:
    deadline = None if timeout_seconds == 0 else time.monotonic() + timeout_seconds
    pauses = metadata.setdefault("coexistence_pauses", [])
    if not isinstance(pauses, list):
        raise RuntimeError("run metadata coexistence_pauses must be a list")
    paused = False
    pause_count = 0
    while True:
        try:
            return process.wait(timeout=COEXISTENCE_POLL_SECONDS), False
        except subprocess.TimeoutExpired:
            available_kib = preflight_tool.mem_available_kib()
            disk_free_bytes = shutil.disk_usage(ROOT).free
            abort_record: dict[str, object] | None = None
            if deadline is not None and time.monotonic() >= deadline:
                abort_record = {
                    "reason": "owned-run-timeout",
                    "timeout_seconds": timeout_seconds,
                }
            elif (
                minimum_free_disk_bytes > 0
                and disk_free_bytes < minimum_free_disk_bytes
            ):
                abort_record = {
                    "reason": "workspace-disk-floor",
                    "free_bytes": disk_free_bytes,
                    "floor_bytes": minimum_free_disk_bytes,
                }
            elif available_kib < COEXISTENCE_ABORT_MEMORY_KIB:
                abort_record = {
                    "reason": "host-memory-floor",
                    "mem_available_kib": available_kib,
                    "floor_kib": COEXISTENCE_ABORT_MEMORY_KIB,
                }
            else:
                try:
                    compute_pids = preflight_tool.nvidia_compute_pids()
                    all_processes, protected, _workspace = (
                        preflight_tool.process_table()
                    )
                    protected_compute_pids = sorted(
                        compute_pids & {candidate.pid for candidate in protected}
                    )
                    owned_compute_pids, _external_compute_pids = partition_compute_pids(
                        compute_pids, metadata, all_processes
                    )
                except Exception as error:
                    abort_record = {
                        "reason": "protected-nvidia-runtime-audit-failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                else:
                    if owned_compute_pids:
                        abort_record = {
                            "reason": "owned-nvidia-compute-became-active",
                            "pids": owned_compute_pids,
                        }
                    elif protected_compute_pids:
                        abort_record = {
                            "reason": "protected-nvidia-compute-became-active",
                            "pids": protected_compute_pids,
                        }
            if paused:
                if abort_record is not None and abort_record["reason"] in {
                    "owned-run-timeout",
                    "owned-nvidia-compute-became-active",
                }:
                    metadata["coexistence_abort"] = abort_record
                    return_code = terminate_owned_process(
                        process, metadata, wait_seconds=5
                    )
                    survivors = stop_run.process_group_members(int(metadata["pgid"]))
                    if survivors:
                        if metadata.get("status") != "paused":
                            stop_run.signal_verified(metadata, signal.SIGSTOP)
                        metadata["status"] = "paused"
                        abort_record["surviving_group_pids"] = survivors
                        abort_record["owned_run_action"] = (
                            "verified-sigterm-sigcont-timeout-then-sigstop"
                        )
                    else:
                        abort_record["owned_run_action"] = "verified-sigterm-sigcont"
                    if metadata_path is not None:
                        atomic_json(metadata_path, metadata)
                    return return_code, True
                recovery_safe = (
                    abort_record is None
                    and available_kib >= COEXISTENCE_RESUME_MEMORY_KIB
                    and (
                        minimum_free_disk_bytes == 0
                        or disk_free_bytes >= minimum_free_disk_bytes
                    )
                )
                if not recovery_safe:
                    continue
                try:
                    stop_run.signal_verified(metadata, signal.SIGCONT)
                except ProcessLookupError:
                    return process.wait(), False
                paused = False
                metadata["status"] = "running"
                pauses[-1]["resumed_at"] = datetime.datetime.now(datetime.UTC).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                if metadata_path is not None:
                    atomic_json(metadata_path, metadata)
                print(
                    "PyPTO coexistence watchdog resumed its verified owned run",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if abort_record is None:
                continue
            if abort_record["reason"] == "owned-nvidia-compute-became-active":
                metadata["coexistence_abort"] = abort_record
                return_code = terminate_owned_process(process, metadata, wait_seconds=5)
                survivors = stop_run.process_group_members(int(metadata["pgid"]))
                if survivors:
                    if metadata.get("status") != "paused":
                        stop_run.signal_verified(metadata, signal.SIGSTOP)
                    metadata["status"] = "paused"
                    abort_record["surviving_group_pids"] = survivors
                    abort_record["owned_run_action"] = (
                        "verified-sigterm-sigcont-timeout-then-sigstop"
                    )
                else:
                    abort_record["owned_run_action"] = "verified-sigterm-sigcont"
                if metadata_path is not None:
                    atomic_json(metadata_path, metadata)
                return return_code, True
            print(
                "PyPTO coexistence watchdog is pausing only its verified owned "
                f"run: {abort_record['reason']}",
                file=sys.stderr,
                flush=True,
            )
            try:
                stop_run.signal_verified(metadata, signal.SIGSTOP)
            except ProcessLookupError:
                return process.wait(), False
            pause_count += 1
            abort_record.update(
                {
                    "owned_run_action": "verified-sigstop",
                    "pause_index": pause_count,
                    "paused_at": datetime.datetime.now(datetime.UTC).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                }
            )
            pauses.append(abort_record)
            metadata["status"] = "paused"
            if metadata_path is not None:
                atomic_json(metadata_path, metadata)
            paused = True


def partition_compute_pids(
    compute_pids: set[int],
    metadata: dict[str, object],
    all_processes: list[preflight_tool.ProcessInfo],
) -> tuple[list[int], list[int]]:
    root_process = next(
        (
            candidate
            for candidate in all_processes
            if candidate.pid == int(metadata["pid"])
        ),
        None,
    )
    if root_process is not None and root_process.start_ticks != int(
        metadata["start_ticks"]
    ):
        root_process = None
    descendants = (
        []
        if root_process is None
        else preflight_tool.process_descendant_closure(all_processes, [root_process])
    )
    expected_pgid = int(metadata["pgid"])
    owned_lineage_pids: set[int] = set()
    for candidate in descendants:
        try:
            candidate_pgid = os.getpgid(candidate.pid)
        except (OSError, ProcessLookupError):
            continue
        if candidate_pgid == expected_pgid:
            owned_lineage_pids.add(candidate.pid)
    owned: set[int] = set()
    for compute_pid in compute_pids:
        if compute_pid in owned_lineage_pids:
            owned.add(compute_pid)
    return sorted(owned), sorted(compute_pids - owned)


def wait_with_gpu_benchmark_watchdog(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
    *,
    timeout_seconds: int = 0,
    minimum_free_disk_bytes: int = 0,
    metadata_path: pathlib.Path | None = None,
) -> tuple[int, bool]:
    deadline = None if timeout_seconds == 0 else time.monotonic() + timeout_seconds
    while True:
        try:
            return process.wait(timeout=GPU_BENCHMARK_POLL_SECONDS), False
        except subprocess.TimeoutExpired:
            violation: dict[str, object] | None = None
            if deadline is not None and time.monotonic() >= deadline:
                violation = {
                    "reason": "gpu-benchmark-timeout",
                    "timeout_seconds": timeout_seconds,
                }
            elif preflight_tool.mem_available_kib() < (
                preflight_tool.STANDARD_HEAVY_MEMORY_FLOOR_KIB
            ):
                violation = {"reason": "gpu-benchmark-memory-floor"}
            elif (
                minimum_free_disk_bytes > 0
                and shutil.disk_usage(ROOT).free < minimum_free_disk_bytes
            ):
                violation = {"reason": "gpu-benchmark-disk-floor"}
            else:
                try:
                    compute_pids = preflight_tool.nvidia_compute_pids()
                    all_processes, protected, _workspace = (
                        preflight_tool.process_table()
                    )
                    _owned, external = partition_compute_pids(
                        compute_pids, metadata, all_processes
                    )
                except Exception as error:
                    violation = {
                        "reason": "gpu-benchmark-nvidia-audit-failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                else:
                    protected_heavy = [
                        candidate
                        for candidate in protected
                        if preflight_tool.is_heavy_command(candidate.command)
                    ]
                    if protected_heavy:
                        violation = {
                            "reason": "protected-heavy-became-active",
                            "pids": [candidate.pid for candidate in protected_heavy],
                        }
                    elif external:
                        violation = {
                            "reason": "external-nvidia-compute-became-active",
                            "pids": external,
                        }
            if violation is None:
                continue
            metadata["gpu_benchmark_abort"] = violation
            return_code = terminate_owned_process(process, metadata, wait_seconds=5)
            survivors = stop_run.process_group_members(int(metadata["pgid"]))
            if survivors:
                if metadata.get("status") != "paused":
                    stop_run.signal_verified(metadata, signal.SIGSTOP)
                violation["surviving_group_pids"] = survivors
                violation["owned_run_action"] = (
                    "verified-sigterm-sigcont-timeout-then-sigstop"
                )
                metadata["status"] = "paused"
            else:
                violation["owned_run_action"] = "verified-sigterm-sigcont"
            if metadata_path is not None:
                atomic_json(metadata_path, metadata)
            return return_code, True


def audit_gpu_smoke_runtime_state(
    metadata: dict[str, object],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return a complete live isolation snapshot or one fail-closed violation."""

    try:
        gpu = preflight_tool.nvidia_identity()
        if gpu.get("compute_capability") != "12.0":
            raise RuntimeError(f"unexpected NVIDIA identity: {gpu}")
        free_memory_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
        compute_pids = preflight_tool.nvidia_compute_pids()
        all_processes, protected, _workspace = preflight_tool.process_table()
        owned, external = partition_compute_pids(compute_pids, metadata, all_processes)
        protected_runtime, unreadable_maps = (
            preflight_tool.protected_nvidia_runtime_mappings(protected)
        )
    except Exception as error:
        return None, {
            "reason": "gpu-smoke-nvidia-audit-failed",
            "error": f"{type(error).__name__}: {error}",
        }
    protected_pid_set = {candidate.pid for candidate in protected}
    protected_compute = sorted(compute_pids & protected_pid_set)
    protected_heavy = [
        candidate
        for candidate in protected
        if preflight_tool.is_heavy_command(candidate.command)
    ]
    gpu_smoke = metadata.get("gpu_smoke")
    authorization_requested = bool(
        isinstance(gpu_smoke, dict) and gpu_smoke.get("requested") is True
    )
    violation: dict[str, object] | None = None
    if unreadable_maps:
        violation = {
            "reason": "gpu-smoke-protected-maps-unreadable",
            "pids": unreadable_maps,
        }
    elif protected_runtime:
        violation = {
            "reason": "protected-nvidia-runtime-became-active",
            "pids": protected_runtime,
        }
    elif protected_compute:
        violation = {
            "reason": "protected-nvidia-compute-became-active",
            "pids": protected_compute,
        }
    elif external:
        violation = {
            "reason": "external-nvidia-compute-became-active",
            "pids": external,
        }
    elif protected_heavy and not authorization_requested:
        violation = {
            "reason": "unauthorized-protected-heavy-became-active",
            "pids": [candidate.pid for candidate in protected_heavy],
        }
    elif free_memory_mib < preflight_tool.GPU_SMOKE_FREE_MEMORY_FLOOR_MIB:
        violation = {
            "reason": "gpu-smoke-device-memory-floor",
            "free_memory_mib": free_memory_mib,
        }
    snapshot = {
        "owned_nvidia_compute_pids": owned,
        "external_nvidia_compute_pids": external,
        "protected_nvidia_compute_pids": protected_compute,
        "protected_nvidia_runtime_mapping_pids": protected_runtime,
        "unreadable_protected_maps": unreadable_maps,
        "protected_heavy_pids": [candidate.pid for candidate in protected_heavy],
        "protected_cpu_lane_authorized": authorization_requested,
        "free_memory_mib": free_memory_mib,
        "gpu": gpu,
    }
    return snapshot, violation


def wait_with_gpu_smoke_watchdog(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    metadata: dict[str, object],
    *,
    timeout_seconds: int = 0,
    minimum_free_disk_bytes: int = 0,
    metadata_path: pathlib.Path | None = None,
) -> tuple[int, bool]:
    """Allow protected CPU work while terminating only this run on GPU drift."""

    deadline = None if timeout_seconds == 0 else time.monotonic() + timeout_seconds
    while True:
        try:
            return process.wait(timeout=GPU_SMOKE_POLL_SECONDS), False
        except subprocess.TimeoutExpired:
            violation: dict[str, object] | None = None
            if deadline is not None and time.monotonic() >= deadline:
                violation = {
                    "reason": "gpu-smoke-timeout",
                    "timeout_seconds": timeout_seconds,
                }
            elif preflight_tool.mem_available_kib() < COEXISTENCE_ABORT_MEMORY_KIB:
                violation = {"reason": "gpu-smoke-host-memory-floor"}
            elif (
                minimum_free_disk_bytes > 0
                and shutil.disk_usage(ROOT).free < minimum_free_disk_bytes
            ):
                violation = {"reason": "gpu-smoke-disk-floor"}
            else:
                snapshot, violation = audit_gpu_smoke_runtime_state(metadata)
                if snapshot is not None:
                    metadata["gpu_smoke_last_audit"] = snapshot
            if violation is None:
                continue
            metadata["gpu_smoke_abort"] = violation
            return_code = terminate_owned_process(process, metadata, wait_seconds=5)
            survivors = stop_run.process_group_members(int(metadata["pgid"]))
            if survivors:
                if metadata.get("status") != "paused":
                    stop_run.signal_verified(metadata, signal.SIGSTOP)
                violation["surviving_group_pids"] = survivors
                violation["owned_run_action"] = (
                    "verified-sigterm-sigcont-timeout-then-sigstop"
                )
                metadata["status"] = "paused"
            else:
                violation["owned_run_action"] = "verified-sigterm-sigcont"
            if metadata_path is not None:
                atomic_json(metadata_path, metadata)
            return return_code, True


@close_registered_environment_lock
def main() -> int:
    global _ACTIVE_ENVIRONMENT_LOCK
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("light", "heavy", "gpu-smoke", "gpu-benchmark"),
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
    parser.add_argument(
        "--allow-protected-cpu-only-coexistence",
        action="store_true",
        help=(
            "use the explicit protected CPU-only coexistence policy and an "
            "owned-run memory watchdog"
        ),
    )
    parser.add_argument(
        "--allow-protected-zero-nvidia-gpu-smoke",
        action="store_true",
        help=(
            "use the correctness-only GPU-smoke policy beside a protected "
            "CPU lane with zero NVIDIA mappings and compute PIDs"
        ),
    )
    parser.add_argument(
        "--exact-pypto-nvidia-smoke",
        action="store_true",
        help="require the fixed PyPTO NvidiaExecutable SM120 direct child",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--minimum-free-disk-gib", type=int, default=0)
    parser.add_argument("--run-id-file", type=pathlib.Path)
    parser.add_argument(
        "--environment-lock-mode",
        choices=("shared", "exclusive"),
        default="shared",
        help=(
            "hold a shared environment-consumer lock, or the exclusive lock "
            "reserved for a mutating Triton replacement transaction"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    fixed_smoke_runner = str(ROOT / nvidia_smoke_contract.RUNNER_RELATIVE_PATH)
    if not args.exact_pypto_nvidia_smoke and any(
        fixed_smoke_runner in token for token in command
    ):
        parser.error(
            "the PyPTO NVIDIA correctness runner is forbidden outside its "
            "exact gpu-smoke mode"
        )
    if args.require_framework_plugins and args.framework_profile != "pypto":
        parser.error("--require-framework-plugins requires --framework-profile pypto")
    if args.allow_protected_cpu_only_coexistence and args.mode != "heavy":
        parser.error("--allow-protected-cpu-only-coexistence requires --mode heavy")
    if args.allow_protected_zero_nvidia_gpu_smoke and args.mode != "gpu-smoke":
        parser.error(
            "--allow-protected-zero-nvidia-gpu-smoke requires --mode gpu-smoke"
        )
    if args.mode == "gpu-smoke" and not args.exact_pypto_nvidia_smoke:
        parser.error("--mode gpu-smoke requires --exact-pypto-nvidia-smoke")
    if args.exact_pypto_nvidia_smoke and args.mode != "gpu-smoke":
        parser.error("--exact-pypto-nvidia-smoke requires --mode gpu-smoke")
    if (
        args.allow_protected_cpu_only_coexistence
        and args.allow_protected_zero_nvidia_gpu_smoke
    ):
        parser.error("protected coexistence policies are mutually exclusive")
    if args.allow_protected_cpu_only_coexistence and args.framework_launch:
        parser.error("protected CPU-only coexistence cannot launch a GPU framework")
    if args.mode == "gpu-smoke" and (
        args.framework_launch or args.require_framework_plugins
    ):
        parser.error("exact GPU smoke cannot bootstrap framework plugins")
    if args.mode == "gpu-smoke" and not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("exact GPU smoke controller requires Python -E -B -S")
    if args.exact_pypto_nvidia_smoke and (
        args.environment != "pypto-nvidia"
        or args.framework_profile != "pypto"
        or args.environment_lock_mode != "shared"
        or args.timeout_seconds != nvidia_smoke_contract.GPU_SMOKE_TIMEOUT_SECONDS
        or args.minimum_free_disk_gib
        != nvidia_smoke_contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB
    ):
        parser.error(
            "exact GPU smoke requires pypto-nvidia/pypto, a shared environment "
            f"lock, timeout {nvidia_smoke_contract.GPU_SMOKE_TIMEOUT_SECONDS}, "
            "and minimum free disk "
            f"{nvidia_smoke_contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB} GiB"
        )
    if args.environment_lock_mode == "exclusive":
        if args.environment != "pypto-nvidia" or args.mode != "heavy":
            parser.error(
                "exclusive environment transactions require "
                "--environment pypto-nvidia --mode heavy"
            )
        if args.framework_launch or args.require_framework_plugins:
            parser.error(
                "exclusive environment transactions cannot launch frameworks/plugins"
            )
    if args.timeout_seconds < 0 or args.minimum_free_disk_gib < 0:
        parser.error("timeout and disk floor must be non-negative")
    if (args.timeout_seconds or args.minimum_free_disk_gib) and not (
        args.allow_protected_cpu_only_coexistence
        or args.allow_protected_zero_nvidia_gpu_smoke
        or args.mode == "gpu-smoke"
        or args.mode == "gpu-benchmark"
    ):
        parser.error(
            "timeout/disk watchdog controls require a watched heavy, GPU-smoke, "
            "or GPU-benchmark mode"
        )
    expected_environment = PROFILE_ENVIRONMENTS[args.framework_profile]
    if args.environment != expected_environment:
        parser.error(
            f"--framework-profile {args.framework_profile} requires "
            f"--environment {expected_environment}"
        )

    environment_prefix = ENVIRONMENTS[args.environment].resolve()
    environment_lock = ENVIRONMENT_LOCKS[args.environment].resolve()
    if args.environment_lock_mode == "exclusive":
        try:
            validate_exclusive_environment_command(command, environment_prefix)
        except ValueError as error:
            parser.error(str(error))
    if args.framework_launch:
        expected_python = (environment_prefix / "bin" / "python").resolve()
        command_python = pathlib.Path(command[0]).resolve()
        if command_python != expected_python:
            parser.error(
                "--framework-launch requires the selected environment Python: "
                f"expected {expected_python}, got {command_python}"
            )
    if args.exact_pypto_nvidia_smoke:
        if pathlib.Path(sys.executable).resolve() != (
            environment_prefix / "bin/python3.14"
        ).resolve(strict=True):
            parser.error("exact GPU smoke controller uses the wrong Python")
        try:
            validate_exact_nvidia_smoke_command(command)
        except ValueError as error:
            parser.error(str(error))

    registration_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    registration_mask = signal.pthread_sigmask(signal.SIG_BLOCK, registration_signals)
    try:
        try:
            environment_access = acquire_environment_lock(
                args.environment,
                args.environment_lock_mode,
            )
        except EnvironmentLockBusy as error:
            print(str(error), file=sys.stderr)
            return 75
        _ACTIVE_ENVIRONMENT_LOCK = environment_access
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, registration_mask)

    preflight_command = [sys.executable]
    if args.mode == "gpu-smoke":
        preflight_command.extend(["-E", "-B", "-S"])
    preflight_command.extend(
        [
            str(ROOT / "tools" / "preflight.py"),
            "--mode",
            args.mode,
            "--json",
        ]
    )
    preflight_environment = None
    if args.mode == "gpu-smoke":
        preflight_environment = {
            "PATH": (
                f"{environment_prefix}/bin:/usr/local/cuda-13.3/bin:"
                f"{SYSTEM_EXECUTABLE_PATH}"
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
    if args.allow_protected_cpu_only_coexistence:
        preflight_command.append("--allow-protected-cpu-only-coexistence")
    if args.allow_protected_zero_nvidia_gpu_smoke:
        preflight_command.append("--allow-protected-zero-nvidia-gpu-smoke")
    preflight_result = subprocess.run(
        preflight_command,
        cwd=ROOT,
        env=preflight_environment,
        check=False,
    )
    if preflight_result.returncode != 0:
        return preflight_result.returncode
    if args.exact_pypto_nvidia_smoke:
        try:
            validate_exact_nvidia_smoke_inputs()
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            print(str(error), file=sys.stderr)
            return 75

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"pypto-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    if args.run_id_file is not None:
        run_id_file = pathlib.Path(os.path.abspath(os.fspath(args.run_id_file)))
        parent = run_id_file.parent.resolve(strict=True)
        if ROOT not in parent.parents and parent != ROOT:
            parser.error("--run-id-file must be below the workspace")
        if run_id_file.exists() or run_id_file.is_symlink():
            parser.error("--run-id-file must not already exist")
        atomic_json(run_id_file, {"run_id": run_id})
    environment = isolated_environment(
        run_id,
        run_dir,
        environment_prefix=environment_prefix,
        framework_profile=args.framework_profile,
        protected_cpu_only_coexistence_requested=(
            args.allow_protected_cpu_only_coexistence
        ),
        protected_zero_nvidia_gpu_smoke_requested=(
            args.allow_protected_zero_nvidia_gpu_smoke
        ),
        exact_nvidia_smoke=args.exact_pypto_nvidia_smoke,
    )
    environment.update(environment_lock_markers(environment_access))
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
    command_requires_plugins = args.require_framework_plugins or (
        args.framework_profile == "pypto" and args.framework_launch
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

    action_preflight = subprocess.run(
        preflight_command,
        cwd=ROOT,
        env=preflight_environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if action_preflight.stderr:
        print(action_preflight.stderr, file=sys.stderr, end="")
    try:
        action_preflight_report = json.loads(action_preflight.stdout)
    except json.JSONDecodeError:
        print(action_preflight.stdout, end="")
        raise RuntimeError("action-boundary preflight did not emit valid JSON")
    if not isinstance(action_preflight_report, dict):
        raise RuntimeError("action-boundary preflight report must be an object")
    expected_requested = args.allow_protected_cpu_only_coexistence
    expected_gpu_smoke_requested = args.allow_protected_zero_nvidia_gpu_smoke
    if action_preflight_report.get("mode") != args.mode:
        raise RuntimeError("action-boundary preflight mode mismatch")
    if (
        action_preflight_report.get("protected_cpu_only_coexistence_requested")
        is not expected_requested
    ):
        raise RuntimeError("action-boundary coexistence request mismatch")
    if (
        action_preflight_report.get("protected_zero_nvidia_gpu_smoke_requested")
        is not expected_gpu_smoke_requested
    ):
        raise RuntimeError("action-boundary GPU-smoke coexistence request mismatch")
    if args.mode == "gpu-smoke":
        if action_preflight_report.get("nvidia_compute_audit_ok") is not True:
            raise RuntimeError("action-boundary NVIDIA compute audit is indeterminate")
        protected_at_boundary = action_preflight_report.get("protected_processes")
        if (
            expected_gpu_smoke_requested
            and isinstance(protected_at_boundary, list)
            and protected_at_boundary
            and action_preflight_report.get("protected_gpu_smoke_waiver_applied")
            is not True
        ):
            raise RuntimeError("action-boundary GPU-smoke waiver was not applied")
    preflight_report_path = run_dir / "preflight.json"
    atomic_json(preflight_report_path, action_preflight_report)
    preflight_report_sha256 = sha256_file(preflight_report_path)
    if action_preflight.returncode != 0:
        print(action_preflight.stdout, end="")
        return action_preflight.returncode
    if action_preflight_report.get("ok") is not True:
        raise RuntimeError("action-boundary preflight returned zero without ok=true")
    minimum_free_disk_bytes = args.minimum_free_disk_gib << 30
    if (
        minimum_free_disk_bytes > 0
        and shutil.disk_usage(ROOT).free < minimum_free_disk_bytes
    ):
        print(
            f"workspace free disk is below {args.minimum_free_disk_gib} GiB floor",
            file=sys.stderr,
        )
        return 75
    waiver_applied = bool(
        action_preflight_report.get("protected_activity_waiver_applied")
    )
    gpu_smoke_waiver_applied = bool(
        action_preflight_report.get("protected_gpu_smoke_waiver_applied")
    )
    gpu_smoke_start_barrier = (
        run_dir / "gpu-smoke-start-barrier.json" if args.mode == "gpu-smoke" else None
    )
    gpu_smoke_gate = (
        run_dir / "gpu-smoke-gate.json" if args.mode == "gpu-smoke" else None
    )
    environment.update(
        {
            "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED": (
                "1" if expected_requested else "0"
            ),
            "PYPTO_PROTECTED_ACTIVITY_WAIVER_APPLIED": ("1" if waiver_applied else "0"),
            "PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED": (
                "1" if expected_gpu_smoke_requested else "0"
            ),
            "PYPTO_PROTECTED_GPU_SMOKE_WAIVER_APPLIED": (
                "1" if gpu_smoke_waiver_applied else "0"
            ),
            "PYPTO_GPU_SMOKE_START_BARRIER": (
                "" if gpu_smoke_start_barrier is None else str(gpu_smoke_start_barrier)
            ),
            "PYPTO_PREFLIGHT_REPORT_PATH": str(preflight_report_path),
            "PYPTO_PREFLIGHT_REPORT_SHA256": preflight_report_sha256,
            "PYPTO_RUN_MODE": args.mode,
        }
    )
    metadata_context = {
        "protected_cpu_only_coexistence_requested": expected_requested,
        "protected_activity_waiver_applied": waiver_applied,
        "protected_zero_nvidia_gpu_smoke_requested": (expected_gpu_smoke_requested),
        "protected_gpu_smoke_waiver_applied": gpu_smoke_waiver_applied,
        "gpu_smoke_start_barrier_path": (
            None if gpu_smoke_start_barrier is None else str(gpu_smoke_start_barrier)
        ),
        "gpu_smoke_gate_path": (
            None if gpu_smoke_gate is None else str(gpu_smoke_gate)
        ),
        "preflight_report_sha256": preflight_report_sha256,
        "preflight_report_path": str(preflight_report_path),
        "preflight_report": action_preflight_report,
        "run_timeout_seconds": args.timeout_seconds,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "environment_access_lock": {
            "path": str(environment_access.path),
            "mode": environment_access.mode,
            "device": environment_access.device,
            "inode": environment_access.inode,
        },
    }

    metadata_path = run_dir / "process.json"
    process: subprocess.Popen[bytes] | None = None
    metadata: dict[str, object] | None = None
    parent_return_code: int | None = None
    child_return_code: int | None = None
    gpu_smoke_admission_failed = False
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
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
                pass_fds=(environment_access.descriptor,),
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
                **metadata_context,
            )
            atomic_json(metadata_path, metadata)
            if gpu_smoke_start_barrier is not None:
                if (
                    gpu_smoke_start_barrier.exists()
                    or gpu_smoke_start_barrier.is_symlink()
                    or gpu_smoke_gate is None
                    or gpu_smoke_gate.exists()
                    or gpu_smoke_gate.is_symlink()
                ):
                    raise RuntimeError("GPU-smoke gate or start barrier already exists")
                pre_release_snapshot: dict[str, object] | None = None
                pre_release_violation: dict[str, object] | None = None
                static_identity: dict[str, object] | None = None
                control_identity: dict[str, object] | None = None
                try:
                    control_identity = validate_exact_nvidia_smoke_inputs()
                    static_identity = preflight_tool.static_torch_identity()
                    if static_identity.get("static_identity_error"):
                        raise RuntimeError(
                            str(static_identity["static_identity_error"])
                        )
                    if process.poll() is not None:
                        raise RuntimeError("GPU-smoke child exited before gate release")
                    stop_run.verify(metadata)
                    pre_release_snapshot, pre_release_violation = (
                        audit_gpu_smoke_runtime_state(metadata)
                    )
                    if (
                        pre_release_violation is None
                        and pre_release_snapshot is not None
                        and pre_release_snapshot.get("owned_nvidia_compute_pids")
                    ):
                        pre_release_violation = {
                            "reason": "owned-nvidia-compute-before-gate-release",
                            "pids": pre_release_snapshot["owned_nvidia_compute_pids"],
                        }
                except Exception as error:
                    pre_release_violation = {
                        "reason": "gpu-smoke-pre-release-audit-failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                if pre_release_snapshot is not None:
                    metadata["gpu_smoke_pre_release_audit"] = pre_release_snapshot
                if pre_release_violation is not None:
                    metadata["gpu_smoke_abort"] = pre_release_violation
                    atomic_json(metadata_path, metadata)
                    child_return_code = terminate_owned_process(
                        process, metadata, wait_seconds=5
                    )
                    parent_return_code = 75
                    gpu_smoke_admission_failed = True
                else:
                    assert pre_release_snapshot is not None
                    assert static_identity is not None
                    assert control_identity is not None
                    gate_document = {
                        "schema": nvidia_smoke_contract.GPU_SMOKE_POLICY_VERSION,
                        "run_id": run_id,
                        "pid": process.pid,
                        "pgid": metadata["pgid"],
                        "start_ticks": metadata["start_ticks"],
                        "command": command,
                        "preflight": metadata["preflight"],
                        "static_identity": static_identity,
                        "control_manifest": control_identity,
                        "runtime_isolation": pre_release_snapshot,
                    }
                    atomic_json(gpu_smoke_gate, gate_document)
                    gate_sha256 = sha256_file(gpu_smoke_gate)
                    barrier_document = {
                        "schema": nvidia_smoke_contract.GPU_SMOKE_POLICY_VERSION,
                        "run_id": run_id,
                        "pid": process.pid,
                        "pgid": metadata["pgid"],
                        "start_ticks": metadata["start_ticks"],
                        "gate_path": str(gpu_smoke_gate),
                        "gate_sha256": gate_sha256,
                    }
                    barrier_sha256 = hashlib.sha256(
                        canonical_json_bytes(barrier_document)
                    ).hexdigest()
                    gpu_smoke_metadata = metadata["gpu_smoke"]
                    assert isinstance(gpu_smoke_metadata, dict)
                    gpu_smoke_metadata.update(
                        {
                            "gate_sha256": gate_sha256,
                            "start_barrier_sha256": barrier_sha256,
                            "release_authorized_at": datetime.datetime.now(
                                datetime.UTC
                            ).strftime("%Y%m%dT%H%M%SZ"),
                        }
                    )
                    atomic_json(metadata_path, metadata)
                    atomic_json(gpu_smoke_start_barrier, barrier_document)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        print(
            f"PYPTO_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}",
            flush=True,
        )
        if gpu_smoke_admission_failed:
            pass
        elif args.allow_protected_cpu_only_coexistence:
            child_return_code, watchdog_aborted = wait_with_coexistence_watchdog(
                process,
                metadata,
                timeout_seconds=args.timeout_seconds,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
                metadata_path=metadata_path,
            )
            parent_return_code = 75 if watchdog_aborted else child_return_code
        elif args.mode == "gpu-smoke":
            child_return_code, smoke_aborted = wait_with_gpu_smoke_watchdog(
                process,
                metadata,
                timeout_seconds=args.timeout_seconds,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
                metadata_path=metadata_path,
            )
            parent_return_code = 75 if smoke_aborted else child_return_code
            post_exit_snapshot, post_exit_violation = audit_gpu_smoke_runtime_state(
                metadata
            )
            if post_exit_snapshot is not None:
                metadata["gpu_smoke_post_exit_audit"] = post_exit_snapshot
            if post_exit_violation is not None:
                metadata["gpu_smoke_abort"] = post_exit_violation
                parent_return_code = 75
        elif args.mode == "gpu-benchmark":
            child_return_code, benchmark_aborted = wait_with_gpu_benchmark_watchdog(
                process,
                metadata,
                timeout_seconds=args.timeout_seconds,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
                metadata_path=metadata_path,
            )
            parent_return_code = 75 if benchmark_aborted else child_return_code
        else:
            child_return_code = process.wait()
            parent_return_code = child_return_code
        if process.poll() is not None and metadata.get("status") != "paused":
            try:
                survivors = stop_run.owned_group_members(metadata)
            except (OSError, RuntimeError) as error:
                metadata["status"] = "group-ownership-ambiguous"
                metadata["group_exit_error"] = f"{type(error).__name__}: {error}"
                parent_return_code = 75
            else:
                if survivors:
                    metadata["surviving_group_pids"] = survivors
                    cleanup_code = terminate_owned_process(
                        process, metadata, wait_seconds=5
                    )
                    metadata["surviving_group_cleanup_code"] = cleanup_code
                    parent_return_code = 75
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
                        **metadata_context,
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
                        **metadata_context,
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
                    **metadata_context,
                )
            terminate_owned_process(process, metadata)
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
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
            atomic_json(metadata_path, metadata)
    if parent_return_code is None:
        raise RuntimeError("run exited without a parent return code")
    return parent_return_code


if __name__ == "__main__":
    sys.exit(main())
