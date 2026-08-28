"""Commands for launching release workers through repository controllers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

from .workload import ReleaseContractError


@dataclass(frozen=True, slots=True)
class ControlledRun:
    command: tuple[str, ...]
    return_code: int
    run_id: str | None


def _python(prefix: Path) -> Path:
    candidate = (prefix / "bin/python").resolve()
    if not candidate.is_file():
        raise ReleaseContractError(f"required environment is missing: {candidate}")
    return candidate


def _runtime(root: Path) -> dict[str, object]:
    path = root / "benchmarks/release/runtime.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict or payload.get("schema") != 1:
        raise ReleaseContractError("release runtime has an unknown schema")
    return payload


def _control_python(root: Path) -> Path:
    prefix = root / str(_runtime(root)["control_prefix"])
    candidate = prefix / "bin/python3.14"
    return candidate.resolve() if candidate.is_file() else _python(prefix)


def pypto_gpu_command(
    root: Path,
    worker: Path,
    worker_args: Sequence[str],
    pointer: Path,
    *,
    timeout_seconds: int,
) -> tuple[str, ...]:
    return gpu_bounded_command(
        root,
        worker,
        worker_args,
        pointer,
        framework_profile="pypto",
        timeout_seconds=timeout_seconds,
    )


def gpu_bounded_command(
    root: Path,
    worker: Path,
    worker_args: Sequence[str],
    pointer: Path,
    *,
    framework_profile: str,
    timeout_seconds: int,
) -> tuple[str, ...]:
    if framework_profile not in {"pypto", "baseline"}:
        raise ValueError(f"unknown framework profile: {framework_profile}")
    runtime = _runtime(root)
    profile = runtime["profiles"][framework_profile]
    environment = str(profile["environment"])
    selected = root / str(profile["prefix"])
    return (
        str(_control_python(root)),
        "-E",
        "-B",
        "-S",
        str((root / "tools/run_pypto_gpu_bounded.py").resolve(strict=True)),
        "--environment",
        environment,
        "--framework-profile",
        framework_profile,
        "--run-id-file",
        str(pointer),
        "--timeout-seconds",
        str(timeout_seconds),
        "--minimum-free-disk-gib",
        "64",
        "--",
        str(_python(selected)),
        "-B",
        str(worker.resolve(strict=True)),
        *worker_args,
    )


def cpu_bounded_command(
    root: Path,
    worker: Path,
    worker_args: Sequence[str],
    pointer: Path,
    *,
    timeout_seconds: int,
) -> tuple[str, ...]:
    runtime = _runtime(root)
    profile = runtime["profiles"]["pypto"]
    environment = str(profile["environment"])
    selected = root / str(profile["prefix"])
    return (
        str(_control_python(root)),
        "-E",
        "-B",
        "-S",
        str((root / "tools/run_pypto_cpu_bounded.py").resolve(strict=True)),
        "--environment",
        environment,
        "--framework-profile",
        "pypto",
        "--run-id-file",
        str(pointer),
        "--timeout-seconds",
        str(timeout_seconds),
        "--minimum-free-disk-gib",
        "64",
        "--",
        str(_python(selected)),
        "-B",
        str(worker.resolve(strict=True)),
        *worker_args,
    )


def isolated_command(
    root: Path,
    worker: Path,
    worker_args: Sequence[str],
    pointer: Path,
    *,
    framework_profile: str,
    timeout_seconds: int,
    cpu_only: bool = False,
) -> tuple[str, ...]:
    if cpu_only:
        if framework_profile != "pypto":
            raise ValueError("release CPU regression requires the PyPTO profile")
        return cpu_bounded_command(
            root,
            worker,
            worker_args,
            pointer,
            timeout_seconds=timeout_seconds,
        )
    return gpu_bounded_command(
        root,
        worker,
        worker_args,
        pointer,
        framework_profile=framework_profile,
        timeout_seconds=timeout_seconds,
    )


def invoke_controlled(
    command_factory, *, root: Path, dry_run: bool = False
) -> ControlledRun:
    control_root = root / "runs/.release-control"
    control_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pypto-release-control-", dir=control_root
    ) as temporary:
        pointer = Path(temporary) / "run-id.json"
        command = tuple(command_factory(pointer))
        if dry_run:
            return ControlledRun(command=command, return_code=0, run_id=None)
        completed = subprocess.run(command, check=False)
        run_id = None
        if pointer.is_file():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            value = payload.get("run_id")
            if isinstance(value, str) and value:
                run_id = value
        return ControlledRun(
            command=command,
            return_code=int(completed.returncode),
            run_id=run_id,
        )
