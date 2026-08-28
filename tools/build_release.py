#!/usr/bin/env python3
"""Build, test, and install the portable PyPTO SM120 release wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.controllers import invoke_controlled, isolated_command  # noqa: E402
from benchmarks.release.workload import (  # noqa: E402
    CPU_JOBS,
    ReleaseContractError,
    atomic_json,
    require_run_directory,
)


RELEASE = "qwen35-sm120-v1"
SOURCE = ROOT / ".sources/pypto"
KERNELS = ROOT / "packages/pypto-kernels"
PLUGINS = ROOT / "packages/pypto-framework-plugins"
BUILD_ROOT = ROOT / "builds" / RELEASE
PYPTO_BUILD = BUILD_ROOT / "pypto"
NATIVE_BUILD = BUILD_ROOT / "native"
WHEEL_DIR = BUILD_ROOT / "wheels"
CUDA_ROOT = Path("/usr/local/cuda-13.3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseContractError(
            f"release build command failed ({completed.returncode}): {command}"
        )


def _one_wheel(distribution: str) -> Path:
    normalized = distribution.replace("-", "_")
    matches = sorted(WHEEL_DIR.glob(f"{normalized}-*.whl"))
    if len(matches) != 1:
        raise ReleaseContractError(
            f"expected one {distribution} wheel, found {[path.name for path in matches]}"
        )
    return matches[0].resolve(strict=True)


def _wheel_record() -> list[dict[str, object]]:
    records = []
    for distribution in ("pypto", "pypto-kernels", "pypto-framework-plugins"):
        path = _one_wheel(distribution)
        records.append(
            {
                "distribution": distribution,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _build_wheels(python: Path) -> None:
    if not SOURCE.is_dir() or not KERNELS.is_dir() or not PLUGINS.is_dir():
        raise ReleaseContractError("release sources/packages have not been materialized")
    if not CUDA_ROOT.is_dir():
        raise ReleaseContractError(f"pinned CUDA toolkit is missing: {CUDA_ROOT}")
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": str(CPU_JOBS),
            "CMAKE_GENERATOR": "Ninja",
            "PYTHONNOUSERSITE": "1",
        }
    )
    pypto_command = [
        str(python),
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(WHEEL_DIR),
        f"--config-setting=build-dir={PYPTO_BUILD}",
        "--config-setting=cmake.define.PYPTO_ENABLE_NVIDIA_BACKEND=ON",
        "--config-setting=cmake.define.BUILD_TESTING=ON",
        "--config-setting=cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"--config-setting=cmake.define.PYPTO_NVIDIA_CUDA_TOOLKIT_ROOT={CUDA_ROOT}",
        str(SOURCE),
    ]
    _run(pypto_command, environment=environment)
    for package in (KERNELS, PLUGINS):
        _run(
            [
                str(python),
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(WHEEL_DIR),
                str(package),
            ],
            environment=environment,
        )
    _wheel_record()


def _install_wheels(python: Path) -> None:
    wheels = {
        name: _one_wheel(name)
        for name in ("pypto", "pypto-kernels", "pypto-framework-plugins")
    }
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheels["pypto"]),
            str(wheels["pypto-kernels"]),
            str(wheels["pypto-framework-plugins"]),
        ]
    )
    _run([str(python), "-m", "pip", "check"])
    _run(
        [
            str(python),
            str(ROOT / "tools/bootstrap_release_environment.py"),
            "--finalize-only",
        ]
    )


def _build_native(python: Path) -> None:
    cmake = (python.parent / "cmake").resolve(strict=True)
    NATIVE_BUILD.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(cmake),
            "-S",
            str(SOURCE),
            "-B",
            str(NATIVE_BUILD),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DBUILD_TESTING=ON",
            "-DPYPTO_ENABLE_NVIDIA_BACKEND=ON",
            f"-DPYPTO_NVIDIA_CUDA_TOOLKIT_ROOT={CUDA_ROOT}",
            f"-DPython_EXECUTABLE={python}",
            f"-DPython3_EXECUTABLE={python}",
            f"-DPYPTO_NATIVE_EXTENSION_OUTPUT_DIRECTORY={NATIVE_BUILD / 'product'}",
        ]
    )
    _run(
        [
            str(cmake),
            "--build",
            str(NATIVE_BUILD),
            "--parallel",
            str(CPU_JOBS),
        ]
    )


def _run_ctest(python: Path) -> None:
    if not (NATIVE_BUILD / "CTestTestfile.cmake").is_file():
        raise ReleaseContractError("PyPTO release build has no configured CTest suite")
    ctest = (python.parent / "ctest").resolve(strict=True)
    inventory = subprocess.run(
        [str(ctest), "--test-dir", str(NATIVE_BUILD), "-N"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if inventory.returncode != 0 or "Total Tests: 13" not in inventory.stdout:
        raise ReleaseContractError(
            "native CTest inventory must contain exactly 13 tests; "
            + inventory.stdout[-1000:]
        )
    _run(
        [
            str(ctest),
            "--test-dir",
            str(NATIVE_BUILD),
            "--output-on-failure",
            "-j24",
        ]
    )


def _worker(stage: str, jobs: int) -> int:
    if jobs != CPU_JOBS:
        raise ReleaseContractError(f"release build requires exactly {CPU_JOBS} jobs")
    if os.environ.get("PYPTO_RUN_MODE") != "cpu-bounded":
        raise ReleaseContractError("release build worker requires cpu-bounded mode")
    run_id, run_dir = require_run_directory(ROOT)
    report_path = run_dir / f"release-build-{stage}.json"
    report: dict[str, object] = {
        "schema": 1,
        "kind": "pypto-sm120-release-build",
        "release": RELEASE,
        "stage": stage,
        "jobs": jobs,
        "run_id": run_id,
        "status": "starting",
    }
    try:
        python = Path(sys.executable).resolve(strict=True)
        expected = (ROOT / "envs/pypto-release/bin/python").resolve(strict=True)
        if python != expected:
            raise ReleaseContractError(
                f"release build selected wrong interpreter: {python}"
            )
        if stage == "wheels":
            _build_wheels(python)
        elif stage == "native":
            _build_native(python)
        elif stage == "ctest":
            _run_ctest(python)
        elif stage == "install":
            _install_wheels(python)
        else:
            raise ReleaseContractError(f"unknown worker stage: {stage}")
        wheels = []
        try:
            wheels = _wheel_record()
        except ReleaseContractError:
            if stage in {"wheels", "install"}:
                raise
        report.update({"status": "complete", "wheels": wheels})
        return_code = 0
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return return_code


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--stage",
        choices=("wheels", "native", "ctest", "install", "all"),
        default="all",
    )
    value.add_argument("--jobs", type=int, default=CPU_JOBS)
    value.add_argument("--timeout-seconds", type=int, default=3600)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--_worker",
        choices=("wheels", "native", "ctest", "install"),
        help=argparse.SUPPRESS,
    )
    value.add_argument("--_jobs", type=int, help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.jobs != CPU_JOBS or args.timeout_seconds <= 0:
        raise ReleaseContractError("release build requires jobs=24 and positive timeout")
    if args._worker:
        return _worker(args._worker, args._jobs)
    stages = (
        ("wheels", "native", "ctest", "install")
        if args.stage == "all"
        else (args.stage,)
    )
    results = []
    for stage in stages:
        worker_args = ("--_worker", stage, "--_jobs", str(CPU_JOBS))
        controlled = invoke_controlled(
            lambda pointer: isolated_command(
                ROOT,
                Path(__file__),
                worker_args,
                pointer,
                framework_profile="pypto",
                timeout_seconds=args.timeout_seconds,
                cpu_only=True,
            ),
            root=ROOT,
            dry_run=args.dry_run,
        )
        results.append(
            {
                "stage": stage,
                "run_id": controlled.run_id,
                "return_code": controlled.return_code,
                "command": list(controlled.command),
            }
        )
        if controlled.return_code != 0:
            break
    complete = len(results) == len(stages) and all(
        item["return_code"] == 0 for item in results
    )
    payload = {
        "schema": 1,
        "kind": "pypto-sm120-release-build-control",
        "release": RELEASE,
        "jobs": CPU_JOBS,
        "dry_run": args.dry_run,
        "runs": results,
        "status": "planned" if args.dry_run else "complete" if complete else "failed",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if payload["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
