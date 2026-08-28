#!/usr/bin/env python3
"""Run the final-revision operator suite under bounded controllers.

The structural suite is CPU-only and always uses 24 pytest workers.  GPU
numerical programs are executed serially.  Legacy programs are copied into the
owned run directory before execution so their historical result writers can
never modify a tracked source file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import traceback
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.controllers import (  # noqa: E402
    invoke_controlled,
    isolated_command,
    pypto_gpu_command,
)
from benchmarks.release.workload import (  # noqa: E402
    CPU_JOBS,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    read_json,
    require_run_directory,
    sha256_file,
)


MANIFEST_PATH = ROOT / "benchmarks/release/operator_manifest.json"


def _git_record(repository: Path, paths: list[str] | None = None) -> dict[str, object]:
    def invoke(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    head = invoke(["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise ReleaseContractError(f"not a Git repository: {repository}")
    status_args = ["status", "--porcelain=v1", "--untracked-files=all"]
    if paths:
        status_args.extend(("--", *paths))
    status = invoke(status_args)
    if status.returncode != 0:
        raise ReleaseContractError(status.stderr.strip() or "git status failed")
    entries = [line for line in status.stdout.splitlines() if line]
    return {
        "repository": str(repository),
        "head": head.stdout.strip(),
        "dirty": bool(entries),
        "status": entries,
    }


def _repository_for(path: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseContractError(f"path is not in a Git repository: {path}")
    return Path(completed.stdout.strip()).resolve(strict=True)


def _relative_to_repository(repository: Path, paths: list[Path]) -> list[str]:
    try:
        return [
            str(path.resolve(strict=True).relative_to(repository)) for path in paths
        ]
    except ValueError as error:
        raise ReleaseContractError("operator source escaped its repository") from error


def _require_portable_output_contract(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    forbidden_roots = (
        "/" + "home" + "/",
        "/" + "root" + "/",
        "work" + "trees/",
    )
    found = [token for token in forbidden_roots if token in text]
    if found:
        raise ReleaseContractError(
            f"operator regression source contains non-portable paths: {source}"
        )
    if '"--output"' not in text and "'--output'" not in text:
        raise ReleaseContractError(
            f"operator regression source lacks required --output contract: {source}"
        )


def _installed_pypto_inputs(
    dso: Path | None, pypto_package: Path | None
) -> tuple[Path, Path]:
    if pypto_package is None:
        controlled = os.environ.get("PYPTO_KERNEL_PACKAGE_PATH")
        if controlled:
            pypto_package = Path(controlled)
        else:
            spec = importlib.util.find_spec("pypto")
            locations = None if spec is None else spec.submodule_search_locations
            if not locations:
                raise ReleaseContractError("installed pypto distribution was not found")
            pypto_package = Path(next(iter(locations)))
    package = pypto_package.resolve(strict=True)
    if dso is None:
        controlled = os.environ.get("PYPTO_KERNEL_DSO_PATH")
        if controlled:
            dso = Path(controlled)
        else:
            candidates = sorted(
                {
                    *package.glob("pypto_core*.so"),
                    *package.parent.glob("pypto_core*.so"),
                }
            )
            if len(candidates) != 1:
                raise ReleaseContractError(
                    "installed PyPTO DSO is ambiguous; pass --dso explicitly"
                )
            dso = candidates[0]
    return dso.resolve(strict=True), package


def _structure_worker(kernel_root: Path, jobs: int | None) -> int:
    run_id, run_dir = require_run_directory(ROOT)
    report_path = run_dir / "operator-structure-regression.json"
    source = kernel_root / "tests/test_operators.py"
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "pypto-release-operator-structure",
        "run_id": run_id,
        "jobs": CPU_JOBS,
        "source": str(source),
        "status": "starting",
    }
    try:
        if jobs != CPU_JOBS:
            raise ReleaseContractError("operator structure jobs must be exactly 24")
        if os.environ.get("PYPTO_RUN_MODE") != "cpu-bounded":
            raise ReleaseContractError("structure worker requires cpu-bounded mode")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ReleaseContractError("structure worker must have CUDA hidden")
        source = source.resolve(strict=True)
        repository = _repository_for(kernel_root)
        git = _git_record(
            repository, _relative_to_repository(repository, [kernel_root])
        )
        if git["dirty"]:
            raise ReleaseContractError(
                "operator structure source is not a committed final revision"
            )
        import pytest

        arguments = [
            str(source),
            "-n",
            str(CPU_JOBS),
            "--basetemp",
            str(run_dir / "pytest-tmp"),
            "-o",
            f"cache_dir={run_dir / 'pytest-cache'}",
            f"--junitxml={run_dir / 'operator-structure-junit.xml'}",
        ]
        code = int(pytest.main(arguments))
        report.update(
            {
                "status": "complete" if code == 0 else "failed",
                "return_code": code,
                "source_sha256": sha256_file(source),
                "revision": git,
                "pytest_arguments": arguments,
            }
        )
        atomic_json(report_path, report)
        return code
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(report_path, report)
        return 1


def _gpu_worker(kernel_root: Path, dso: Path | None, pypto_package: Path | None) -> int:
    run_id, run_dir = require_run_directory(ROOT)
    report_path = run_dir / "operator-numerical-regression.json"
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "pypto-release-operator-numerical",
        "run_id": run_id,
        "execution": "serial",
        "status": "starting",
    }
    try:
        if os.environ.get("PYPTO_RUN_MODE") != "gpu-bounded":
            raise ReleaseContractError("numerical worker requires gpu-bounded mode")
        kernel_root = kernel_root.resolve(strict=True)
        dso, pypto_package = _installed_pypto_inputs(dso, pypto_package)
        manifest = read_json(MANIFEST_PATH)
        suites = manifest.get("gpu_suites")
        if type(suites) is not list or not suites:
            raise ReleaseContractError("operator manifest has no GPU suites")
        sources = [
            (kernel_root / "benchmarks" / Path(str(item["path"])).name).resolve(
                strict=True
            )
            for item in suites
        ]
        repository = _repository_for(kernel_root)
        revision = _git_record(
            repository, _relative_to_repository(repository, [kernel_root])
        )
        if revision["dirty"]:
            raise ReleaseContractError(
                "GPU operator sources are not a committed final revision"
            )
        for source in sources:
            _require_portable_output_contract(source)
        environment = os.environ.copy()
        environment.update(
            {
                "PYPTO_KERNEL_DSO_PATH": str(dso),
                "PYPTO_PLUGINS_PYPTO_DSO": str(dso),
                "PYPTO_KERNEL_PACKAGE_PATH": str(pypto_package),
                "PYPTO_ALLOW_FALLBACK": "0",
                "PYPTO_STRICT_COVERAGE": "1",
            }
        )
        results: list[dict[str, object]] = []
        for raw, source in zip(suites, sources):
            if kernel_root not in source.parents:
                raise ReleaseContractError(f"suite escaped kernel root: {source}")
            log_path = run_dir / f"{source.stem}.log"
            result_path = run_dir / str(raw["result"])
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(source),
                        "--output",
                        str(result_path),
                    ],
                    cwd=run_dir,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            item: dict[str, object] = {
                "source": (
                    str(source.relative_to(ROOT))
                    if ROOT in source.parents
                    else str(source)
                ),
                "source_sha256": sha256_file(source),
                "return_code": int(completed.returncode),
                "log": str(log_path),
            }
            if not result_path.is_file():
                item.update({"passed": False, "error": "result JSON was not written"})
            else:
                payload = read_json(result_path)
                success_field = str(raw["success_field"])
                item.update(
                    {
                        "result": str(result_path),
                        "result_sha256": sha256_file(result_path),
                        "passed": bool(
                            completed.returncode == 0
                            and payload.get(success_field) is True
                        ),
                        "success_field": success_field,
                        "case_count": len(payload.get("cases", [])),
                    }
                )
            results.append(item)
            if not item["passed"]:
                break
        passed = len(results) == len(suites) and all(item["passed"] for item in results)
        report.update(
            {
                "status": "complete" if passed else "failed",
                "all_correct": passed,
                "dso": {"path": str(dso), "sha256": sha256_file(dso)},
                "pypto_package": str(pypto_package),
                "revision": revision,
                "suites": results,
            }
        )
        atomic_json(report_path, report)
        return 0 if passed else 1
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "all_correct": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(report_path, report)
        return 1


def _public(args: argparse.Namespace) -> int:
    kernel_root = args.kernel_root.resolve()
    worker = Path(__file__).resolve()
    planned = []
    if args.stage in {"all", "structure"}:
        planned.append(
            (
                "structure",
                lambda pointer: isolated_command(
                    ROOT,
                    worker,
                    (
                        "--_worker-structure",
                        "--_jobs",
                        str(CPU_JOBS),
                        "--kernel-root",
                        str(kernel_root),
                    ),
                    pointer,
                    framework_profile="pypto",
                    timeout_seconds=args.timeout_seconds,
                    cpu_only=True,
                ),
            )
        )
    if args.stage in {"all", "gpu"}:
        dso = None if args.dso is None else args.dso.resolve()
        package = None if args.pypto_package is None else args.pypto_package.resolve()
        worker_values = [
            "--_worker-gpu",
            "--kernel-root",
            str(kernel_root),
        ]
        if dso is not None:
            worker_values.extend(("--dso", str(dso)))
        if package is not None:
            worker_values.extend(("--pypto-package", str(package)))
        planned.append(
            (
                "gpu",
                lambda pointer: pypto_gpu_command(
                    ROOT,
                    worker,
                    tuple(worker_values),
                    pointer,
                    timeout_seconds=args.timeout_seconds,
                ),
            )
        )
    records = []
    for stage, factory in planned:
        controlled = invoke_controlled(factory, root=ROOT, dry_run=args.dry_run)
        records.append(
            {
                "stage": stage,
                "command": list(controlled.command),
                "return_code": controlled.return_code,
                "run_id": controlled.run_id,
            }
        )
        if controlled.return_code != 0:
            break
    summary = {
        "schema": SCHEMA_VERSION,
        "kind": "pypto-release-operator-regression-control",
        "jobs": CPU_JOBS,
        "gpu_execution": "serial",
        "dry_run": args.dry_run,
        "stages": records,
        "status": (
            "planned"
            if args.dry_run
            else "complete"
            if len(records) == len(planned)
            and all(item["return_code"] == 0 for item in records)
            else "failed"
        ),
    }
    if not args.dry_run:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = (
            ROOT
            / "runs"
            / (f"release-operator-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}")
            / "summary.json"
        )
        atomic_json(path, summary)
        summary["summary_path"] = str(path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["status"] in {"planned", "complete"} else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--stage", choices=("all", "structure", "gpu"), default="all")
    value.add_argument(
        "--kernel-root", type=Path, default=ROOT / "packages/pypto-kernels"
    )
    value.add_argument("--dso", type=Path)
    value.add_argument("--pypto-package", type=Path)
    value.add_argument("--timeout-seconds", type=int, default=7200)
    value.add_argument("--dry-run", action="store_true")
    workers = value.add_mutually_exclusive_group()
    workers.add_argument(
        "--_worker-structure", action="store_true", help=argparse.SUPPRESS
    )
    workers.add_argument("--_worker-gpu", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--_jobs", type=int, help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args._worker_structure:
        return _structure_worker(args.kernel_root.resolve(strict=True), args._jobs)
    if args._worker_gpu:
        return _gpu_worker(args.kernel_root, args.dso, args.pypto_package)
    return _public(args)


if __name__ == "__main__":
    raise SystemExit(main())
