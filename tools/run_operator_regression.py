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
from pathlib import Path, PurePosixPath
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
from benchmarks.release.evidence_identity import collect_run_identity  # noqa: E402
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
KERNEL_MANIFEST_PREFIX = PurePosixPath("packages/pypto-kernels")
PLUGIN_MANIFEST_PREFIX = PurePosixPath("packages/pypto-framework-plugins")
ALLOWED_SUITE_KEYS = {
    "id",
    "path",
    "result",
    "success_field",
    "arguments",
    "path_arguments",
    "case_expectations",
    "expected_case_count",
    "run_id_required",
}


def _release_model_path(value: Path) -> Path:
    resolved = value.resolve(strict=True)
    expected = (ROOT / "models/Qwen3.5-9B").resolve(strict=True)
    if resolved != expected:
        raise ReleaseContractError("operator release model must be the frozen 9B path")
    return resolved


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


def _manifest_relative_path(value: object, label: str) -> PurePosixPath:
    if type(value) is not str or not value:
        raise ReleaseContractError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ReleaseContractError(f"{label} must be a normalized relative path")
    return path


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ReleaseContractError(f"{label} escaped its allowed root")
    return resolved


def _suite_source(raw: object, kernel_root: Path) -> tuple[Path, Path]:
    manifest_path = _manifest_relative_path(raw, "operator suite path")
    parts = manifest_path.parts
    kernel_parts = KERNEL_MANIFEST_PREFIX.parts
    plugin_parts = PLUGIN_MANIFEST_PREFIX.parts
    if parts[: len(kernel_parts)] == kernel_parts:
        relative = Path(*parts[len(kernel_parts) :])
        source_scope = kernel_root.resolve(strict=True)
        source = _within(source_scope / relative, source_scope, "kernel suite")
    elif parts[: len(plugin_parts)] == plugin_parts:
        source_scope = (ROOT / Path(*plugin_parts)).resolve(strict=True)
        relative = Path(*parts[len(plugin_parts) :])
        source = _within(source_scope / relative, source_scope, "plugin suite")
    else:
        raise ReleaseContractError(
            "operator suite must belong to the kernel or framework-plugin package"
        )
    if source.parent.name != "benchmarks" or source.suffix != ".py":
        raise ReleaseContractError("operator suite must be a Python benchmark source")
    return source, source_scope


def _suite_arguments(raw: dict[str, object]) -> list[str]:
    values = raw.get("arguments", [])
    if type(values) is not list or any(type(value) is not str for value in values):
        raise ReleaseContractError("operator suite arguments must be a string list")
    if "--output" in values or any("\x00" in value for value in values):
        raise ReleaseContractError("operator suite arguments violate output ownership")
    result = list(values)
    path_arguments = raw.get("path_arguments", [])
    if type(path_arguments) is not list:
        raise ReleaseContractError("operator path_arguments must be a list")
    for item in path_arguments:
        if type(item) is not dict or set(item) != {"flag", "path"}:
            raise ReleaseContractError("operator path argument has an unknown schema")
        flag = item["flag"]
        if (
            type(flag) is not str
            or not flag.startswith("--")
            or flag == "--output"
        ):
            raise ReleaseContractError("operator path argument flag is invalid")
        relative = _manifest_relative_path(item["path"], "operator input path")
        resolved = _within(ROOT / Path(*relative.parts), ROOT, "operator input")
        result.extend((flag, str(resolved)))
    return result


def _validate_case_expectations(
    payload: dict[str, object], raw: dict[str, object]
) -> None:
    expectations = raw.get("case_expectations", [])
    if type(expectations) is not list:
        raise ReleaseContractError("case_expectations must be a list")
    cases = payload.get("cases")
    if expectations and (
        type(cases) is not list or any(type(case) is not dict for case in cases)
    ):
        raise ReleaseContractError("operator result cases have an unknown schema")
    expected_count = raw.get("expected_case_count")
    if expected_count is not None and (
        type(expected_count) is not int
        or expected_count <= 0
        or type(cases) is not list
        or len(cases) != expected_count
    ):
        raise ReleaseContractError("operator result case count changed")
    for expectation in expectations:
        if type(expectation) is not dict or set(expectation) != {"where", "equals"}:
            raise ReleaseContractError("case expectation has an unknown schema")
        where = expectation["where"]
        equals = expectation["equals"]
        if type(where) is not dict or not where or type(equals) is not dict:
            raise ReleaseContractError("case expectation predicates must be objects")
        matches = [
            case
            for case in cases
            if all(case.get(field) == value for field, value in where.items())
        ]
        if len(matches) != 1:
            raise ReleaseContractError(
                f"case expectation matched {len(matches)} records: {where!r}"
            )
        mismatches = {
            field: {"expected": value, "observed": matches[0].get(field)}
            for field, value in equals.items()
            if matches[0].get(field) != value
        }
        if mismatches:
            raise ReleaseContractError(
                f"case expectation failed for {where!r}: {mismatches!r}"
            )


def _operator_suites(
    kernel_root: Path,
) -> list[tuple[dict[str, object], Path, Path, list[str]]]:
    manifest = read_json(MANIFEST_PATH)
    if set(manifest) != {"schema", "structure", "gpu_suites"}:
        raise ReleaseContractError("operator manifest has an unknown top-level schema")
    if manifest.get("schema") != 2:
        raise ReleaseContractError("operator manifest schema is not supported")
    _structure_sources(kernel_root)
    suites = manifest.get("gpu_suites")
    if type(suites) is not list or not suites:
        raise ReleaseContractError("operator manifest has no GPU suites")
    result = []
    suite_ids = set()
    result_names = set()
    source_paths = set()
    for raw in suites:
        if type(raw) is not dict or not set(raw).issubset(ALLOWED_SUITE_KEYS):
            raise ReleaseContractError("operator suite has an unknown schema")
        if not {"id", "path", "result", "success_field"}.issubset(raw):
            raise ReleaseContractError("operator suite is missing required fields")
        suite_id = raw["id"]
        if (
            type(suite_id) is not str
            or not suite_id
            or not all(character.isalnum() or character in "-_" for character in suite_id)
            or suite_id in suite_ids
        ):
            raise ReleaseContractError("operator suite id is invalid or duplicated")
        suite_ids.add(suite_id)
        success_field = raw["success_field"]
        if type(success_field) is not str or not success_field:
            raise ReleaseContractError("operator suite success field is invalid")
        if type(raw.get("run_id_required", True)) is not bool:
            raise ReleaseContractError("operator suite run_id_required is invalid")
        result_name = _manifest_relative_path(raw["result"], "operator result")
        if (
            len(result_name.parts) != 1
            or result_name.suffix != ".json"
            or result_name.name in result_names
        ):
            raise ReleaseContractError("operator result name is invalid or duplicated")
        result_names.add(result_name.name)
        source, source_scope = _suite_source(raw["path"], kernel_root)
        if source in source_paths:
            raise ReleaseContractError("operator suite source is duplicated")
        source_paths.add(source)
        result.append((raw, source, source_scope, _suite_arguments(raw)))
    return result


def _structure_sources(kernel_root: Path) -> list[Path]:
    manifest = read_json(MANIFEST_PATH)
    if (
        set(manifest) != {"schema", "structure", "gpu_suites"}
        or manifest.get("schema") != 2
    ):
        raise ReleaseContractError("operator manifest schema is not supported")
    structure = manifest.get("structure")
    if (
        type(structure) is not dict
        or set(structure) != {"jobs", "paths"}
        or structure.get("jobs") != CPU_JOBS
        or type(structure.get("paths")) is not list
        or not structure["paths"]
    ):
        raise ReleaseContractError("operator structure manifest has an unknown schema")
    sources = []
    seen = set()
    for value in structure["paths"]:
        relative = _manifest_relative_path(value, "operator structure source")
        parts = relative.parts
        prefix = KERNEL_MANIFEST_PREFIX.parts
        if parts[: len(prefix)] != prefix:
            raise ReleaseContractError("operator structure source escaped kernel package")
        source = _within(
            kernel_root / Path(*parts[len(prefix) :]),
            kernel_root,
            "operator structure source",
        )
        if source.parent.name != "tests" or source.suffix != ".py" or source in seen:
            raise ReleaseContractError("operator structure source is invalid or duplicated")
        sources.append(source)
        seen.add(source)
    return sources


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


def _structure_worker(
    kernel_root: Path,
    jobs: int | None,
    model_path: Path = ROOT / "models/Qwen3.5-9B",
) -> int:
    run_id, run_dir = require_run_directory(ROOT)
    report_path = run_dir / "operator-structure-regression.json"
    sources = _structure_sources(kernel_root)
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "pypto-release-operator-structure",
        "run_id": run_id,
        "jobs": CPU_JOBS,
        "sources": [str(source) for source in sources],
        "status": "starting",
    }
    try:
        if jobs != CPU_JOBS:
            raise ReleaseContractError("operator structure jobs must be exactly 24")
        if os.environ.get("PYPTO_RUN_MODE") != "cpu-bounded":
            raise ReleaseContractError("structure worker requires cpu-bounded mode")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ReleaseContractError("structure worker must have CUDA hidden")
        repository = _repository_for(kernel_root)
        git = _git_record(
            repository, _relative_to_repository(repository, [kernel_root])
        )
        if git["dirty"]:
            raise ReleaseContractError(
                "operator structure source is not a committed final revision"
            )
        report["evidence_identity"] = collect_run_identity(
            ROOT, "pypto", _release_model_path(model_path)
        )
        import pytest

        arguments = [
            *(str(source) for source in sources),
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
                "source_sha256": {
                    str(source): sha256_file(source) for source in sources
                },
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


def _gpu_worker(
    kernel_root: Path,
    dso: Path | None,
    pypto_package: Path | None,
    model_path: Path = ROOT / "models/Qwen3.5-9B",
) -> int:
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
        suites = _operator_suites(kernel_root)
        source_revisions = []
        seen_scopes = set()
        for _raw, _source, source_scope, _arguments in suites:
            if source_scope in seen_scopes:
                continue
            seen_scopes.add(source_scope)
            repository = _repository_for(source_scope)
            revision = _git_record(
                repository,
                _relative_to_repository(repository, [source_scope]),
            )
            revision["scope"] = str(source_scope)
            if revision["dirty"]:
                raise ReleaseContractError(
                    f"GPU operator source scope is not final: {source_scope}"
                )
            source_revisions.append(revision)
        report["evidence_identity"] = collect_run_identity(
            ROOT, "pypto", _release_model_path(model_path)
        )
        report["operator_model_configs"] = {
            model: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for model in ("Qwen3.5-0.8B", "Qwen3.5-9B")
            for path in [(ROOT / "models" / model / "config.json").resolve(strict=True)]
        }
        for _raw, source, _source_scope, _arguments in suites:
            _require_portable_output_contract(source)
        temporary_root = run_dir / "temporary"
        temporary_root.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(
            {
                "PYPTO_KERNEL_DSO_PATH": str(dso),
                "PYPTO_PLUGINS_PYPTO_DSO": str(dso),
                "PYPTO_KERNEL_PACKAGE_PATH": str(pypto_package),
                "PYPTO_ALLOW_FALLBACK": "0",
                "PYPTO_STRICT_COVERAGE": "1",
                "PYPTO_WORKSPACE_ROOT": str(ROOT),
                "PYPTO_ENV_PREFIX": sys.prefix,
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "torchinductor-cache"),
                "TRITON_CACHE_DIR": str(run_dir / "triton-cache"),
                "CUDA_CACHE_PATH": str(run_dir / "cuda-cache"),
                "SGLANG_CACHE_DIR": str(run_dir / "sglang-cache"),
                "XDG_CACHE_HOME": str(run_dir / "cache"),
                "TMPDIR": str(temporary_root),
            }
        )
        results: list[dict[str, object]] = []
        for raw, source, _source_scope, arguments in suites:
            suite_id = str(raw["id"])
            log_path = run_dir / f"{suite_id}.log"
            result_path = run_dir / str(raw["result"])
            if result_path.parent != run_dir or result_path.exists():
                raise ReleaseContractError("operator result path is not freshly owned")
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(source),
                        "--output",
                        str(result_path),
                        *arguments,
                    ],
                    cwd=run_dir,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            item: dict[str, object] = {
                "suite_id": suite_id,
                "source": (
                    str(source.relative_to(ROOT))
                    if ROOT in source.parents
                    else str(source)
                ),
                "source_sha256": sha256_file(source),
                "return_code": int(completed.returncode),
                "log": str(log_path),
                "arguments": arguments,
            }
            if not result_path.is_file():
                item.update({"passed": False, "error": "result JSON was not written"})
            else:
                payload = read_json(result_path)
                success_field = str(raw["success_field"])
                run_id_required = bool(raw.get("run_id_required", True))
                observed_run_id = payload.get("run_id")
                if run_id_required and observed_run_id != run_id:
                    raise ReleaseContractError(
                        f"operator suite {suite_id} did not bind the controller run id"
                    )
                if not run_id_required and observed_run_id not in (None, run_id):
                    raise ReleaseContractError(
                        f"operator suite {suite_id} reported a foreign run id"
                    )
                if payload.get(success_field) is not True:
                    raise ReleaseContractError(
                        f"operator suite {suite_id} success field is not true"
                    )
                _validate_case_expectations(payload, raw)
                item.update(
                    {
                        "result": str(result_path),
                        "result_sha256": sha256_file(result_path),
                        "passed": completed.returncode == 0,
                        "success_field": success_field,
                        "case_count": len(payload.get("cases", [])),
                        "run_id_binding": (
                            "suite-result"
                            if observed_run_id == run_id
                            else "controller-owned-fresh-result"
                        ),
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
                "source_revisions": source_revisions,
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
    model_path = args.model_path.resolve()
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
                        "--model-path",
                        str(model_path),
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
            "--model-path",
            str(model_path),
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
        report_name = (
            "operator-structure-regression.json"
            if stage == "structure"
            else "operator-numerical-regression.json"
        )
        report_path = (
            None
            if controlled.run_id is None
            else str(ROOT / "runs" / controlled.run_id / report_name)
        )
        records.append(
            {
                "stage": stage,
                "command": list(controlled.command),
                "return_code": controlled.return_code,
                "run_id": controlled.run_id,
                "report": report_path,
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
    value.add_argument(
        "--model-path", type=Path, default=ROOT / "models/Qwen3.5-9B"
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
        return _structure_worker(
            args.kernel_root.resolve(strict=True),
            args._jobs,
            args.model_path.resolve(strict=True),
        )
    if args._worker_gpu:
        return _gpu_worker(
            args.kernel_root,
            args.dso,
            args.pypto_package,
            args.model_path.resolve(strict=True),
        )
    return _public(args)


if __name__ == "__main__":
    raise SystemExit(main())
