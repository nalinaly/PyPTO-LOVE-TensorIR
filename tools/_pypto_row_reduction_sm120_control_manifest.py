#!/usr/bin/env python3
"""Validate the separately published RowReductionV3 SM120 control manifest."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "pypto-row-reduction-sm120-controls-v1"
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_row_reduction_sm120_v1.json")
BASE_ADMISSION_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_fused_pointwise_sm120_v2.json"
)
BASE_ADMISSION_MANIFEST_SIZE = 1_553
BASE_ADMISSION_MANIFEST_SHA256 = (
    "d3b16079c811dd2fbe610ba264d81117e8c4a44886b74caaddb684df2d467036"
)
BASE_ADMISSION_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py"
)
BASE_ADMISSION_VALIDATOR_SIZE = 10_807
BASE_ADMISSION_VALIDATOR_SHA256 = (
    "6c2737daf653ac237a2da0081ad05b9d3e14593e2862582e74898f92c7c94ebf"
)
CONTROL_PATHS = (
    "benchmarks/operators/pypto_row_reduction_sm120.py",
    "tools/_pypto_row_reduction_sm120_contract.py",
    "tools/generate_pypto_row_reduction_anchors.py",
    "state/contracts/pypto_row_reduction_compile_anchors_v1.json",
    "tools/_pypto_row_reduction_sm120_control_manifest.py",
    "tools/run_pypto_row_reduction_sm120_isolated.py",
    "tools/finalize_pypto_row_reduction_sm120.py",
)
PYTHON_SOURCE_PATHS = (
    *[path for path in CONTROL_PATHS if path.endswith(".py")],
    "tests/test_pypto_row_reduction_sm120.py",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ControlManifestError(RuntimeError):
    """The row-reduction controls do not match reviewed bytes."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ControlManifestError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_exact(name: str, path: Path, size: int, digest: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControlManifestError(
            f"exact base admission source is noncanonical: {path}"
        )
    raw = path.read_bytes()
    if len(raw) != size or sha256_bytes(raw) != digest:
        raise ControlManifestError(f"exact base admission source differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


base_admission = load_exact(
    "_pypto_row_reduction_base_admission_validator",
    ROOT / BASE_ADMISSION_VALIDATOR_RELATIVE_PATH,
    BASE_ADMISSION_VALIDATOR_SIZE,
    BASE_ADMISSION_VALIDATOR_SHA256,
)


def control_bytecode_cache_entries(root: Path) -> list[str]:
    entries: set[str] = set()
    for relative in PYTHON_SOURCE_PATHS:
        source = root / relative
        for suffix in (".pyc", ".pyo"):
            candidate = source.with_suffix(suffix)
            if candidate.exists() or candidate.is_symlink():
                entries.add(candidate.relative_to(root).as_posix())
        cache = source.parent / "__pycache__"
        if not cache.is_dir():
            continue
        for candidate in cache.iterdir():
            if candidate.name.startswith(source.stem + ".") and candidate.name.endswith(
                (".pyc", ".pyo")
            ):
                entries.add(candidate.relative_to(root).as_posix())
    return sorted(entries)


def reject_control_bytecode_cache(root: Path) -> None:
    entries = control_bytecode_cache_entries(root)
    if entries:
        raise ControlManifestError(
            "row-reduction control bytecode/cache entries are forbidden: "
            + ", ".join(entries)
        )


def git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=text, capture_output=True
    ).stdout


def load_canonical(
    path: Path, size: int, digest: str, description: str
) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ControlManifestError(f"{description} bytes differ")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ControlManifestError(f"{description} is not canonical JSON")
    return value


def validate_compile_anchors(root: Path, contract: ModuleType) -> dict[str, object]:
    path = root / contract.COMPILE_ANCHORS_RELATIVE_PATH
    anchors = load_canonical(
        path,
        contract.COMPILE_ANCHORS_SIZE,
        contract.COMPILE_ANCHORS_SHA256,
        "row-reduction compile anchors",
    )
    records = anchors.get("records")
    policy = anchors.get("matrix_policy")
    if (
        anchors.get("schema_version") != 1
        or anchors.get("kind") != "pypto-row-reduction-compile-anchors-v1"
        or not isinstance(records, list)
        or len(records) != 10
        or sha256_bytes(canonical_json(records)) != anchors.get("records_sha256")
        or not isinstance(policy, dict)
        or policy.get("case_count") != 10
        or policy.get("executions") != 20
        or policy.get("fresh_executable_lifetimes") != 20
        or policy.get("input_guard_elements_per_side") != 4096
        or policy.get("output_guard_elements_per_side") != 16
        or [record.get("case") for record in records] != list(contract.CASE_ORDER)
        or anchors.get("generator")
        != {
            "path": contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix(),
            "bytes": contract.ANCHOR_GENERATOR_SIZE,
            "sha256": contract.ANCHOR_GENERATOR_SHA256,
        }
    ):
        raise ControlManifestError("row-reduction compile-anchor schema differs")
    runs = anchors.get("anchor_runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or runs[0]["run_id"] == runs[1]["run_id"]
    ):
        raise ControlManifestError("row-reduction anchor-run set differs")
    for run in runs:
        if (
            run.get("return_code") != 0
            or run.get("cuda_visible_devices") != ""
            or run.get("nvidia_visible_devices") != "void"
            or run.get("torch_cuda_initialized_before_and_after") is not False
        ):
            raise ControlManifestError("row-reduction anchor-run identity differs")
        for name in ("preflight", "process", "record"):
            item = run.get(name)
            if not isinstance(item, dict):
                raise ControlManifestError("row-reduction anchor sidecar is malformed")
            sidecar = root / str(item["path"])
            if (
                sidecar.is_symlink()
                or not sidecar.is_file()
                or sidecar.stat().st_size != item["bytes"]
                or stat.S_IMODE(sidecar.stat().st_mode) != item["mode"]
                or sha256_file(sidecar) != item["sha256"]
            ):
                raise ControlManifestError("row-reduction anchor sidecar differs")
    return {
        "path": contract.COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
        "bytes": contract.COMPILE_ANCHORS_SIZE,
        "sha256": contract.COMPILE_ANCHORS_SHA256,
        "records_sha256": anchors["records_sha256"],
        "normalized_records_sha256": anchors["normalized_records_sha256"],
        "anchor_run_ids": [run["run_id"] for run in runs],
    }


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestError("workspace contains a symlinked path")
    reject_control_bytecode_cache(root)
    base_manifest = root / BASE_ADMISSION_MANIFEST_RELATIVE_PATH
    if (
        base_manifest.is_symlink()
        or not base_manifest.is_file()
        or base_manifest.stat().st_size != BASE_ADMISSION_MANIFEST_SIZE
        or sha256_file(base_manifest) != BASE_ADMISSION_MANIFEST_SHA256
    ):
        raise ControlManifestError("accepted policy-2 admission manifest differs")
    base_identity = base_admission.validate_control_manifest(root)
    contract = load_exact(
        "_pypto_row_reduction_sm120_contract_for_control",
        root / "tools/_pypto_row_reduction_sm120_contract.py",
        (root / "tools/_pypto_row_reduction_sm120_contract.py").stat().st_size,
        sha256_file(root / "tools/_pypto_row_reduction_sm120_contract.py"),
    )
    load_canonical(
        root / contract.CP48_REPORT_RELATIVE_PATH,
        contract.CP48_REPORT_SIZE,
        contract.CP48_REPORT_SHA256,
        "accepted CP48 compiler/Cubin report",
    )
    anchors = validate_compile_anchors(root, contract)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestError("reviewed row-reduction control manifest is missing")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    if not isinstance(manifest, dict) or canonical_json(manifest) != raw:
        raise ControlManifestError(
            "row-reduction control manifest is not canonical JSON"
        )
    if set(manifest) != {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "base_admission_manifest_sha256",
        "cp48_report_sha256",
        "files",
    }:
        raise ControlManifestError("row-reduction control-manifest schema differs")
    commit = manifest.get("implementation_commit")
    tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("base_admission_manifest_sha256")
        != BASE_ADMISSION_MANIFEST_SHA256
        or manifest.get("cp48_report_sha256") != contract.CP48_REPORT_SHA256
        or not isinstance(commit, str)
        or COMMIT_PATTERN.fullmatch(commit) is None
        or not isinstance(tree, str)
        or COMMIT_PATTERN.fullmatch(tree) is None
        or str(git(root, "rev-parse", f"{commit}^{{tree}}")).strip() != tree
    ):
        raise ControlManifestError("row-reduction implementation identity differs")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current_head],
        cwd=root,
        check=False,
    ).returncode:
        raise ControlManifestError("row-reduction implementation is not an ancestor")
    if str(git(root, "status", "--porcelain=v1", "--untracked-files=all")):
        raise ControlManifestError("root control repository is not clean")
    if subprocess.run(
        ["git", "diff", "--quiet", f"{commit}..{current_head}", "--", *CONTROL_PATHS],
        cwd=root,
        check=False,
    ).returncode:
        raise ControlManifestError(
            "row-reduction controls changed after implementation"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(CONTROL_PATHS):
        raise ControlManifestError("row-reduction control file set differs")
    normalized: list[dict[str, object]] = []
    for record, expected_path in zip(files, CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("row-reduction control record is malformed")
        path = root / expected_path
        if (
            record.get("path") != expected_path
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != record.get("mode")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ControlManifestError(
                f"live row-reduction control differs: {expected_path}"
            )
        committed = git(root, "show", f"{commit}:{expected_path}", text=False)
        assert isinstance(committed, bytes)
        if (
            len(committed) != record["bytes"]
            or sha256_bytes(committed) != record["sha256"]
        ):
            raise ControlManifestError(
                f"committed row control differs: {expected_path}"
            )
        normalized.append(dict(record))
    return {
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(raw),
        "manifest_sha256": sha256_bytes(raw),
        "implementation_commit": commit,
        "implementation_tree": tree,
        "current_head": current_head,
        "current_tree": str(git(root, "rev-parse", "HEAD^{tree}")).strip(),
        "root_clean": True,
        "base_admission": base_identity,
        "compile_anchors": anchors,
        "files": normalized,
    }
