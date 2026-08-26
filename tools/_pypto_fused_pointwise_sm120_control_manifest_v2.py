#!/usr/bin/env python3
"""Validate the separately published fused-pointwise SM120 v2 control manifest."""

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
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_KIND = "pypto-fused-pointwise-sm120-controls-v2"
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_fused_pointwise_sm120_v2.json")
BASE_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_fused_pointwise_sm120_control_manifest.py"
)
BASE_VALIDATOR_SIZE = 29_977
BASE_VALIDATOR_SHA256 = (
    "299356cf6361fd1372e1fb77ddd626c2d4f84609abd565a3ea3be0bbe26c98c9"
)
BASE_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_fused_pointwise_sm120_v1.json"
)
BASE_MANIFEST_SIZE = 2_193
BASE_MANIFEST_SHA256 = (
    "ce20dd3ac6796bee16235913b8b296ae8c4781167c35f08de7c19ac7977a6896"
)
CONTROL_PATHS = (
    "benchmarks/operators/pypto_fused_pointwise_sm120_v2.py",
    "tools/_pypto_fused_pointwise_sm120_contract_v2.py",
    "tools/preflight_gpu_smoke_v2.py",
    "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py",
    "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py",
    "tools/finalize_pypto_fused_pointwise_sm120_v2.py",
)
PYTHON_SOURCE_PATHS = CONTROL_PATHS + ("tests/test_pypto_fused_pointwise_sm120_v2.py",)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ControlManifestV2Error(RuntimeError):
    """The live v2 controls do not match their reviewed publication."""


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
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ControlManifestV2Error(f"duplicate control-manifest key: {key}")
        output[key] = value
    return output


def _load_exact(name: str, path: Path, size: int, sha256: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControlManifestV2Error(f"exact base control is noncanonical: {path}")
    raw = path.read_bytes()
    if len(raw) != size or sha256_bytes(raw) != sha256:
        raise ControlManifestV2Error(f"exact base control differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = sha256
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base_validator = _load_exact(
    "_pypto_fused_pointwise_sm120_control_manifest_v1_base",
    ROOT / BASE_VALIDATOR_RELATIVE_PATH,
    BASE_VALIDATOR_SIZE,
    BASE_VALIDATOR_SHA256,
)


def control_bytecode_cache_entries(root: Path) -> list[str]:
    entries: set[str] = set()
    for relative in PYTHON_SOURCE_PATHS:
        source = root / relative
        for suffix in (".pyc", ".pyo"):
            direct = source.with_suffix(suffix)
            if direct.exists() or direct.is_symlink():
                entries.add(direct.relative_to(root).as_posix())
        cache = source.parent / "__pycache__"
        if not cache.is_dir():
            continue
        stem = source.stem
        for candidate in cache.iterdir():
            if candidate.name == f"{stem}.pyc" or (
                candidate.name.startswith(f"{stem}.")
                and candidate.name.endswith((".pyc", ".pyo"))
            ):
                entries.add(candidate.relative_to(root).as_posix())
    return sorted(entries)


def reject_control_bytecode_cache(root: Path) -> None:
    entries = control_bytecode_cache_entries(root)
    if entries:
        raise ControlManifestV2Error(
            "v2 fused-pointwise control bytecode/cache entries are forbidden: "
            + ", ".join(entries)
        )


def git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=text,
        capture_output=True,
    ).stdout


def _base_identity(root: Path) -> dict[str, object]:
    manifest = root / BASE_MANIFEST_RELATIVE_PATH
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size != BASE_MANIFEST_SIZE
        or sha256_file(manifest) != BASE_MANIFEST_SHA256
    ):
        raise ControlManifestV2Error("frozen v1 control manifest differs")
    return base_validator.validate_control_manifest(root)


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestV2Error("workspace contains a symlinked path")
    reject_control_bytecode_cache(root)
    base_identity = _base_identity(root)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestV2Error("reviewed v2 control manifest is missing")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestV2Error("v2 control manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or canonical_json(manifest) != raw:
        raise ControlManifestV2Error("v2 control manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "base_v1_manifest_sha256",
        "files",
    }:
        raise ControlManifestV2Error("v2 control manifest schema differs")
    implementation_commit = manifest.get("implementation_commit")
    implementation_tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("base_v1_manifest_sha256") != BASE_MANIFEST_SHA256
        or not isinstance(implementation_commit, str)
        or COMMIT_PATTERN.fullmatch(implementation_commit) is None
        or not isinstance(implementation_tree, str)
        or COMMIT_PATTERN.fullmatch(implementation_tree) is None
    ):
        raise ControlManifestV2Error("v2 implementation identity differs")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(CONTROL_PATHS):
        raise ControlManifestV2Error("v2 control file set is incomplete")
    if str(git(root, "rev-parse", f"{implementation_commit}^{{tree}}")).strip() != (
        implementation_tree
    ):
        raise ControlManifestV2Error("v2 implementation commit/tree join failed")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    current_tree = str(git(root, "rev-parse", "HEAD^{tree}")).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, current_head],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ControlManifestV2Error("v2 implementation is not an ancestor")
    if str(git(root, "status", "--porcelain=v1", "--untracked-files=all")):
        raise ControlManifestV2Error("root control repository is not clean")
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            f"{implementation_commit}..{current_head}",
            "--",
            *CONTROL_PATHS,
        ],
        cwd=root,
        check=False,
    )
    if changed.returncode != 0:
        raise ControlManifestV2Error("v2 controls changed after implementation")

    normalized: list[dict[str, object]] = []
    for record, expected_path in zip(files, CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestV2Error("v2 control file record is malformed")
        if record.get("path") != expected_path:
            raise ControlManifestV2Error("v2 control file order differs")
        size = record.get("bytes")
        digest = record.get("sha256")
        mode = record.get("mode")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode not in {0o644, 0o755}
        ):
            raise ControlManifestV2Error("v2 control file identity is invalid")
        path = root / expected_path
        resolved = path.resolve(strict=True)
        identity = resolved.stat()
        if (
            resolved != path
            or path.is_symlink()
            or not stat.S_ISREG(identity.st_mode)
            or identity.st_size != size
            or stat.S_IMODE(identity.st_mode) != mode
            or sha256_file(path) != digest
        ):
            raise ControlManifestV2Error(f"live v2 control differs: {expected_path}")
        committed = git(
            root, "show", f"{implementation_commit}:{expected_path}", text=False
        )
        assert isinstance(committed, bytes)
        if len(committed) != size or sha256_bytes(committed) != digest:
            raise ControlManifestV2Error(
                f"committed v2 control differs: {expected_path}"
            )
        normalized.append(dict(record))
    return {
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(raw),
        "manifest_sha256": sha256_bytes(raw),
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "current_head": current_head,
        "current_tree": current_tree,
        "root_clean": True,
        "base_v1": base_identity,
        "files": normalized,
    }
