#!/usr/bin/env python3
"""Validate the reviewed v1 frontend-SM120 root-control manifest."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "pypto-frontend-vector-add-sm120-controls-v1"
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_frontend_vector_add_sm120_v1.json")
PARENT_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_nvidia_executable_sm120_v4.json"
)
PARENT_MANIFEST_SIZE = 1_569
PARENT_MANIFEST_SHA256 = (
    "a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf"
)
CONTROL_PATHS = (
    "benchmarks/operators/pypto_frontend_vector_add_sm120.py",
    "tools/_pypto_frontend_vector_add_sm120_contract.py",
    "tools/_pypto_frontend_sm120_control_manifest.py",
    "tools/run_pypto_frontend_sm120_isolated.py",
    "tools/finalize_pypto_frontend_vector_add_sm120.py",
    "tools/preflight.py",
    "tools/run_isolated.py",
    "tools/stop_run.py",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ControlManifestError(RuntimeError):
    """The checked-out smoke controls are not the reviewed implementation."""


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
            raise ControlManifestError(f"duplicate control-manifest key: {key}")
        output[key] = value
    return output


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=text,
        capture_output=True,
    )
    return result.stdout


def validate_parent_control_manifest(root: Path) -> dict[str, object]:
    """Bind the accepted v4 transaction used as this lane's primitive."""

    path = root / PARENT_MANIFEST_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ControlManifestError("accepted v4 parent control manifest is missing")
    raw = path.read_bytes()
    if len(raw) != PARENT_MANIFEST_SIZE or sha256_bytes(raw) != PARENT_MANIFEST_SHA256:
        raise ControlManifestError("accepted v4 parent control manifest differs")
    try:
        parent = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError("accepted v4 parent manifest is not JSON") from error
    if (
        not isinstance(parent, dict)
        or canonical_json(parent) != raw
        or set(parent)
        != {
            "schema_version",
            "kind",
            "implementation_commit",
            "implementation_tree",
            "files",
        }
        or parent.get("schema_version") != 4
        or parent.get("kind") != "pypto-nvidia-executable-sm120-controls-v4"
        or not isinstance(parent.get("files"), list)
    ):
        raise ControlManifestError("accepted v4 parent manifest schema differs")
    records = parent["files"]
    assert isinstance(records, list)
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    primitive_paths = CONTROL_PATHS[-3:]
    if set(primitive_paths) - set(by_path):
        raise ControlManifestError("accepted v4 parent omits an isolation primitive")
    return {
        "path": PARENT_MANIFEST_RELATIVE_PATH.as_posix(),
        "bytes": len(raw),
        "sha256": PARENT_MANIFEST_SHA256,
        "schema_version": parent["schema_version"],
        "kind": parent["kind"],
        "implementation_commit": parent["implementation_commit"],
        "implementation_tree": parent["implementation_tree"],
        "primitive_files": [dict(by_path[name]) for name in primitive_paths],
    }


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestError("workspace contains a symlinked path")
    parent_identity = validate_parent_control_manifest(root)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestError(
            "reviewed frontend smoke control manifest is missing"
        )
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError("control manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or canonical_json(manifest) != manifest_raw:
        raise ControlManifestError("control manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "files",
    }:
        raise ControlManifestError("control manifest top-level schema differs")
    implementation_commit = manifest.get("implementation_commit")
    implementation_tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or not isinstance(implementation_commit, str)
        or COMMIT_PATTERN.fullmatch(implementation_commit) is None
        or not isinstance(implementation_tree, str)
        or COMMIT_PATTERN.fullmatch(implementation_tree) is None
    ):
        raise ControlManifestError("control manifest implementation identity differs")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(CONTROL_PATHS):
        raise ControlManifestError("control manifest file set is incomplete")
    if str(git(root, "rev-parse", f"{implementation_commit}^{{tree}}")).strip() != (
        implementation_tree
    ):
        raise ControlManifestError("implementation commit/tree join failed")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    current_tree = str(git(root, "rev-parse", "HEAD^{tree}")).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, current_head],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ControlManifestError("reviewed implementation is not an ancestor")
    dirty = str(git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    if dirty:
        raise ControlManifestError("root control repository is not clean")
    changed_controls = subprocess.run(
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
    if changed_controls.returncode != 0:
        raise ControlManifestError(
            "reviewed control files changed after implementation"
        )

    normalized_files: list[dict[str, object]] = []
    parent_primitives = {
        record["path"]: record for record in parent_identity["primitive_files"]
    }
    for record, expected_path in zip(files, CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("control manifest file record is malformed")
        if record.get("path") != expected_path:
            raise ControlManifestError("control manifest file order differs")
        if (
            expected_path in parent_primitives
            and record != parent_primitives[expected_path]
        ):
            raise ControlManifestError(
                f"isolation primitive differs from accepted v4: {expected_path}"
            )
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
            raise ControlManifestError("control manifest file identity is invalid")
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
            raise ControlManifestError(f"live control file differs: {expected_path}")
        committed = git(
            root, "show", f"{implementation_commit}:{expected_path}", text=False
        )
        assert isinstance(committed, bytes)
        if len(committed) != size or sha256_bytes(committed) != digest:
            raise ControlManifestError(
                f"implementation commit blob differs: {expected_path}"
            )
        normalized_files.append(dict(record))
    return {
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "current_head": current_head,
        "current_tree": current_tree,
        "root_clean": True,
        "parent_control_manifest": parent_identity,
        "files": normalized_files,
    }
