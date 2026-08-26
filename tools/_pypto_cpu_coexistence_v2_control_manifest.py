#!/usr/bin/env python3
"""Validate the separately reviewed CPU coexistence policy-v2 manifest."""

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
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
GIT_ENVIRONMENT = {"PATH": "/usr/bin:/bin"}


class ControlManifestError(RuntimeError):
    """The CPU-v2 control bytes are absent, dirty, or unreviewed."""


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
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ControlManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_source(name: str, path: Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControlManifestError(f"CPU-v2 source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = load_source(
    "_pypto_cpu_coexistence_v2_contract_for_control",
    ROOT / "tools/_pypto_cpu_coexistence_v2_contract.py",
)
base_nvidia_control = contract.load_exact(
    "_pypto_cpu_v2_base_nvidia_control_for_control",
    ROOT / contract.BASE_NVIDIA_CONTROL_RELATIVE_PATH,
    contract.BASE_NVIDIA_CONTROL_SIZE,
    contract.BASE_NVIDIA_CONTROL_SHA256,
)


def git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=text,
        capture_output=True,
        env=GIT_ENVIRONMENT,
    ).stdout


def reject_bytecode_cache(root: Path) -> None:
    entries: list[str] = []
    for relative in (*contract.CONTROL_PATHS, "tests/test_pypto_cpu_coexistence_v2.py"):
        source = root / relative
        for suffix in (".pyc", ".pyo"):
            candidate = source.with_suffix(suffix)
            if candidate.exists() or candidate.is_symlink():
                entries.append(candidate.relative_to(root).as_posix())
        cache = source.parent / "__pycache__"
        if cache.is_dir():
            entries.extend(
                item.relative_to(root).as_posix()
                for item in cache.iterdir()
                if item.name.startswith(source.stem + ".")
                and item.name.endswith((".pyc", ".pyo"))
            )
    if entries:
        raise ControlManifestError(
            "CPU-v2 bytecode/cache entries are forbidden: "
            + ", ".join(sorted(set(entries)))
        )


def validate_base_dependencies(root: Path) -> dict[str, object]:
    dependencies = {
        "preflight": (
            root / contract.BASE_PREFLIGHT_RELATIVE_PATH,
            contract.BASE_PREFLIGHT_SIZE,
            contract.BASE_PREFLIGHT_SHA256,
        ),
        "run_isolated": (
            root / contract.BASE_ISOLATION_RELATIVE_PATH,
            contract.BASE_ISOLATION_SIZE,
            contract.BASE_ISOLATION_SHA256,
        ),
        "stop_run": (
            root / contract.BASE_STOP_RELATIVE_PATH,
            contract.BASE_STOP_SIZE,
            contract.BASE_STOP_SHA256,
        ),
        "nvidia_contract": (
            root / contract.BASE_NVIDIA_CONTRACT_RELATIVE_PATH,
            contract.BASE_NVIDIA_CONTRACT_SIZE,
            contract.BASE_NVIDIA_CONTRACT_SHA256,
        ),
        "nvidia_control": (
            root / contract.BASE_NVIDIA_CONTROL_RELATIVE_PATH,
            contract.BASE_NVIDIA_CONTROL_SIZE,
            contract.BASE_NVIDIA_CONTROL_SHA256,
        ),
        "nvidia_manifest": (
            root / contract.BASE_NVIDIA_MANIFEST_RELATIVE_PATH,
            contract.BASE_NVIDIA_MANIFEST_SIZE,
            contract.BASE_NVIDIA_MANIFEST_SHA256,
        ),
    }
    output: dict[str, object] = {}
    for name, (path, size, digest) in dependencies.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ControlManifestError(f"CPU-v2 exact base dependency differs: {name}")
        output[name] = {
            "path": path.relative_to(root).as_posix(),
            "bytes": size,
            "sha256": digest,
        }
    return output


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestError("CPU-v2 workspace contains a symlinked path")
    reject_bytecode_cache(root)
    base_dependencies = validate_base_dependencies(root)
    manifest_path = root / contract.MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestError("reviewed CPU-v2 control manifest is missing")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    if not isinstance(manifest, dict) or canonical_json(manifest) != raw:
        raise ControlManifestError("CPU-v2 manifest is not canonical JSON")
    expected_keys = {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "base_dependencies",
        "files",
    }
    if set(manifest) != expected_keys:
        raise ControlManifestError("CPU-v2 manifest key set differs")
    commit = manifest.get("implementation_commit")
    tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != contract.SCHEMA_VERSION
        or manifest.get("kind") != contract.POLICY_KIND
        or manifest.get("base_dependencies") != base_dependencies
        or not isinstance(commit, str)
        or COMMIT_PATTERN.fullmatch(commit) is None
        or not isinstance(tree, str)
        or COMMIT_PATTERN.fullmatch(tree) is None
        or str(git(root, "rev-parse", f"{commit}^{{tree}}")).strip() != tree
    ):
        raise ControlManifestError("CPU-v2 implementation identity differs")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current_head],
        cwd=root,
        check=False,
        env=GIT_ENVIRONMENT,
    ).returncode:
        raise ControlManifestError("CPU-v2 implementation is not an ancestor")
    if str(git(root, "status", "--porcelain=v1", "--untracked-files=all")):
        raise ControlManifestError("root control repository is not clean")
    try:
        base_control_identity = base_nvidia_control.validate_control_manifest(root)
    except Exception as error:
        raise ControlManifestError(
            "accepted NVIDIA v4 control identity differs"
        ) from error
    if (
        base_control_identity.get("manifest_sha256")
        != contract.BASE_NVIDIA_MANIFEST_SHA256
    ):
        raise ControlManifestError("accepted NVIDIA v4 manifest identity differs")
    if subprocess.run(
        ["git", "diff", "--quiet", f"{commit}..{current_head}", "--", *contract.CONTROL_PATHS],
        cwd=root,
        check=False,
        env=GIT_ENVIRONMENT,
    ).returncode:
        raise ControlManifestError("CPU-v2 controls changed after implementation")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(contract.CONTROL_PATHS):
        raise ControlManifestError("CPU-v2 control file set differs")
    normalized: list[dict[str, object]] = []
    for record, relative in zip(files, contract.CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("CPU-v2 control record is malformed")
        path = root / relative
        if (
            record.get("path") != relative
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.stat().st_size != record.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != record.get("mode")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ControlManifestError(f"live CPU-v2 control differs: {relative}")
        committed = git(root, "show", f"{commit}:{relative}", text=False)
        assert isinstance(committed, bytes)
        if (
            len(committed) != record["bytes"]
            or sha256_bytes(committed) != record["sha256"]
        ):
            raise ControlManifestError(f"committed CPU-v2 control differs: {relative}")
        normalized.append(dict(record))
    return {
        "manifest_path": contract.MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(raw),
        "manifest_sha256": sha256_bytes(raw),
        "implementation_commit": commit,
        "implementation_tree": tree,
        "current_head": current_head,
        "current_tree": str(git(root, "rev-parse", "HEAD^{tree}")).strip(),
        "root_clean": True,
        "base_dependencies": base_dependencies,
        "base_nvidia_control": base_control_identity,
        "files": normalized,
    }
