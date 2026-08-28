#!/usr/bin/env python3
"""Fail-closed verification for the vendored source-release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "vendor" / "source-lock.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATCH_FROM = re.compile(rb"^From ([0-9a-f]{40}) Mon Sep 17 00:00:00 2001$")
FULL_INDEX = re.compile(rb"^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: [0-7]{6})?$")


class SourceReleaseError(RuntimeError):
    """Raised when a source-release invariant is not satisfied."""


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Release tools never depend on, or mutate, the caller's global Git config.
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _run(
    command: list[str],
    *,
    cwd: pathlib.Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_git_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SourceReleaseError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result


def _git(cwd: pathlib.Path, *arguments: str) -> str:
    return _run(["git", *arguments], cwd=cwd).stdout.strip()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex40(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise SourceReleaseError(f"{label} must be a lowercase 40-digit Git SHA")
    return value


def _relative_path(root: pathlib.Path, value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise SourceReleaseError(f"{label} must be a non-empty relative path")
    portable = pathlib.PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or "\\" in value:
        raise SourceReleaseError(f"{label} is not a portable relative path: {value}")
    candidate = (root / pathlib.Path(*portable.parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SourceReleaseError(f"{label} escapes the release root: {value}")
    return candidate


def load_lock(lock_path: pathlib.Path = DEFAULT_LOCK) -> dict[str, Any]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceReleaseError(
            f"cannot read source lock {lock_path}: {error}"
        ) from error
    if not isinstance(lock, dict) or lock.get("schema") != 1:
        raise SourceReleaseError("source lock schema must be 1")
    _reject_absolute_strings(lock, "source_lock")
    if lock.get("release") != "qwen35-sm120-v1":
        raise SourceReleaseError("unexpected source release identity")
    repositories = lock.get("repositories")
    if not isinstance(repositories, dict):
        raise SourceReleaseError("source lock repositories must be an object")
    bundled = lock.get("bundled_repositories")
    if bundled != ["pypto", "tensor_ir"]:
        raise SourceReleaseError(
            "bundled_repositories must be exactly ['pypto', 'tensor_ir']"
        )
    for name in bundled:
        spec = repositories.get(name)
        if not isinstance(spec, dict):
            raise SourceReleaseError(f"missing repository specification: {name}")
        _validate_bundled_repository_spec(spec, name)
    sglang = repositories.get("sglang")
    if not isinstance(sglang, dict):
        raise SourceReleaseError("missing sglang repository specification")
    _validate_checkout_spec(sglang, "sglang")
    materialization = lock.get("materialization")
    if not isinstance(materialization, dict):
        raise SourceReleaseError("materialization must be an object")
    for name in ("pypto", "tensor_ir", "sglang"):
        _relative_path(pathlib.Path("."), materialization.get(name), name)
    submodules = lock.get("pypto_submodules")
    if not isinstance(submodules, list) or not submodules:
        raise SourceReleaseError("pypto_submodules must be a non-empty list")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, entry in enumerate(submodules):
        if not isinstance(entry, dict):
            raise SourceReleaseError(f"pypto_submodules[{index}] must be an object")
        name = entry.get("name")
        path = entry.get("path")
        url = entry.get("url")
        if not isinstance(name, str) or not name or name in seen_names:
            raise SourceReleaseError(f"invalid or duplicate submodule name: {name}")
        if not isinstance(path, str) or path in seen_paths:
            raise SourceReleaseError(f"invalid or duplicate submodule path: {path}")
        _relative_path(pathlib.Path("."), path, f"pypto_submodules[{index}].path")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise SourceReleaseError(f"submodule URL must use HTTPS: {url}")
        _require_hex40(entry.get("commit"), f"pypto_submodules[{index}].commit")
        _require_hex40(entry.get("tree"), f"pypto_submodules[{index}].tree")
        local_source = entry.get("local_source")
        if local_source is not None and local_source != "tensor_ir":
            raise SourceReleaseError(
                f"unsupported local submodule source: {local_source}"
            )
        seen_names.add(name)
        seen_paths.add(path)
    expected_gitlinks = {
        str(entry["path"]): str(entry["commit"]) for entry in submodules
    }
    if repositories["pypto"].get("gitlinks") != expected_gitlinks:
        raise SourceReleaseError(
            "pypto.gitlinks must exactly match the locked submodule commits"
        )
    if not isinstance(lock.get("environment_lock"), dict):
        raise SourceReleaseError("environment_lock must be an artifact object")
    environment_artifacts = lock.get("environment_artifacts")
    if not isinstance(environment_artifacts, list) or len(environment_artifacts) != 3:
        raise SourceReleaseError("environment_artifacts must lock exactly three files")
    return lock


def _reject_absolute_strings(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_strings(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, f"{label}[{index}]")
    elif isinstance(value, str) and pathlib.PurePath(value).is_absolute():
        raise SourceReleaseError(f"absolute host path is forbidden in {label}: {value}")


def _validate_checkout_spec(spec: Mapping[str, Any], label: str) -> None:
    _require_hex40(spec.get("head_commit"), f"{label}.head_commit")
    _require_hex40(spec.get("head_tree"), f"{label}.head_tree")
    origin = spec.get("origin_url")
    if not isinstance(origin, str) or not origin.startswith("https://"):
        raise SourceReleaseError(f"{label}.origin_url must be an HTTPS URL")


def _validate_bundled_repository_spec(spec: Mapping[str, Any], label: str) -> None:
    _validate_checkout_spec(spec, label)
    _require_hex40(spec.get("base_commit"), f"{label}.base_commit")
    _require_hex40(spec.get("base_tree"), f"{label}.base_tree")
    count = spec.get("commit_count")
    if not isinstance(count, int) or count <= 0:
        raise SourceReleaseError(f"{label}.commit_count must be positive")
    if spec.get("shallow_boundary") != spec.get("base_commit"):
        raise SourceReleaseError(
            f"{label}.shallow_boundary must equal its audited base commit"
        )
    for artifact_name in ("bundle", "patch_series"):
        if not isinstance(spec.get(artifact_name), dict):
            raise SourceReleaseError(f"{label}.{artifact_name} must be an object")


def _verify_regular_artifact(
    root: pathlib.Path, metadata: Mapping[str, Any], label: str
) -> pathlib.Path:
    path = _relative_path(root, metadata.get("path"), f"{label}.path")
    if not path.is_file() or path.is_symlink():
        raise SourceReleaseError(f"{label} is not a regular file: {path}")
    expected_size = metadata.get("bytes")
    if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        raise SourceReleaseError(
            f"{label} size differs: expected {expected_size}, got {path.stat().st_size}"
        )
    expected_hash = metadata.get("sha256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise SourceReleaseError(
            f"{label} SHA-256 differs: expected {expected_hash}, got {actual_hash}"
        )
    return path


def _bundle_clone_probe(
    root: pathlib.Path,
    bundle_path: pathlib.Path,
    spec: Mapping[str, Any],
    gitlinks: Mapping[str, str] | None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="source-bundle-probe-") as temporary:
        repository = pathlib.Path(temporary) / "repository"
        repository.mkdir()
        _git(repository, "init", "--quiet", "--initial-branch=release")
        # Both audited upstream checkouts started at a deliberate shallow
        # boundary.  Recording that boundary before fetch makes the bundle
        # self-contained without vendoring unrelated pre-base history.
        (repository / ".git" / "shallow").write_text(
            f"{spec['shallow_boundary']}\n", encoding="ascii"
        )
        head_ref = str(spec["bundle"]["head_ref"])
        _git(
            repository,
            "fetch",
            "--quiet",
            str(bundle_path),
            f"{head_ref}:refs/remotes/release/head",
            f"{spec['bundle']['base_ref']}:refs/tags/release-base",
        )
        _git(repository, "checkout", "--quiet", "--detach", spec["head_commit"])
        actual_tree = _git(repository, "rev-parse", "HEAD^{tree}")
        if actual_tree != spec["head_tree"]:
            raise SourceReleaseError(
                f"bundle clone tree differs: expected {spec['head_tree']}, got {actual_tree}"
            )
        _git(repository, "cat-file", "-e", f"{spec['base_commit']}^{{commit}}")
        base_tree = _git(repository, "rev-parse", f"{spec['base_commit']}^{{tree}}")
        if base_tree != spec["base_tree"]:
            raise SourceReleaseError(
                f"bundle base tree differs: expected {spec['base_tree']}, got {base_tree}"
            )
        count = int(
            _git(
                repository,
                "rev-list",
                "--count",
                f"{spec['base_commit']}..{spec['head_commit']}",
            )
        )
        if count != spec["commit_count"]:
            raise SourceReleaseError(
                f"bundle commit count differs: expected {spec['commit_count']}, got {count}"
            )
        object_walk = _git(
            repository,
            "rev-list",
            "--objects",
            "--missing=print",
            spec["head_commit"],
        )
        missing = [line for line in object_walk.splitlines() if line.startswith("?")]
        if missing:
            raise SourceReleaseError(
                f"bundle is missing reachable objects: {missing[:5]}"
            )
        for path, expected_commit in (gitlinks or {}).items():
            tree_entry = _git(repository, "ls-tree", spec["head_commit"], path)
            fields = tree_entry.split(maxsplit=3)
            if len(fields) != 4 or fields[:3] != ["160000", "commit", expected_commit]:
                raise SourceReleaseError(
                    f"bundle gitlink differs for {path}: expected {expected_commit}, "
                    f"got {tree_entry}"
                )
        if _git(repository, "rev-parse", "--is-shallow-repository") != "true":
            raise SourceReleaseError("bundle probe lost its declared shallow boundary")
        _git(repository, "fsck", "--connectivity-only", "--no-dangling")
    return {
        "base_tree": spec["base_tree"],
        "head_tree": spec["head_tree"],
        "commit_count": count,
        "reachable_objects": len(object_walk.splitlines()),
        "gitlinks_verified": len(gitlinks or {}),
        "shallow_boundary": spec["shallow_boundary"],
    }


def verify_bundle(
    root: pathlib.Path,
    repository_name: str,
    spec: Mapping[str, Any],
    *,
    clone_probe: bool = True,
    gitlinks: Mapping[str, str] | None = None,
) -> dict[str, object]:
    bundle = spec["bundle"]
    if not isinstance(bundle, dict):
        raise SourceReleaseError(f"{repository_name}.bundle is invalid")
    bundle_path = _verify_regular_artifact(root, bundle, f"{repository_name}.bundle")
    verification = _run(["git", "bundle", "verify", str(bundle_path)], cwd=root)
    heads = _git(root, "bundle", "list-heads", str(bundle_path)).splitlines()
    actual_heads: dict[str, str] = {}
    for line in heads:
        commit, ref = line.split(maxsplit=1)
        actual_heads[ref] = commit
    expected_heads = {
        str(bundle["head_ref"]): str(spec["head_commit"]),
        str(bundle["base_ref"]): str(spec["base_commit"]),
    }
    if actual_heads != expected_heads:
        raise SourceReleaseError(
            f"{repository_name} bundle refs differ: "
            f"expected {expected_heads}, got {actual_heads}"
        )
    materialization_probe = (
        _bundle_clone_probe(root, bundle_path, spec, gitlinks) if clone_probe else None
    )
    return {
        "path": str(bundle["path"]),
        "sha256": bundle["sha256"],
        "bytes": bundle["bytes"],
        "clone_probe": clone_probe,
        "materialization_probe": materialization_probe,
        "git_verify": bool(verification.returncode == 0),
    }


def _patch_source_commit(path: pathlib.Path) -> str:
    with path.open("rb") as source:
        first_line = source.readline().rstrip(b"\n")
    match = PATCH_FROM.fullmatch(first_line)
    if match is None:
        raise SourceReleaseError(f"invalid format-patch From line: {path}")
    return match.group(1).decode("ascii")


def _verify_full_indexes(path: pathlib.Path) -> None:
    for line in path.read_bytes().splitlines():
        if line.startswith(b"Binary files "):
            raise SourceReleaseError(
                f"non-replayable binary diff found instead of --binary output: {path}"
            )
        if not line.startswith(b"index "):
            continue
        match = FULL_INDEX.fullmatch(line)
        if match is None or len(match.group(1)) != 40 or len(match.group(2)) != 40:
            raise SourceReleaseError(f"non-full Git index line in {path}: {line!r}")


def verify_patch_series(
    root: pathlib.Path,
    repository_name: str,
    spec: Mapping[str, Any],
) -> dict[str, object]:
    metadata = spec["patch_series"]
    if not isinstance(metadata, dict):
        raise SourceReleaseError(f"{repository_name}.patch_series is invalid")
    manifest_metadata = {
        "path": metadata.get("manifest"),
        "sha256": metadata.get("manifest_sha256"),
        "bytes": metadata.get("manifest_bytes"),
    }
    manifest_path = _verify_regular_artifact(
        root, manifest_metadata, f"{repository_name}.series_manifest"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourceReleaseError(f"invalid series manifest {manifest_path}: {error}")
    expected_header = {
        "schema": 1,
        "repository": repository_name,
        "base_commit": spec["base_commit"],
        "base_tree": spec["base_tree"],
        "head_commit": spec["head_commit"],
        "head_tree": spec["head_tree"],
        "commit_count": spec["commit_count"],
        "format_patch_options": [
            "--binary",
            "--full-index",
            "--no-signature",
            "--numbered-files",
        ],
    }
    for name, expected in expected_header.items():
        if manifest.get(name) != expected:
            raise SourceReleaseError(
                f"{repository_name} series {name} differs: "
                f"expected {expected!r}, got {manifest.get(name)!r}"
            )
    patches = manifest.get("patches")
    if not isinstance(patches, list) or len(patches) != spec["commit_count"]:
        raise SourceReleaseError(
            f"{repository_name} series must contain {spec['commit_count']} patches"
        )
    patch_directory = _relative_path(
        root, metadata.get("directory"), f"{repository_name}.patch_series.directory"
    )
    if not patch_directory.is_dir() or patch_directory.is_symlink():
        raise SourceReleaseError(f"patch directory is invalid: {patch_directory}")
    expected_files = {manifest_path.resolve()}
    seen_commits: set[str] = set()
    for expected_order, patch_metadata in enumerate(patches, start=1):
        if not isinstance(patch_metadata, dict):
            raise SourceReleaseError("patch metadata entries must be objects")
        if patch_metadata.get("order") != expected_order:
            raise SourceReleaseError(
                f"{repository_name} patch order is not contiguous at {expected_order}"
            )
        expected_name = f"{expected_order:04d}.patch"
        if patch_metadata.get("file") != expected_name:
            raise SourceReleaseError(
                f"{repository_name} patch {expected_order} must be {expected_name}"
            )
        commit = _require_hex40(
            patch_metadata.get("source_commit"),
            f"{repository_name}.patches[{expected_order}].source_commit",
        )
        if commit in seen_commits:
            raise SourceReleaseError(f"duplicate source commit in series: {commit}")
        seen_commits.add(commit)
        patch_path = (patch_directory / expected_name).resolve()
        expected_files.add(patch_path)
        _verify_regular_artifact(
            root,
            {
                "path": patch_path.relative_to(root.resolve()).as_posix(),
                "sha256": patch_metadata.get("sha256"),
                "bytes": patch_metadata.get("bytes"),
            },
            f"{repository_name}.patches[{expected_order}]",
        )
        if _patch_source_commit(patch_path) != commit:
            raise SourceReleaseError(
                f"source commit does not match patch header: {patch_path}"
            )
        _verify_full_indexes(patch_path)
    actual_files = {
        path.resolve() for path in patch_directory.iterdir() if path.is_file()
    }
    if actual_files != expected_files:
        extras = sorted(str(path) for path in actual_files - expected_files)
        missing = sorted(str(path) for path in expected_files - actual_files)
        raise SourceReleaseError(
            f"{repository_name} patch directory differs: extras={extras}, missing={missing}"
        )
    return {
        "manifest": str(metadata["manifest"]),
        "patch_count": len(patches),
        "head_commit": spec["head_commit"],
        "head_tree": spec["head_tree"],
    }


def verify_release_artifacts(
    root: pathlib.Path = ROOT,
    lock: Mapping[str, Any] | None = None,
    *,
    clone_probe: bool = True,
) -> dict[str, object]:
    if lock is None:
        lock = load_lock(root / "vendor" / "source-lock.json")
    repositories = lock["repositories"]
    results: dict[str, object] = {}
    for name in lock["bundled_repositories"]:
        spec = repositories[name]
        gitlinks = spec.get("gitlinks") if name == "pypto" else None
        results[name] = {
            "bundle": verify_bundle(
                root,
                name,
                spec,
                clone_probe=clone_probe,
                gitlinks=gitlinks,
            ),
            "patch_series": verify_patch_series(root, name, spec),
        }
    environment_metadata = lock["environment_lock"]
    environment_path = _verify_regular_artifact(
        root, environment_metadata, "environment_lock"
    )
    try:
        environment_lock = json.loads(environment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourceReleaseError(f"invalid environment lock: {error}") from error
    _reject_absolute_strings(environment_lock, "environment_lock")
    if (
        environment_lock.get("schema") != 1
        or environment_lock.get("release") != lock["release"]
        or environment_lock.get("build_parallelism") != 24
    ):
        raise SourceReleaseError("environment lock identity or parallelism differs")
    if (
        environment_lock.get("source_identities", {}).get("sglang")
        != repositories["sglang"]["head_commit"]
    ):
        raise SourceReleaseError("environment and source locks disagree on SGLang")
    locked_environment_files: dict[str, Mapping[str, Any]] = {}
    for index, metadata in enumerate(lock["environment_artifacts"]):
        if not isinstance(metadata, dict):
            raise SourceReleaseError(f"environment_artifacts[{index}] is invalid")
        path = _verify_regular_artifact(
            root, metadata, f"environment_artifacts[{index}]"
        )
        locked_environment_files[path.name] = metadata
    expected_environment_files = {
        "python-artifacts.json",
        "conda-linux-64.lock",
        "python-requirements.lock",
    }
    if set(locked_environment_files) != expected_environment_files:
        raise SourceReleaseError(
            f"environment artifact set differs: {set(locked_environment_files)}"
        )
    artifact_path = root / str(
        locked_environment_files["python-artifacts.json"]["path"]
    )
    artifact_lock = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        artifact_lock.get("schema") != 1
        or artifact_lock.get("release") != lock["release"]
    ):
        raise SourceReleaseError("Python artifact lock identity differs")
    for label, filename in (
        ("conda_lock", "conda-linux-64.lock"),
        ("python_requirements_lock", "python-requirements.lock"),
    ):
        if artifact_lock.get(label) != locked_environment_files[filename]:
            raise SourceReleaseError(f"Python artifact lock disagrees on {label}")
    python_artifacts = artifact_lock.get("artifacts")
    if not isinstance(python_artifacts, dict) or set(python_artifacts) != {
        "torch",
        "triton",
    }:
        raise SourceReleaseError("Python artifact set must be exactly Torch and Triton")
    for name, metadata in python_artifacts.items():
        if (
            not isinstance(metadata, dict)
            or not str(metadata.get("url", "")).startswith("https://")
            or HEX64.fullmatch(str(metadata.get("sha256", ""))) is None
            or not isinstance(metadata.get("bytes"), int)
            or metadata["bytes"] <= 0
        ):
            raise SourceReleaseError(f"invalid locked Python artifact: {name}")
    conda_path = root / str(locked_environment_files["conda-linux-64.lock"]["path"])
    conda_lines = [
        line.strip()
        for line in conda_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not conda_lines or conda_lines[0] != "@EXPLICIT":
        raise SourceReleaseError("Conda lock is not explicit")
    if any(re.search(r"#[0-9a-f]{64}$", line) is None for line in conda_lines[1:]):
        raise SourceReleaseError("every Conda artifact must carry a SHA-256 fragment")
    requirements_path = root / str(
        locked_environment_files["python-requirements.lock"]["path"]
    )
    requirements_lines = requirements_path.read_text(encoding="utf-8").splitlines()
    requirements = [
        line for line in requirements_lines if line and not line.startswith(("#", " "))
    ]
    hashes = [line.strip() for line in requirements_lines if "--hash=sha256:" in line]
    if len(requirements) != 175 or len(hashes) != len(requirements):
        raise SourceReleaseError(
            "Python requirements must contain 175 one-hash pinned artifacts"
        )
    for name in ("torch", "triton"):
        expected_hash = python_artifacts[name]["sha256"]
        if f"--hash=sha256:{expected_hash}" not in hashes:
            raise SourceReleaseError(f"requirements do not lock {name} artifact hash")
    if (
        environment_lock.get("runtime_packages", {}).get("triton")
        != python_artifacts["triton"]["version"]
    ):
        raise SourceReleaseError("runtime and artifact locks disagree on Triton")
    if artifact_lock.get("known_metadata_conflicts") != []:
        raise SourceReleaseError("formal environment must be pip-check clean")
    results["environment"] = {
        "path": environment_metadata["path"],
        "sha256": environment_metadata["sha256"],
        "build_parallelism": 24,
        "locked_artifacts": len(conda_lines) - 1 + len(requirements),
        "fresh_creation_status": artifact_lock["fresh_creation"]["status"],
    }
    return results


def _checkout_identity(path: pathlib.Path) -> dict[str, str]:
    if not path.is_dir():
        raise SourceReleaseError(f"source checkout is missing: {path}")
    worktree = _git(path, "rev-parse", "--is-inside-work-tree")
    if worktree != "true":
        raise SourceReleaseError(f"not a Git worktree: {path}")
    status = _git(path, "status", "--porcelain=v2", "--untracked-files=all")
    if status:
        raise SourceReleaseError(f"source checkout is dirty: {path}: {status}")
    return {
        "head_commit": _git(path, "rev-parse", "HEAD"),
        "head_tree": _git(path, "rev-parse", "HEAD^{tree}"),
    }


def verify_checkout(
    path: pathlib.Path,
    spec: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    actual = _checkout_identity(path)
    expected = {
        "head_commit": str(spec["head_commit"]),
        "head_tree": str(spec["head_tree"]),
    }
    if actual != expected:
        raise SourceReleaseError(
            f"{label} checkout identity differs: expected {expected}, got {actual}"
        )
    return actual


def _gitmodules(path: pathlib.Path) -> dict[str, dict[str, str]]:
    modules_file = path / ".gitmodules"
    if not modules_file.is_file():
        return {}
    output = _git(
        path,
        "config",
        "--file",
        str(modules_file),
        "--get-regexp",
        r"^submodule\..*\.(path|url)$",
    )
    modules: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        key, value = line.split(maxsplit=1)
        match = re.fullmatch(r"submodule\.(.*)\.(path|url)", key)
        if match is None:
            raise SourceReleaseError(f"unexpected .gitmodules key: {key}")
        modules.setdefault(match.group(1), {})[match.group(2)] = value
    return modules


def verify_materialized_sources(
    destination: pathlib.Path,
    lock: Mapping[str, Any],
) -> dict[str, object]:
    materialization = lock["materialization"]
    repositories = lock["repositories"]
    pypto_path = destination / materialization["pypto"]
    tensor_ir_path = destination / materialization["tensor_ir"]
    sglang_path = destination / materialization["sglang"]
    result: dict[str, object] = {
        "pypto": verify_checkout(pypto_path, repositories["pypto"], "pypto"),
        "tensor_ir": verify_checkout(
            tensor_ir_path, repositories["tensor_ir"], "tensor_ir"
        ),
        "sglang": verify_checkout(sglang_path, repositories["sglang"], "sglang"),
    }
    locked_submodules = lock.get("pypto_submodules")
    if not isinstance(locked_submodules, list):
        raise SourceReleaseError("pypto_submodules must be a list")
    actual_modules = _gitmodules(pypto_path)
    expected_modules = {
        str(entry["name"]): {
            "path": str(entry["path"]),
            "url": str(entry["url"]),
        }
        for entry in locked_submodules
    }
    if actual_modules != expected_modules:
        raise SourceReleaseError(
            f"PyPTO .gitmodules differs: expected {expected_modules}, "
            f"got {actual_modules}"
        )
    submodule_results: dict[str, object] = {}
    for entry in locked_submodules:
        path = pypto_path / str(entry["path"])
        actual = verify_checkout(
            path,
            {
                "head_commit": entry["commit"],
                "head_tree": entry["tree"],
            },
            f"PyPTO submodule {entry['path']}",
        )
        gitlink = _git(pypto_path, "rev-parse", f"HEAD:{entry['path']}")
        if gitlink != entry["commit"]:
            raise SourceReleaseError(
                f"PyPTO gitlink differs for {entry['path']}: {gitlink}"
            )
        submodule_results[str(entry["path"])] = actual
    result["pypto_submodules"] = submodule_results
    return result


def replay_patch_series(
    root: pathlib.Path,
    repository_name: str,
    spec: Mapping[str, Any],
) -> dict[str, object]:
    metadata = spec["patch_series"]
    manifest_path = _relative_path(
        root, metadata["manifest"], f"{repository_name}.patch_series.manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch_directory = _relative_path(
        root, metadata["directory"], f"{repository_name}.patch_series.directory"
    )
    bundle_path = _relative_path(
        root, spec["bundle"]["path"], f"{repository_name}.bundle.path"
    )
    with tempfile.TemporaryDirectory(prefix=f"{repository_name}-patch-replay-") as tmp:
        repository = pathlib.Path(tmp) / "repository"
        repository.mkdir()
        _git(repository, "init", "--quiet", "--initial-branch=release")
        (repository / ".git" / "shallow").write_text(
            f"{spec['shallow_boundary']}\n", encoding="ascii"
        )
        _git(
            repository,
            "fetch",
            "--quiet",
            str(bundle_path),
            f"{spec['bundle']['base_ref']}:refs/tags/replay-base",
        )
        _git(repository, "checkout", "--quiet", "--detach", spec["base_commit"])
        patch_paths = [
            str(patch_directory / entry["file"]) for entry in manifest["patches"]
        ]
        _run(
            [
                "git",
                "-c",
                "user.name=PyPTO Source Release",
                "-c",
                "user.email=source-release.invalid",
                "am",
                "--quiet",
                "--committer-date-is-author-date",
                *patch_paths,
            ],
            cwd=repository,
        )
        actual_tree = _git(repository, "rev-parse", "HEAD^{tree}")
        if actual_tree != spec["head_tree"]:
            raise SourceReleaseError(
                f"{repository_name} replay tree differs: "
                f"expected {spec['head_tree']}, got {actual_tree}"
            )
        count = int(_git(repository, "rev-list", "--count", "replay-base..HEAD"))
        if count != spec["commit_count"]:
            raise SourceReleaseError(
                f"{repository_name} replay count differs: "
                f"expected {spec['commit_count']}, got {count}"
            )
        if _git(repository, "status", "--porcelain=v2", "--untracked-files=all"):
            raise SourceReleaseError(f"{repository_name} replay is dirty")
    return {
        "repository": repository_name,
        "patch_count": spec["commit_count"],
        "result_tree": spec["head_tree"],
    }


def replay_all_patch_series(
    root: pathlib.Path,
    lock: Mapping[str, Any],
) -> dict[str, object]:
    return {
        name: replay_patch_series(root, name, lock["repositories"][name])
        for name in lock["bundled_repositories"]
    }


def _json_dump(value: Mapping[str, object]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--lock", type=pathlib.Path)
    parser.add_argument("--replay-patches", action="store_true")
    parser.add_argument("--sources", type=pathlib.Path)
    parser.add_argument(
        "--skip-clone-probe",
        action="store_true",
        help="Skip the fresh-fetch bundle probe (intended only for focused unit tests).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    lock_path = (
        args.lock.resolve() if args.lock else root / "vendor" / "source-lock.json"
    )
    try:
        lock = load_lock(lock_path)
        report: dict[str, object] = {
            "schema": 1,
            "release": lock["release"],
            "artifacts": verify_release_artifacts(
                root, lock, clone_probe=not args.skip_clone_probe
            ),
        }
        if args.replay_patches:
            report["patch_replay"] = replay_all_patch_series(root, lock)
        if args.sources is not None:
            report["sources"] = verify_materialized_sources(
                args.sources.resolve(), lock
            )
    except SourceReleaseError as error:
        print(f"source release verification failed: {error}", file=sys.stderr)
        return 1
    _json_dump(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
