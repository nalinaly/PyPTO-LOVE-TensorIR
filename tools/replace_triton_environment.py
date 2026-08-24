#!/usr/bin/env python3
"""Transactionally replace the project environment's editable Triton.

The transaction is intentionally narrower than a package manager.  It accepts
only the exact workspace wheel already accepted by ``audit_triton_wheel.py``,
``probe_triton_wheel.py`` and the finalized exclusive-run SM120 smoke.
Installation is
delegated to the probe tool's standard-library Wheel/RECORD installer; this
module never invokes pip and never performs network I/O.

``--plan`` is read-only.  It validates every SHA anchor, inventories the old
editable distribution and reports the exact old/new ownership sets without
creating the backup root or evidence file.  ``--apply`` first seals a
content-addressed backup below the workspace build tree.  It then deletes only
prefix-local paths owned by the old RECORD, installs the audited wheel, and
audits the result in new Python processes.  Any failure before atomic success
evidence publication removes only audit-owned new paths and restores the old
bytes, modes and symlink targets.

Before the first deletion, ``--apply`` durably publishes a backup manifest and
phase journal.  INT/TERM/HUP enter rollback; ``--recover`` resumes an
interrupted rollback after SIGKILL/WSL termination, and ``--rollback`` performs
the same idempotent manifest-driven restoration explicitly.  Apply/recovery
hold one workspace flock and reject foreign processes using the prefix or old
external libtriton at every action boundary.

The external editable source and its native objects are evidence/backup inputs
only.  They are never mutation targets.  The tool never signals another
process and never touches user caches, upstream source, or another environment.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import hashlib
import importlib.util
import io
import json
import os
import errno
import fcntl
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import tempfile
import urllib.parse
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
REPLACEMENT_NAME = "triton-project-environment-replacement"
GPU_SMOKE_NAME = "reference-only-triton-sm120"
INSTALL_METHOD = "probe-stdlib-safe-wheel-installer"
BACKUP_NAME = "triton-editable-environment-backup"
ENVIRONMENT_LOCK_NAME = "environment-pypto-nvidia.lock"
INHERITED_LOCK_ENVIRONMENT = {
    "fd": "PYPTO_ENVIRONMENT_LOCK_FD",
    "mode": "PYPTO_ENVIRONMENT_LOCK_MODE",
    "path": "PYPTO_ENVIRONMENT_LOCK_PATH",
    "dev": "PYPTO_ENVIRONMENT_LOCK_DEV",
    "ino": "PYPTO_ENVIRONMENT_LOCK_INO",
    "controller_pid": "PYPTO_ENVIRONMENT_LOCK_CONTROLLER_PID",
    "controller_start_ticks": "PYPTO_ENVIRONMENT_LOCK_CONTROLLER_START_TICKS",
}


def _load_probe_module():
    """Load the sibling tool even when this file is imported by path in tests."""

    name = "_pypto_replace_triton_probe_tool"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("probe_triton_wheel.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the stdlib wheel installer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


class ReplacementError(RuntimeError):
    """An ownership, provenance, transaction or rollback invariant failed."""

    def __init__(
        self,
        message: str,
        *,
        rollback: dict[str, object] | None = None,
        evidence_sha256: str | None = None,
        evidence_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rollback = rollback
        self.evidence_sha256 = evidence_sha256
        self.evidence_status = evidence_status


@dataclass(frozen=True, slots=True)
class ReplacementLimits:
    max_evidence_bytes: int = 64 << 20
    max_record_bytes: int = 16 << 20
    max_record_entries: int = 200_000
    max_backup_file_bytes: int = 4 << 30
    max_backup_bytes: int = 16 << 30
    max_output_bytes: int = 8 << 20
    timeout_seconds: int = 180


@dataclass(frozen=True, slots=True)
class ReplacementRequest:
    workspace: Path
    prefix: Path
    wheel: Path
    wheel_audit_evidence: Path
    expected_wheel_audit_evidence_sha256: str
    wheel_probe_evidence: Path
    expected_wheel_probe_evidence_sha256: str
    gpu_smoke_evidence: Path
    expected_gpu_smoke_evidence_sha256: str
    environment_lock: Path
    expected_environment_lock_sha256: str
    backup_root: Path
    evidence: Path


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    kind: str
    mode: int
    size: int
    sha256: str | None
    link_target_b64: str | None
    record_digest: str
    record_size: str
    roles: tuple[str, ...]

    def document(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "mode": f"{self.mode:04o}",
            "path": self.path,
            "record_digest": self.record_digest,
            "record_size": self.record_size,
            "roles": list(self.roles),
            "size": self.size,
        }
        if self.sha256 is not None:
            value["sha256"] = self.sha256
        if self.link_target_b64 is not None:
            value["link_target_b64"] = self.link_target_b64
        return value


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    path: str
    mode: int

    def document(self) -> dict[str, object]:
        return {"mode": f"{self.mode:04o}", "path": self.path}


@dataclass(frozen=True, slots=True)
class NativeEntry:
    path: str
    mode: int
    size: int
    sha256: str

    def document(self) -> dict[str, object]:
        return {
            "kind": "regular",
            "mode": f"{self.mode:04o}",
            "path": self.path,
            "role": "external-native-observation-only",
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class OldInventory:
    distribution_version: str
    dist_info: str
    direct_url: dict[str, object]
    editable_source: str
    package_root: str
    finder_module: str
    external_source_identity: dict[str, object]
    files: tuple[FileEntry, ...]
    removable_directories: tuple[DirectoryEntry, ...]
    native_files: tuple[NativeEntry, ...]

    def document(self) -> dict[str, object]:
        value = {
            "direct_url": self.direct_url,
            "dist_info": self.dist_info,
            "distribution_version": self.distribution_version,
            "editable_source": self.editable_source,
            "external_source_identity": self.external_source_identity,
            "files": [entry.document() for entry in self.files],
            "files_count": len(self.files),
            "finder_module": self.finder_module,
            "native_files": [entry.document() for entry in self.native_files],
            "native_files_count": len(self.native_files),
            "package_root": self.package_root,
            "removable_directories": [
                entry.document() for entry in self.removable_directories
            ],
        }
        value["sha256"] = sha256_bytes(canonical_json(value).encode("ascii"))
        return value


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    environment_lock: dict[str, object]
    environment_lock_raw: bytes
    audit_document: dict[str, object]
    audit_raw: bytes
    anchor: dict[str, object]
    probe_document: dict[str, object]
    probe_raw: bytes
    smoke_document: dict[str, object]
    smoke_raw: bytes
    versions_lock: dict[str, str]
    versions_lock_path: Path


@dataclass(frozen=True, slots=True)
class PreparedReplacement:
    request: ReplacementRequest
    workspace: Path
    prefix: Path
    python: Path
    scheme: object
    inputs: ValidatedInputs
    old_inventory: OldInventory
    new_paths: tuple[str, ...]
    new_directories: tuple[str, ...]
    preexisting_directories: tuple[str, ...]
    torch_before: dict[str, object]
    plan_document: dict[str, object]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReplacementError(
            f"{description} must be 64 lowercase hexadecimal characters"
        )
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplacementError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    return (allow_root and path == root) or root in path.parents


def _workspace_relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _prefix_relative(path: Path, prefix: Path) -> str:
    return path.relative_to(prefix).as_posix()


def _safe_relative(value: str, description: str) -> PurePosixPath:
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ReplacementError(f"{description} path is unsafe: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".") for part in path.parts)
        or (path.parts and re.match(r"^[A-Za-z]:", path.parts[0]))
    ):
        raise ReplacementError(f"{description} path is non-canonical: {value!r}")
    return path


def _path_without_symlink_parents(path: Path, root: Path) -> None:
    lexical = _absolute_lexical(path)
    if not _is_within(lexical, root, allow_root=True):
        raise ReplacementError(f"path escaped its mutation root: {lexical}")
    relative = lexical.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ReplacementError(f"mutation path has a symlink parent: {current}")
        if not stat.S_ISDIR(mode):
            raise ReplacementError(f"mutation path parent is not a directory: {current}")


def _read_regular_file_once(
    path: Path, description: str, *, max_bytes: int
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReplacementError(
            f"cannot open {description} as a non-symlink: {path}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ReplacementError(f"{description} is not a regular file: {path}")
        if file_stat.st_size > max_bytes:
            raise ReplacementError(f"{description} exceeds its size limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1 << 20, max_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ReplacementError(f"{description} exceeds its size limit: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_canonical_json(
    path: Path, description: str, *, max_bytes: int
) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_file_once(path, description, max_bytes=max_bytes)
    try:
        text = raw.decode("utf-8", errors="strict")
        document = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplacementError(f"{description} is not strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ReplacementError(f"{description} root must be an object")
    if text != canonical_json(document):
        raise ReplacementError(f"{description} is not canonical JSON")
    return document, raw


def _require_workspace(path: Path) -> Path:
    if not path.is_absolute():
        raise ReplacementError("--workspace must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError(f"workspace is absent: {lexical}") from error
    if lexical != resolved or not resolved.is_dir() or resolved.is_symlink():
        raise ReplacementError("workspace must be a real non-symlink directory")
    return resolved


def _require_workspace_file(path: Path, workspace: Path, description: str) -> Path:
    if not path.is_absolute():
        raise ReplacementError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError(f"{description} is absent: {lexical}") from error
    if (
        lexical != resolved
        or not _is_within(resolved, workspace)
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise ReplacementError(
            f"{description} must be a regular, non-symlink workspace file"
        )
    return resolved


def _require_prefix(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise ReplacementError("environment prefix must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError(f"environment prefix is absent: {lexical}") from error
    if (
        lexical != resolved
        or resolved.is_symlink()
        or not resolved.is_dir()
        or not _is_within(resolved, workspace)
    ):
        raise ReplacementError("environment prefix must be a real workspace directory")
    return resolved


def _require_fresh_build_root(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise ReplacementError("backup root must be absolute")
    lexical = _absolute_lexical(path)
    builds = workspace / "builds"
    if not builds.is_dir() or builds.is_symlink():
        raise ReplacementError("workspace builds directory is absent or unsafe")
    if not _is_within(lexical, builds):
        raise ReplacementError("backup root must be below workspace/builds")
    parent = lexical.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ReplacementError(f"backup-root parent is absent: {parent}") from error
    if parent != resolved_parent or not _is_within(parent, builds, allow_root=True):
        raise ReplacementError("backup-root parent must be a real build directory")
    if lexical.exists() or lexical.is_symlink():
        raise ReplacementError(f"backup root is not fresh: {lexical}")
    return lexical


def _require_evidence_output(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise ReplacementError("evidence output must be absolute")
    lexical = _absolute_lexical(path)
    parent = lexical.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ReplacementError(f"evidence parent is absent: {parent}") from error
    if parent != resolved_parent or not _is_within(parent, workspace, allow_root=True):
        raise ReplacementError("evidence parent must be a real workspace directory")
    if lexical.exists() or lexical.is_symlink():
        raise ReplacementError(f"evidence already exists: {lexical}")
    return lexical


def _identity(path: Path, workspace: Path) -> dict[str, object]:
    return {
        "path": _workspace_relative(path, workspace),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _require_identity(
    record: object,
    *,
    path: Path,
    workspace: Path,
    description: str,
    expected_sha256: str | None = None,
) -> None:
    if not isinstance(record, dict):
        raise ReplacementError(f"{description} identity is absent")
    actual = _identity(path, workspace)
    if record != actual:
        raise ReplacementError(f"{description} identity/path differs from accepted evidence")
    if expected_sha256 is not None and actual["sha256"] != expected_sha256:
        raise ReplacementError(f"{description} differs from its SHA256 anchor")


def _validate_probe_evidence(
    document: dict[str, object],
    *,
    request: ReplacementRequest,
    workspace: Path,
    environment_lock_sha256: str,
    audit_sha256: str,
    wheel_sha256: str,
) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("probe") != probe.PROBE_NAME
        or document.get("acceptance") != "accepted"
    ):
        raise ReplacementError("wheel-probe evidence is not accepted exact-probe evidence")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplacementError("wheel-probe input identities are absent")
    base_python = inputs.get("base_python")
    if not isinstance(base_python, dict):
        raise ReplacementError("wheel-probe base-Python identity is absent")
    base_path_value = base_python.get("path")
    if not isinstance(base_path_value, str):
        raise ReplacementError("wheel-probe base-Python path is absent")
    base_relative = _safe_relative(base_path_value, "probe base Python")
    _require_identity(
        base_python,
        path=workspace.joinpath(*base_relative.parts),
        workspace=workspace,
        description="probe base Python",
        expected_sha256=probe.BASE_PYTHON_SHA256,
    )
    _require_identity(
        inputs.get("environment_lock"),
        path=request.environment_lock,
        workspace=workspace,
        description="probe ENVIRONMENT.lock",
        expected_sha256=environment_lock_sha256,
    )
    wheel = inputs.get("wheel")
    if not isinstance(wheel, dict):
        raise ReplacementError("wheel-probe wheel identity is absent")
    audit_identity = wheel.get("audit_evidence")
    _require_identity(
        audit_identity,
        path=request.wheel_audit_evidence,
        workspace=workspace,
        description="probe wheel-audit evidence",
        expected_sha256=audit_sha256,
    )
    expected_wheel = _identity(request.wheel, workspace)
    if {
        "filename": wheel.get("filename"),
        "path": wheel.get("path"),
        "sha256": wheel.get("sha256"),
        "size": wheel.get("size"),
    } != {"filename": request.wheel.name, **expected_wheel}:
        raise ReplacementError("wheel-probe evidence names different wheel bytes")
    if expected_wheel["sha256"] != wheel_sha256:
        raise ReplacementError("wheel-probe wheel SHA differs from the accepted audit")

    installation = document.get("installation")
    if (
        not isinstance(installation, dict)
        or installation.get("fresh_prefix") is not True
        or installation.get("method") != "stdlib-safe-wheel-installer"
    ):
        raise ReplacementError("wheel-probe did not use the accepted fresh stdlib install")
    runtime = document.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("gpu_execution") is not False:
        raise ReplacementError("wheel-probe runtime scope is not CPU-only")
    processes = runtime.get("processes")
    triton_key = runtime.get("triton_key")
    if (
        runtime.get("processes_count") != 2
        or not isinstance(processes, list)
        or len(processes) != 2
        or processes[0] != processes[1]
        or not isinstance(triton_key, dict)
        or triton_key.get("stable_across_processes") is not True
        or not isinstance(triton_key.get("value"), str)
        or not triton_key["value"]
        or triton_key.get("sha256")
        != sha256_bytes(str(triton_key["value"]).encode("utf-8"))
    ):
        raise ReplacementError("wheel-probe lacks two stable independent runtime reports")


def _validate_smoke_provenance_paths(
    provenance: dict[str, object], probe_prefix: Path
) -> None:
    modules = provenance.get("module_paths")
    maps = provenance.get("libtriton_maps")
    if not isinstance(modules, dict) or not modules:
        raise ReplacementError("GPU-smoke module-path provenance is absent")
    if not isinstance(maps, list) or not maps:
        raise ReplacementError("GPU-smoke libtriton map provenance is absent")
    values: list[str] = []
    for name, paths in modules.items():
        if (
            not isinstance(name, str)
            or (name != "triton" and not name.startswith("triton."))
            or not isinstance(paths, list)
            or not paths
            or any(not isinstance(item, str) for item in paths)
        ):
            raise ReplacementError("GPU-smoke module-path provenance is malformed")
        values.extend(paths)
    if any(not isinstance(item, str) for item in maps):
        raise ReplacementError("GPU-smoke libtriton map provenance is malformed")
    values.extend(maps)
    for value in values:
        marker = "$PROBE_PREFIX/"
        if not value.startswith(marker):
            raise ReplacementError("GPU-smoke Triton path is not probe-normalized")
        relative = _safe_relative(
            value.removeprefix(marker), "GPU-smoke probe provenance"
        )
        try:
            resolved = probe_prefix.joinpath(*relative.parts).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ReplacementError("GPU-smoke provenance path is absent") from error
        if not _is_within(resolved, probe_prefix):
            raise ReplacementError(f"GPU-smoke Triton provenance escaped probe: {resolved}")


def _validate_finalized_smoke_run(
    document: dict[str, object], *, workspace: Path
) -> dict[str, object]:
    provisional_identity = document.get("provisional_evidence")
    exclusive = document.get("exclusive_run")
    if not isinstance(provisional_identity, dict) or not isinstance(exclusive, dict):
        raise ReplacementError("GPU-smoke is not finalizer-bound evidence")
    provisional_path_value = provisional_identity.get("path")
    if not isinstance(provisional_path_value, str):
        raise ReplacementError("GPU-smoke provisional path is absent")
    provisional_path = workspace.joinpath(
        *_safe_relative(provisional_path_value, "provisional smoke").parts
    )
    _require_identity(
        provisional_identity,
        path=provisional_path,
        workspace=workspace,
        description="provisional GPU smoke",
    )
    provisional, _ = load_canonical_json(
        provisional_path,
        "provisional GPU-smoke evidence",
        max_bytes=64 << 20,
    )
    if (
        provisional.get("acceptance")
        != "gpu-execution-complete-awaiting-run-finalization"
    ):
        raise ReplacementError("linked provisional GPU smoke has wrong status")
    derived = dict(document)
    derived.pop("exclusive_run", None)
    derived.pop("provisional_evidence", None)
    derived["acceptance"] = provisional["acceptance"]
    if derived != provisional:
        raise ReplacementError("final GPU smoke differs from linked provisional bytes")

    run_id = exclusive.get("run_id")
    process_identity = exclusive.get("process")
    preflight_identity = exclusive.get("preflight")
    finalizer_identity = exclusive.get("finalizer")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}", run_id)
        is None
        or not isinstance(process_identity, dict)
        or not isinstance(preflight_identity, dict)
        or not isinstance(finalizer_identity, dict)
        or exclusive.get("gpu_benchmark_abort") is not None
        or exclusive.get("coexistence_pauses") != []
    ):
        raise ReplacementError("GPU-smoke exclusive-run finalization is malformed")
    run_root = workspace / "runs" / run_id
    process_path = run_root / "process.json"
    preflight_path = run_root / "preflight.json"
    if process_identity.get("path") != _workspace_relative(process_path, workspace):
        raise ReplacementError("final GPU-smoke process path is not its exact run")
    if preflight_identity.get("path") != _workspace_relative(preflight_path, workspace):
        raise ReplacementError("final GPU-smoke preflight path is not its exact run")
    process, process_raw = load_canonical_json(
        process_path, "GPU-smoke process metadata", max_bytes=16 << 20
    )
    preflight, preflight_raw = load_canonical_json(
        preflight_path, "GPU-smoke preflight", max_bytes=16 << 20
    )
    process_sha = sha256_bytes(process_raw)
    preflight_sha = sha256_bytes(preflight_raw)
    command = process.get("command")
    if (
        process_identity.get("document_sha256") != process_sha
        or process_identity.get("status") != "exited"
        or process_identity.get("return_code") != 0
        or process_identity.get("mode") != "gpu-benchmark"
        or not isinstance(command, list)
        or any(not isinstance(item, str) for item in command)
        or process_identity.get("command_sha256")
        != sha256_bytes(canonical_json(command).encode("ascii"))
    ):
        raise ReplacementError("GPU-smoke finalized process identity is inconsistent")
    process_preflight = process.get("preflight")
    if (
        process.get("run_id") != run_id
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("mode") != "gpu-benchmark"
        or process.get("gpu_benchmark_abort") is not None
        or process.get("coexistence_pauses", []) != []
        or not isinstance(process.get("coexistence"), dict)
        or process["coexistence"].get("requested") is not False
        or not isinstance(process_preflight, dict)
        or process_preflight.get("path") != str(preflight_path)
        or process_preflight.get("sha256") != preflight_sha
    ):
        raise ReplacementError("GPU-smoke process did not exit exclusively with rc0")
    if (
        preflight_identity.get("document_sha256") != preflight_sha
        or preflight_identity.get("ok") is not True
        or preflight_identity.get("mode") != "gpu-benchmark"
        or preflight.get("ok") is not True
        or preflight.get("mode") != "gpu-benchmark"
        or preflight.get("nvidia_compute_pids") != []
        or preflight.get("protected_heavy_processes") != []
        or preflight.get("protected_cpu_only_coexistence_requested") is not False
    ):
        raise ReplacementError("GPU-smoke finalized preflight was not exclusive")
    runner = workspace / "benchmarks/operators/triton_reference_sm120.py"
    if not any(str(runner) in item for item in command) or not any(
        str(provisional_path) in item for item in command
    ):
        raise ReplacementError("GPU-smoke process command did not bind runner/evidence")
    run_context = provisional.get("run_context")
    if (
        not isinstance(run_context, dict)
        or run_context.get("run_id") != run_id
        or run_context.get("mode") != "gpu-benchmark"
        or run_context.get("pgid") != process.get("pgid")
        or not isinstance(run_context.get("pid"), int)
        or isinstance(run_context.get("pid"), bool)
        or run_context.get("provisional_evidence_path")
        != _workspace_relative(provisional_path, workspace)
        or run_context.get("preflight")
        != {
            "path": _workspace_relative(preflight_path, workspace),
            "sha256": preflight_sha,
            "size": len(preflight_raw),
        }
    ):
        raise ReplacementError("provisional GPU-smoke run_context does not join final run")
    finalizer_path_value = finalizer_identity.get("path")
    if not isinstance(finalizer_path_value, str):
        raise ReplacementError("GPU-smoke finalizer identity is absent")
    finalizer_path = workspace.joinpath(
        *_safe_relative(finalizer_path_value, "smoke finalizer").parts
    )
    _require_identity(
        finalizer_identity,
        path=finalizer_path,
        workspace=workspace,
        description="GPU-smoke finalizer",
    )
    return provisional


def _validate_gpu_smoke_evidence(
    document: dict[str, object],
    *,
    request: ReplacementRequest,
    workspace: Path,
    probe_document: dict[str, object],
    probe_sha256: str,
    anchor: dict[str, object],
) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("smoke") != GPU_SMOKE_NAME
        or document.get("acceptance") != "accepted"
    ):
        raise ReplacementError("GPU-smoke evidence is not accepted reference-only evidence")
    _validate_finalized_smoke_run(document, workspace=workspace)
    if document.get("scope") != {
        "coverage_result": False,
        "performance_result": False,
        "provider": "triton",
        "pypto_kernel": False,
        "reference_only": True,
    }:
        raise ReplacementError("GPU-smoke scope is not reference-only Triton")
    inputs = document.get("inputs")
    probe_inputs = probe_document.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(probe_inputs, dict):
        raise ReplacementError("GPU-smoke input provenance is absent")
    _require_identity(
        inputs.get("probe_evidence"),
        path=request.wheel_probe_evidence,
        workspace=workspace,
        description="GPU-smoke probe evidence",
        expected_sha256=probe_sha256,
    )
    if inputs.get("wheel") != probe_inputs.get("wheel"):
        raise ReplacementError("GPU-smoke wheel input differs from accepted probe")
    if inputs.get("environment_lock") != probe_inputs.get("environment_lock"):
        raise ReplacementError("GPU-smoke environment lock differs from accepted probe")
    if inputs.get("base_python") != probe_inputs.get("base_python"):
        raise ReplacementError("GPU-smoke base Python differs from accepted probe")
    if inputs.get("torch_site_packages") != probe_inputs.get("torch_site_packages"):
        raise ReplacementError("GPU-smoke Torch tree differs from accepted probe")
    runner = inputs.get("runner")
    if not isinstance(runner, dict):
        raise ReplacementError("GPU-smoke runner identity is absent")
    runner_path_value = runner.get("path")
    if not isinstance(runner_path_value, str):
        raise ReplacementError("GPU-smoke runner path is absent")
    runner_relative = _safe_relative(runner_path_value, "GPU-smoke runner")
    runner_path = workspace.joinpath(*runner_relative.parts)
    _require_identity(
        runner,
        path=runner_path,
        workspace=workspace,
        description="GPU-smoke runner",
    )

    probe_installation = probe_document.get("installation")
    if not isinstance(probe_installation, dict):
        raise ReplacementError("accepted probe installation identity is absent")
    prefix_value = inputs.get("probe_prefix")
    if isinstance(prefix_value, dict):
        prefix_value = prefix_value.get("path")
    expected_prefix = probe_installation.get("prefix")
    if not isinstance(prefix_value, str) or prefix_value != expected_prefix:
        raise ReplacementError("GPU-smoke prefix differs from the accepted probe prefix")
    probe_prefix = workspace.joinpath(*_safe_relative(prefix_value, "probe prefix").parts)
    try:
        resolved_probe_prefix = probe_prefix.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("accepted GPU-smoke probe prefix is absent") from error
    if not _is_within(resolved_probe_prefix, workspace):
        raise ReplacementError("accepted GPU-smoke probe prefix escaped workspace")

    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        raise ReplacementError("GPU-smoke runtime evidence is absent")
    device = runtime.get("device")
    target = runtime.get("target")
    if (
        not isinstance(device, dict)
        or device.get("name") != "NVIDIA GeForce RTX 5090 Laptop GPU"
        or device.get("compute_capability") != [12, 0]
        or not isinstance(device.get("index"), int)
        or isinstance(device.get("index"), bool)
        or target != {"arch": 120, "backend": "cuda", "warp_size": 32}
        or runtime.get("gpu_execution") is not True
    ):
        raise ReplacementError("GPU-smoke did not execute exact SM120 target")
    ptxas = runtime.get("ptxas_blackwell")
    if (
        not isinstance(ptxas, dict)
        or ptxas.get("sha256") != anchor["ptxas_blackwell"]["sha256"]
        or ptxas.get("reported_release") != "13.1"
        or ptxas.get("audited_full_version") != "13.1.80"
        or ptxas.get("wheel_owned") is not True
    ):
        raise ReplacementError("GPU-smoke did not use the audited ptxas-blackwell")
    probe_processes = probe_document.get("runtime", {}).get("processes", [])  # type: ignore[union-attr]
    if (
        not isinstance(probe_processes, list)
        or not probe_processes
        or not isinstance(probe_processes[0], dict)
        or ptxas.get("path") != probe_processes[0].get("ptxas_blackwell", {}).get("path")
    ):
        raise ReplacementError("GPU-smoke ptxas path differs from accepted probe")
    correctness = runtime.get("correctness")
    if (
        not isinstance(correctness, dict)
        or correctness.get("dtype") != "float32"
        or correctness.get("comparison") != "torch.equal"
        or correctness.get("equal") is not True
        or correctness.get("n_elements") != 65_537
        or correctness.get("block_size") != 256
        or correctness.get("kernel") != "masked-vector-add"
        or correctness.get("reference_provider") != "torch"
    ):
        raise ReplacementError("GPU-smoke vector-add correctness proof is incomplete")
    synchronization = runtime.get("synchronization")
    if synchronization != {
        "after_comparison": True,
        "after_kernel": True,
        "before_launch": True,
        "error": None,
    }:
        raise ReplacementError("GPU-smoke synchronization did not finish cleanly")
    compiled_cache = runtime.get("compiled_cache")
    if (
        not isinstance(compiled_cache, dict)
        or compiled_cache.get("fresh_before_run") is not True
        or not isinstance(compiled_cache.get("artifacts"), list)
        or compiled_cache.get("artifacts_count")
        != len(compiled_cache.get("artifacts", []))
        or not isinstance(compiled_cache.get("artifacts_sha256"), str)
        or compiled_cache.get("cubin_count", 0) < 1
    ):
        raise ReplacementError("GPU-smoke compiled-cache provenance is incomplete")
    cache_path_value = compiled_cache.get("path")
    artifacts = compiled_cache.get("artifacts")
    if not isinstance(cache_path_value, str) or not isinstance(artifacts, list):
        raise ReplacementError("GPU-smoke compiled-cache path/artifacts are absent")
    cache_relative = _safe_relative(cache_path_value, "GPU-smoke cache")
    if ".." in cache_relative.parts:
        raise ReplacementError("GPU-smoke cache escaped workspace")
    cache_root = workspace.joinpath(*cache_relative.parts)
    try:
        resolved_cache_root = cache_root.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("GPU-smoke cache root is absent") from error
    if not _is_within(resolved_cache_root, workspace):
        raise ReplacementError("GPU-smoke cache root escaped workspace")
    cache_root = resolved_cache_root
    actual_artifacts: list[dict[str, object]] = []
    cubin_count = 0
    total_bytes = 0
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ReplacementError("GPU-smoke cache artifact is malformed")
        relative = _safe_relative(item["path"], "GPU-smoke cache artifact")
        if ".." in relative.parts:
            raise ReplacementError("GPU-smoke cache artifact escaped cache root")
        artifact = cache_root.joinpath(*relative.parts)
        if artifact.is_symlink() or not artifact.is_file():
            raise ReplacementError("GPU-smoke cache artifact is absent/indirect")
        actual = {
            "path": item["path"],
            "sha256": sha256_file(artifact),
            "size": artifact.stat().st_size,
        }
        if actual != item:
            raise ReplacementError("GPU-smoke cache artifact bytes changed")
        actual_artifacts.append(actual)
        total_bytes += int(actual["size"])
        if str(actual["path"]).casefold().endswith(".cubin"):
            cubin_count += 1
    if (
        actual_artifacts != artifacts
        or compiled_cache.get("artifacts_count") != len(actual_artifacts)
        or compiled_cache.get("artifacts_sha256")
        != sha256_bytes(canonical_json(actual_artifacts).encode("ascii"))
        or compiled_cache.get("total_bytes") != total_bytes
        or compiled_cache.get("cubin_count") != cubin_count
        or cubin_count < 1
    ):
        raise ReplacementError("GPU-smoke compiled-cache aggregate is inconsistent")
    provenance = runtime.get("provenance")
    if not isinstance(provenance, dict):
        raise ReplacementError("GPU-smoke runtime provenance is absent")
    editable = provenance.get("editable")
    if (
        not isinstance(editable, dict)
        or editable.get("loaded_modules") != []
        or not isinstance(editable.get("carriers"), dict)
        or any(value != [] for value in editable["carriers"].values())
    ):
        raise ReplacementError("GPU-smoke loaded an editable Triton carrier")
    _validate_smoke_provenance_paths(provenance, resolved_probe_prefix)
    base_identity = inputs.get("base_python")
    python_runtime = provenance.get("python")
    torch_runtime = provenance.get("torch_runtime")
    probe_torch = probe_document.get("runtime", {}).get("processes", [])[0].get("torch", {})  # type: ignore[union-attr,index]
    if (
        not isinstance(base_identity, dict)
        or not isinstance(python_runtime, dict)
        or python_runtime.get("executable") != "$PROBE_PREFIX/bin/python"
        or python_runtime.get("resolved_executable") != base_identity.get("path")
        or python_runtime.get("resolved_sha256") != base_identity.get("sha256")
        or python_runtime.get("resolved_size") != base_identity.get("size")
        or not isinstance(torch_runtime, dict)
        or torch_runtime.get("version") != probe_torch.get("version")
        or torch_runtime.get("git_version") != probe_torch.get("git_version")
        or torch_runtime.get("cuda") != probe_torch.get("cuda")
        or torch_runtime.get("hip") != probe_torch.get("hip")
    ):
        raise ReplacementError("GPU-smoke Python/Torch runtime provenance changed")
    integrity = runtime.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("stable") is not True
        or integrity.get("before") != integrity.get("after")
        or not isinstance(integrity.get("before"), dict)
    ):
        raise ReplacementError("GPU-smoke before/after integrity is not stable")
    snapshot = integrity["before"]
    linked = snapshot.get("linked_inputs")
    installed = snapshot.get("installed_triton")
    torch_tree = snapshot.get("torch_tree")
    environment_snapshot = snapshot.get("environment_lock")
    if (
        not isinstance(linked, dict)
        or linked.get("base_python") != inputs.get("base_python")
        or linked.get("probe_evidence") != inputs.get("probe_evidence")
        or not isinstance(linked.get("wheel"), dict)
        or linked["wheel"].get("sha256") != inputs["wheel"].get("sha256")
        or linked["wheel"].get("audit_evidence_sha256")
        != inputs["wheel"].get("audit_evidence", {}).get("sha256")
        or not isinstance(environment_snapshot, dict)
        or environment_snapshot.get("sha256")
        != inputs["environment_lock"].get("sha256")
        or not isinstance(torch_tree, dict)
        or torch_tree.get("torch_tree_sha256")
        != inputs["torch_site_packages"].get("torch_tree_sha256")
        or torch_tree.get("torch_tree_files")
        != inputs["torch_site_packages"].get("torch_tree_files")
        or torch_tree.get("torch_tree_bytes")
        != inputs["torch_site_packages"].get("torch_tree_bytes")
        or not isinstance(installed, dict)
        or any(
            not isinstance(installed.get(name), int)
            or installed.get(name, 0) <= 0
            for name in (
                "native_entries_count",
                "package_entries_count",
                "record_entries_count",
            )
        )
        or any(
            not isinstance(installed.get(name), str)
            or not installed.get(name)
            for name in (
                "native_entries_sha256",
                "package_entries_sha256",
                "record_entries_sha256",
            )
        )
    ):
        raise ReplacementError("GPU-smoke integrity snapshot is incomplete or unbound")


def validate_input_chain(
    request: ReplacementRequest,
    *,
    workspace: Path,
    limits: ReplacementLimits,
) -> ValidatedInputs:
    versions_lock_path = _require_workspace_file(
        workspace / "VERSIONS.lock", workspace, "VERSIONS.lock"
    )
    versions_raw = _read_regular_file_once(
        versions_lock_path, "VERSIONS.lock", max_bytes=4 << 20
    )
    try:
        versions_text = versions_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReplacementError("VERSIONS.lock is not strict UTF-8") from error
    versions_lock: dict[str, str] = {}
    for line_number, raw_line in enumerate(versions_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ReplacementError(f"VERSIONS.lock line {line_number} is malformed")
        name, value = line.split("=", 1)
        if not name or name in versions_lock:
            raise ReplacementError(f"VERSIONS.lock key is empty/duplicate: {name!r}")
        versions_lock[name] = value
    if versions_lock.get("triton.producer.python_sha256") != probe.BASE_PYTHON_SHA256:
        raise ReplacementError("VERSIONS.lock base-Python SHA256 differs from probe")
    environment_lock, environment_lock_raw = load_canonical_json(
        request.environment_lock,
        "ENVIRONMENT.lock",
        max_bytes=limits.max_evidence_bytes,
    )
    environment_lock_sha = validate_sha256(
        request.expected_environment_lock_sha256, "expected ENVIRONMENT.lock SHA256"
    )
    if sha256_bytes(environment_lock_raw) != environment_lock_sha:
        raise ReplacementError("ENVIRONMENT.lock differs from its SHA256 anchor")

    audit_document, audit_raw = load_canonical_json(
        request.wheel_audit_evidence,
        "wheel-audit evidence",
        max_bytes=limits.max_evidence_bytes,
    )
    audit_sha = validate_sha256(
        request.expected_wheel_audit_evidence_sha256,
        "expected wheel-audit evidence SHA256",
    )
    try:
        anchor = probe.validate_audit_anchor(
            audit_document,
            audit_raw,
            expected_evidence_sha256=audit_sha,
            wheel_path=request.wheel,
            workspace=workspace,
            limits=probe.ProbeLimits(),
        )
    except probe.ProbeError as error:
        raise ReplacementError(f"wheel-audit anchor failed: {error}") from error

    probe_document, probe_raw = load_canonical_json(
        request.wheel_probe_evidence,
        "wheel-probe evidence",
        max_bytes=limits.max_evidence_bytes,
    )
    probe_sha = validate_sha256(
        request.expected_wheel_probe_evidence_sha256,
        "expected wheel-probe evidence SHA256",
    )
    if sha256_bytes(probe_raw) != probe_sha:
        raise ReplacementError("wheel-probe evidence differs from its SHA256 anchor")
    _validate_probe_evidence(
        probe_document,
        request=request,
        workspace=workspace,
        environment_lock_sha256=environment_lock_sha,
        audit_sha256=audit_sha,
        wheel_sha256=str(anchor["wheel_sha256"]),
    )

    smoke_document, smoke_raw = load_canonical_json(
        request.gpu_smoke_evidence,
        "GPU-smoke evidence",
        max_bytes=limits.max_evidence_bytes,
    )
    smoke_sha = validate_sha256(
        request.expected_gpu_smoke_evidence_sha256,
        "expected GPU-smoke evidence SHA256",
    )
    if sha256_bytes(smoke_raw) != smoke_sha:
        raise ReplacementError("GPU-smoke evidence differs from its SHA256 anchor")
    _validate_gpu_smoke_evidence(
        smoke_document,
        request=request,
        workspace=workspace,
        probe_document=probe_document,
        probe_sha256=probe_sha,
        anchor=anchor,
    )
    return ValidatedInputs(
        environment_lock=environment_lock,
        environment_lock_raw=environment_lock_raw,
        audit_document=audit_document,
        audit_raw=audit_raw,
        anchor=anchor,
        probe_document=probe_document,
        probe_raw=probe_raw,
        smoke_document=smoke_document,
        smoke_raw=smoke_raw,
        versions_lock=versions_lock,
        versions_lock_path=versions_lock_path,
    )


SCHEME_PROGRAM = r"""
import json
import pathlib
import sys
import sysconfig

print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "version": list(sys.version_info[:3]),
    "paths": {
        "data": sysconfig.get_path("data"),
        "headers": str(
            pathlib.Path(sys.prefix)
            / "include"
            / "site"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "triton"
        ),
        "platlib": sysconfig.get_path("platlib"),
        "purelib": sysconfig.get_path("purelib"),
        "scripts": sysconfig.get_path("scripts"),
    },
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


@contextmanager
def _workspace_temporary_io(workspace: Path):
    """Keep parent/child temporary files in an ephemeral workspace run root."""

    runs = workspace / "runs"
    if not runs.is_dir() or runs.is_symlink():
        raise ReplacementError("workspace runs directory is absent or unsafe")
    temporary = Path(tempfile.mkdtemp(prefix="replace-triton-temp.", dir=runs))
    previous = tempfile.tempdir
    tempfile.tempdir = str(temporary)
    try:
        # Assert the exact default used by probe.run_command's TemporaryFile.
        with tempfile.TemporaryFile() as handle:
            descriptor_target = Path(f"/proc/self/fd/{handle.fileno()}")
            if os.path.lexists(descriptor_target):
                value = os.readlink(descriptor_target).removesuffix(" (deleted)")
                if not _is_within(_absolute_lexical(Path(value)), temporary):
                    raise ReplacementError("TemporaryFile escaped workspace runs temp")
        yield temporary
    finally:
        tempfile.tempdir = previous
        leftovers = list(temporary.iterdir()) if temporary.exists() else []
        if leftovers:
            raise ReplacementError(
                f"workspace temporary directory retained files: {leftovers}"
            )
        try:
            temporary.rmdir()
        except FileNotFoundError:
            pass


@contextmanager
def _temporary_file_default(directory: Path):
    if directory.is_symlink() or not directory.is_dir():
        raise ReplacementError("temporary-file directory is absent or unsafe")
    previous = tempfile.tempdir
    tempfile.tempdir = str(directory)
    try:
        with tempfile.TemporaryFile() as handle:
            value = os.readlink(f"/proc/self/fd/{handle.fileno()}").removesuffix(
                " (deleted)"
            )
            if not _is_within(_absolute_lexical(Path(value)), directory):
                raise ReplacementError("TemporaryFile escaped transaction scratch")
        yield
    finally:
        tempfile.tempdir = previous


def _readonly_environment(temporary: Path) -> dict[str, str]:
    """A deterministic environment that creates no workspace/prefix paths."""

    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        "TRITON_CACHE_DIR": "/nonexistent/triton-cache",
        "XDG_CACHE_HOME": "/nonexistent/cache",
    }


def _transaction_environment(scratch: Path) -> dict[str, str]:
    home = scratch / "home"
    cache = scratch / "cache"
    temporary = scratch / "tmp"
    for path in (home, cache, temporary, cache / "triton"):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ReplacementError(f"transaction scratch path is unsafe: {path}")
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "XDG_CACHE_HOME": str(cache),
    }


def query_target_scheme(
    prefix: Path,
    environment_lock: Mapping[str, object],
    *,
    workspace: Path,
    base_python_identity: Mapping[str, object],
    limits: ReplacementLimits,
) -> tuple[Path, object]:
    python = prefix / "bin" / "python"
    if not python.exists() or not os.access(python, os.X_OK):
        raise ReplacementError("environment prefix has no executable bin/python")
    try:
        resolved_python = python.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("environment Python link is broken") from error
    locked_python_value = environment_lock.get("python_executable")
    if not isinstance(locked_python_value, str):
        raise ReplacementError("ENVIRONMENT.lock python_executable is absent")
    locked_python = Path(locked_python_value)
    try:
        resolved_locked_python = locked_python.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("ENVIRONMENT.lock Python executable is absent") from error
    if resolved_python != resolved_locked_python:
        raise ReplacementError("environment Python differs from ENVIRONMENT.lock")
    base_path_value = base_python_identity.get("path")
    if not isinstance(base_path_value, str):
        raise ReplacementError("probe base-Python path is absent")
    base_relative = _safe_relative(base_path_value, "probe base Python")
    base_python = workspace.joinpath(*base_relative.parts)
    expected_base_python = prefix / "bin" / "python3.14"
    if (
        base_python != expected_base_python
        or base_python.is_symlink()
        or not base_python.is_file()
        or not os.access(base_python, os.X_OK)
    ):
        raise ReplacementError(
            "probe base Python is not exact prefix/bin/python3.14"
        )
    _require_identity(
        base_python_identity,
        path=base_python,
        workspace=workspace,
        description="probe base Python",
        expected_sha256=probe.BASE_PYTHON_SHA256,
    )
    try:
        resolved_base_python = base_python.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("probe base Python is absent") from error
    if (
        resolved_python != resolved_base_python
        or resolved_locked_python != resolved_base_python
        or sha256_file(resolved_python) != probe.BASE_PYTHON_SHA256
    ):
        raise ReplacementError("replacement interpreter differs from probe base Python")
    try:
        with _workspace_temporary_io(workspace) as temporary:
            report = probe.run_json_command(
                [str(python), "-I", "-B", "-S", "-c", SCHEME_PROGRAM],
                environment=_readonly_environment(temporary),
                limits=probe.ProbeLimits(
                    max_output_bytes=limits.max_output_bytes,
                    timeout_seconds=limits.timeout_seconds,
                ),
                description="read-only target sysconfig probe",
            )
    except probe.ProbeError as error:
        raise ReplacementError(f"target sysconfig probe failed: {error}") from error
    if report.get("prefix") != str(prefix):
        raise ReplacementError("target Python sys.prefix differs from requested prefix")
    executable = report.get("executable")
    if not isinstance(executable, str) or _absolute_lexical(Path(executable)) != python:
        raise ReplacementError("target Python sys.executable is not prefix/bin/python")
    version = report.get("version")
    if version != [3, 14, 6] or environment_lock.get("python_abi") != "cp314":
        raise ReplacementError("target Python is not locked CPython 3.14.6/cp314")
    paths = report.get("paths")
    if not isinstance(paths, dict):
        raise ReplacementError("target sysconfig paths are absent")
    normalized: dict[str, Path] = {}
    for name in ("data", "headers", "platlib", "purelib", "scripts"):
        value = paths.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ReplacementError(f"target sysconfig {name} path is invalid")
        path = _absolute_lexical(Path(value))
        if not _is_within(path, prefix, allow_root=True):
            raise ReplacementError(f"target sysconfig {name} escaped prefix")
        _path_without_symlink_parents(path, prefix)
        normalized[name] = path
    scheme = probe.InstallScheme(
        prefix=prefix,
        purelib=normalized["purelib"],
        platlib=normalized["platlib"],
        scripts=normalized["scripts"],
        headers=normalized["headers"],
        data=normalized["data"],
        python_version=(3, 14, 6),
    )
    return python, scheme


def _single_metadata_header(message: object, name: str) -> str:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ReplacementError(f"old Triton METADATA must contain one {name}")
    return values[0].strip()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _decode_record_sha(value: str, relative: str) -> str:
    if not value.startswith("sha256="):
        raise ReplacementError(f"old RECORD uses a non-SHA256 digest: {relative}")
    encoded = value.removeprefix("sha256=")
    if not encoded or "=" in encoded or re.fullmatch(r"[A-Za-z0-9_-]+", encoded) is None:
        raise ReplacementError(f"old RECORD SHA256 is non-canonical: {relative}")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as error:
        raise ReplacementError(f"old RECORD SHA256 cannot be decoded: {relative}") from error
    if len(decoded) != 32:
        raise ReplacementError(f"old RECORD SHA256 has wrong length: {relative}")
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != encoded:
        raise ReplacementError(f"old RECORD SHA256 is non-canonical: {relative}")
    return decoded.hex()


def _record_target(relative: str, scheme: object) -> Path:
    path = _safe_relative(relative, "old RECORD")
    target = _absolute_lexical(scheme.platlib.joinpath(*path.parts))
    if not _is_within(target, scheme.prefix):
        raise ReplacementError(f"old RECORD target escaped environment prefix: {relative}")
    _path_without_symlink_parents(target, scheme.prefix)
    return target


def _lstat_file_entry(
    path: Path,
    prefix: Path,
    *,
    record_digest: str,
    record_size: str,
    roles: Iterable[str],
) -> FileEntry:
    _path_without_symlink_parents(path, prefix)
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise ReplacementError(f"old RECORD target is absent: {path}") from error
    mode = stat.S_IMODE(file_stat.st_mode)
    if stat.S_ISREG(file_stat.st_mode):
        digest = sha256_file(path)
        size = file_stat.st_size
        kind = "regular"
        link_target = None
    elif stat.S_ISLNK(file_stat.st_mode):
        target_bytes = os.fsencode(os.readlink(path))
        digest = None
        size = len(target_bytes)
        kind = "symlink"
        link_target = base64.b64encode(target_bytes).decode("ascii")
    else:
        raise ReplacementError(f"old RECORD target is not regular/symlink: {path}")
    if record_digest:
        expected_digest = _decode_record_sha(record_digest, str(path))
        if kind != "regular" or digest != expected_digest:
            raise ReplacementError(f"old RECORD byte ownership mismatch: {path}")
        if re.fullmatch(r"0|[1-9][0-9]*", record_size) is None:
            raise ReplacementError(f"old RECORD size is non-canonical: {path}")
        if size != int(record_size):
            raise ReplacementError(f"old RECORD size ownership mismatch: {path}")
    elif record_size:
        raise ReplacementError(f"old RECORD has size without digest: {path}")
    return FileEntry(
        path=_prefix_relative(path, prefix),
        kind=kind,
        mode=mode,
        size=size,
        sha256=digest,
        link_target_b64=link_target,
        record_digest=record_digest,
        record_size=record_size,
        roles=tuple(sorted(set(roles))),
    )


def _finder_literal_assignment(tree: ast.Module, name: str) -> object:
    matches: list[object] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            matches.append(ast.literal_eval(node.value))
        except (TypeError, ValueError) as error:
            raise ReplacementError(f"editable finder {name} is not literal data") from error
    if len(matches) != 1:
        raise ReplacementError(f"editable finder must define {name} exactly once")
    return matches[0]


def _parse_editable_finder(
    finder: Path, editable_source: Path
) -> tuple[str, Path]:
    raw = _read_regular_file_once(finder, "editable Triton finder", max_bytes=4 << 20)
    try:
        text = raw.decode("utf-8", errors="strict")
        tree = ast.parse(text, filename=str(finder))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ReplacementError("editable Triton finder is not strict Python source") from error
    mapping = _finder_literal_assignment(tree, "MAPPING")
    namespaces = _finder_literal_assignment(tree, "NAMESPACES")
    if not isinstance(mapping, dict) or set(mapping) != {"triton"}:
        raise ReplacementError("editable finder does not map exactly the Triton root")
    if not isinstance(namespaces, dict) or any(
        not isinstance(key, str)
        or not key.startswith("triton.")
        or not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        for key, value in namespaces.items()
    ):
        raise ReplacementError("editable finder namespace map is malformed")
    package_value = mapping["triton"]
    if not isinstance(package_value, str) or not Path(package_value).is_absolute():
        raise ReplacementError("editable finder Triton root is not absolute")
    package_lexical = _absolute_lexical(Path(package_value))
    try:
        package_root = package_lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("editable finder Triton root is absent") from error
    if package_lexical != package_root or not package_root.is_dir():
        raise ReplacementError("editable finder Triton root must be a real directory")
    if not _is_within(package_root, editable_source):
        raise ReplacementError("editable finder Triton root escaped direct_url source")
    for values in namespaces.values():
        for value in values:
            try:
                namespace = Path(value).resolve(strict=True)
            except OSError as error:
                raise ReplacementError("editable namespace target is absent") from error
            if not _is_within(namespace, package_root):
                raise ReplacementError("editable namespace target escaped Triton package")
    module_name = finder.stem
    return module_name, package_root


def tree_identity(root: Path) -> dict[str, object]:
    """Hash the complete directory tree including modes and symlink targets."""

    if root.is_symlink() or not root.is_dir():
        raise ReplacementError(f"tree identity root is absent or unsafe: {root}")
    digest_value = hashlib.sha256()
    byte_count = 0
    directory_count = 0
    file_count = 0
    symlink_count = 0
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        path_stat = path.lstat()
        mode = stat.S_IMODE(path_stat.st_mode)
        digest_value.update(len(relative).to_bytes(8, "little"))
        digest_value.update(relative)
        digest_value.update(mode.to_bytes(4, "little"))
        if stat.S_ISDIR(path_stat.st_mode):
            digest_value.update(b"D")
            directory_count += 1
        elif stat.S_ISREG(path_stat.st_mode):
            digest_value.update(b"F")
            digest_value.update(path_stat.st_size.to_bytes(8, "little"))
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 << 20), b""):
                    digest_value.update(chunk)
            byte_count += path_stat.st_size
            file_count += 1
        elif stat.S_ISLNK(path_stat.st_mode):
            target = os.fsencode(os.readlink(path))
            digest_value.update(b"L")
            digest_value.update(len(target).to_bytes(8, "little"))
            digest_value.update(target)
            symlink_count += 1
        else:
            raise ReplacementError(f"external package tree has special file: {path}")
    return {
        "bytes": byte_count,
        "directories": directory_count,
        "files": file_count,
        "paths": len(paths),
        "sha256": digest_value.hexdigest(),
        "symlinks": symlink_count,
    }


def _run_readonly_git(
    workspace: Path,
    source: Path,
    arguments: Sequence[str],
    *,
    description: str,
    limits: ReplacementLimits,
) -> str:
    git = Path("/usr/bin/git")
    if not git.is_file() or not os.access(git, os.X_OK):
        raise ReplacementError("absolute /usr/bin/git is unavailable")
    with _workspace_temporary_io(workspace) as temporary:
        environment = _readonly_environment(temporary)
        environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        try:
            stdout, stderr = probe.run_command(
                [str(git), "-C", str(source), *arguments],
                environment=environment,
                limits=probe.ProbeLimits(
                    max_output_bytes=limits.max_output_bytes,
                    timeout_seconds=limits.timeout_seconds,
                ),
                description=description,
            )
        except probe.ProbeError as error:
            raise ReplacementError(f"{description} failed: {error}") from error
    if stderr:
        raise ReplacementError(f"{description} produced unexpected stderr")
    try:
        return stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReplacementError(f"{description} output is not UTF-8") from error


def external_source_identity(
    workspace: Path,
    source: Path,
    package_root: Path,
    *,
    limits: ReplacementLimits,
) -> dict[str, object]:
    head = _run_readonly_git(
        workspace,
        source,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        description="external Triton HEAD query",
        limits=limits,
    ).strip()
    tree = _run_readonly_git(
        workspace,
        source,
        ["rev-parse", "--verify", "HEAD^{tree}"],
        description="external Triton tree query",
        limits=limits,
    ).strip()
    origin = _run_readonly_git(
        workspace,
        source,
        ["remote", "get-url", "origin"],
        description="external Triton origin query",
        limits=limits,
    ).strip()
    porcelain = _run_readonly_git(
        workspace,
        source,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        description="external Triton dirty-state query",
        limits=limits,
    )
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ReplacementError("external Triton HEAD is not a commit SHA")
    if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise ReplacementError("external Triton tree is not a tree SHA")
    if not origin or "\x00" in origin or "\n" in origin:
        raise ReplacementError("external Triton origin is absent or malformed")
    lines = porcelain.splitlines()
    return {
        "dirty": bool(lines),
        "git_head": head,
        "git_origin": origin,
        "git_status_lines": lines,
        "git_status_sha256": sha256_bytes(porcelain.encode("utf-8")),
        "git_tree": tree,
        "package_root": str(package_root),
        "package_tree": tree_identity(package_root),
        "source": str(source),
    }


def verify_external_source_identity(
    prepared: PreparedReplacement, *, limits: ReplacementLimits
) -> None:
    observed = external_source_identity(
        prepared.workspace,
        Path(prepared.old_inventory.editable_source),
        Path(prepared.old_inventory.package_root),
        limits=limits,
    )
    if observed != prepared.old_inventory.external_source_identity:
        raise ReplacementError("external editable Triton source/package tree drifted")


def _record_owned_directories(
    files: Sequence[FileEntry], prefix: Path, scheme: object
) -> tuple[DirectoryEntry, ...]:
    file_paths = {prefix / entry.path for entry in files}
    candidates: set[Path] = set()
    protected = {
        prefix,
        scheme.platlib,
        scheme.purelib,
        scheme.scripts,
        scheme.headers,
        scheme.data,
    }
    for path in file_paths:
        parent = path.parent
        while _is_within(parent, prefix) and parent not in protected:
            candidates.add(parent)
            parent = parent.parent

    removable: set[Path] = set()
    for directory in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ReplacementError(f"old ownership parent is not a real directory: {directory}")
        children = set(directory.iterdir())
        allowed = {
            path for path in file_paths if path.parent == directory
        } | {path for path in removable if path.parent == directory}
        if children <= allowed:
            removable.add(directory)
    entries = []
    for directory in sorted(removable):
        entries.append(
            DirectoryEntry(
                path=_prefix_relative(directory, prefix),
                mode=stat.S_IMODE(directory.stat().st_mode),
            )
        )
    return tuple(entries)


def _top_level_triton_carriers(site: Path, finder_module: str) -> set[Path]:
    carriers: set[Path] = set()
    for path in site.iterdir():
        lower = path.name.casefold()
        if (
            lower == "triton"
            or re.fullmatch(r"triton[-_.].*\.(?:dist|egg)-info", lower) is not None
            or "triton" in lower
            and (
                lower.endswith((".pth", ".egg-link"))
                or "editable" in lower
                or "finder" in lower
            )
            or path.name == f"{finder_module}.py"
        ):
            carriers.add(path)
    pycache = site / "__pycache__"
    if pycache.is_dir() and not pycache.is_symlink():
        for path in pycache.iterdir():
            if finder_module in path.name or "triton" in path.name.casefold():
                carriers.add(path)
    return carriers


def inventory_old_editable(
    scheme: object,
    *,
    workspace: Path,
    limits: ReplacementLimits,
) -> OldInventory:
    site = scheme.platlib
    if site != scheme.purelib:
        raise ReplacementError("target purelib/platlib split is unsupported for Triton")
    if not site.is_dir() or site.is_symlink():
        raise ReplacementError("target site-packages is absent or unsafe")
    dist_infos = sorted(site.glob("triton-*.dist-info"))
    if len(dist_infos) != 1 or not dist_infos[0].is_dir() or dist_infos[0].is_symlink():
        raise ReplacementError("old environment must contain exactly one Triton dist-info")
    dist_info = dist_infos[0]
    metadata_raw = _read_regular_file_once(
        dist_info / "METADATA", "old Triton METADATA", max_bytes=4 << 20
    )
    message = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
    if _normalized_distribution_name(_single_metadata_header(message, "Name")) != "triton":
        raise ReplacementError("old distribution is not Triton")
    version = _single_metadata_header(message, "Version")
    if version != probe.TRITON_DISTRIBUTION_VERSION:
        raise ReplacementError("old editable distribution version is not the exact pin")

    direct_path = dist_info / "direct_url.json"
    direct_raw = _read_regular_file_once(
        direct_path, "old Triton direct_url.json", max_bytes=1 << 20
    )
    try:
        direct_url = json.loads(
            direct_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplacementError("old Triton direct_url.json is invalid") from error
    if not isinstance(direct_url, dict) or direct_url.get("dir_info") != {"editable": True}:
        raise ReplacementError("old Triton is not an exact editable distribution")
    url = direct_url.get("url")
    if not isinstance(url, str):
        raise ReplacementError("old editable direct URL is absent")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise ReplacementError("old editable direct URL is not a local directory")
    source_lexical = _absolute_lexical(Path(urllib.parse.unquote(parsed.path)))
    try:
        editable_source = source_lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("old editable source is absent") from error
    if (
        source_lexical != editable_source
        or not editable_source.is_dir()
        or _is_within(editable_source, workspace, allow_root=True)
    ):
        raise ReplacementError("old Triton source is not a real external directory")

    record_path = dist_info / "RECORD"
    record_raw = _read_regular_file_once(
        record_path, "old Triton RECORD", max_bytes=limits.max_record_bytes
    )
    try:
        record_text = record_raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ReplacementError("old Triton RECORD is not strict UTF-8 CSV") from error
    if not rows or len(rows) > limits.max_record_entries:
        raise ReplacementError("old Triton RECORD count is invalid")
    targets: dict[Path, tuple[str, str, str]] = {}
    relative_rows: set[str] = set()
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in relative_rows:
            raise ReplacementError("old Triton RECORD has malformed/duplicate rows")
        relative_rows.add(row[0])
        target = _record_target(row[0], scheme)
        if target in targets:
            raise ReplacementError("old Triton RECORD aliases one deletion target")
        targets[target] = (row[0], row[1], row[2])
    if record_path not in targets or targets[record_path][1:] != ("", ""):
        raise ReplacementError("old Triton RECORD does not own itself canonically")
    if direct_path not in targets:
        raise ReplacementError("old Triton RECORD does not own direct_url.json")

    pth_paths = sorted(
        path
        for path in targets
        if path.parent == site
        and path.name.casefold().startswith("__editable__.triton-")
        and path.suffix.casefold() == ".pth"
    )
    finder_paths = sorted(
        path
        for path in targets
        if path.parent == site
        and path.name.casefold().startswith("__editable___triton_")
        and path.name.casefold().endswith("_finder.py")
    )
    if len(pth_paths) != 1 or len(finder_paths) != 1:
        raise ReplacementError("old Triton must own one editable pth and one finder")
    pth_raw = _read_regular_file_once(
        pth_paths[0], "editable Triton pth", max_bytes=1 << 20
    )
    try:
        pth_text = pth_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReplacementError("editable Triton pth is not strict UTF-8") from error
    finder_module = finder_paths[0].stem
    active_lines = [
        line.strip()
        for line in pth_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_pth = f"import {finder_module}; {finder_module}.install()"
    if active_lines != [expected_pth]:
        raise ReplacementError("editable Triton pth does not install its exact finder")
    parsed_finder_module, package_root = _parse_editable_finder(
        finder_paths[0], editable_source
    )
    if parsed_finder_module != finder_module:
        raise ReplacementError("editable Triton finder module identity is inconsistent")

    role_map: dict[Path, set[str]] = {path: {"record-owned"} for path in targets}
    role_map[record_path].add("record")
    role_map[direct_path].add("direct-url")
    role_map[pth_paths[0]].add("editable-pth")
    role_map[finder_paths[0]].add("editable-finder")
    for path in targets:
        if _is_within(path, dist_info):
            role_map[path].add("dist-info")
        if path.parent.name == "__pycache__" and finder_module in path.name:
            role_map[path].add("editable-finder-bytecode")

    files: list[FileEntry] = []
    backup_bytes = 0
    for path in sorted(targets):
        _, record_digest, record_size = targets[path]
        entry = _lstat_file_entry(
            path,
            scheme.prefix,
            record_digest=record_digest,
            record_size=record_size,
            roles=role_map[path],
        )
        if entry.size > limits.max_backup_file_bytes:
            raise ReplacementError(f"old backup member exceeds size limit: {path}")
        backup_bytes += entry.size
        if backup_bytes > limits.max_backup_bytes:
            raise ReplacementError("old RECORD backup exceeds aggregate size limit")
        files.append(entry)
    for entry in files:
        if entry.kind != "regular":
            continue
        destination = scheme.prefix / entry.path
        rollback_temporary = destination.with_name(
            f".{destination.name}.pypto-triton-rollback.partial"
        )
        if rollback_temporary.exists() or rollback_temporary.is_symlink():
            raise ReplacementError(
                f"reserved rollback temporary path is occupied: {rollback_temporary}"
            )

    owned_paths = set(targets)
    for root in (dist_info, site / "triton"):
        if not root.exists() and not root.is_symlink():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ReplacementError(f"old Triton tree is not a real directory: {root}")
        for path in root.rglob("*"):
            if path.is_file() or path.is_symlink():
                if path not in owned_paths:
                    raise ReplacementError(f"old Triton tree contains unowned file: {path}")
    carriers = _top_level_triton_carriers(site, finder_module)
    unowned_carriers = sorted(
        path
        for path in carriers
        if path not in owned_paths
        and not (
            path in (dist_info, site / "triton")
            and path.is_dir()
            and not path.is_symlink()
        )
    )
    if unowned_carriers:
        raise ReplacementError(f"old Triton has unowned carrier(s): {unowned_carriers}")
    if set(pth_paths + finder_paths) - carriers:
        raise ReplacementError("editable carrier discovery is inconsistent")

    native_paths = sorted((package_root / "_C").glob("libtriton*.so"))
    if not native_paths:
        raise ReplacementError("external editable Triton has no libtriton native object")
    native_files: list[NativeEntry] = []
    for native in native_paths:
        try:
            native_resolved = native.resolve(strict=True)
        except OSError as error:
            raise ReplacementError("external libtriton path is absent") from error
        native_stat = native.lstat()
        if native != native_resolved or not stat.S_ISREG(native_stat.st_mode):
            raise ReplacementError("external libtriton must be a regular non-symlink file")
        if not _is_within(native_resolved, package_root):
            raise ReplacementError("external libtriton escaped editable package")
        if native_stat.st_size > limits.max_backup_file_bytes:
            raise ReplacementError("external libtriton exceeds backup file limit")
        backup_bytes += native_stat.st_size
        if backup_bytes > limits.max_backup_bytes:
            raise ReplacementError("old Triton backup exceeds aggregate size limit")
        native_files.append(
            NativeEntry(
                path=str(native_resolved),
                mode=stat.S_IMODE(native_stat.st_mode),
                size=native_stat.st_size,
                sha256=sha256_file(native_resolved),
            )
        )

    source_identity = external_source_identity(
        workspace,
        editable_source,
        package_root,
        limits=limits,
    )
    directories = _record_owned_directories(files, scheme.prefix, scheme)
    return OldInventory(
        distribution_version=version,
        dist_info=_prefix_relative(dist_info, scheme.prefix),
        direct_url=direct_url,
        editable_source=str(editable_source),
        package_root=str(package_root),
        finder_module=finder_module,
        external_source_identity=source_identity,
        files=tuple(files),
        removable_directories=directories,
        native_files=tuple(native_files),
    )


def capture_torch_identity(
    scheme: object,
    environment_lock: dict[str, object],
    *,
    workspace: Path,
) -> dict[str, object]:
    try:
        return probe.validate_torch_site(
            scheme.platlib,
            environment_lock,
            workspace=workspace,
        )
    except probe.ProbeError as error:
        raise ReplacementError(f"frozen Torch tree validation failed: {error}") from error


def _new_install_inventory(
    scheme: object, anchor: dict[str, object]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    members = anchor.get("member_map")
    if not isinstance(members, dict) or not members:
        raise ReplacementError("internal audited wheel member map is absent")
    destinations: set[Path] = set()
    for archive_path in members:
        if not isinstance(archive_path, str):
            raise ReplacementError("audited wheel member name is malformed")
        try:
            destination = probe.wheel_member_destination(archive_path, scheme)
        except probe.ProbeError as error:
            raise ReplacementError(f"new wheel destination is unsafe: {error}") from error
        if destination in destinations:
            raise ReplacementError("new wheel aliases one environment destination")
        destinations.add(destination)
    try:
        direct_url = probe.wheel_member_destination(probe.DIRECT_URL_PATH, scheme)
    except probe.ProbeError as error:
        raise ReplacementError(f"new direct_url destination is unsafe: {error}") from error
    if direct_url in destinations:
        raise ReplacementError("audited wheel already owns generated direct_url.json")
    destinations.add(direct_url)

    directories: set[Path] = set()
    for destination in destinations:
        if not _is_within(destination, scheme.prefix):
            raise ReplacementError("new wheel destination escaped environment prefix")
        parent = destination.parent
        while _is_within(parent, scheme.prefix):
            directories.add(parent)
            parent = parent.parent
    return (
        tuple(sorted(_prefix_relative(path, scheme.prefix) for path in destinations)),
        tuple(sorted(_prefix_relative(path, scheme.prefix) for path in directories)),
    )


def _validate_new_collisions(
    *,
    prefix: Path,
    old_inventory: OldInventory,
    new_paths: Sequence[str],
    new_directories: Sequence[str],
) -> None:
    old_files = {prefix / entry.path for entry in old_inventory.files}
    old_directories = {
        prefix / entry.path for entry in old_inventory.removable_directories
    }
    for relative in new_paths:
        path = prefix / relative
        _path_without_symlink_parents(path, prefix)
        if path.exists() or path.is_symlink():
            if path not in old_files:
                raise ReplacementError(
                    f"new wheel destination is not old-RECORD-owned: {path}"
                )
    for relative in new_directories:
        directory = prefix / relative
        _path_without_symlink_parents(directory, prefix)
        if directory.exists() or directory.is_symlink():
            if directory in old_files:
                raise ReplacementError(
                    f"new wheel directory collides with an old file: {directory}"
                )
            if directory.is_symlink() or not directory.is_dir():
                raise ReplacementError(f"new wheel parent is unsafe: {directory}")
            if directory in old_directories:
                continue
            # Shared package ancestors may exist.  Their contents are never
            # deletion targets, and mkdir(exist_ok=True) does not replace them.


def _input_identity_document(
    prepared_inputs: ValidatedInputs,
    request: ReplacementRequest,
    workspace: Path,
) -> dict[str, object]:
    return {
        "environment_lock": _identity(request.environment_lock, workspace),
        "gpu_smoke_evidence": _identity(request.gpu_smoke_evidence, workspace),
        "wheel": _identity(request.wheel, workspace),
        "wheel_audit_evidence": _identity(
            request.wheel_audit_evidence, workspace
        ),
        "wheel_probe_evidence": _identity(
            request.wheel_probe_evidence, workspace
        ),
        "versions_lock": _identity(prepared_inputs.versions_lock_path, workspace),
    }


def _stable_input_digests(prepared: PreparedReplacement) -> None:
    expected = _input_identity_document(
        prepared.inputs, prepared.request, prepared.workspace
    )
    if expected != prepared.plan_document["inputs"]:
        raise ReplacementError("an accepted transaction input changed after planning")


def prepare_replacement(
    request: ReplacementRequest,
    *,
    limits: ReplacementLimits = ReplacementLimits(),
) -> PreparedReplacement:
    if limits.timeout_seconds <= 0 or limits.timeout_seconds > 600:
        raise ReplacementError("timeout must be between 1 and 600 seconds")
    workspace = _require_workspace(request.workspace)
    prefix = _require_prefix(request.prefix, workspace)

    # Normalize every immutable input before parsing cross-evidence paths.
    wheel = _require_workspace_file(request.wheel, workspace, "wheel")
    audit_evidence = _require_workspace_file(
        request.wheel_audit_evidence, workspace, "wheel-audit evidence"
    )
    probe_evidence = _require_workspace_file(
        request.wheel_probe_evidence, workspace, "wheel-probe evidence"
    )
    smoke_evidence = _require_workspace_file(
        request.gpu_smoke_evidence, workspace, "GPU-smoke evidence"
    )
    environment_lock_path = _require_workspace_file(
        request.environment_lock, workspace, "ENVIRONMENT.lock"
    )
    backup_root = _require_fresh_build_root(request.backup_root, workspace)
    evidence = _require_evidence_output(request.evidence, workspace)
    if _is_within(evidence, prefix, allow_root=True):
        raise ReplacementError("replacement evidence must be outside environment prefix")
    immutable_paths = {
        wheel,
        audit_evidence,
        probe_evidence,
        smoke_evidence,
        environment_lock_path,
    }
    if evidence in immutable_paths:
        raise ReplacementError("replacement evidence aliases an immutable input")
    normalized_request = ReplacementRequest(
        workspace=workspace,
        prefix=prefix,
        wheel=wheel,
        wheel_audit_evidence=audit_evidence,
        expected_wheel_audit_evidence_sha256=(
            request.expected_wheel_audit_evidence_sha256
        ),
        wheel_probe_evidence=probe_evidence,
        expected_wheel_probe_evidence_sha256=(
            request.expected_wheel_probe_evidence_sha256
        ),
        gpu_smoke_evidence=smoke_evidence,
        expected_gpu_smoke_evidence_sha256=(
            request.expected_gpu_smoke_evidence_sha256
        ),
        environment_lock=environment_lock_path,
        expected_environment_lock_sha256=(
            request.expected_environment_lock_sha256
        ),
        backup_root=backup_root,
        evidence=evidence,
    )
    inputs = validate_input_chain(
        normalized_request, workspace=workspace, limits=limits
    )
    locked_destination = inputs.environment_lock.get("destination_prefix")
    if not isinstance(locked_destination, str):
        raise ReplacementError("ENVIRONMENT.lock destination_prefix is absent")
    locked_relative = _safe_relative(locked_destination, "locked destination prefix")
    locked_prefix = workspace.joinpath(*locked_relative.parts)
    try:
        resolved_locked_prefix = locked_prefix.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("locked destination prefix is absent") from error
    if resolved_locked_prefix != prefix:
        raise ReplacementError("requested prefix differs from ENVIRONMENT.lock")

    probe_inputs = inputs.probe_document.get("inputs")
    if not isinstance(probe_inputs, dict) or not isinstance(
        probe_inputs.get("base_python"), dict
    ):
        raise ReplacementError("accepted probe base-Python input disappeared")
    python, scheme = query_target_scheme(
        prefix,
        inputs.environment_lock,
        workspace=workspace,
        base_python_identity=probe_inputs["base_python"],
        limits=limits,
    )
    old_inventory = inventory_old_editable(
        scheme, workspace=workspace, limits=limits
    )
    new_paths, new_directories = _new_install_inventory(scheme, inputs.anchor)
    preexisting_directories = tuple(
        relative
        for relative in new_directories
        if (prefix / relative).is_dir() and not (prefix / relative).is_symlink()
    )
    _validate_new_collisions(
        prefix=prefix,
        old_inventory=old_inventory,
        new_paths=new_paths,
        new_directories=new_directories,
    )
    torch_before = capture_torch_identity(
        scheme, inputs.environment_lock, workspace=workspace
    )
    input_document = _input_identity_document(inputs, normalized_request, workspace)
    plan: dict[str, object] = {
        "backup": {
            "root": _workspace_relative(backup_root, workspace),
            "will_copy_external_native": True,
            "will_mutate_external_native": False,
        },
        "environment": {
            "prefix": _workspace_relative(prefix, workspace),
            "python": _workspace_relative(python, workspace),
            "scheme": {
                name: _prefix_relative(getattr(scheme, name), prefix)
                for name in ("data", "headers", "platlib", "purelib", "scripts")
            },
            "torch_before": torch_before,
        },
        "inputs": input_document,
        "mode": "plan",
        "mutation": False,
        "new_inventory": {
            "directories": list(new_directories),
            "preexisting_directories": list(preexisting_directories),
            "paths": list(new_paths),
            "paths_count": len(new_paths),
            "wheel_sha256": inputs.anchor["wheel_sha256"],
        },
        "old_inventory": old_inventory.document(),
        "replacement": REPLACEMENT_NAME,
        "schema_version": SCHEMA_VERSION,
    }
    return PreparedReplacement(
        request=normalized_request,
        workspace=workspace,
        prefix=prefix,
        python=python,
        scheme=scheme,
        inputs=inputs,
        old_inventory=old_inventory,
        new_paths=new_paths,
        new_directories=new_directories,
        preexisting_directories=preexisting_directories,
        torch_before=torch_before,
        plan_document=plan,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_canonical_json_no_replace(path: Path, value: object) -> str:
    encoded = canonical_json(value).encode("ascii")
    digest = sha256_bytes(encoded)
    if path.exists() or path.is_symlink():
        raise ReplacementError(f"evidence already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReplacementError(f"evidence already exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


@contextmanager
def workspace_transaction_lock(
    workspace: Path,
    *,
    required_mode: str = "exclusive",
    create_if_missing: bool = True,
):
    if required_mode not in ("shared", "exclusive"):
        raise ReplacementError("unknown environment lock requirement")
    lock_path = workspace / "runs" / ENVIRONMENT_LOCK_NAME
    if not lock_path.parent.is_dir() or lock_path.parent.is_symlink():
        raise ReplacementError("workspace transaction-lock directory is unsafe")
    markers = {
        name: os.environ.get(variable)
        for name, variable in INHERITED_LOCK_ENVIRONMENT.items()
    }
    present = {name for name, value in markers.items() if value is not None}
    if present:
        if present != set(INHERITED_LOCK_ENVIRONMENT):
            raise ReplacementError("inherited environment-lock marker set is incomplete")
        inherited_mode = markers["mode"]
        allowed_modes = (
            {"exclusive"}
            if required_mode == "exclusive"
            else {"shared", "exclusive"}
        )
        if inherited_mode not in allowed_modes or markers["path"] != str(lock_path):
            raise ReplacementError(
                f"replacement requires inherited {required_mode} environment lock"
            )
        try:
            descriptor = int(str(markers["fd"]))
            expected_dev = int(str(markers["dev"]))
            expected_ino = int(str(markers["ino"]))
            controller_pid = int(str(markers["controller_pid"]))
            controller_start_ticks = int(str(markers["controller_start_ticks"]))
        except ValueError as error:
            raise ReplacementError("inherited environment-lock marker is malformed") from error
        if (
            descriptor < 0
            or controller_pid <= 0
            or controller_start_ticks < 0
            or controller_pid != os.getppid()
        ):
            raise ReplacementError("inherited environment-lock FD is invalid")
        try:
            inherited_stat = os.fstat(descriptor)
            path_stat = lock_path.lstat()
        except OSError as error:
            raise ReplacementError("inherited environment-lock FD/path is absent") from error
        if (
            not stat.S_ISREG(inherited_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or lock_path.is_symlink()
            or inherited_stat.st_dev != path_stat.st_dev
            or inherited_stat.st_ino != path_stat.st_ino
            or inherited_stat.st_dev != expected_dev
            or inherited_stat.st_ino != expected_ino
        ):
            raise ReplacementError("inherited environment-lock FD identity mismatch")
        try:
            controller_stat_text = Path(
                f"/proc/{controller_pid}/stat"
            ).read_text(errors="strict")
            observed_controller_ticks = _proc_start_ticks(controller_stat_text)
            controller_fd_stat = Path(
                f"/proc/{controller_pid}/fd/{descriptor}"
            ).stat()
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ReplacementError(
                "inherited environment-lock controller identity is absent"
            ) from error
        if (
            observed_controller_ticks != controller_start_ticks
            or controller_fd_stat.st_dev != inherited_stat.st_dev
            or controller_fd_stat.st_ino != inherited_stat.st_ino
            or not _controller_holds_flock(
                controller_pid,
                inherited_stat,
                str(inherited_mode),
            )
        ):
            raise ReplacementError(
                "inherited environment-lock controller identity mismatch"
            )
        probe_descriptor = os.open(
            lock_path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            try:
                probe_operation = (
                    fcntl.LOCK_SH
                    if inherited_mode == "exclusive"
                    else fcntl.LOCK_EX
                )
                fcntl.flock(probe_descriptor, probe_operation | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(probe_descriptor, fcntl.LOCK_UN)
                qualifier = (
                    "exclusively" if inherited_mode == "exclusive" else "shared"
                )
                raise ReplacementError(
                    f"inherited environment lock is not held {qualifier}"
                )
        finally:
            os.close(probe_descriptor)
        yield {
            "dev": inherited_stat.st_dev,
            "fd": descriptor,
            "inherited": True,
            "ino": inherited_stat.st_ino,
            "mode": inherited_mode,
            "path": _workspace_relative(lock_path, workspace),
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "controller": {
                "pid": controller_pid,
                "start_ticks": controller_start_ticks,
            },
        }
        return
    flags = os.O_RDWR | (os.O_CREAT if create_if_missing else 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        operation = fcntl.LOCK_EX if required_mode == "exclusive" else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplacementError("another Triton environment transaction holds the lock") from error
        yield {
            "dev": os.fstat(descriptor).st_dev,
            "fd": descriptor,
            "inherited": False,
            "ino": os.fstat(descriptor).st_ino,
            "mode": required_mode,
            "path": _workspace_relative(lock_path, workspace),
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "controller": None,
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _proc_process_group(stat_text: str) -> int:
    closing = stat_text.rfind(")")
    if closing < 0:
        raise ValueError("missing proc stat command terminator")
    fields = stat_text[closing + 1 :].strip().split()
    if len(fields) < 3:
        raise ValueError("short proc stat")
    return int(fields[2])


def _proc_start_ticks(stat_text: str) -> int:
    closing = stat_text.rfind(")")
    if closing < 0:
        raise ValueError("missing proc stat command terminator")
    fields = stat_text[closing + 1 :].strip().split()
    if len(fields) < 20:
        raise ValueError("short proc stat for start ticks")
    return int(fields[19])


def _proc_uid(status_text: str) -> int:
    for line in status_text.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1])
    raise ValueError("proc status has no Uid")


def _controller_holds_flock(
    controller_pid: int,
    identity: os.stat_result,
    mode: str,
) -> bool:
    expected_kind = "WRITE" if mode == "exclusive" else "READ"
    try:
        lines = Path("/proc/locks").read_text(errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReplacementError("cannot audit inherited lock in /proc/locks") from error
    for line in lines:
        fields = line.split()
        if len(fields) < 8 or fields[1] != "FLOCK":
            continue
        try:
            owner = int(fields[4])
            major_text, minor_text, inode_text = fields[5].split(":", 2)
            device_major = int(major_text, 16)
            device_minor = int(minor_text, 16)
            inode = int(inode_text)
        except (ValueError, IndexError):
            continue
        if (
            owner == controller_pid
            and fields[3] == expected_kind
            and device_major == os.major(identity.st_dev)
            and device_minor == os.minor(identity.st_dev)
            and inode == identity.st_ino
        ):
            return True
    return False


def scan_environment_users(
    prefix: Path,
    external_natives: Sequence[NativeEntry],
    *,
    proc_root: Path = Path("/proc"),
    allowed_pgid: int | None = None,
    exempt_identities: Mapping[int, int] | None = None,
) -> list[dict[str, object]]:
    """Read /proc only; never signal, stop, or otherwise manage a process."""

    allowed = os.getpgrp() if allowed_pgid is None else allowed_pgid
    native_paths = {Path(entry.path) for entry in external_natives}
    exemptions = {} if exempt_identities is None else dict(exempt_identities)
    prefix_bytes = os.fsencode(str(prefix))
    blockers: list[dict[str, object]] = []
    for process_dir in sorted(
        (path for path in proc_root.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        pid = int(process_dir.name)
        try:
            stat_text = (process_dir / "stat").read_text(errors="strict")
            pgid = _proc_process_group(stat_text)
            uid = _proc_uid((process_dir / "status").read_text(errors="strict"))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError, ValueError) as error:
            if pid == os.getpid():
                raise ReplacementError("cannot audit current process identity") from error
            continue
        if pgid == allowed:
            continue
        expected_start = exemptions.get(pid)
        if expected_start is not None:
            try:
                observed_start = _proc_start_ticks(stat_text)
            except ValueError:
                observed_start = -1
            if observed_start == expected_start:
                continue
        reasons: list[str] = []
        for link_name, reason in (("exe", "prefix-executable"), ("cwd", "prefix-cwd")):
            try:
                resolved = (process_dir / link_name).resolve(strict=True)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if _is_within(resolved, prefix, allow_root=True):
                reasons.append(reason)
            if resolved in native_paths:
                reasons.append(f"external-native-{link_name}")
        try:
            maps_text = (process_dir / "maps").read_text(errors="strict")
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, UnicodeDecodeError):
            maps_text = ""
        for line in maps_text.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            value = fields[5].removesuffix(" (deleted)")
            path = _absolute_lexical(Path(value))
            if _is_within(path, prefix, allow_root=True):
                reasons.append("prefix-mapping")
            if path in native_paths:
                reasons.append("external-libtriton-mapping")
        try:
            environment = (process_dir / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            environment = b""
        if prefix_bytes in environment:
            reasons.append("prefix-environment")
        if reasons:
            blockers.append(
                {
                    "pgid": pgid,
                    "pid": pid,
                    "reasons": sorted(set(reasons)),
                    "uid": uid,
                }
            )
    return blockers


def assert_exclusive_prefix(
    prepared: PreparedReplacement,
    phase: str,
    *,
    exempt_identities: Mapping[int, int] | None = None,
) -> dict[str, object]:
    blockers = scan_environment_users(
        prepared.prefix,
        prepared.old_inventory.native_files,
        exempt_identities=exempt_identities,
    )
    if blockers:
        raise ReplacementError(
            f"foreign process uses environment/native at {phase}: {blockers}"
        )
    return {"blockers": [], "phase": phase, "run_pgid": os.getpgrp()}


class TransactionInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"transaction interrupted by signal {signum}")
        self.signum = signum


class TransactionSignalGuard:
    def __init__(self) -> None:
        self.signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        self.previous: dict[int, object] = {}
        self.rollback_active = False
        self.received: list[int] = []

    def __enter__(self):
        self.previous = {kind: signal.getsignal(kind) for kind in self.signals}

        def handler(signum: int, _frame: object) -> None:
            self.received.append(signum)
            if not self.rollback_active:
                raise TransactionInterrupted(signum)

        for kind in self.signals:
            signal.signal(kind, handler)
        return self

    def begin_rollback(self) -> None:
        self.rollback_active = True

    def __exit__(self, _type, _value, _traceback) -> None:
        for kind, previous in self.previous.items():
            signal.signal(kind, previous)


def _copy_regular_to_blob(
    source: Path,
    blob: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if blob.exists():
        if (
            blob.is_symlink()
            or not blob.is_file()
            or blob.stat().st_size != expected_size
            or sha256_file(blob) != expected_sha256
        ):
            raise ReplacementError("content-addressed backup blob is inconsistent")
        return
    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        value = getattr(os, name, 0)
        source_flags |= value
        destination_flags |= value
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(source, source_flags)
        destination_descriptor = os.open(blob, destination_flags, 0o400)
    except OSError as error:
        if source_descriptor is not None:
            os.close(source_descriptor)
        raise ReplacementError(f"cannot create backup blob for {source}") from error
    digest = hashlib.sha256()
    total = 0
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ReplacementError(f"backup source ceased to be regular: {source}")
        while chunk := os.read(source_descriptor, 8 << 20):
            digest.update(chunk)
            total += len(chunk)
            written = 0
            while written < len(chunk):
                written += os.write(destination_descriptor, chunk[written:])
        os.fsync(destination_descriptor)
    except Exception:
        try:
            blob.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        assert source_descriptor is not None
        assert destination_descriptor is not None
        os.close(source_descriptor)
        os.close(destination_descriptor)
    if total != expected_size or digest.hexdigest() != expected_sha256:
        try:
            blob.unlink()
        except FileNotFoundError:
            pass
        raise ReplacementError(f"backup source changed while copying: {source}")
    os.chmod(blob, 0o400)


def _entry_matches(path: Path, entry: FileEntry) -> bool:
    try:
        file_stat = path.lstat()
    except OSError:
        return False
    if stat.S_IMODE(file_stat.st_mode) != entry.mode:
        return False
    if entry.kind == "regular":
        return (
            stat.S_ISREG(file_stat.st_mode)
            and file_stat.st_size == entry.size
            and entry.sha256 is not None
            and sha256_file(path) == entry.sha256
        )
    if entry.kind == "symlink":
        return (
            stat.S_ISLNK(file_stat.st_mode)
            and base64.b64encode(os.fsencode(os.readlink(path))).decode("ascii")
            == entry.link_target_b64
        )
    return False


def _native_matches(entry: NativeEntry) -> bool:
    path = Path(entry.path)
    try:
        file_stat = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(file_stat.st_mode)
        and not path.is_symlink()
        and stat.S_IMODE(file_stat.st_mode) == entry.mode
        and file_stat.st_size == entry.size
        and sha256_file(path) == entry.sha256
    )


def verify_old_inventory(
    prepared: PreparedReplacement,
    *,
    limits: ReplacementLimits = ReplacementLimits(),
) -> None:
    for entry in prepared.old_inventory.files:
        path = prepared.prefix / entry.path
        _path_without_symlink_parents(path, prepared.prefix)
        if not _entry_matches(path, entry):
            raise ReplacementError(f"old RECORD ownership changed: {path}")
    for entry in prepared.old_inventory.removable_directories:
        path = prepared.prefix / entry.path
        try:
            path_stat = path.lstat()
        except OSError as error:
            raise ReplacementError(f"old owned directory disappeared: {path}") from error
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or stat.S_IMODE(path_stat.st_mode) != entry.mode
        ):
            raise ReplacementError(f"old owned directory changed: {path}")
    for native in prepared.old_inventory.native_files:
        if not _native_matches(native):
            raise ReplacementError(f"external native changed: {native.path}")
    verify_external_source_identity(prepared, limits=limits)


RUNTIME_AUDIT_PROGRAM = r"""
import importlib.metadata
import json
import pathlib
import sys

prefix = pathlib.Path(sys.argv[1]).resolve(strict=True)
site = pathlib.Path(sys.argv[2]).resolve(strict=True)
finder_module = sys.argv[3]

def below(value, root):
    path = pathlib.Path(value).resolve(strict=True)
    return path == root or root in path.parents

def carrier_modules(value):
    return sorted({
        item for item in (
            getattr(value, "__module__", None),
            getattr(type(value), "__module__", None),
        ) if isinstance(item, str) and item
    })

if pathlib.Path(sys.prefix).resolve(strict=True) != prefix:
    raise RuntimeError("runtime audit escaped target prefix")
triton_distributions = [
    dist for dist in importlib.metadata.distributions(path=[str(site)])
    if (dist.metadata.get("Name") or "").lower().replace("_", "-") == "triton"
]
if len(triton_distributions) != 1:
    raise RuntimeError("runtime audit does not see one Triton distribution")
dist = triton_distributions[0]

import triton
import triton._C.libtriton
import triton.backends
import triton.compiler
import triton.language
import triton.runtime
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler.compiler import make_backend
from triton.runtime.cache import triton_key as direct_triton_key

target = GPUTarget("cuda", 120, 32)
backend = make_backend(target)
options = backend.parse_options({})
selected_ptxas = get_ptxas(120)
direct_key = direct_triton_key()

import torch
from torch.utils._triton import get_triton_version, has_triton_package
from torch._inductor.runtime import triton_compat
from torch._inductor.codecache import triton_key as torch_triton_key

compat_key = triton_compat.triton_key()
torch_key = torch_triton_key()
module_paths = {}
native_submodules = []
libtriton_owner = pathlib.Path(
    sys.modules["triton._C.libtriton"].__file__
).resolve(strict=True)
for name, module in sorted(sys.modules.items()):
    if name != "triton" and not name.startswith("triton."):
        continue
    values = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        values.append(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        values.extend(item for item in module_path if isinstance(item, str))
    resolved = sorted({str(pathlib.Path(value).resolve(strict=True)) for value in values})
    if not resolved:
        if not name.startswith("triton._C.libtriton."):
            raise RuntimeError(f"loaded Triton module has no path: {name}")
        resolved = [str(libtriton_owner)]
        native_submodules.append(name)
    module_paths[name] = resolved

mapped = []
for line in pathlib.Path("/proc/self/maps").read_text(errors="strict").splitlines():
    if "libtriton" not in line.lower():
        continue
    fields = line.split(maxsplit=5)
    if len(fields) != 6 or not fields[5].startswith("/"):
        raise RuntimeError(f"unrecognized libtriton map: {line}")
    value = fields[5].removesuffix(" (deleted)")
    mapped.append(str(pathlib.Path(value).resolve(strict=True)))
mapped = sorted(set(mapped))
if not mapped:
    raise RuntimeError("runtime audit mapped no libtriton")

carriers = {}
for carrier_name, values in (
    ("meta_path", sys.meta_path),
    ("path_hooks", sys.path_hooks),
    ("path_importer_cache", sys.path_importer_cache.values()),
):
    carriers[carrier_name] = sorted({
        module_name
        for value in values
        for module_name in carrier_modules(value)
        if module_name == finder_module
        or ("triton" in module_name.lower() and "editable" in module_name.lower())
    })
loaded_editable = sorted(
    name for name in sys.modules
    if name == finder_module
    or ("triton" in name.lower() and "editable" in name.lower())
)
symbols = {
    name: getattr(triton_compat, name, None) is not None
    for name in ("Config", "CompiledKernel", "GPUTarget", "JITFunction", "tl", "triton_key")
}
print(json.dumps({
    "backend": {
        "arch": getattr(options, "arch", None),
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "target": [target.backend, target.arch, target.warp_size],
    },
    "distribution": {
        "dist_info": str(pathlib.Path(dist._path).resolve(strict=True)),
        "name": dist.metadata.get("Name"),
        "version": dist.version,
    },
    "editable": {"carriers": carriers, "loaded_modules": loaded_editable},
    "keys": {"direct": direct_key, "torch_compat": compat_key, "torch_inductor": torch_key},
    "libtriton_maps": mapped,
    "module_paths": module_paths,
    "module_version": triton.__version__,
    "native_submodules": native_submodules,
    "ptxas_blackwell": {
        "path": str(pathlib.Path(selected_ptxas.path).resolve(strict=True)),
        "reported_release": selected_ptxas.version,
    },
    "sys_path": list(sys.path),
    "torch": {
        "cuda": torch.version.cuda,
        "file": str(pathlib.Path(torch.__file__).resolve(strict=True)),
        "git_version": torch.version.git_version,
        "has_triton_package": has_triton_package(),
        "hip": torch.version.hip,
        "triton_compat_has_triton": triton_compat.HAS_TRITON,
        "triton_compat_symbols": symbols,
        "triton_version": list(get_triton_version()),
        "version": str(torch.__version__),
    },
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def run_normal_runtime_audit(
    prepared: PreparedReplacement,
    *,
    scratch: Path,
    limits: ReplacementLimits,
) -> dict[str, object]:
    environment = _transaction_environment(scratch)
    try:
        with _temporary_file_default(scratch / "tmp"):
            return probe.run_json_command(
                [
                    str(prepared.python),
                    "-I",
                    "-B",
                    "-c",
                    RUNTIME_AUDIT_PROGRAM,
                    str(prepared.prefix),
                    str(prepared.scheme.platlib),
                    prepared.old_inventory.finder_module,
                ],
                environment=environment,
                limits=probe.ProbeLimits(
                    max_output_bytes=limits.max_output_bytes,
                    timeout_seconds=limits.timeout_seconds,
                ),
                description="fresh normal-site Triton runtime audit",
            )
    except probe.ProbeError as error:
        raise ReplacementError(f"normal-site runtime audit failed: {error}") from error


def validate_old_runtime_audit(
    report: dict[str, object], prepared: PreparedReplacement
) -> None:
    distribution = report.get("distribution")
    expected_dist = prepared.prefix / prepared.old_inventory.dist_info
    if (
        not isinstance(distribution, dict)
        or distribution.get("name") != "triton"
        or distribution.get("version") != prepared.old_inventory.distribution_version
        or not isinstance(distribution.get("dist_info"), str)
        or Path(distribution["dist_info"]).resolve(strict=True) != expected_dist
    ):
        raise ReplacementError("pre-transaction runtime distribution differs from inventory")
    editable = report.get("editable")
    if not isinstance(editable, dict):
        raise ReplacementError("pre-transaction editable runtime report is absent")
    carriers = editable.get("carriers")
    loaded = editable.get("loaded_modules")
    finder = prepared.old_inventory.finder_module
    if (
        not isinstance(carriers, dict)
        or not isinstance(loaded, list)
        or finder not in loaded
        or finder not in set(
            item
            for values in carriers.values()
            if isinstance(values, list)
            for item in values
        )
    ):
        raise ReplacementError("pre-transaction editable finder is not active as inventoried")
    module_paths = report.get("module_paths")
    if not isinstance(module_paths, dict) or not module_paths:
        raise ReplacementError("pre-transaction Triton module inventory is absent")
    package_root = Path(prepared.old_inventory.package_root)
    for values in module_paths.values():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ReplacementError("pre-transaction Triton module inventory is malformed")
        for value in values:
            resolved = Path(value).resolve(strict=True)
            native_paths = {Path(item.path) for item in prepared.old_inventory.native_files}
            if not _is_within(resolved, package_root, allow_root=True) and resolved not in native_paths:
                raise ReplacementError("pre-transaction Triton import escaped editable source")
    expected_native = {Path(item.path) for item in prepared.old_inventory.native_files}
    maps = report.get("libtriton_maps")
    if not isinstance(maps, list) or {Path(item).resolve(strict=True) for item in maps} != expected_native:
        raise ReplacementError("pre-transaction mapped native differs from sealed inventory")


def create_backup(
    prepared: PreparedReplacement,
    *,
    old_runtime: dict[str, object],
) -> tuple[dict[str, object], str]:
    root = prepared.request.backup_root
    if root.is_symlink() or not root.is_dir():
        raise ReplacementError("backup root is absent or unsafe")
    blobs = root / "blobs"
    scratch = root / "scratch"
    blobs.mkdir(mode=0o700)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ReplacementError("transaction scratch root is absent or unsafe")
    entries: list[dict[str, object]] = []
    for entry in prepared.old_inventory.files:
        source = prepared.prefix / entry.path
        if not _entry_matches(source, entry):
            raise ReplacementError(f"old file changed before backup: {source}")
        record = entry.document()
        if entry.kind == "regular":
            assert entry.sha256 is not None
            blob = blobs / entry.sha256
            _copy_regular_to_blob(
                source,
                blob,
                expected_sha256=entry.sha256,
                expected_size=entry.size,
            )
            record["blob"] = f"blobs/{entry.sha256}"
        entries.append(record)
    native_entries: list[dict[str, object]] = []
    for entry in prepared.old_inventory.native_files:
        if not _native_matches(entry):
            raise ReplacementError(f"external native changed before backup: {entry.path}")
        blob = blobs / entry.sha256
        _copy_regular_to_blob(
            Path(entry.path),
            blob,
            expected_sha256=entry.sha256,
            expected_size=entry.size,
        )
        native_entries.append({**entry.document(), "blob": f"blobs/{entry.sha256}"})
    _fsync_directory(blobs)
    manifest: dict[str, object] = {
        "backup": BACKUP_NAME,
        "directories": [
            item.document() for item in prepared.old_inventory.removable_directories
        ],
        "environment_prefix": _workspace_relative(
            prepared.prefix, prepared.workspace
        ),
        "input_identities": prepared.plan_document["inputs"],
        "files": entries,
        "files_count": len(entries),
        "native_files": native_entries,
        "native_files_count": len(native_entries),
        "native_mutation_allowed": False,
        "new_inventory": {
            "directories": list(prepared.new_directories),
            "paths": list(prepared.new_paths),
            "preexisting_directories": list(prepared.preexisting_directories),
        },
        "old_inventory": prepared.old_inventory.document(),
        "old_inventory_sha256": prepared.old_inventory.document()["sha256"],
        "old_runtime": old_runtime,
        "prepared_inputs": {
            "anchor": prepared.inputs.anchor,
            "audit_document": prepared.inputs.audit_document,
            "environment_lock": prepared.inputs.environment_lock,
            "probe_document": prepared.inputs.probe_document,
            "smoke_document": prepared.inputs.smoke_document,
            "versions_lock": prepared.inputs.versions_lock,
        },
        "python": _workspace_relative(prepared.python, prepared.workspace),
        "scheme": {
            name: _prefix_relative(getattr(prepared.scheme, name), prepared.prefix)
            for name in ("data", "headers", "platlib", "purelib", "scripts")
        },
        "terminal_evidence": _workspace_relative(
            prepared.request.evidence, prepared.workspace
        ),
        "torch_before": prepared.torch_before,
        "schema_version": SCHEMA_VERSION,
    }
    manifest_path = root / "manifest.json"
    digest = publish_canonical_json_no_replace(manifest_path, manifest)
    return manifest, digest


def _atomic_replace_canonical_json(path: Path, value: object) -> str:
    encoded = canonical_json(value).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return sha256_bytes(encoded)


def load_journal(backup_root: Path) -> dict[str, object]:
    journal, _ = load_canonical_json(
        backup_root / "journal.json",
        "replacement journal",
        max_bytes=4 << 20,
    )
    if (
        journal.get("schema_version") != 1
        or journal.get("transaction") != REPLACEMENT_NAME
        or not isinstance(journal.get("generation"), int)
        or not isinstance(journal.get("phase"), str)
    ):
        raise ReplacementError("replacement journal schema is invalid")
    return journal


def update_journal(
    backup_root: Path,
    *,
    phase: str,
    manifest_sha256: str | None,
    mutation_started: bool,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    journal_path = backup_root / "journal.json"
    if journal_path.exists():
        previous = load_journal(backup_root)
        generation = int(previous["generation"]) + 1
    else:
        generation = 1
    value: dict[str, object] = {
        "generation": generation,
        "manifest_sha256": manifest_sha256,
        "mutation_started": mutation_started,
        "phase": phase,
        "run_pgid": os.getpgrp(),
        "run_pid": os.getpid(),
        "schema_version": 1,
        "transaction": REPLACEMENT_NAME,
    }
    if extra:
        value.update(extra)
    _atomic_replace_canonical_json(journal_path, value)
    return value


def create_initialization_staging(backup_root: Path) -> Path:
    """Create a sibling staging root without making backup_root visible."""

    if backup_root.exists() or backup_root.is_symlink():
        raise ReplacementError(f"backup root is not fresh: {backup_root}")
    parent = backup_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ReplacementError("backup-root parent is absent or unsafe")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{backup_root.name}.initializing.",
            dir=parent,
        )
    )
    os.chmod(staging, 0o700)
    _fsync_directory(parent)
    return staging


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ReplacementError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ReplacementError(f"backup root ceased to be fresh: {destination}")
    raise ReplacementError(
        f"cannot atomically publish initialized backup root: "
        f"{os.strerror(error_number)}"
    )


def _cleanup_initialization_staging(staging: Path) -> None:
    if not staging.exists() and not staging.is_symlink():
        return
    if staging.is_symlink() or not staging.is_dir():
        raise ReplacementError("backup initialization staging became unsafe")
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        allowed = path.name == "journal.json" or re.fullmatch(
            r"\.journal\.json\.[A-Za-z0-9_-]+\.partial",
            path.name,
        ) is not None
        if not allowed or path.is_symlink() or not path.is_file():
            raise ReplacementError(
                f"backup initialization staging contains unknown entry: {path}"
            )
        path.unlink()
    _fsync_directory(staging)
    staging.rmdir()
    _fsync_directory(staging.parent)


def initialize_backup_root(
    backup_root: Path,
    *,
    lock_evidence: Mapping[str, object],
) -> None:
    """Publish a backup root for which journal visibility is atomic."""

    staging = create_initialization_staging(backup_root)
    published = False
    try:
        update_journal(
            staging,
            phase="initializing",
            manifest_sha256=None,
            mutation_started=False,
            extra={"lock": dict(lock_evidence)},
        )
        _fsync_directory(staging)
        _fsync_directory(staging.parent)
        _rename_directory_no_replace(staging, backup_root)
        published = True
        _fsync_directory(backup_root.parent)
    finally:
        if not published:
            _cleanup_initialization_staging(staging)


def _require_existing_backup_root(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise ReplacementError("backup root must be absolute")
    lexical = _absolute_lexical(path)
    builds = workspace / "builds"
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("backup root is absent") from error
    if (
        lexical != resolved
        or resolved.is_symlink()
        or not resolved.is_dir()
        or not _is_within(resolved, builds)
    ):
        raise ReplacementError("backup root is not a real workspace build directory")
    return resolved


def _mode_from_document(value: object, description: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-7]{4}", value) is None:
        raise ReplacementError(f"{description} mode is malformed")
    return int(value, 8)


def _file_entry_from_document(value: object) -> FileEntry:
    if not isinstance(value, dict):
        raise ReplacementError("backup file entry is not an object")
    roles = value.get("roles")
    if not isinstance(roles, list) or any(not isinstance(item, str) for item in roles):
        raise ReplacementError("backup file roles are malformed")
    path = value.get("path")
    kind = value.get("kind")
    size = value.get("size")
    sha = value.get("sha256")
    link = value.get("link_target_b64")
    record_digest_value = value.get("record_digest")
    record_size_value = value.get("record_size")
    if (
        not isinstance(path, str)
        or kind not in ("regular", "symlink")
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or (sha is not None and not isinstance(sha, str))
        or (link is not None and not isinstance(link, str))
        or not isinstance(record_digest_value, str)
        or not isinstance(record_size_value, str)
    ):
        raise ReplacementError("backup file entry is malformed")
    _safe_relative(path, "backup file")
    if kind == "regular":
        validate_sha256(sha, "backup file SHA256")
    elif link is None:
        raise ReplacementError("backup symlink lacks target")
    return FileEntry(
        path=path,
        kind=kind,
        mode=_mode_from_document(value.get("mode"), "backup file"),
        size=size,
        sha256=sha,
        link_target_b64=link,
        record_digest=record_digest_value,
        record_size=record_size_value,
        roles=tuple(roles),
    )


def _old_inventory_from_document(value: object) -> OldInventory:
    if not isinstance(value, dict):
        raise ReplacementError("backup old inventory is absent")
    expected_sha = validate_sha256(value.get("sha256"), "old inventory SHA256")
    unsigned = dict(value)
    unsigned.pop("sha256")
    if sha256_bytes(canonical_json(unsigned).encode("ascii")) != expected_sha:
        raise ReplacementError("backup old-inventory digest is inconsistent")
    files_value = value.get("files")
    directories_value = value.get("removable_directories")
    native_value = value.get("native_files")
    if (
        not isinstance(files_value, list)
        or not isinstance(directories_value, list)
        or not isinstance(native_value, list)
    ):
        raise ReplacementError("backup old inventory collections are malformed")
    files = tuple(_file_entry_from_document(item) for item in files_value)
    directories: list[DirectoryEntry] = []
    for item in directories_value:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ReplacementError("backup directory entry is malformed")
        _safe_relative(item["path"], "backup directory")
        directories.append(
            DirectoryEntry(
                path=item["path"],
                mode=_mode_from_document(item.get("mode"), "backup directory"),
            )
        )
    natives: list[NativeEntry] = []
    for item in native_value:
        if not isinstance(item, dict):
            raise ReplacementError("backup native entry is malformed")
        native_path = item.get("path")
        native_size = item.get("size")
        native_sha = validate_sha256(item.get("sha256"), "backup native SHA256")
        if (
            not isinstance(native_path, str)
            or not Path(native_path).is_absolute()
            or not isinstance(native_size, int)
            or isinstance(native_size, bool)
            or native_size < 0
        ):
            raise ReplacementError("backup native entry is malformed")
        natives.append(
            NativeEntry(
                path=native_path,
                mode=_mode_from_document(item.get("mode"), "backup native"),
                size=native_size,
                sha256=native_sha,
            )
        )
    direct = value.get("direct_url")
    source_identity = value.get("external_source_identity")
    string_fields = {
        name: value.get(name)
        for name in (
            "distribution_version",
            "dist_info",
            "editable_source",
            "package_root",
            "finder_module",
        )
    }
    if (
        not isinstance(direct, dict)
        or not isinstance(source_identity, dict)
        or any(not isinstance(item, str) for item in string_fields.values())
    ):
        raise ReplacementError("backup old-inventory identity is malformed")
    return OldInventory(
        distribution_version=string_fields["distribution_version"],
        dist_info=string_fields["dist_info"],
        direct_url=direct,
        editable_source=string_fields["editable_source"],
        package_root=string_fields["package_root"],
        finder_module=string_fields["finder_module"],
        external_source_identity=source_identity,
        files=files,
        removable_directories=tuple(directories),
        native_files=tuple(natives),
    )


def _workspace_path_from_identity(
    identity: Mapping[str, object], workspace: Path, description: str
) -> Path:
    value = identity.get("path")
    if not isinstance(value, str):
        raise ReplacementError(f"{description} path is absent")
    relative = _safe_relative(value, description)
    return workspace.joinpath(*relative.parts)


def prepared_from_backup(
    workspace: Path,
    backup_root: Path,
    *,
    evidence_override: Path | None = None,
) -> tuple[PreparedReplacement, dict[str, object], str]:
    journal = load_journal(backup_root)
    expected_manifest_sha = validate_sha256(
        journal.get("manifest_sha256"), "journal manifest SHA256"
    )
    manifest_path = backup_root / "manifest.json"
    manifest, manifest_raw = load_canonical_json(
        manifest_path, "replacement backup manifest", max_bytes=64 << 20
    )
    if sha256_bytes(manifest_raw) != expected_manifest_sha:
        raise ReplacementError("backup manifest differs from journal anchor")
    if manifest.get("backup") != BACKUP_NAME or manifest.get("schema_version") != 1:
        raise ReplacementError("backup manifest schema is invalid")
    input_identities = manifest.get("input_identities")
    prepared_inputs_value = manifest.get("prepared_inputs")
    if not isinstance(input_identities, dict) or not isinstance(prepared_inputs_value, dict):
        raise ReplacementError("backup manifest input provenance is absent")
    required_identities = (
        "environment_lock",
        "gpu_smoke_evidence",
        "wheel",
        "wheel_audit_evidence",
        "wheel_probe_evidence",
        "versions_lock",
    )
    identities: dict[str, dict[str, object]] = {}
    for name in required_identities:
        value = input_identities.get(name)
        if not isinstance(value, dict):
            raise ReplacementError(f"backup input identity is absent: {name}")
        identities[name] = value
    prefix_value = manifest.get("environment_prefix")
    python_value = manifest.get("python")
    terminal_value = manifest.get("terminal_evidence")
    if not all(isinstance(item, str) for item in (prefix_value, python_value, terminal_value)):
        raise ReplacementError("backup environment paths are malformed")
    prefix = _require_prefix(
        workspace.joinpath(*_safe_relative(prefix_value, "backup prefix").parts),
        workspace,
    )
    python = workspace.joinpath(*_safe_relative(python_value, "backup Python").parts)
    try:
        resolved_recovery_python = python.resolve(strict=True)
    except OSError as error:
        raise ReplacementError("backup environment Python is absent") from error
    if (
        not _is_within(python, prefix)
        or not _is_within(resolved_recovery_python, prefix)
        or not os.access(python, os.X_OK)
    ):
        raise ReplacementError("backup environment Python is absent or escaped prefix")
    evidence = (
        evidence_override
        if evidence_override is not None
        else workspace.joinpath(*_safe_relative(terminal_value, "terminal evidence").parts)
    )
    request = ReplacementRequest(
        workspace=workspace,
        prefix=prefix,
        wheel=_workspace_path_from_identity(identities["wheel"], workspace, "wheel"),
        wheel_audit_evidence=_workspace_path_from_identity(
            identities["wheel_audit_evidence"], workspace, "wheel audit"
        ),
        expected_wheel_audit_evidence_sha256=str(
            identities["wheel_audit_evidence"].get("sha256")
        ),
        wheel_probe_evidence=_workspace_path_from_identity(
            identities["wheel_probe_evidence"], workspace, "wheel probe"
        ),
        expected_wheel_probe_evidence_sha256=str(
            identities["wheel_probe_evidence"].get("sha256")
        ),
        gpu_smoke_evidence=_workspace_path_from_identity(
            identities["gpu_smoke_evidence"], workspace, "GPU smoke"
        ),
        expected_gpu_smoke_evidence_sha256=str(
            identities["gpu_smoke_evidence"].get("sha256")
        ),
        environment_lock=_workspace_path_from_identity(
            identities["environment_lock"], workspace, "environment lock"
        ),
        expected_environment_lock_sha256=str(
            identities["environment_lock"].get("sha256")
        ),
        backup_root=backup_root,
        evidence=evidence,
    )
    scheme_value = manifest.get("scheme")
    if not isinstance(scheme_value, dict):
        raise ReplacementError("backup install scheme is absent")
    scheme_paths: dict[str, Path] = {}
    for name in ("data", "headers", "platlib", "purelib", "scripts"):
        value = scheme_value.get(name)
        if not isinstance(value, str):
            raise ReplacementError("backup install scheme is malformed")
        scheme_paths[name] = prefix.joinpath(
            *_safe_relative(value, f"backup scheme {name}").parts
        )
        if not _is_within(scheme_paths[name], prefix, allow_root=True):
            raise ReplacementError(f"backup install scheme {name} escaped prefix")
        _path_without_symlink_parents(scheme_paths[name], prefix)
    scheme = probe.InstallScheme(
        prefix=prefix,
        purelib=scheme_paths["purelib"],
        platlib=scheme_paths["platlib"],
        scripts=scheme_paths["scripts"],
        headers=scheme_paths["headers"],
        data=scheme_paths["data"],
        python_version=(3, 14, 6),
    )
    old_inventory = _old_inventory_from_document(manifest.get("old_inventory"))
    new_inventory = manifest.get("new_inventory")
    if not isinstance(new_inventory, dict):
        raise ReplacementError("backup new inventory is absent")
    collections: dict[str, tuple[str, ...]] = {}
    for name in ("paths", "directories", "preexisting_directories"):
        values = new_inventory.get(name)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ReplacementError(f"backup new inventory {name} is malformed")
        for item in values:
            _safe_relative(item, f"backup new inventory {name}")
        collections[name] = tuple(values)
    environment_lock = prepared_inputs_value.get("environment_lock")
    audit_document = prepared_inputs_value.get("audit_document")
    probe_document = prepared_inputs_value.get("probe_document")
    smoke_document = prepared_inputs_value.get("smoke_document")
    anchor = prepared_inputs_value.get("anchor")
    versions_lock = prepared_inputs_value.get("versions_lock")
    if not all(
        isinstance(item, dict)
        for item in (
            environment_lock,
            audit_document,
            probe_document,
            smoke_document,
            anchor,
            versions_lock,
        )
    ):
        raise ReplacementError("backup prepared-input documents are malformed")
    inputs = ValidatedInputs(
        environment_lock=environment_lock,
        environment_lock_raw=b"",
        audit_document=audit_document,
        audit_raw=b"",
        anchor=anchor,
        probe_document=probe_document,
        probe_raw=b"",
        smoke_document=smoke_document,
        smoke_raw=b"",
        versions_lock=versions_lock,
        versions_lock_path=_workspace_path_from_identity(
            identities["versions_lock"], workspace, "VERSIONS.lock"
        ),
    )
    torch_before = manifest.get("torch_before")
    if not isinstance(torch_before, dict):
        raise ReplacementError("backup frozen Torch identity is absent")
    plan = {"inputs": input_identities, "mode": "recovery"}
    prepared = PreparedReplacement(
        request=request,
        workspace=workspace,
        prefix=prefix,
        python=python,
        scheme=scheme,
        inputs=inputs,
        old_inventory=old_inventory,
        new_paths=collections["paths"],
        new_directories=collections["directories"],
        preexisting_directories=collections["preexisting_directories"],
        torch_before=torch_before,
        plan_document=plan,
    )
    return prepared, manifest, expected_manifest_sha


def _remove_old_inventory(prepared: PreparedReplacement) -> list[str]:
    verify_old_inventory(prepared)
    removed: list[str] = []
    for entry in sorted(
        prepared.old_inventory.files,
        key=lambda item: (len(Path(item.path).parts), item.path),
        reverse=True,
    ):
        path = prepared.prefix / entry.path
        _path_without_symlink_parents(path, prepared.prefix)
        if not _entry_matches(path, entry):
            raise ReplacementError(f"old deletion target changed: {path}")
        path.unlink()
        removed.append(entry.path)
        _fsync_directory(path.parent)
    directory_map = {
        item.path: item for item in prepared.old_inventory.removable_directories
    }
    for relative in sorted(
        directory_map,
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    ):
        path = prepared.prefix / relative
        _path_without_symlink_parents(path, prepared.prefix)
        entry = directory_map[relative]
        path_stat = path.lstat()
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or stat.S_IMODE(path_stat.st_mode) != entry.mode
        ):
            raise ReplacementError(f"old directory deletion target changed: {path}")
        try:
            path.rmdir()
        except OSError as error:
            raise ReplacementError(
                f"old owned directory is not empty at deletion: {path}"
            ) from error
        _fsync_directory(path.parent)
    for entry in prepared.old_inventory.native_files:
        if not _native_matches(entry):
            raise ReplacementError("external native changed during prefix deletion")
    return removed


def _read_installed_record(
    scheme: object, *, max_bytes: int
) -> tuple[Path, bytes, dict[str, tuple[str, str, Path]]]:
    record_path = scheme.platlib / probe.DIST_INFO / "RECORD"
    raw = _read_regular_file_once(
        record_path, "installed Triton RECORD", max_bytes=max_bytes
    )
    try:
        text = raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ReplacementError("installed Triton RECORD is invalid") from error
    records: dict[str, tuple[str, str, Path]] = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in records:
            raise ReplacementError("installed Triton RECORD has malformed/duplicate rows")
        try:
            target = probe._installed_record_target(row[0], scheme)
        except probe.ProbeError as error:
            raise ReplacementError(f"installed RECORD target is unsafe: {error}") from error
        records[row[0]] = (row[1], row[2], target)
    return record_path, raw, records


def _target_triton_carriers(scheme: object) -> list[str]:
    artifacts: list[str] = []
    finder_re = re.compile(r"__editable___triton_.*_finder(?:\.py|\..*\.pyc)$", re.I)
    for root in {scheme.platlib, scheme.purelib}:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            lower = path.name.casefold()
            if (
                lower.endswith(".egg-link") and "triton" in lower
                or lower.endswith(".pth") and "triton" in lower
                or finder_re.fullmatch(path.name) is not None
            ):
                artifacts.append(_prefix_relative(path, scheme.prefix))
        pycache = root / "__pycache__"
        if pycache.is_dir() and not pycache.is_symlink():
            for path in pycache.iterdir():
                if finder_re.fullmatch(path.name) is not None:
                    artifacts.append(_prefix_relative(path, scheme.prefix))
    return sorted(set(artifacts))


def verify_new_install(
    prepared: PreparedReplacement,
    installation: dict[str, object],
    *,
    limits: ReplacementLimits,
) -> dict[str, object]:
    scheme = prepared.scheme
    carriers = _target_triton_carriers(scheme)
    if carriers:
        raise ReplacementError(f"installed environment retained Triton carriers: {carriers}")
    dist_infos = sorted(
        path
        for root in {scheme.platlib, scheme.purelib}
        for path in root.glob("triton-*.dist-info")
    )
    expected_dist = scheme.platlib / probe.DIST_INFO
    if dist_infos != [expected_dist] or expected_dist.is_symlink():
        raise ReplacementError("installed environment does not own exact Triton dist-info")
    metadata_raw = _read_regular_file_once(
        expected_dist / "METADATA", "installed Triton METADATA", max_bytes=4 << 20
    )
    message = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
    if (
        _normalized_distribution_name(_single_metadata_header(message, "Name"))
        != "triton"
        or _single_metadata_header(message, "Version")
        != probe.TRITON_DISTRIBUTION_VERSION
    ):
        raise ReplacementError("installed Triton distribution identity is wrong")

    direct_path = expected_dist / "direct_url.json"
    direct_document, direct_raw = load_canonical_json(
        direct_path,
        "installed Triton direct_url.json",
        max_bytes=1 << 20,
    )
    expected_direct = probe.direct_url_document(
        prepared.request.wheel, prepared.inputs.anchor["wheel_sha256"]
    )
    if direct_document != expected_direct:
        raise ReplacementError("installed direct_url.json does not name audited wheel")

    record_path, record_raw, records = _read_installed_record(
        scheme, max_bytes=limits.max_record_bytes
    )
    installed_record = installation.get("installed_record")
    if (
        not isinstance(installed_record, dict)
        or installed_record.get("sha256") != sha256_bytes(record_raw)
        or installed_record.get("size") != len(record_raw)
    ):
        raise ReplacementError("installed RECORD changed after stdlib install")
    archive_members = installation.get("archive_members")
    if not isinstance(archive_members, list):
        raise ReplacementError("stdlib installation manifest is absent")
    expected: dict[str, tuple[str, int]] = {}
    for member in archive_members:
        if (
            not isinstance(member, dict)
            or not isinstance(member.get("installed_path"), str)
            or not isinstance(member.get("sha256"), str)
            or not isinstance(member.get("size"), int)
        ):
            raise ReplacementError("stdlib installation manifest is malformed")
        expected[member["installed_path"]] = (member["sha256"], member["size"])
    direct_relative = probe._record_relative(direct_path, scheme.platlib)
    expected[direct_relative] = (sha256_bytes(direct_raw), len(direct_raw))
    record_relative = probe._record_relative(record_path, scheme.platlib)
    if set(records) != set(expected) | {record_relative}:
        raise ReplacementError("installed RECORD ownership set differs from installer")
    entries: list[dict[str, object]] = []
    for relative in sorted(records):
        encoded_digest, encoded_size, target = records[relative]
        if target.is_symlink() or not target.is_file():
            raise ReplacementError(f"installed RECORD target is not regular: {relative}")
        if relative == record_relative:
            if encoded_digest or encoded_size:
                raise ReplacementError("installed RECORD self-row is not empty")
            digest = sha256_file(target)
            size = target.stat().st_size
        else:
            try:
                digest = probe._decode_record_digest(encoded_digest, relative)
            except probe.ProbeError as error:
                raise ReplacementError(f"installed RECORD digest failed: {error}") from error
            if re.fullmatch(r"0|[1-9][0-9]*", encoded_size) is None:
                raise ReplacementError(f"installed RECORD size is invalid: {relative}")
            size = int(encoded_size)
            if (
                expected.get(relative) != (digest, size)
                or sha256_file(target) != digest
                or target.stat().st_size != size
            ):
                raise ReplacementError(f"installed RECORD bytes differ: {relative}")
        entries.append({"path": relative, "sha256": digest, "size": size})
    owned = set(records)
    for root in (scheme.platlib / "triton", expected_dist):
        if not root.is_dir() or root.is_symlink():
            raise ReplacementError(f"installed Triton tree is absent/unsafe: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ReplacementError(f"installed Triton tree contains symlink: {path}")
            if path.is_file():
                relative = probe._record_relative(path, scheme.platlib)
                if relative not in owned:
                    raise ReplacementError(
                        f"installed Triton file is not RECORD-owned: {relative}"
                    )
    return {
        "direct_url": direct_document,
        "editable_artifacts": [],
        "entries": entries,
        "entries_count": len(entries),
        "record_sha256": sha256_bytes(record_raw),
        "record_size": len(record_raw),
    }


def fsync_new_install(prepared: PreparedReplacement) -> None:
    directories: set[Path] = set()
    for relative in prepared.new_paths:
        path = prepared.prefix / relative
        if not path.is_file() or path.is_symlink():
            raise ReplacementError(f"new install durability target is absent: {path}")
        directories.add(path.parent)
    for relative in prepared.new_directories:
        path = prepared.prefix / relative
        if path.is_dir() and not path.is_symlink():
            directories.add(path)
            directories.add(path.parent)
    for directory in sorted(directories, key=lambda path: (len(path.parts), str(path))):
        _path_without_symlink_parents(directory, prepared.prefix)
        _fsync_directory(directory)


def _runtime_contract(report: dict[str, object]) -> dict[str, object]:
    contract = json.loads(canonical_json(report))
    contract.pop("sys_path", None)
    torch = contract.get("torch")
    if isinstance(torch, dict):
        torch["file"] = "$TORCH_SITE_PACKAGES/torch/__init__.py"
    return contract


def _remove_runtime_view(
    view: Path, evidence: Mapping[str, object], *, prefix: Path
) -> None:
    included = evidence.get("included_entries")
    if not isinstance(included, list) or any(not isinstance(item, str) for item in included):
        raise ReplacementError("runtime-view ownership manifest is malformed")
    for name in sorted(included, reverse=True):
        if "/" in name or name in ("", ".", ".."):
            raise ReplacementError("runtime-view ownership name is unsafe")
        path = view / name
        _path_without_symlink_parents(path, prefix)
        if not path.is_symlink():
            raise ReplacementError(f"runtime-view entry ceased to be a symlink: {path}")
        path.unlink()
    try:
        view.rmdir()
    except OSError as error:
        raise ReplacementError("runtime-view contains an unowned entry") from error
    _fsync_directory(view.parent)


def run_isolated_post_audits(
    prepared: PreparedReplacement,
    *,
    scratch: Path,
    limits: ReplacementLimits,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    view = prepared.prefix / ".torch-runtime-view"
    if view.exists() or view.is_symlink():
        raise ReplacementError("post-audit runtime-view destination is not fresh")
    try:
        runtime_view, view_evidence = probe.create_torch_runtime_view(
            prepared.scheme.platlib, prepared.prefix
        )
    except probe.ProbeError as error:
        raise ReplacementError(f"cannot create post-audit Torch view: {error}") from error
    reports: list[dict[str, object]] = []
    try:
        for _ in range(2):
            try:
                environment = _transaction_environment(scratch)
                with _temporary_file_default(scratch / "tmp"):
                    raw = probe.run_json_command(
                        [
                            str(prepared.python),
                            "-I",
                            "-B",
                            "-S",
                            "-c",
                            probe.RUNTIME_PROBE_PROGRAM,
                            str(prepared.scheme.platlib),
                            str(runtime_view),
                            str(prepared.prefix),
                            str(prepared.scheme.platlib),
                        ],
                        environment=environment,
                        limits=probe.ProbeLimits(
                            max_output_bytes=limits.max_output_bytes,
                            timeout_seconds=limits.timeout_seconds,
                        ),
                        description="isolated target Triton/Torch post-audit",
                    )
                normalized = probe.validate_runtime_report(
                    raw,
                    scheme=prepared.scheme,
                    torch_site=prepared.scheme.platlib,
                    anchor=prepared.inputs.anchor,
                )
            except probe.ProbeError as error:
                raise ReplacementError(f"isolated target post-audit failed: {error}") from error
            reports.append(normalized)
    finally:
        _remove_runtime_view(view, view_evidence, prefix=prepared.prefix)
    if len(reports) != 2 or reports[0] != reports[1]:
        raise ReplacementError("two target post-audit processes are not identical")
    probe_runtime = prepared.inputs.probe_document.get("runtime")
    if not isinstance(probe_runtime, dict):
        raise ReplacementError("accepted probe runtime record disappeared")
    accepted_reports = probe_runtime.get("processes")
    if (
        not isinstance(accepted_reports, list)
        or len(accepted_reports) != 2
        or _runtime_contract(reports[0]) != _runtime_contract(accepted_reports[0])
    ):
        raise ReplacementError("target runtime contract differs from accepted fresh probe")
    return reports, view_evidence


def validate_new_normal_runtime(
    report: dict[str, object], prepared: PreparedReplacement
) -> dict[str, object]:
    distribution = report.get("distribution")
    expected_dist = prepared.scheme.platlib / probe.DIST_INFO
    if (
        not isinstance(distribution, dict)
        or distribution.get("name") != "triton"
        or distribution.get("version") != probe.TRITON_DISTRIBUTION_VERSION
        or not isinstance(distribution.get("dist_info"), str)
        or Path(distribution["dist_info"]).resolve(strict=True) != expected_dist
        or report.get("module_version") != probe.TRITON_MODULE_VERSION
    ):
        raise ReplacementError("normal-site post-audit imported wrong Triton distribution")
    editable = report.get("editable")
    if (
        not isinstance(editable, dict)
        or editable.get("loaded_modules") != []
        or not isinstance(editable.get("carriers"), dict)
        or any(value != [] for value in editable["carriers"].values())
    ):
        raise ReplacementError("normal-site post-audit retained editable Triton finder")
    module_paths = report.get("module_paths")
    if not isinstance(module_paths, dict) or "triton._C.libtriton" not in module_paths:
        raise ReplacementError("normal-site post-audit module inventory is absent")
    for name, values in module_paths.items():
        if (
            not isinstance(name, str)
            or (name != "triton" and not name.startswith("triton."))
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
        ):
            raise ReplacementError("normal-site post-audit module inventory is malformed")
        for value in values:
            if not _is_within(Path(value).resolve(strict=True), prepared.prefix):
                raise ReplacementError("normal-site post-audit import escaped prefix")
    expected_native = {
        probe.wheel_member_destination(path, prepared.scheme)
        for path in prepared.inputs.anchor["libtriton_members"]
    }
    maps = report.get("libtriton_maps")
    if (
        not isinstance(maps, list)
        or {Path(value).resolve(strict=True) for value in maps} != expected_native
    ):
        raise ReplacementError("normal-site post-audit mapped non-wheel libtriton")
    backend = report.get("backend")
    if backend != {
        "arch": "sm120",
        "class": "triton.backends.nvidia.compiler.CUDABackend",
        "target": ["cuda", 120, 32],
    }:
        raise ReplacementError("normal-site post-audit did not construct SM120 backend")
    ptxas = report.get("ptxas_blackwell")
    expected_ptxas = probe.wheel_member_destination(
        probe.PTXAS_BLACKWELL_PATH, prepared.scheme
    )
    if (
        not isinstance(ptxas, dict)
        or not isinstance(ptxas.get("path"), str)
        or Path(ptxas["path"]).resolve(strict=True) != expected_ptxas
        or ptxas.get("reported_release") != "13.1"
        or sha256_file(expected_ptxas)
        != prepared.inputs.anchor["ptxas_blackwell"]["sha256"]
    ):
        raise ReplacementError("normal-site post-audit selected wrong ptxas-blackwell")
    torch = report.get("torch")
    if (
        not isinstance(torch, dict)
        or torch.get("version") != probe.TORCH_VERSION
        or torch.get("git_version") != probe.TORCH_GIT_VERSION
        or torch.get("cuda") != probe.TORCH_CUDA_VERSION
        or torch.get("hip") is not None
        or torch.get("has_triton_package") is not True
        or torch.get("triton_compat_has_triton") is not True
        or torch.get("triton_version") != [3, 7]
        or not isinstance(torch.get("triton_compat_symbols"), dict)
        or any(value is not True for value in torch["triton_compat_symbols"].values())
    ):
        raise ReplacementError("normal-site Torch Triton APIs did not all pass")
    keys = report.get("keys")
    if (
        not isinstance(keys, dict)
        or any(not isinstance(keys.get(name), str) or not keys.get(name) for name in (
            "direct", "torch_compat", "torch_inductor"
        ))
        or len(set(keys.values())) != 1
    ):
        raise ReplacementError("normal-site Torch/Triton keys disagree")
    sys_path = report.get("sys_path")
    if not isinstance(sys_path, list) or any(not isinstance(value, str) for value in sys_path):
        raise ReplacementError("normal-site sys.path report is malformed")
    old_source = Path(prepared.old_inventory.editable_source)
    for value in sys_path:
        if prepared.old_inventory.finder_module in value:
            raise ReplacementError("normal-site sys.path retained editable finder placeholder")
        if value and Path(value).is_absolute():
            try:
                resolved = Path(value).resolve(strict=True)
            except OSError:
                continue
            if _is_within(resolved, old_source, allow_root=True):
                raise ReplacementError("normal-site sys.path retained external Triton source")
    normalized = json.loads(canonical_json(report))
    normalized["distribution"]["dist_info"] = "$PREFIX/" + _prefix_relative(
        expected_dist, prepared.prefix
    )
    normalized["module_paths"] = {
        name: ["$PREFIX/" + _prefix_relative(Path(value).resolve(), prepared.prefix) for value in values]
        for name, values in module_paths.items()
    }
    normalized["libtriton_maps"] = sorted(
        "$PREFIX/" + _prefix_relative(Path(value).resolve(), prepared.prefix)
        for value in maps
    )
    normalized["torch"]["file"] = "$PREFIX/" + _prefix_relative(
        Path(torch["file"]).resolve(strict=True), prepared.prefix
    )
    normalized["ptxas_blackwell"]["path"] = "$PREFIX/" + _prefix_relative(
        Path(normalized["ptxas_blackwell"]["path"]).resolve(strict=True),
        prepared.prefix,
    )
    return normalized


def _cleanup_transient_runtime_view(prepared: PreparedReplacement) -> None:
    """Remove only a view that this transaction proved absent before mutation."""

    view = prepared.prefix / ".torch-runtime-view"
    if not view.exists() and not view.is_symlink():
        return
    if view.is_symlink() or not view.is_dir():
        raise ReplacementError("transient runtime-view path became unsafe")
    for path in sorted(view.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_symlink():
            raise ReplacementError(f"transient runtime-view has unowned entry: {path}")
        try:
            target = path.resolve(strict=True)
        except OSError as error:
            raise ReplacementError(f"transient runtime-view link is broken: {path}") from error
        if not _is_within(target, prepared.scheme.platlib):
            raise ReplacementError(f"transient runtime-view escaped target site: {path}")
        path.unlink()
    view.rmdir()
    _fsync_directory(view.parent)


def _cleanup_new_install(prepared: PreparedReplacement) -> list[str]:
    _cleanup_transient_runtime_view(prepared)
    old_by_path = {
        prepared.prefix / entry.path: entry for entry in prepared.old_inventory.files
    }
    removed: list[str] = []
    for relative in sorted(
        prepared.new_paths,
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    ):
        path = prepared.prefix / relative
        _path_without_symlink_parents(path, prepared.prefix)
        if not path.exists() and not path.is_symlink():
            continue
        old_entry = old_by_path.get(path)
        if old_entry is not None and _entry_matches(path, old_entry):
            continue
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise ReplacementError(f"new cleanup target is not a regular file: {path}")
        path.unlink()
        removed.append(relative)
        _fsync_directory(path.parent)
    old_directories = {
        prepared.prefix / entry.path
        for entry in prepared.old_inventory.removable_directories
    }
    preexisting_directories = {
        prepared.prefix / relative for relative in prepared.preexisting_directories
    }
    for relative in sorted(
        prepared.new_directories,
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    ):
        directory = prepared.prefix / relative
        _path_without_symlink_parents(directory, prepared.prefix)
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ReplacementError(f"new cleanup directory is unsafe: {directory}")
        if directory in preexisting_directories and directory not in old_directories:
            continue
        try:
            directory.rmdir()
        except OSError:
            # A shared pre-existing ancestor or still-present old file is not a
            # transaction-owned deletion target.  Leave it in place.
            continue
        if directory not in old_directories:
            removed.append(relative + "/")
        _fsync_directory(directory.parent)
    return removed


def _restore_regular_from_blob(
    blob: Path, destination: Path, *, mode: int, sha256: str, size: int
) -> None:
    if (
        blob.is_symlink()
        or not blob.is_file()
        or blob.stat().st_size != size
        or sha256_file(blob) != sha256
    ):
        raise ReplacementError(f"backup blob failed verification: {blob}")
    temporary = destination.with_name(
        f".{destination.name}.pypto-triton-rollback.partial"
    )
    if temporary.exists() or temporary.is_symlink():
        raise ReplacementError(f"rollback temporary path is occupied: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, mode)
    except OSError as error:
        raise ReplacementError(f"cannot create rollback temporary: {temporary}") from error
    digest = hashlib.sha256()
    total = 0
    try:
        with blob.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
                written = 0
                while written < len(chunk):
                    written += os.write(descriptor, chunk[written:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if total != size or digest.hexdigest() != sha256:
        raise ReplacementError(f"rollback temporary differs from backup: {temporary}")
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ReplacementError(f"rollback destination became occupied: {destination}") from error
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _cleanup_rollback_temporaries(prepared: PreparedReplacement) -> list[str]:
    removed: list[str] = []
    for entry in prepared.old_inventory.files:
        if entry.kind != "regular":
            continue
        destination = prepared.prefix / entry.path
        temporary = destination.with_name(
            f".{destination.name}.pypto-triton-rollback.partial"
        )
        _path_without_symlink_parents(temporary, prepared.prefix)
        if not temporary.exists() and not temporary.is_symlink():
            continue
        temporary_stat = temporary.lstat()
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise ReplacementError(f"rollback temporary is unsafe: {temporary}")
        temporary.unlink()
        removed.append(_prefix_relative(temporary, prepared.prefix))
        _fsync_directory(temporary.parent)
    return removed


def restore_old_inventory(
    prepared: PreparedReplacement,
    *,
    limits: ReplacementLimits = ReplacementLimits(),
    verify_inputs: bool = True,
    expected_old_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    rollback_temporaries = _cleanup_rollback_temporaries(prepared)
    removed_new = _cleanup_new_install(prepared)
    backup_root = prepared.request.backup_root
    blobs = backup_root / "blobs"
    restored_directories: list[str] = []
    for entry in sorted(
        prepared.old_inventory.removable_directories,
        key=lambda item: (len(Path(item.path).parts), item.path),
    ):
        path = prepared.prefix / entry.path
        _path_without_symlink_parents(path, prepared.prefix)
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise ReplacementError(f"rollback directory target is occupied: {path}")
            os.chmod(path, entry.mode)
            continue
        path.mkdir(mode=entry.mode)
        os.chmod(path, entry.mode)
        restored_directories.append(entry.path)
        _fsync_directory(path.parent)

    restored_files: list[str] = []
    for entry in sorted(prepared.old_inventory.files, key=lambda item: item.path):
        path = prepared.prefix / entry.path
        _path_without_symlink_parents(path, prepared.prefix)
        if path.exists() or path.is_symlink():
            if _entry_matches(path, entry):
                continue
            raise ReplacementError(f"rollback target is occupied by changed bytes: {path}")
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if entry.kind == "regular":
            if entry.sha256 is None:
                raise ReplacementError("regular rollback entry lacks SHA256")
            _restore_regular_from_blob(
                blobs / entry.sha256,
                path,
                mode=entry.mode,
                sha256=entry.sha256,
                size=entry.size,
            )
        elif entry.kind == "symlink":
            if entry.link_target_b64 is None:
                raise ReplacementError("symlink rollback entry lacks target")
            try:
                target = base64.b64decode(entry.link_target_b64, validate=True)
            except (ValueError, base64.binascii.Error) as error:
                raise ReplacementError("symlink backup target is malformed") from error
            os.symlink(target, os.fsencode(path))
        else:
            raise ReplacementError("unknown rollback entry kind")
        restored_files.append(entry.path)
        _fsync_directory(path.parent)
    verify_old_inventory(prepared, limits=limits)
    torch_after = capture_torch_identity(
        prepared.scheme,
        prepared.inputs.environment_lock,
        workspace=prepared.workspace,
    )
    if torch_after != prepared.torch_before:
        raise ReplacementError("Torch tree changed across rollback")
    if verify_inputs:
        _stable_input_digests(prepared)
    runtime_verified = False
    if expected_old_runtime is not None:
        observed_runtime = run_normal_runtime_audit(
            prepared,
            scratch=prepared.request.backup_root / "scratch" / "rollback-runtime",
            limits=limits,
        )
        validate_old_runtime_audit(observed_runtime, prepared)
        if observed_runtime != expected_old_runtime:
            raise ReplacementError("rollback runtime provenance differs from sealed old state")
        runtime_verified = True
    return {
        "new_paths_removed": removed_new,
        "rollback_temporaries_removed": rollback_temporaries,
        "old_directories_restored": restored_directories,
        "old_files_restored": restored_files,
        "old_inventory_verified": True,
        "old_runtime_verified": runtime_verified,
        "torch_tree_unchanged": True,
    }


def _post_audit(
    prepared: PreparedReplacement,
    installation: dict[str, object],
    *,
    limits: ReplacementLimits,
) -> dict[str, object]:
    scratch = prepared.request.backup_root / "scratch" / "post"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ReplacementError("post-audit scratch path is unsafe")
    record_before = verify_new_install(prepared, installation, limits=limits)
    isolated_reports, view_evidence = run_isolated_post_audits(
        prepared, scratch=scratch / "isolated", limits=limits
    )
    normal_raw = run_normal_runtime_audit(
        prepared, scratch=scratch / "normal", limits=limits
    )
    normal_runtime = validate_new_normal_runtime(normal_raw, prepared)
    record_after = verify_new_install(prepared, installation, limits=limits)
    if record_before != record_after:
        raise ReplacementError("fresh runtime audits changed installed Triton RECORD/tree")
    torch_after = capture_torch_identity(
        prepared.scheme,
        prepared.inputs.environment_lock,
        workspace=prepared.workspace,
    )
    if torch_after != prepared.torch_before:
        raise ReplacementError("frozen Torch tree changed during Triton replacement")
    for native in prepared.old_inventory.native_files:
        if not _native_matches(native):
            raise ReplacementError("external editable native changed during replacement")
    verify_external_source_identity(prepared, limits=limits)
    _stable_input_digests(prepared)
    return {
        "isolated_processes": isolated_reports,
        "isolated_processes_count": 2,
        "normal_site_process": normal_runtime,
        "record_verification": record_after,
        "torch_after": torch_after,
        "torch_tree_unchanged": True,
        "torch_runtime_view": view_evidence,
    }


def _failure_evidence(
    prepared: PreparedReplacement,
    *,
    backup_manifest_sha256: str | None,
    error: BaseException,
    rollback: dict[str, object],
) -> dict[str, object]:
    return {
        "acceptance": "rejected",
        "backup": {
            "manifest": (
                {
                    "path": _workspace_relative(
                        prepared.request.backup_root / "manifest.json",
                        prepared.workspace,
                    ),
                    "sha256": backup_manifest_sha256,
                }
                if backup_manifest_sha256 is not None
                else None
            ),
            "root": _workspace_relative(
                prepared.request.backup_root, prepared.workspace
            ),
        },
        "error": {
            "message": str(error),
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
        },
        "inputs": prepared.plan_document["inputs"],
        "mode": "apply",
        "replacement": REPLACEMENT_NAME,
        "rollback": rollback,
        "schema_version": SCHEMA_VERSION,
        "status": "rolled-back" if rollback.get("verified") is True else "rollback-failed",
    }


def apply_replacement(
    request: ReplacementRequest,
    *,
    limits: ReplacementLimits = ReplacementLimits(),
) -> tuple[dict[str, object], str]:
    """Apply one exact replacement, publishing success or rolled-back evidence."""
    workspace = _require_workspace(request.workspace)
    with workspace_transaction_lock(workspace) as lock_evidence:
        controller = lock_evidence.get("controller")
        controller_exemption = (
            {int(controller["pid"]): int(controller["start_ticks"])}
            if isinstance(controller, dict)
            else {}
        )

        def exclusive_boundary(phase: str) -> dict[str, object]:
            return assert_exclusive_prefix(
                prepared,
                phase,
                exempt_identities=controller_exemption,
            )

        prepared = prepare_replacement(request, limits=limits)
        backup_manifest_sha: str | None = None
        backup_manifest: dict[str, object] | None = None
        mutation_started = False
        success_evidence: dict[str, object] | None = None
        success_sha: str | None = None
        boundary_audits: list[dict[str, object]] = []
        with TransactionSignalGuard() as signal_guard:
            try:
                initialize_backup_root(
                    prepared.request.backup_root,
                    lock_evidence=lock_evidence,
                )
                scratch = prepared.request.backup_root / "scratch"
                scratch.mkdir(mode=0o700)
                boundary_audits.append(exclusive_boundary("before-backup"))
                old_runtime = run_normal_runtime_audit(
                    prepared, scratch=scratch / "old-runtime", limits=limits
                )
                validate_old_runtime_audit(old_runtime, prepared)
                backup_manifest, backup_manifest_sha = create_backup(
                    prepared, old_runtime=old_runtime
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="prepared",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=False,
                    extra={"lock": lock_evidence},
                )
                verify_old_inventory(prepared, limits=limits)
                _stable_input_digests(prepared)
                boundary_audits.append(
                    exclusive_boundary("before-old-removal")
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="removing-old",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                mutation_started = True
                removed_old = _remove_old_inventory(prepared)
                boundary_audits.append(
                    exclusive_boundary("after-old-removal")
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="old-removed",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="installing-new",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                try:
                    installation = probe.install_audited_wheel(
                        prepared.request.wheel,
                        prepared.scheme,
                        prepared.inputs.anchor,
                        limits=probe.ProbeLimits(),
                    )
                except probe.ProbeError as error:
                    raise ReplacementError(
                        f"stdlib wheel installation failed: {error}"
                    ) from error
                fsync_new_install(prepared)
                boundary_audits.append(
                    exclusive_boundary("after-new-install")
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="new-installed",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                update_journal(
                    prepared.request.backup_root,
                    phase="post-auditing",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                post = _post_audit(prepared, installation, limits=limits)
                verify_external_source_identity(prepared, limits=limits)
                for native in prepared.old_inventory.native_files:
                    if not _native_matches(native):
                        raise ReplacementError("external native drifted before commit")
                boundary_audits.append(
                    exclusive_boundary("after-post-audit")
                )
                _stable_input_digests(prepared)
                update_journal(
                    prepared.request.backup_root,
                    phase="post-audited",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                success_evidence = {
                    "acceptance": "accepted",
                    "backup": {
                        "journal": _workspace_relative(
                            prepared.request.backup_root / "journal.json",
                            prepared.workspace,
                        ),
                        "manifest": {
                            "path": _workspace_relative(
                                prepared.request.backup_root / "manifest.json",
                                prepared.workspace,
                            ),
                            "sha256": backup_manifest_sha,
                            "size": (
                                prepared.request.backup_root / "manifest.json"
                            ).stat().st_size,
                        },
                        "root": _workspace_relative(
                            prepared.request.backup_root, prepared.workspace
                        ),
                    },
                    "exclusive_prefix_audits": boundary_audits,
                    "inputs": prepared.plan_document["inputs"],
                    "installation": {
                        **installation,
                        "method": INSTALL_METHOD,
                        "old_record_owned_paths_removed": removed_old,
                        "prefix": _workspace_relative(
                            prepared.prefix, prepared.workspace
                        ),
                    },
                    "mode": "apply",
                    "post_audit": post,
                    "replacement": REPLACEMENT_NAME,
                    "rollback": {"required": False},
                    "schema_version": SCHEMA_VERSION,
                    "status": "committed",
                }
                boundary_audits.append(
                    exclusive_boundary("before-commit")
                )
                success_evidence["exclusive_prefix_audits"] = boundary_audits
                update_journal(
                    prepared.request.backup_root,
                    phase="committing",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                )
                success_sha = publish_canonical_json_no_replace(
                    prepared.request.evidence, success_evidence
                )
                # Atomic success evidence is the commit point.  Recovery treats
                # a crash before this journal advance as committed, not rollback.
                signal_guard.begin_rollback()
                update_journal(
                    prepared.request.backup_root,
                    phase="committed",
                    manifest_sha256=backup_manifest_sha,
                    mutation_started=True,
                    extra={
                        "terminal_evidence_sha256": success_sha,
                        "terminal_evidence_status": "committed",
                    },
                )
                return success_evidence, success_sha
            except BaseException as original:
                signal_guard.begin_rollback()
                # A signal between the atomic link and journal advance cannot
                # revoke an already published commit.
                if success_evidence is not None and prepared.request.evidence.exists():
                    try:
                        observed, raw = load_canonical_json(
                            prepared.request.evidence,
                            "terminal replacement evidence",
                            max_bytes=limits.max_evidence_bytes,
                        )
                    except ReplacementError:
                        observed = {}
                        raw = b""
                    if observed == success_evidence:
                        success_sha = sha256_bytes(raw)
                        update_journal(
                            prepared.request.backup_root,
                            phase="committed",
                            manifest_sha256=backup_manifest_sha,
                            mutation_started=True,
                            extra={
                                "terminal_evidence_sha256": success_sha,
                                "terminal_evidence_status": "committed",
                            },
                        )
                        return success_evidence, success_sha
                rollback: dict[str, object]
                if mutation_started and backup_manifest is not None:
                    try:
                        boundary_audits.append(
                            exclusive_boundary("before-rollback")
                        )
                        update_journal(
                            prepared.request.backup_root,
                            phase="rolling-back",
                            manifest_sha256=backup_manifest_sha,
                            mutation_started=True,
                        )
                        details = restore_old_inventory(
                            prepared,
                            limits=limits,
                            expected_old_runtime=backup_manifest["old_runtime"],
                        )
                        boundary_audits.append(
                            exclusive_boundary("after-rollback")
                        )
                        rollback = {
                            **details,
                            "attempted": True,
                            "verified": True,
                        }
                        update_journal(
                            prepared.request.backup_root,
                            phase="rolled-back",
                            manifest_sha256=backup_manifest_sha,
                            mutation_started=True,
                            extra={"rollback": rollback},
                        )
                    except BaseException as rollback_error:
                        rollback = {
                            "attempted": True,
                            "error": {
                                "message": str(rollback_error),
                                "type": (
                                    f"{type(rollback_error).__module__}."
                                    f"{type(rollback_error).__qualname__}"
                                ),
                            },
                            "verified": False,
                        }
                        update_journal(
                            prepared.request.backup_root,
                            phase="rollback-failed",
                            manifest_sha256=backup_manifest_sha,
                            mutation_started=True,
                            extra={"rollback": rollback},
                        )
                else:
                    rollback = {"attempted": False, "verified": True}
                evidence_sha: str | None = None
                evidence_status = (
                    "rolled-back"
                    if rollback.get("verified") is True
                    else "rollback-failed"
                )
                if prepared.request.backup_root.exists():
                    failure = _failure_evidence(
                        prepared,
                        backup_manifest_sha256=backup_manifest_sha,
                        error=original,
                        rollback=rollback,
                    )
                    try:
                        evidence_sha = publish_canonical_json_no_replace(
                            prepared.request.evidence, failure
                        )
                        if backup_manifest_sha is not None:
                            update_journal(
                                prepared.request.backup_root,
                                phase=evidence_status,
                                manifest_sha256=backup_manifest_sha,
                                mutation_started=mutation_started,
                                extra={
                                    "rollback": rollback,
                                    "terminal_evidence_sha256": evidence_sha,
                                    "terminal_evidence_status": evidence_status,
                                },
                            )
                    except BaseException as publication_error:
                        raise ReplacementError(
                            "replacement failed and terminal evidence publication "
                            f"failed: {original}; {publication_error}",
                            rollback=rollback,
                            evidence_status=evidence_status,
                        ) from original
                if rollback.get("verified") is not True:
                    raise ReplacementError(
                        f"replacement failed and rollback could not be proven: {original}",
                        rollback=rollback,
                        evidence_sha256=evidence_sha,
                        evidence_status="rollback-failed",
                    ) from original
                if isinstance(original, (KeyboardInterrupt, SystemExit)):
                    raise original
                raise ReplacementError(
                    f"replacement failed; old environment restored: {original}",
                    rollback=rollback,
                    evidence_sha256=evidence_sha,
                    evidence_status="rolled-back",
                ) from original


def recover_replacement(
    workspace_path: Path,
    backup_root_path: Path,
    *,
    force_rollback: bool,
    evidence: Path | None = None,
    limits: ReplacementLimits = ReplacementLimits(),
) -> tuple[dict[str, object], str | None]:
    workspace = _require_workspace(workspace_path)
    backup_root = _require_existing_backup_root(backup_root_path, workspace)
    if evidence is not None:
        if not evidence.is_absolute():
            raise ReplacementError("recovery evidence path must be absolute")
        evidence = _absolute_lexical(evidence)
        try:
            evidence_parent = evidence.parent.resolve(strict=True)
        except OSError as error:
            raise ReplacementError("recovery evidence parent is absent") from error
        if evidence.parent != evidence_parent or not _is_within(
            evidence_parent, workspace, allow_root=True
        ):
            raise ReplacementError("recovery evidence parent escaped workspace")
    with workspace_transaction_lock(workspace) as lock_evidence:
        controller = lock_evidence.get("controller")
        controller_exemption = (
            {int(controller["pid"]): int(controller["start_ticks"])}
            if isinstance(controller, dict)
            else {}
        )
        journal = load_journal(backup_root)
        phase = str(journal["phase"])
        if journal.get("manifest_sha256") is None:
            if journal.get("mutation_started") is True:
                raise ReplacementError("journal claims mutation without a backup manifest")
            if force_rollback:
                raise ReplacementError("no backup manifest exists for --rollback")
            result = {
                "mode": "recover",
                "replacement": REPLACEMENT_NAME,
                "schema_version": 1,
                "status": "no-mutation-before-backup",
            }
            digest_value: str | None = None
            if evidence is not None:
                if evidence.exists():
                    existing, raw = load_canonical_json(
                        evidence,
                        "no-mutation recovery evidence",
                        max_bytes=limits.max_evidence_bytes,
                    )
                    if existing != result:
                        raise ReplacementError("existing recovery evidence differs")
                    digest_value = sha256_bytes(raw)
                else:
                    digest_value = publish_canonical_json_no_replace(
                        evidence, result
                    )
                update_journal(
                    backup_root,
                    phase="no-mutation-before-backup",
                    manifest_sha256=None,
                    mutation_started=False,
                    extra={
                        "recovered_by": lock_evidence,
                        "recovery_evidence_sha256": digest_value,
                    },
                )
            return result, digest_value
        prepared, manifest, manifest_sha = prepared_from_backup(
            workspace,
            backup_root,
            evidence_override=evidence,
        )
        original_terminal = workspace.joinpath(
            *_safe_relative(
                str(manifest["terminal_evidence"]), "terminal evidence"
            ).parts
        )
        original_terminal = _absolute_lexical(original_terminal)
        if not _is_within(original_terminal, workspace):
            raise ReplacementError("backup terminal-evidence path escaped workspace")
        recovery_evidence = (
            evidence
            if evidence is not None
            else backup_root / "recovery-evidence.json"
        )
        if recovery_evidence == original_terminal:
            raise ReplacementError("recovery evidence must not replace original terminal evidence")
        if _is_within(recovery_evidence, prepared.prefix, allow_root=True):
            raise ReplacementError("recovery evidence must be outside environment prefix")
        if not force_rollback and original_terminal.is_file():
            terminal, raw = load_canonical_json(
                original_terminal,
                "terminal replacement evidence",
                max_bytes=limits.max_evidence_bytes,
            )
            terminal_backup = terminal.get("backup")
            terminal_manifest = (
                terminal_backup.get("manifest")
                if isinstance(terminal_backup, dict)
                else None
            )
            if (
                terminal.get("replacement") == REPLACEMENT_NAME
                and terminal.get("status") == "committed"
                and terminal.get("acceptance") == "accepted"
                and isinstance(terminal_manifest, dict)
                and terminal_manifest.get("sha256") == manifest_sha
            ):
                installation = terminal.get("installation")
                expected_post = terminal.get("post_audit")
                if not isinstance(installation, dict) or not isinstance(
                    expected_post, dict
                ):
                    raise ReplacementError(
                        "committed terminal evidence lacks installation/post-audit"
                    )
                assert_exclusive_prefix(
                    prepared,
                    "recover-verify-committed-before",
                    exempt_identities=controller_exemption,
                )
                observed_post = _post_audit(
                    prepared,
                    installation,
                    limits=limits,
                )
                assert_exclusive_prefix(
                    prepared,
                    "recover-verify-committed-after",
                    exempt_identities=controller_exemption,
                )
                if observed_post != expected_post:
                    raise ReplacementError(
                        "current committed prefix differs from terminal post-audit; "
                        "use explicit --rollback"
                    )
                terminal_sha = sha256_bytes(raw)
                result: dict[str, object] = {
                    "acceptance": "recovered",
                    "backup": {
                        "manifest_sha256": manifest_sha,
                        "root": _workspace_relative(backup_root, workspace),
                    },
                    "mode": "recover",
                    "original_terminal_evidence": {
                        "path": _workspace_relative(original_terminal, workspace),
                        "sha256": terminal_sha,
                        "size": len(raw),
                    },
                    "replacement": REPLACEMENT_NAME,
                    "schema_version": 1,
                    "status": "committed-verified",
                }
                if recovery_evidence.exists():
                    existing, recovery_raw = load_canonical_json(
                        recovery_evidence,
                        "recovery evidence",
                        max_bytes=limits.max_evidence_bytes,
                    )
                    if existing != result:
                        raise ReplacementError("existing recovery evidence differs")
                    digest_value = sha256_bytes(recovery_raw)
                else:
                    digest_value = publish_canonical_json_no_replace(
                        recovery_evidence, result
                    )
                update_journal(
                    backup_root,
                    phase="committed",
                    manifest_sha256=manifest_sha,
                    mutation_started=True,
                    extra={
                        "recovered_by": lock_evidence,
                        "recovery_evidence_sha256": digest_value,
                        "terminal_evidence_sha256": terminal_sha,
                        "terminal_evidence_status": "committed",
                    },
                )
                return result, digest_value
        if not force_rollback and phase == "committed":
            raise ReplacementError("committed journal lost its accepted terminal evidence")
        with TransactionSignalGuard() as guard:
            guard.begin_rollback()
            assert_exclusive_prefix(
                prepared,
                "recovery-before-rollback",
                exempt_identities=controller_exemption,
            )
            update_journal(
                backup_root,
                phase="rolling-back",
                manifest_sha256=manifest_sha,
                mutation_started=bool(journal.get("mutation_started")),
                extra={"recovered_by": lock_evidence},
            )
            old_runtime = manifest.get("old_runtime")
            if not isinstance(old_runtime, dict):
                raise ReplacementError("backup manifest old runtime is absent")
            details = restore_old_inventory(
                prepared,
                limits=limits,
                verify_inputs=False,
                expected_old_runtime=old_runtime,
            )
            assert_exclusive_prefix(
                prepared,
                "recovery-after-rollback",
                exempt_identities=controller_exemption,
            )
            result: dict[str, object] = {
                "acceptance": "recovered",
                "backup": {
                    "manifest_sha256": manifest_sha,
                    "root": _workspace_relative(backup_root, workspace),
                },
                "mode": "rollback" if force_rollback else "recover",
                "replacement": REPLACEMENT_NAME,
                "rollback": {
                    "old_inventory_verified": details["old_inventory_verified"],
                    "old_runtime_verified": details["old_runtime_verified"],
                    "torch_tree_unchanged": details["torch_tree_unchanged"],
                    "verified": True,
                },
                "schema_version": 1,
                "status": "rolled-back",
            }
            digest_value: str | None = None
            if recovery_evidence.exists():
                existing, raw = load_canonical_json(
                    recovery_evidence,
                    "recovery evidence",
                    max_bytes=limits.max_evidence_bytes,
                )
                if existing != result:
                    raise ReplacementError("existing recovery evidence differs")
                digest_value = sha256_bytes(raw)
            else:
                if not _is_within(recovery_evidence, workspace):
                    raise ReplacementError("recovery evidence escaped workspace")
                digest_value = publish_canonical_json_no_replace(
                    recovery_evidence, result
                )
            update_journal(
                backup_root,
                phase="rolled-back",
                manifest_sha256=manifest_sha,
                mutation_started=bool(journal.get("mutation_started")),
                extra={
                    "recovered_by": lock_evidence,
                    "recovery_evidence_sha256": digest_value,
                    "rollback": result["rollback"],
                },
            )
            return result, digest_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--wheel-audit-evidence", type=Path)
    parser.add_argument("--expected-wheel-audit-evidence-sha256")
    parser.add_argument("--wheel-probe-evidence", type=Path)
    parser.add_argument("--expected-wheel-probe-evidence-sha256")
    parser.add_argument("--gpu-smoke-evidence", type=Path)
    parser.add_argument("--expected-gpu-smoke-evidence-sha256")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--expected-environment-lock-sha256")
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--recover", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def _request_from_args(args: argparse.Namespace) -> ReplacementRequest:
    required = {
        "prefix": args.prefix,
        "wheel": args.wheel,
        "wheel_audit_evidence": args.wheel_audit_evidence,
        "expected_wheel_audit_evidence_sha256": (
            args.expected_wheel_audit_evidence_sha256
        ),
        "wheel_probe_evidence": args.wheel_probe_evidence,
        "expected_wheel_probe_evidence_sha256": (
            args.expected_wheel_probe_evidence_sha256
        ),
        "gpu_smoke_evidence": args.gpu_smoke_evidence,
        "expected_gpu_smoke_evidence_sha256": (
            args.expected_gpu_smoke_evidence_sha256
        ),
        "environment_lock": args.environment_lock,
        "expected_environment_lock_sha256": (
            args.expected_environment_lock_sha256
        ),
        "evidence": args.evidence,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ReplacementError(
            f"--plan/--apply require transaction arguments: {missing}"
        )
    return ReplacementRequest(
        workspace=args.workspace,
        prefix=args.prefix,
        wheel=args.wheel,
        wheel_audit_evidence=args.wheel_audit_evidence,
        expected_wheel_audit_evidence_sha256=(
            args.expected_wheel_audit_evidence_sha256
        ),
        wheel_probe_evidence=args.wheel_probe_evidence,
        expected_wheel_probe_evidence_sha256=(
            args.expected_wheel_probe_evidence_sha256
        ),
        gpu_smoke_evidence=args.gpu_smoke_evidence,
        expected_gpu_smoke_evidence_sha256=(
            args.expected_gpu_smoke_evidence_sha256
        ),
        environment_lock=args.environment_lock,
        expected_environment_lock_sha256=args.expected_environment_lock_sha256,
        backup_root=args.backup_root,
        evidence=args.evidence,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = ReplacementLimits(timeout_seconds=args.timeout_seconds)
    request: ReplacementRequest | None = None
    try:
        if args.recover or args.rollback:
            value, digest = recover_replacement(
                args.workspace,
                args.backup_root,
                force_rollback=args.rollback,
                evidence=args.evidence,
                limits=limits,
            )
            print(
                canonical_json(
                    {
                        "evidence": (
                            None
                            if digest is None
                            else str(
                                args.evidence
                                if args.evidence is not None
                                else args.backup_root / "recovery-evidence.json"
                            )
                        ),
                        "evidence_sha256": digest,
                        "status": value["status"],
                    }
                ),
                end="",
            )
            return 0
        request = _request_from_args(args)
        if args.plan:
            workspace = _require_workspace(request.workspace)
            with workspace_transaction_lock(
                workspace,
                required_mode="shared",
                create_if_missing=False,
            ):
                prepared = prepare_replacement(request, limits=limits)
            print(canonical_json(prepared.plan_document), end="")
            return 0
        evidence, digest = apply_replacement(request, limits=limits)
    except (ReplacementError, OSError, ValueError, csv.Error) as error:
        print(f"Triton environment replacement failed: {error}", file=sys.stderr)
        if isinstance(error, ReplacementError) and error.evidence_sha256 is not None:
            status = error.evidence_status or "rollback-failed"
            print(
                canonical_json(
                    {
                        "evidence": str(
                            request.evidence if request is not None else args.evidence
                        ),
                        "evidence_sha256": error.evidence_sha256,
                        "status": status,
                    }
                ),
                end="",
            )
        return 1
    print(
        canonical_json(
            {
                "evidence": str(request.evidence),
                "evidence_sha256": digest,
                "status": evidence["status"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
