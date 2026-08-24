#!/usr/bin/env python3
"""Run one provenance-bound, reference-only Triton SM120 vector-add smoke.

This program is intentionally not a benchmark despite living below
``benchmarks/``.  It records no duration, throughput, bandwidth, or warm-run
measurement.  Its only purpose is to prove that the exact fresh Triton wheel
probe can compile and execute a minimal FP32 kernel on the pinned RTX 5090
Laptop GPU, with Torch used solely as the numerical reference.

Invoke the program with the Python recorded by ``probe_triton_wheel.py`` and
with isolated/no-site processing enabled, for example::

    PROBE/bin/python -I -B -S \
      benchmarks/operators/triton_reference_sm120.py \
      --workspace /absolute/workspace \
      --probe-evidence /absolute/workspace/reports/data/probe.json \
      --expected-probe-evidence-sha256 SHA256 \
      --probe-prefix /absolute/workspace/runs/fresh-probe \
      --probe-site /absolute/workspace/runs/fresh-probe/lib/python3.14/site-packages \
      --torch-runtime-view /absolute/workspace/runs/fresh-probe/.torch-runtime-view \
      --cache-dir /absolute/workspace/caches/reference-sm120-FRESH \
      --evidence /absolute/workspace/reports/data/reference-sm120.json

The probe site and its filtered Torch view are inserted manually.  ``site``
processing, ambient/editable Triton carriers, a pre-existing cache, the wrong
GPU, an external ptxas, a numerical mismatch, or an asynchronous CUDA error
all fail before final evidence publication.  Accepted evidence is canonical
JSON published atomically without replacing an existing path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SMOKE_NAME = "reference-only-triton-sm120"
PROBE_NAME = "triton-fresh-wheel"
WHEEL_AUDIT_NAME = "triton-workspace-wheel"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"
EXPECTED_CAPABILITY = (12, 0)
EXPECTED_TARGET = ("cuda", 120, 32)
EXPECTED_PTXAS_FULL_VERSION = "13.1.80"
EXPECTED_PTXAS_RELEASE = "13.1"
PTXAS_BLACKWELL_RELATIVE = Path(
    "triton/backends/nvidia/bin/ptxas-blackwell"
)
VECTOR_ELEMENTS = 65_537
BLOCK_SIZE = 256

_FORBIDDEN_TRITON_ENVIRONMENT = (
    "TRITON_CACHE_MANAGER",
    "TRITON_INTERPRET",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_OVERRIDE_ARCH",
    "TRITON_OVERRIDE_DIR",
    "TRITON_PTXAS_BLACKWELL_PATH",
    "TRITON_PTXAS_PATH",
    "TRITON_REMOTE_CACHE_BACKEND",
)


class SmokeError(RuntimeError):
    """A reference-smoke provenance or runtime invariant was not proven."""


@dataclass(frozen=True, slots=True)
class SmokeRequest:
    workspace: Path
    probe_evidence: Path
    expected_probe_evidence_sha256: str
    probe_prefix: Path
    probe_site: Path
    torch_runtime_view: Path
    cache_dir: Path
    evidence: Path


@dataclass(frozen=True, slots=True)
class ProbeAnchor:
    workspace: Path
    base_python: Path
    environment_lock: Path
    prefix: Path
    probe_site: Path
    torch_runtime_view: Path
    torch_source_site: Path
    triton_dist_info: Path
    python: Path
    probe_evidence: Path
    probe_evidence_sha256: str
    probe_evidence_size: int
    probe_document: dict[str, object]
    ptxas_path: Path
    ptxas_sha256: str
    libtriton_maps: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PreparedSmoke:
    anchor: ProbeAnchor
    cache_dir: Path
    evidence: Path
    integrity_before: dict[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    torch: ModuleType
    triton: ModuleType
    language: ModuleType
    ptxas_evidence: dict[str, object]


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


def _regular_file_identity(
    path: Path,
    description: str,
    *,
    content_digest: Any | None = None,
) -> tuple[str, int, bytes]:
    """Hash a stable regular file without retaining its bytes in memory."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SmokeError(
            f"cannot open {description} as a non-symlink: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmokeError(f"{description} is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        prefix = b""
        while chunk := os.read(descriptor, 8 << 20):
            if not prefix:
                prefix = chunk[:4]
            digest.update(chunk)
            if content_digest is not None:
                content_digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != before.st_size:
            raise SmokeError(f"{description} changed while it was read: {path}")
        return digest.hexdigest(), size, prefix
    finally:
        os.close(descriptor)


def _read_regular_file_once(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SmokeError(
            f"cannot open {description} as a non-symlink: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmokeError(f"{description} is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SmokeError(f"{description} changed while it was read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return _regular_file_identity(path, "file")[0]


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> object:
    raise SmokeError(f"non-finite JSON value is forbidden: {value}")


def load_canonical_json(
    path: Path, description: str
) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_file_once(path, description)
    try:
        text = raw.decode("utf-8", errors="strict")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise SmokeError(f"{description} root must be an object")
    if text != canonical_json(document):
        raise SmokeError(f"{description} is not canonical JSON")
    return document, raw


def _validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SmokeError(
            f"{description} must be 64 lowercase hexadecimal characters"
        )
    return value


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    return (allow_root and path == root) or root in path.parents


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError as error:
        raise SmokeError(f"path escaped the workspace: {path}") from error


def _require_workspace(path: Path) -> Path:
    if not path.is_absolute():
        raise SmokeError("--workspace must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"workspace is absent: {lexical}") from error
    if lexical != resolved or not resolved.is_dir():
        raise SmokeError("workspace must be a real, non-symlink directory")
    return resolved


def _require_workspace_file(
    path: Path, workspace: Path, description: str
) -> Path:
    if not path.is_absolute():
        raise SmokeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise SmokeError(f"{description} must be a real workspace-owned path")
    if resolved.is_symlink() or not resolved.is_file():
        raise SmokeError(f"{description} must be a regular non-symlink file")
    return resolved


def _require_workspace_directory(
    path: Path, workspace: Path, description: str
) -> Path:
    if not path.is_absolute():
        raise SmokeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise SmokeError(f"{description} must be a real workspace-owned directory")
    if resolved.is_symlink() or not resolved.is_dir():
        raise SmokeError(f"{description} must be a real non-symlink directory")
    return resolved


def _workspace_path_from_document(
    value: object,
    workspace: Path,
    description: str,
    *,
    directory: bool,
) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise SmokeError(f"{description} must be a non-empty workspace-relative path")
    lexical = _absolute_lexical(workspace / value)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise SmokeError(f"{description} escaped the workspace")
    if directory:
        if not resolved.is_dir() or resolved.is_symlink():
            raise SmokeError(f"{description} is not a real directory")
    elif not resolved.is_file() or resolved.is_symlink():
        raise SmokeError(f"{description} is not a regular non-symlink file")
    return resolved


def _require_fresh_workspace_path(
    path: Path, workspace: Path, description: str
) -> Path:
    if not path.is_absolute():
        raise SmokeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    if lexical == workspace or not _is_within(lexical, workspace):
        raise SmokeError(f"{description} must be a child of the workspace")
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"{description} parent is absent: {lexical.parent}") from error
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or not _is_within(parent, workspace, allow_root=True)
    ):
        raise SmokeError(f"{description} parent must be a real workspace directory")
    if lexical != parent / lexical.name:
        raise SmokeError(f"{description} path is not lexical and canonical")
    if lexical.exists() or lexical.is_symlink():
        raise SmokeError(f"{description} is not fresh: {lexical}")
    return lexical


def _require_evidence_output(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise SmokeError("evidence path must be absolute")
    lexical = _absolute_lexical(path)
    if lexical == workspace or not _is_within(lexical, workspace):
        raise SmokeError("evidence output must be a child of the workspace")
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as error:
        raise SmokeError("evidence parent is absent") from error
    if lexical != parent / lexical.name or not parent.is_dir() or parent.is_symlink():
        raise SmokeError("evidence parent must be a real workspace directory")
    if not _is_within(parent, workspace, allow_root=True):
        raise SmokeError("evidence output escaped the workspace")
    if lexical.exists() or lexical.is_symlink():
        raise SmokeError(f"evidence already exists: {lexical}")
    return lexical


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SmokeError(f"{description} is absent or not an object")
    return value


def _sequence(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise SmokeError(f"{description} is absent or not an array")
    return value


def _verify_linked_file_identity(
    identity: object,
    workspace: Path,
    description: str,
) -> Path:
    document = _mapping(identity, description)
    path = _workspace_path_from_document(
        document.get("path"), workspace, f"{description} path", directory=False
    )
    expected_sha256 = _validate_sha256(
        document.get("sha256"), f"{description} SHA256"
    )
    size = document.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SmokeError(f"{description} size is invalid")
    actual_sha256, actual_size, _ = _regular_file_identity(path, description)
    if actual_size != size or actual_sha256 != expected_sha256:
        raise SmokeError(f"{description} differs from the fresh-probe anchor")
    return path


def _expand_probe_token(value: object, prefix: Path, description: str) -> Path:
    if not isinstance(value, str):
        raise SmokeError(f"{description} path is malformed")
    marker = "$PROBE_PREFIX/"
    if not value.startswith(marker):
        raise SmokeError(f"{description} is not normalized below $PROBE_PREFIX")
    relative = Path(value.removeprefix(marker))
    if relative.is_absolute() or ".." in relative.parts:
        raise SmokeError(f"{description} escaped $PROBE_PREFIX")
    try:
        resolved = (prefix / relative).resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"{description} is absent") from error
    if not _is_within(resolved, prefix):
        raise SmokeError(f"{description} escaped the probe prefix")
    return resolved


def _forbidden_view_entry(path: Path) -> str | None:
    lower = path.name.casefold()
    if lower == "__pycache__":
        return "bytecode-cache"
    if lower in ("sitecustomize.py", "usercustomize.py"):
        return "site-startup-code"
    if lower.endswith((".pth", ".egg-link")):
        return "site-path-carrier"
    if "editable" in lower:
        return "editable-carrier"
    if (
        lower in ("triton", "triton.py")
        or re.fullmatch(r"triton[-_.].*\.(?:dist|egg)-info", lower) is not None
        or re.fullmatch(r"triton[-_.].*\.egg-info", lower) is not None
    ):
        return "ambient-triton-carrier"
    return None


def _validate_filtered_torch_view(
    view: Path,
    source_site: Path,
    view_evidence: dict[str, object],
) -> None:
    included = _sequence(
        view_evidence.get("included_entries"),
        "probe Torch-runtime-view included entries",
    )
    if (
        any(not isinstance(name, str) or not name or "/" in name for name in included)
        or included != sorted(set(included))
    ):
        raise SmokeError("probe Torch-runtime-view entry inventory is malformed")
    actual = sorted(path.name for path in view.iterdir())
    if actual != included:
        raise SmokeError("filtered Torch runtime view changed after the fresh probe")
    count = view_evidence.get("included_entries_count")
    digest = view_evidence.get("included_entries_sha256")
    if count != len(actual) or digest != sha256_bytes(canonical_json(actual).encode("ascii")):
        raise SmokeError("filtered Torch runtime view differs from its probe digest")
    for name in actual:
        path = view / name
        reason = _forbidden_view_entry(path)
        if reason is not None:
            raise SmokeError(f"filtered Torch runtime view contains {reason}: {name}")
        if not path.is_symlink():
            raise SmokeError(f"filtered Torch runtime view entry is not a projection: {name}")
        try:
            target = path.resolve(strict=True)
        except OSError as error:
            raise SmokeError(f"filtered Torch runtime view link is broken: {name}") from error
        if not _is_within(target, source_site):
            raise SmokeError(f"filtered Torch runtime view escaped its source site: {name}")
    if "torch" not in actual or not any(
        re.fullmatch(r"torch-.*\.dist-info", name, flags=re.IGNORECASE)
        for name in actual
    ):
        raise SmokeError("filtered Torch runtime view omitted Torch package metadata")


def _validate_probe_site(site: Path) -> None:
    for path in site.iterdir():
        lower = path.name.casefold()
        if (
            lower.endswith((".pth", ".egg-link"))
            or "editable" in lower
            or lower in ("sitecustomize.py", "usercustomize.py")
        ):
            raise SmokeError(f"probe site contains ambient/editable carrier: {path.name}")
    package = site / "triton"
    if not package.is_dir() or package.is_symlink():
        raise SmokeError("probe site has no real wheel-owned Triton package")


def _single_metadata_header(message: object, name: str) -> str:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) != 1 or not isinstance(values[0], str):
        raise SmokeError(f"Torch METADATA must contain exactly one {name}")
    return values[0].strip()


def _torch_tree_paths(torch_root: Path, dist_info: Path) -> list[Path]:
    return sorted(
        path
        for root in (torch_root, dist_info)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _torch_tree_identity(
    torch_site: Path, torch_root: Path, dist_info: Path
) -> dict[str, object]:
    """Reproduce the exact complete-tree digest recorded by the fresh probe."""

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    files = _torch_tree_paths(torch_root, dist_info)
    for path in files:
        relative = path.relative_to(torch_site).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        if path.is_symlink():
            target = os.readlink(path).encode()
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "little"))
            digest.update(target)
            continue
        digest.update(b"F")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise SmokeError(f"Torch tree entry disappeared: {path}") from error
        digest.update(size.to_bytes(8, "little"))
        _, streamed_size, _ = _regular_file_identity(
            path, "Torch tree file", content_digest=digest
        )
        if streamed_size != size:
            raise SmokeError(f"Torch tree file size changed: {path}")
        file_count += 1
        byte_count += size
    if files != _torch_tree_paths(torch_root, dist_info):
        raise SmokeError("complete Torch tree changed while it was inventoried")
    return {
        "torch_tree_bytes": byte_count,
        "torch_tree_files": file_count,
        "torch_tree_sha256": digest.hexdigest(),
    }


def _environment_and_torch_snapshot(anchor: ProbeAnchor) -> dict[str, object]:
    inputs = _mapping(anchor.probe_document.get("inputs"), "probe inputs")
    environment_identity = _mapping(
        inputs.get("environment_lock"), "probe environment-lock identity"
    )
    environment, environment_raw = load_canonical_json(
        anchor.environment_lock, "environment lock"
    )
    if (
        sha256_bytes(environment_raw)
        != _validate_sha256(
            environment_identity.get("sha256"), "environment-lock SHA256"
        )
        or len(environment_raw) != environment_identity.get("size")
    ):
        raise SmokeError("ENVIRONMENT.lock changed after the fresh probe")

    torch_input = _mapping(
        inputs.get("torch_site_packages"), "probe Torch-site identity"
    )
    torch_root = anchor.torch_source_site / "torch"
    torch_init = torch_root / "__init__.py"
    if (
        not torch_root.is_dir()
        or torch_root.is_symlink()
        or not torch_init.is_file()
        or torch_init.is_symlink()
    ):
        raise SmokeError("complete Torch package root is absent or indirect")
    dist_infos = sorted(anchor.torch_source_site.glob("torch-*.dist-info"))
    if (
        len(dist_infos) != 1
        or not dist_infos[0].is_dir()
        or dist_infos[0].is_symlink()
    ):
        raise SmokeError("Torch source site no longer has exactly one dist-info")
    dist_info = dist_infos[0]
    metadata_raw = _read_regular_file_once(
        dist_info / "METADATA", "Torch METADATA"
    )
    metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
    if _single_metadata_header(metadata, "Name").casefold() != "torch":
        raise SmokeError("Torch METADATA distribution name changed")
    version = _single_metadata_header(metadata, "Version")
    tree_identity = _torch_tree_identity(
        anchor.torch_source_site, torch_root, dist_info
    )
    observed = {
        **tree_identity,
        "dist_info_name": dist_info.name,
        "metadata_sha256": sha256_bytes(metadata_raw),
        "module_sha256": sha256_file(torch_init),
        "version": version,
    }
    expected = {
        name: torch_input.get(name)
        for name in (
            "dist_info_name",
            "metadata_sha256",
            "module_sha256",
            "torch_tree_bytes",
            "torch_tree_files",
            "torch_tree_sha256",
            "version",
        )
    }
    if observed != expected:
        raise SmokeError("complete Torch package tree differs from probe evidence")

    process = _mapping(
        _sequence(
            _mapping(anchor.probe_document.get("runtime"), "probe runtime").get(
                "processes"
            ),
            "probe processes",
        )[0],
        "probe runtime process",
    )
    runtime_torch = _mapping(process.get("torch"), "probe Torch runtime")
    expected_environment = {
        "cuda": runtime_torch.get("cuda"),
        "hip": runtime_torch.get("hip"),
        "torch": version,
        "torch_git": runtime_torch.get("git_version"),
        **tree_identity,
    }
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise SmokeError("ENVIRONMENT.lock Torch version/git/CUDA/tree identity changed")
    if (
        environment.get("python_abi") != "cp314"
        or environment.get("python_implementation") != "CPython"
        or not str(environment.get("python", "")).startswith("3.14.6 ")
    ):
        raise SmokeError("ENVIRONMENT.lock Python identity changed")
    expected_paths = {
        "python_executable": anchor.base_python,
        "torch_dist_info": dist_info,
        "torch_file": torch_init,
        "torch_package_root": torch_root,
    }
    for name, expected_path in expected_paths.items():
        value = environment.get(name)
        if not isinstance(value, str):
            raise SmokeError(f"ENVIRONMENT.lock {name} is absent")
        try:
            resolved = Path(value).resolve(strict=True)
        except OSError as error:
            raise SmokeError(f"ENVIRONMENT.lock {name} target is absent") from error
        if resolved != expected_path:
            raise SmokeError(f"ENVIRONMENT.lock {name} changed")

    return {
        "environment_lock": {
            "cuda": environment["cuda"],
            "hip": environment["hip"],
            "path": _workspace_relative(anchor.environment_lock, anchor.workspace),
            "sha256": sha256_bytes(environment_raw),
            "size": len(environment_raw),
            "torch": environment["torch"],
            "torch_git": environment["torch_git"],
        },
        "torch_tree": {
            **observed,
            "cuda": environment["cuda"],
            "git_version": environment["torch_git"],
            "hip": environment["hip"],
            "path": _workspace_relative(anchor.torch_source_site, anchor.workspace),
        },
    }


def _installed_record_target(
    value: object, probe_site: Path, prefix: Path
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SmokeError("installed RECORD path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".") for part in relative.parts)
    ):
        raise SmokeError(f"installed RECORD path is non-canonical: {value!r}")
    lexical = _absolute_lexical(probe_site.joinpath(*relative.parts))
    if not _is_within(lexical, prefix):
        raise SmokeError(f"installed RECORD path escaped the probe prefix: {value}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SmokeError(f"installed RECORD target is absent: {value}") from error
    if resolved != lexical or not resolved.is_file() or resolved.is_symlink():
        raise SmokeError(f"installed RECORD target is not a regular owned file: {value}")
    return value, resolved


def _tree_regular_paths(root: Path, description: str) -> set[Path]:
    if not root.is_dir() or root.is_symlink():
        raise SmokeError(f"{description} root is absent or indirect")
    paths: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SmokeError(f"{description} contains a symlink: {path}")
        if path.is_file():
            paths.add(path.resolve(strict=True))
        elif not path.is_dir():
            raise SmokeError(f"{description} contains a non-regular entry: {path}")
    return paths


def _installed_triton_snapshot(
    anchor: ProbeAnchor, audit_document: dict[str, object]
) -> dict[str, object]:
    installation = _mapping(
        anchor.probe_document.get("installation"), "probe installation"
    )
    verification = _mapping(
        installation.get("record_verification"), "installed RECORD verification"
    )
    expected_entries = _sequence(
        verification.get("entries"), "installed RECORD entries"
    )
    if (
        verification.get("editable_artifacts") != []
        or verification.get("entries_count") != len(expected_entries)
    ):
        raise SmokeError("installed RECORD verification summary is malformed")

    current_entries: list[dict[str, object]] = []
    target_by_record: dict[str, Path] = {}
    native_records: list[dict[str, object]] = []
    for value in expected_entries:
        expected = _mapping(value, "installed RECORD entry")
        record_path, target = _installed_record_target(
            expected.get("path"), anchor.probe_site, anchor.prefix
        )
        if record_path in target_by_record:
            raise SmokeError(f"duplicate installed RECORD path: {record_path}")
        expected_sha256 = _validate_sha256(
            expected.get("sha256"), f"installed RECORD {record_path} SHA256"
        )
        expected_size = expected.get("size")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise SmokeError(f"installed RECORD {record_path} size is invalid")
        actual_sha256, actual_size, prefix = _regular_file_identity(
            target, "installed Triton RECORD-owned file"
        )
        current = {
            "path": record_path,
            "sha256": actual_sha256,
            "size": actual_size,
        }
        if current != expected:
            raise SmokeError(f"installed Triton file changed: {record_path}")
        current_entries.append(current)
        target_by_record[record_path] = target
        if prefix == b"\x7fELF":
            native_records.append(current)
    current_entries.sort(key=lambda record: str(record["path"]))
    if current_entries != expected_entries:
        raise SmokeError("installed RECORD entry ordering/content changed")

    package_paths = _tree_regular_paths(
        anchor.probe_site / "triton", "installed Triton package"
    ) | _tree_regular_paths(anchor.triton_dist_info, "installed Triton dist-info")
    owned_package_paths = {
        target
        for target in target_by_record.values()
        if _below(target, anchor.probe_site / "triton")
        or _below(target, anchor.triton_dist_info)
    }
    if package_paths != owned_package_paths:
        raise SmokeError("installed Triton package/dist-info tree is not exactly RECORD-owned")
    package_records = [
        record
        for record in current_entries
        if target_by_record[str(record["path"])] in package_paths
    ]

    wheel_audit = _mapping(audit_document.get("wheel"), "wheel-audit wheel")
    elf_paths = _sequence(wheel_audit.get("elf_paths"), "wheel-audit ELF paths")
    if (
        not elf_paths
        or any(not isinstance(path, str) for path in elf_paths)
        or elf_paths != sorted(set(elf_paths))
    ):
        raise SmokeError("wheel-audit ELF inventory is malformed")
    archive_members = _sequence(
        installation.get("archive_members"), "installed archive members"
    )
    installed_by_archive: dict[str, str] = {}
    for value in archive_members:
        member = _mapping(value, "installed archive member")
        archive_path = member.get("archive_path")
        installed_path = member.get("installed_path")
        if not isinstance(archive_path, str) or not isinstance(installed_path, str):
            raise SmokeError("installed archive member path is malformed")
        if archive_path in installed_by_archive:
            raise SmokeError(f"duplicate installed archive member: {archive_path}")
        installed_by_archive[archive_path] = installed_path
    try:
        expected_native_paths = {installed_by_archive[str(path)] for path in elf_paths}
    except KeyError as error:
        raise SmokeError("wheel-audit ELF is absent from installed archive members") from error
    actual_native_paths = {str(record["path"]) for record in native_records}
    if actual_native_paths != expected_native_paths:
        raise SmokeError("installed native tree differs from wheel-audit ELF inventory")
    native_records.sort(key=lambda record: str(record["path"]))

    return {
        "native_bytes": sum(int(record["size"]) for record in native_records),
        "native_entries_count": len(native_records),
        "native_entries_sha256": sha256_bytes(
            canonical_json(native_records).encode("ascii")
        ),
        "package_entries_count": len(package_records),
        "package_entries_sha256": sha256_bytes(
            canonical_json(package_records).encode("ascii")
        ),
        "record_entries_count": len(current_entries),
        "record_entries_sha256": sha256_bytes(
            canonical_json(current_entries).encode("ascii")
        ),
    }


def _integrity_snapshot(anchor: ProbeAnchor) -> dict[str, object]:
    inputs = _mapping(anchor.probe_document.get("inputs"), "probe inputs")
    probe_sha256, probe_size, _ = _regular_file_identity(
        anchor.probe_evidence, "probe evidence"
    )
    if (
        probe_sha256 != anchor.probe_evidence_sha256
        or probe_size != anchor.probe_evidence_size
    ):
        raise SmokeError("probe evidence changed during the reference smoke")

    if anchor.python.resolve(strict=True) != anchor.base_python:
        raise SmokeError("probe bin/python no longer resolves to the audited base Python")
    base_identity = _mapping(inputs.get("base_python"), "probe base-Python identity")
    _verify_linked_file_identity(base_identity, anchor.workspace, "base Python")

    wheel_identity = _mapping(inputs.get("wheel"), "probe wheel identity")
    _verify_linked_file_identity(
        wheel_identity, anchor.workspace, "audited Triton wheel"
    )
    audit_identity = _mapping(
        wheel_identity.get("audit_evidence"), "probe wheel-audit identity"
    )
    audit_path = _verify_linked_file_identity(
        audit_identity, anchor.workspace, "wheel-audit evidence"
    )
    audit_document, audit_raw = load_canonical_json(
        audit_path, "wheel-audit evidence"
    )
    if (
        audit_document.get("acceptance") != "accepted"
        or audit_document.get("audit") != WHEEL_AUDIT_NAME
        or audit_document.get("schema_version") != 1
        or sha256_bytes(audit_raw) != audit_identity.get("sha256")
    ):
        raise SmokeError("wheel-audit evidence identity changed")

    base_sha256, base_size, _ = _regular_file_identity(
        anchor.base_python, "base Python"
    )
    linked = {
        "base_python": {
            "path": _workspace_relative(anchor.base_python, anchor.workspace),
            "sha256": base_sha256,
            "size": base_size,
        },
        "probe_evidence": {
            "path": _workspace_relative(anchor.probe_evidence, anchor.workspace),
            "sha256": probe_sha256,
            "size": probe_size,
        },
        "wheel": {
            "audit_evidence_sha256": audit_identity["sha256"],
            "sha256": wheel_identity["sha256"],
        },
    }
    return {
        **_environment_and_torch_snapshot(anchor),
        "installed_triton": _installed_triton_snapshot(anchor, audit_document),
        "linked_inputs": linked,
    }


def _validate_probe_anchor(
    document: dict[str, object],
    raw: bytes,
    *,
    workspace: Path,
    probe_path: Path,
    expected_probe_sha256: str,
    prefix: Path,
    probe_site: Path,
    torch_runtime_view: Path,
) -> ProbeAnchor:
    if (
        document.get("acceptance") != "accepted"
        or document.get("probe") != PROBE_NAME
        or document.get("schema_version") != 1
    ):
        raise SmokeError("probe evidence is not accepted triton-fresh-wheel schema 1")
    if sha256_bytes(raw) != expected_probe_sha256:
        raise SmokeError("probe evidence differs from its expected SHA256 anchor")

    inputs = _mapping(document.get("inputs"), "probe inputs")
    base_python_identity = _mapping(
        inputs.get("base_python"), "probe base-Python identity"
    )
    base_python = _verify_linked_file_identity(
        base_python_identity, workspace, "base Python"
    )
    environment_lock = _mapping(
        inputs.get("environment_lock"), "probe environment-lock identity"
    )
    environment_lock_path = _verify_linked_file_identity(
        environment_lock, workspace, "environment lock"
    )
    wheel = _mapping(inputs.get("wheel"), "probe wheel identity")
    _verify_linked_file_identity(wheel, workspace, "audited Triton wheel")
    _verify_linked_file_identity(
        wheel.get("audit_evidence"), workspace, "wheel-audit evidence"
    )
    torch_input = _mapping(
        inputs.get("torch_site_packages"), "probe Torch-site identity"
    )
    torch_source_site = _workspace_path_from_document(
        torch_input.get("path"),
        workspace,
        "probe Torch source site",
        directory=True,
    )

    installation = _mapping(document.get("installation"), "probe installation")
    if installation.get("fresh_prefix") is not True:
        raise SmokeError("probe evidence does not attest a fresh prefix")
    recorded_prefix = _workspace_path_from_document(
        installation.get("prefix"), workspace, "probe prefix", directory=True
    )
    if recorded_prefix != prefix:
        raise SmokeError("--probe-prefix differs from probe evidence")
    scheme = _mapping(installation.get("scheme"), "probe install scheme")
    recorded_site = _workspace_path_from_document(
        scheme.get("platlib"), workspace, "probe platlib", directory=True
    )
    if recorded_site != probe_site or not _is_within(recorded_site, prefix):
        raise SmokeError("--probe-site differs from the wheel-owned probe platlib")
    recorded_python_value = installation.get("python")
    if not isinstance(recorded_python_value, str) or Path(recorded_python_value).is_absolute():
        raise SmokeError("probe Python path is malformed")
    recorded_python = _absolute_lexical(workspace / recorded_python_value)
    if not recorded_python.exists() and not recorded_python.is_symlink():
        raise SmokeError("probe Python is absent")
    if not _is_within(recorded_python, prefix):
        raise SmokeError("probe Python escaped the fresh prefix")
    try:
        resolved_probe_python = recorded_python.resolve(strict=True)
    except OSError as error:
        raise SmokeError("probe Python link is broken") from error
    if resolved_probe_python != base_python:
        raise SmokeError("probe bin/python no longer resolves to probe base_python")

    view_evidence = _mapping(
        installation.get("torch_runtime_view"), "probe Torch runtime view"
    )
    recorded_view = _workspace_path_from_document(
        view_evidence.get("path"),
        workspace,
        "filtered Torch runtime view",
        directory=True,
    )
    if recorded_view != torch_runtime_view or not _is_within(recorded_view, prefix):
        raise SmokeError("--torch-runtime-view differs from probe evidence")
    _validate_filtered_torch_view(recorded_view, torch_source_site, view_evidence)
    _validate_probe_site(recorded_site)

    runtime = _mapping(document.get("runtime"), "probe runtime")
    processes = _sequence(runtime.get("processes"), "probe runtime processes")
    if (
        runtime.get("gpu_execution") is not False
        or runtime.get("processes_count") != 2
        or len(processes) != 2
        or processes[0] != processes[1]
    ):
        raise SmokeError("probe did not retain two identical non-GPU process reports")
    process = _mapping(processes[0], "probe runtime process")
    distribution = _mapping(process.get("distribution"), "probe distribution")
    triton_dist_info = _expand_probe_token(
        distribution.get("dist_info"), prefix, "probe Triton dist-info"
    )
    if not _below(triton_dist_info, probe_site) or not triton_dist_info.is_dir():
        raise SmokeError("probe Triton dist-info escaped the wheel-owned site")
    backend = _mapping(process.get("backend"), "probe backend")
    if backend.get("target") != list(EXPECTED_TARGET) or backend.get("arch") != "sm120":
        raise SmokeError("probe evidence is not bound to the exact SM120 backend")
    editable = _mapping(process.get("editable"), "probe editable report")
    carriers = _mapping(editable.get("carriers"), "probe editable carriers")
    if editable.get("loaded_modules") != [] or any(value != [] for value in carriers.values()):
        raise SmokeError("probe evidence retained an editable importer carrier")

    ptxas = _mapping(process.get("ptxas_blackwell"), "probe ptxas-blackwell")
    ptxas_path = _expand_probe_token(
        ptxas.get("path"), prefix, "probe ptxas-blackwell"
    )
    expected_ptxas = probe_site / PTXAS_BLACKWELL_RELATIVE
    if ptxas_path != expected_ptxas.resolve(strict=True):
        raise SmokeError("probe ptxas-blackwell is not the wheel-owned binary")
    ptxas_sha256 = _validate_sha256(
        ptxas.get("sha256"), "probe ptxas-blackwell SHA256"
    )
    if (
        ptxas.get("audited_full_version") != EXPECTED_PTXAS_FULL_VERSION
        or ptxas.get("reported_release") != EXPECTED_PTXAS_RELEASE
        or sha256_file(ptxas_path) != ptxas_sha256
    ):
        raise SmokeError("wheel-owned ptxas-blackwell changed after the fresh probe")

    map_values = _sequence(process.get("libtriton_maps"), "probe libtriton maps")
    libtriton_maps = tuple(
        sorted(
            {
                _expand_probe_token(value, prefix, "probe libtriton map")
                for value in map_values
            },
            key=str,
        )
    )
    if not libtriton_maps:
        raise SmokeError("probe evidence has no wheel-owned libtriton mapping")
    module_paths = _mapping(process.get("module_paths"), "probe module paths")
    if "triton._C.libtriton" not in module_paths:
        raise SmokeError("probe evidence omitted triton._C.libtriton provenance")
    for name, values in module_paths.items():
        if not isinstance(name, str):
            raise SmokeError("probe module name is malformed")
        for value in _sequence(values, f"probe module paths for {name}"):
            _expand_probe_token(value, prefix, f"probe module {name}")

    return ProbeAnchor(
        workspace=workspace,
        base_python=base_python,
        environment_lock=environment_lock_path,
        prefix=prefix,
        probe_site=probe_site,
        torch_runtime_view=torch_runtime_view,
        torch_source_site=torch_source_site,
        triton_dist_info=triton_dist_info,
        python=recorded_python,
        probe_evidence=probe_path,
        probe_evidence_sha256=expected_probe_sha256,
        probe_evidence_size=len(raw),
        probe_document=document,
        ptxas_path=ptxas_path,
        ptxas_sha256=ptxas_sha256,
        libtriton_maps=libtriton_maps,
    )


def prepare_smoke(request: SmokeRequest) -> PreparedSmoke:
    workspace = _require_workspace(request.workspace)

    # No-replace is deliberately the first request-level gate.  A repeated
    # invocation must not import Triton, initialize CUDA, or create a cache.
    evidence = _require_evidence_output(request.evidence, workspace)
    probe_path = _require_workspace_file(
        request.probe_evidence, workspace, "probe evidence"
    )
    expected_probe_sha256 = _validate_sha256(
        request.expected_probe_evidence_sha256,
        "expected probe-evidence SHA256",
    )
    prefix = _require_workspace_directory(
        request.probe_prefix, workspace, "probe prefix"
    )
    probe_site = _require_workspace_directory(
        request.probe_site, workspace, "probe site"
    )
    torch_runtime_view = _require_workspace_directory(
        request.torch_runtime_view, workspace, "filtered Torch runtime view"
    )
    cache_dir = _require_fresh_workspace_path(
        request.cache_dir, workspace, "Triton cache"
    )
    if _is_within(cache_dir, prefix) or _is_within(prefix, cache_dir):
        raise SmokeError("fresh Triton cache must be disjoint from the probe prefix")
    if evidence == cache_dir or _is_within(evidence, cache_dir):
        raise SmokeError("evidence output must be outside the Triton cache")

    document, raw = load_canonical_json(probe_path, "probe evidence")
    anchor = _validate_probe_anchor(
        document,
        raw,
        workspace=workspace,
        probe_path=probe_path,
        expected_probe_sha256=expected_probe_sha256,
        prefix=prefix,
        probe_site=probe_site,
        torch_runtime_view=torch_runtime_view,
    )
    integrity_before = _integrity_snapshot(anchor)
    return PreparedSmoke(
        anchor=anchor,
        cache_dir=cache_dir,
        evidence=evidence,
        integrity_before=integrity_before,
    )


def _validate_isolated_invocation(anchor: ProbeAnchor) -> None:
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise SmokeError("runtime must be invoked with both -I and -S")
    try:
        runtime_prefix = Path(sys.prefix).resolve(strict=True)
    except OSError as error:
        raise SmokeError("runtime sys.prefix is absent") from error
    if runtime_prefix != anchor.prefix:
        raise SmokeError("runtime sys.prefix differs from the fresh probe prefix")
    if _absolute_lexical(Path(sys.executable)) != anchor.python:
        raise SmokeError("runtime executable differs from the probe Python")
    try:
        resolved_executable = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise SmokeError("runtime executable link is broken") from error
    if resolved_executable != anchor.base_python:
        raise SmokeError("runtime executable does not resolve to probe base_python")
    if "" in sys.path:
        raise SmokeError("isolated runtime retained an empty import path")
    forbidden = {str(anchor.probe_site), str(anchor.torch_runtime_view)}
    if any(value in forbidden for value in sys.path):
        raise SmokeError("probe imports were exposed before manual path insertion")
    for value in sys.path:
        if not isinstance(value, str):
            raise SmokeError("runtime sys.path contains a non-string entry")
        lower = value.casefold()
        if "site-packages" in lower or "dist-packages" in lower:
            raise SmokeError(f"isolated runtime retained an ambient package path: {value}")
    # -B is recommended in the command line; this assignment also prevents
    # package-tree mutation when the caller supplied exactly -I -S.
    sys.dont_write_bytecode = True


@contextmanager
def _fresh_cache_environment(cache_dir: Path) -> Iterator[None]:
    present = [name for name in _FORBIDDEN_TRITON_ENVIRONMENT if name in os.environ]
    if present:
        raise SmokeError(f"ambient Triton runtime override is forbidden: {present}")
    old = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = old


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _carrier_modules(value: object) -> list[str]:
    return sorted(
        {
            item
            for item in (
                getattr(value, "__module__", None),
                getattr(type(value), "__module__", None),
            )
            if isinstance(item, str) and item
        }
    )


def _editable_report() -> dict[str, object]:
    carriers: dict[str, list[str]] = {}
    for carrier_name, values in (
        ("meta_path", sys.meta_path),
        ("path_hooks", sys.path_hooks),
        ("path_importer_cache", sys.path_importer_cache.values()),
    ):
        found = sorted(
            {
                module_name
                for value in values
                for module_name in _carrier_modules(value)
                if "editable" in module_name.casefold()
            }
        )
        if found:
            raise SmokeError(
                f"editable importer carrier is active: {carrier_name}={found}"
            )
        carriers[carrier_name] = found
    loaded = sorted(
        name
        for name in sys.modules
        if name.startswith(("_editable", "__editable"))
        or "editable_finder" in name.casefold()
    )
    if loaded:
        raise SmokeError(f"editable finder modules are loaded: {loaded}")
    return {"carriers": carriers, "loaded_modules": loaded}


def _load_runtime(anchor: ProbeAnchor) -> RuntimeContext:
    preloaded = sorted(
        name
        for name in sys.modules
        if name in ("torch", "triton")
        or name.startswith(("torch.", "triton."))
    )
    if preloaded:
        raise SmokeError(f"Torch/Triton was loaded before explicit path setup: {preloaded}")

    sys.path.insert(0, str(anchor.probe_site))
    distributions = [
        distribution
        for distribution in importlib.metadata.distributions(
            path=[str(anchor.probe_site), str(anchor.torch_runtime_view)]
        )
        if (distribution.metadata.get("Name") or "").casefold().replace("_", "-")
        == "triton"
    ]
    if len(distributions) != 1:
        raise SmokeError("explicit probe paths do not expose exactly one Triton distribution")
    distribution_path = Path(distributions[0]._path).resolve(strict=True)
    if not _below(distribution_path, anchor.probe_site):
        raise SmokeError("Triton distribution metadata escaped the probe site")

    triton = importlib.import_module("triton")
    importlib.import_module("triton._C.libtriton")
    language = importlib.import_module("triton.language")
    compiler = importlib.import_module("triton.backends.nvidia.compiler")
    selected_ptxas = compiler.get_ptxas(120)
    selected_path = Path(selected_ptxas.path).resolve(strict=True)
    if (
        selected_path != anchor.ptxas_path
        or not _below(selected_path, anchor.prefix)
        or sha256_file(selected_path) != anchor.ptxas_sha256
        or selected_ptxas.version != EXPECTED_PTXAS_RELEASE
    ):
        raise SmokeError("SM120 did not select the probe-bound wheel ptxas-blackwell")

    sys.path.append(str(anchor.torch_runtime_view))
    torch = importlib.import_module("torch")
    torch_file = Path(torch.__file__).resolve(strict=True)
    if not _below(torch_file, anchor.torch_source_site):
        raise SmokeError("Torch import escaped the filtered runtime-view source site")

    process = _mapping(
        _sequence(
            _mapping(anchor.probe_document["runtime"], "probe runtime").get("processes"),
            "probe processes",
        )[0],
        "probe runtime process",
    )
    expected_torch = _mapping(process.get("torch"), "probe Torch runtime")
    if (
        str(torch.__version__) != expected_torch.get("version")
        or torch.version.git_version != expected_torch.get("git_version")
        or torch.version.cuda != expected_torch.get("cuda")
        or torch.version.hip is not None
    ):
        raise SmokeError("Torch runtime identity changed after the fresh probe")

    _editable_report()
    return RuntimeContext(
        torch=torch,
        triton=triton,
        language=language,
        ptxas_evidence={
            "audited_full_version": EXPECTED_PTXAS_FULL_VERSION,
            "path": _normalize_runtime_path(selected_path, anchor),
            "reported_release": selected_ptxas.version,
            "sha256": anchor.ptxas_sha256,
            "wheel_owned": True,
        },
    )


# ``from __future__ import annotations`` keeps this import-free at module load.
# The exact wheel-owned ``triton.language`` module is assigned immediately
# before wrapping the source function with ``triton.jit``.
tl: Any = None


def _vector_add_kernel(
    x_pointer,
    y_pointer,
    output_pointer,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_pointer + offsets, mask=mask)
    y = tl.load(y_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, x + y, mask=mask)


def _execute_vector_add(context: RuntimeContext) -> dict[str, object]:
    torch = context.torch
    triton = context.triton
    if torch.cuda.is_available() is not True or torch.cuda.device_count() < 1:
        raise SmokeError("CUDA device 0 is unavailable")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if name != EXPECTED_DEVICE_NAME or capability != EXPECTED_CAPABILITY:
        raise SmokeError(
            f"wrong GPU: name={name!r}, capability={capability!r}"
        )

    target = triton.runtime.driver.active.get_current_target()
    target_tuple = (target.backend, target.arch, target.warp_size)
    if target_tuple != EXPECTED_TARGET:
        raise SmokeError(f"active Triton target is not SM120: {target_tuple!r}")

    device = torch.device("cuda", 0)
    x = torch.arange(VECTOR_ELEMENTS, dtype=torch.float32, device=device)
    y = torch.full((VECTOR_ELEMENTS,), 0.5, dtype=torch.float32, device=device)
    reference = x + y
    output = torch.empty_like(x)

    synchronization = {
        "after_comparison": False,
        "after_kernel": False,
        "before_launch": False,
        "error": None,
    }
    try:
        torch.cuda.synchronize(0)
        synchronization["before_launch"] = True

        global tl
        tl = context.language
        kernel = triton.jit(_vector_add_kernel)
        grid = (triton.cdiv(VECTOR_ELEMENTS, BLOCK_SIZE),)
        kernel[grid](
            x,
            y,
            output,
            VECTOR_ELEMENTS,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize(0)
        synchronization["after_kernel"] = True
        equal = bool(torch.equal(output, reference))
        torch.cuda.synchronize(0)
        synchronization["after_comparison"] = True
    except Exception as error:
        synchronization["error"] = f"{type(error).__module__}.{type(error).__qualname__}: {error}"
        raise SmokeError(f"Triton SM120 vector-add execution failed: {error}") from error
    if not equal:
        raise SmokeError("Triton FP32 vector add did not match Torch with torch.equal")

    return {
        "correctness": {
            "block_size": BLOCK_SIZE,
            "comparison": "torch.equal",
            "dtype": "float32",
            "equal": True,
            "kernel": "masked-vector-add",
            "n_elements": VECTOR_ELEMENTS,
            "reference_provider": "torch",
        },
        "device": {
            "compute_capability": list(capability),
            "index": 0,
            "name": name,
        },
        "synchronization": synchronization,
        "target": {
            "arch": target.arch,
            "backend": target.backend,
            "warp_size": target.warp_size,
        },
    }


def _normalize_runtime_path(path: Path, anchor: ProbeAnchor) -> str:
    resolved = path.resolve(strict=True)
    for marker, root in (
        ("$PROBE_PREFIX", anchor.prefix),
        ("$TORCH_SITE_PACKAGES", anchor.torch_source_site),
        ("$TORCH_RUNTIME_VIEW", anchor.torch_runtime_view),
    ):
        if _below(resolved, root):
            relative = resolved.relative_to(root).as_posix()
            return marker if relative == "." else f"{marker}/{relative}"
    return str(resolved)


def _collect_runtime_provenance(
    context: RuntimeContext, anchor: ProbeAnchor
) -> dict[str, object]:
    module_paths: dict[str, list[str]] = {}
    libtriton_owner = Path(
        sys.modules["triton._C.libtriton"].__file__
    ).resolve(strict=True)
    for name, module in sorted(sys.modules.items()):
        if name != "triton" and not name.startswith("triton."):
            continue
        values: list[str] = []
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            values.append(module_file)
        module_path = getattr(module, "__path__", None)
        if module_path is not None:
            values.extend(value for value in module_path if isinstance(value, str))
        resolved = sorted(
            {Path(value).resolve(strict=True) for value in values}, key=str
        )
        if not resolved and name.startswith("triton._C.libtriton."):
            resolved = [libtriton_owner]
        if not resolved or any(not _below(path, anchor.prefix) for path in resolved):
            raise SmokeError(f"loaded Triton module escaped wheel ownership: {name}")
        module_paths[name] = [
            _normalize_runtime_path(path, anchor) for path in resolved
        ]
    if "triton._C.libtriton" not in module_paths:
        raise SmokeError("runtime module inventory omitted triton._C.libtriton")

    mapped: set[Path] = set()
    maps_path = Path("/proc/self/maps")
    for line in maps_path.read_text(encoding="utf-8", errors="strict").splitlines():
        if "libtriton" not in line.casefold():
            continue
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            raise SmokeError(f"unrecognized libtriton map: {line}")
        if fields[5].endswith(" (deleted)"):
            raise SmokeError(f"mapped libtriton was deleted: {fields[5]}")
        path = Path(fields[5]).resolve(strict=True)
        if not _below(path, anchor.prefix):
            raise SmokeError(f"mapped libtriton escaped the probe prefix: {path}")
        mapped.add(path)
    if mapped != set(anchor.libtriton_maps):
        raise SmokeError("runtime libtriton mapping differs from the fresh probe")

    probe_site_value = str(anchor.probe_site)
    view_value = str(anchor.torch_runtime_view)
    if (
        not sys.path
        or sys.path[0] != probe_site_value
        or sys.path.count(probe_site_value) != 1
        or sys.path.count(view_value) != 1
        or sys.path[-1] != view_value
    ):
        raise SmokeError("manual probe-site/Torch-view import order changed")
    for value in sys.path[1:-1]:
        lower = value.casefold()
        if "site-packages" in lower or "dist-packages" in lower:
            raise SmokeError(f"runtime gained an ambient package path: {value}")

    torch_file = Path(context.torch.__file__).resolve(strict=True)
    if not _below(torch_file, anchor.torch_source_site):
        raise SmokeError("runtime Torch module escaped its filtered source site")
    resolved_executable = Path(sys.executable).resolve(strict=True)
    if resolved_executable != anchor.base_python:
        raise SmokeError("runtime sys.executable resolution changed")
    base_sha256, base_size, _ = _regular_file_identity(
        resolved_executable, "resolved runtime executable"
    )
    return {
        "editable": _editable_report(),
        "libtriton_maps": [
            _normalize_runtime_path(path, anchor) for path in sorted(mapped, key=str)
        ],
        "module_paths": module_paths,
        "python": {
            "executable": "$PROBE_PREFIX/bin/python",
            "resolved_executable": _workspace_relative(
                resolved_executable, anchor.workspace
            ),
            "resolved_sha256": base_sha256,
            "resolved_size": base_size,
        },
        "sys_path": [
            _normalize_runtime_path(Path(value), anchor)
            if value and Path(value).exists()
            else value
            for value in sys.path
        ],
        "torch_file": _normalize_runtime_path(torch_file, anchor),
        "torch_runtime": {
            "cuda": context.torch.version.cuda,
            "file": _normalize_runtime_path(torch_file, anchor),
            "git_version": context.torch.version.git_version,
            "hip": context.torch.version.hip,
            "version": str(context.torch.__version__),
        },
    }


def _cache_records(cache_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    stack = [cache_dir]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name, reverse=True):
                path = Path(entry.path)
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise SmokeError(f"Triton cache contains a symlink: {path}")
                if stat.S_ISDIR(entry_stat.st_mode):
                    stack.append(path)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise SmokeError(f"Triton cache contains a non-regular artifact: {path}")
                artifact_sha256, artifact_size, _ = _regular_file_identity(
                    path, "Triton cache artifact"
                )
                records.append(
                    {
                        "path": path.relative_to(cache_dir).as_posix(),
                        "sha256": artifact_sha256,
                        "size": artifact_size,
                    }
                )
    return sorted(records, key=lambda record: str(record["path"]))


def _collect_compiled_cache(
    cache_dir: Path, workspace: Path
) -> dict[str, object]:
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise SmokeError("Triton did not retain its fresh workspace cache")
    artifacts = _cache_records(cache_dir)
    cubin_count = sum(
        1 for record in artifacts if str(record["path"]).casefold().endswith(".cubin")
    )
    if not artifacts or cubin_count < 1:
        raise SmokeError("fresh Triton cache contains no compiled CUBIN artifact")
    return {
        "artifacts": artifacts,
        "artifacts_count": len(artifacts),
        "artifacts_sha256": sha256_bytes(canonical_json(artifacts).encode("ascii")),
        "cubin_count": cubin_count,
        "fresh_before_run": True,
        "path": _workspace_relative(cache_dir, workspace),
        "total_bytes": sum(int(record["size"]) for record in artifacts),
    }


def _path_identity(path: Path, workspace: Path) -> dict[str, object]:
    identity_sha256, identity_size, _ = _regular_file_identity(
        path, "identity input"
    )
    try:
        rendered_path = path.relative_to(workspace).as_posix()
    except ValueError:
        # This only occurs in synthetic unit workspaces.  A production request
        # points --workspace at ROOT and therefore records a relative path.
        rendered_path = str(path)
    return {
        "path": rendered_path,
        "sha256": identity_sha256,
        "size": identity_size,
    }


def publish_canonical_json_no_replace(output: Path, value: object) -> str:
    encoded = canonical_json(value).encode("ascii")
    digest = sha256_bytes(encoded)
    if output.exists() or output.is_symlink():
        raise SmokeError(f"evidence already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise SmokeError(f"evidence already exists: {output}") from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def _capture_gpu_run_context(
    workspace: Path, provisional_evidence: Path
) -> dict[str, object]:
    run_id = os.environ.get("PYPTO_RUN_ID")
    mode = os.environ.get("PYPTO_RUN_MODE")
    preflight_value = os.environ.get("PYPTO_PREFLIGHT_REPORT_PATH")
    preflight_sha = os.environ.get("PYPTO_PREFLIGHT_REPORT_SHA256")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(
            r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}", run_id
        )
        is None
        or mode != "gpu-benchmark"
        or not preflight_value
    ):
        raise SmokeError("smoke is not inside an owned gpu-benchmark run")
    preflight_path = _require_workspace_file(
        Path(preflight_value), workspace, "gpu-benchmark preflight"
    )
    preflight, preflight_raw = load_canonical_json(
        preflight_path, "gpu-benchmark preflight"
    )
    actual_sha = sha256_bytes(preflight_raw)
    if preflight_sha != actual_sha:
        raise SmokeError("gpu-benchmark preflight environment digest mismatch")
    if (
        preflight.get("ok") is not True
        or preflight.get("mode") != "gpu-benchmark"
        or preflight.get("nvidia_compute_pids") != []
        or preflight.get("protected_heavy_processes") != []
        or preflight.get("protected_cpu_only_coexistence_requested") is not False
    ):
        raise SmokeError("gpu-benchmark preflight was not initially exclusive")
    return {
        "mode": mode,
        "pgid": os.getpgrp(),
        "pid": os.getpid(),
        "preflight": {
            "path": _workspace_relative(preflight_path, workspace),
            "sha256": actual_sha,
            "size": len(preflight_raw),
        },
        "provisional_evidence_path": _workspace_relative(
            provisional_evidence, workspace
        ),
        "run_id": run_id,
    }


def run_smoke(request: SmokeRequest) -> tuple[dict[str, object], str]:
    prepared = prepare_smoke(request)
    anchor = prepared.anchor
    run_context = _capture_gpu_run_context(
        anchor.workspace, prepared.evidence
    )
    _validate_isolated_invocation(anchor)
    with _fresh_cache_environment(prepared.cache_dir):
        try:
            prepared.cache_dir.mkdir(mode=0o700)
        except FileExistsError as error:
            raise SmokeError("Triton cache ceased to be fresh") from error
        context = _load_runtime(anchor)
        execution = _execute_vector_add(context)
        provenance = _collect_runtime_provenance(context, anchor)
        compiled_cache = _collect_compiled_cache(
            prepared.cache_dir, anchor.workspace
        )

    integrity_after = _integrity_snapshot(anchor)
    if integrity_after != prepared.integrity_before:
        raise SmokeError(
            "base/ENV/Torch/installed-Triton identity changed during the smoke"
        )

    inputs = _mapping(anchor.probe_document.get("inputs"), "probe inputs")
    evidence: dict[str, object] = {
        "acceptance": "gpu-execution-complete-awaiting-run-finalization",
        "inputs": {
            "base_python": inputs["base_python"],
            "environment_lock": inputs["environment_lock"],
            "probe_evidence": {
                "path": _workspace_relative(
                    anchor.probe_evidence, anchor.workspace
                ),
                "sha256": anchor.probe_evidence_sha256,
                "size": anchor.probe_evidence_size,
            },
            "probe_prefix": _workspace_relative(anchor.prefix, anchor.workspace),
            "probe_site": _workspace_relative(anchor.probe_site, anchor.workspace),
            "runner": _path_identity(Path(__file__).resolve(), anchor.workspace),
            "torch_runtime_view": _workspace_relative(
                anchor.torch_runtime_view, anchor.workspace
            ),
            "torch_site_packages": inputs["torch_site_packages"],
            "wheel": inputs["wheel"],
        },
        "runtime": {
            "compiled_cache": compiled_cache,
            "correctness": execution["correctness"],
            "device": execution["device"],
            "gpu_execution": True,
            "integrity": {
                "after": integrity_after,
                "before": prepared.integrity_before,
                "stable": True,
            },
            "provenance": provenance,
            "ptxas_blackwell": context.ptxas_evidence,
            "synchronization": execution["synchronization"],
            "target": execution["target"],
        },
        "schema_version": SCHEMA_VERSION,
        "run_context": run_context,
        "scope": {
            "coverage_result": False,
            "performance_result": False,
            "provider": "triton",
            "pypto_kernel": False,
            "reference_only": True,
        },
        "smoke": SMOKE_NAME,
    }
    digest = publish_canonical_json_no_replace(prepared.evidence, evidence)
    return evidence, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--probe-evidence", type=Path, required=True)
    parser.add_argument("--expected-probe-evidence-sha256", required=True)
    parser.add_argument("--probe-prefix", type=Path, required=True)
    parser.add_argument("--probe-site", type=Path, required=True)
    parser.add_argument("--torch-runtime-view", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = SmokeRequest(
        workspace=args.workspace,
        probe_evidence=args.probe_evidence,
        expected_probe_evidence_sha256=args.expected_probe_evidence_sha256,
        probe_prefix=args.probe_prefix,
        probe_site=args.probe_site,
        torch_runtime_view=args.torch_runtime_view,
        cache_dir=args.cache_dir,
        evidence=args.evidence,
    )
    try:
        _, digest = run_smoke(request)
    except (SmokeError, OSError, RuntimeError, ValueError) as error:
        print(f"Triton reference-only SM120 smoke failed: {error}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "evidence": str(request.evidence),
                "evidence_sha256": digest,
                "smoke": SMOKE_NAME,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
