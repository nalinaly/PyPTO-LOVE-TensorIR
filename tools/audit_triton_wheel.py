#!/usr/bin/env python3
"""Fail-closed audit for the workspace-built, PyTorch-pinned Triton wheel.

The auditor never imports the wheel.  It first verifies the ZIP and wheel
metadata in place, then manually extracts every member into a fresh
workspace-owned temporary directory.  Native files are discovered by ELF
magic rather than by suffix.  Wheel-owned executables are version-probed in a
network- and device-isolated bubblewrap sandbox.

All provenance inputs must be canonical JSON.  The accepted evidence is also
canonical JSON and is published atomically without replacing an existing
path.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA_VERSION = 1
TRITON_REPOSITORY = "https://github.com/triton-lang/triton.git"
TRITON_COMMIT = "5d6048aa0a324e090ada215b609ea76620133845"
TRITON_TREE = "448265acc1eff726c2e528813552865b33546cc9"
TRITON_LLVM_COMMIT = "ac5dc54d509169d387fcfd495d71853d81c46484"
TRITON_DISTRIBUTION_VERSION = "3.7.1+git5d6048aa"
TRITON_MODULE_VERSION = "3.7.1"
LIBDEVICE_SHA256 = (
    "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee"
)
SOURCE_ARCHIVE_SHA256 = (
    "2ebfd3f7e98dee2e8524b9b210716fbe1f07759b6d89307280a9b10ae359b43e"
)
AUDIT_TOOL_SHA256 = {
    "bwrap": "0abea81db798ebf6b4742ac0664802d97521547a353c2a0dbdc21d76cbbfd2c0",
    "ldd": "b6c9a28572ea3920442c2f5b1ea11b0999adc407913ffdd7f92e530dfc051894",
    "readelf": "c857339616bbbfa5eba32733e22365048903fbaf6ed2126b897dd138bcb741fc",
}
AUDIT_TOOL_PATHS = {
    "bwrap": Path("/usr/bin/bwrap"),
    "ldd": Path("/usr/bin/ldd"),
    "readelf": Path("/usr/bin/x86_64-linux-gnu-readelf"),
}
PRODUCER_CMAKE_SHA256 = (
    "576c050dab1e1418b6703b5cfb523330567683dad0c60a5ff9cc23128143812e"
)
PRODUCER_CMAKE_PAYLOAD = (
    ROOT
    / "envs/pypto-nvidia/lib/python3.14/site-packages/cmake/data/bin/cmake"
)
PRODUCER_CMAKE_WRAPPER_SHA256 = (
    "aadd40ffd6b8bc9dac19f6dadc7ee0800cdbb3cf72f5b1f1b8b24e37f61e97da"
)
PRODUCER_NINJA_SHA256 = (
    "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67"
)
PRODUCER_SITE_IDENTITY = {
    "bytes": 9485023,
    "directories": 134,
    "distributions": [
        "build",
        "lit",
        "packaging",
        "pyproject-hooks",
        "setuptools",
        "wheel",
    ],
    "files": 963,
    "sha256": "b0ccdda495e52c61a5fc3c05c87677dee95c435cc641c626f6cc343b0dc4a6f0",
    "symlinks": 0,
}
PRODUCER_IDENTITY_SCHEMA = 2
PRODUCER_IDENTITY_SCOPE = "workspace-prefix-and-selected-host-build-tools"


def producer_cmake_wrapper(payload: Path) -> bytes:
    return f'#!/bin/sh\nexec {payload} "$@"\n'.encode("ascii")


NVIDIA_TOOL_VERSIONS = {
    "ptxas": "12.8.93",
    "ptxas-blackwell": "13.1.80",
    "cuobjdump": "13.1.80",
    "nvdisasm": "13.1.80",
}

PRODUCER_PACKAGE_VERSIONS = {
    "build": "1.5.0",
    "cmake": "3.31.10",
    "lit": "18.1.8",
    "ninja": "1.13.0",
    "packaging": "26.2",
    "pyproject-hooks": "1.2.0",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}

DIST_INFO = f"triton-{TRITON_DISTRIBUTION_VERSION}.dist-info"
METADATA_PATH = f"{DIST_INFO}/METADATA"
WHEEL_PATH = f"{DIST_INFO}/WHEEL"
RECORD_PATH = f"{DIST_INFO}/RECORD"
MODULE_PATH = "triton/__init__.py"
FILECHECK_PATH = "triton/FileCheck"
NVIDIA_BIN_PREFIX = "triton/backends/nvidia/bin"
NVIDIA_LIB_PREFIX = "triton/backends/nvidia/lib"
LIBDEVICE_PATH = f"{NVIDIA_LIB_PREFIX}/libdevice.10.bc"
REQUIRED_HEADERS = (
    "triton/backends/nvidia/include/cuda.h",
    "triton/backends/nvidia/include/cupti.h",
)


class AuditError(RuntimeError):
    """An acceptance invariant was not proven."""


@dataclass(frozen=True, slots=True)
class AuditExpectations:
    distribution_version: str = TRITON_DISTRIBUTION_VERSION
    module_version: str = TRITON_MODULE_VERSION
    commit: str = TRITON_COMMIT
    tree: str = TRITON_TREE
    llvm_commit: str = TRITON_LLVM_COMMIT
    libdevice_sha256: str = LIBDEVICE_SHA256
    source_archive_sha256: str = SOURCE_ARCHIVE_SHA256
    nvidia_tool_versions: tuple[tuple[str, str], ...] = tuple(
        NVIDIA_TOOL_VERSIONS.items()
    )

    def tool_versions(self) -> dict[str, str]:
        return dict(self.nvidia_tool_versions)


@dataclass(frozen=True, slots=True)
class AuditLimits:
    max_wheel_bytes: int = 4 << 30
    max_members: int = 100_000
    max_member_bytes: int = 2 << 30
    max_expanded_bytes: int = 12 << 30
    max_compression_ratio: int = 2_000
    max_metadata_bytes: int = 1 << 20
    max_wheel_metadata_bytes: int = 1 << 20
    max_record_bytes: int = 64 << 20
    max_command_output_bytes: int = 16 << 20
    command_address_space_bytes: int = 4 << 30
    command_timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class AuditToolPaths:
    readelf: Path
    ldd: Path
    bwrap: Path


@dataclass(frozen=True, slots=True)
class AuditRequest:
    workspace: Path
    wheel: Path
    dependency_manifest: Path
    reviewed_dependency_manifest_sha256: str
    source_provenance: Path
    source_archive: Path
    source_input: Path
    build_input: Path
    built_source: Path
    producer_provenance: Path
    producer_site_identity: Path
    producer_site: Path
    producer_bin: Path
    expected_producer_identity_sha256: str
    evidence: Path
    temp_root: Path
    tools: AuditToolPaths


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def compact_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    counts = {"bytes": 0, "directories": 0, "files": 0, "symlinks": 0}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(mode.to_bytes(4, "little"))
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "little"))
            digest.update(target)
            counts["symlinks"] += 1
        elif path.is_dir():
            digest.update(b"D")
            counts["directories"] += 1
        elif path.is_file():
            size = path.stat().st_size
            digest.update(b"F")
            digest.update(size.to_bytes(8, "little"))
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    digest.update(chunk)
            counts["files"] += 1
            counts["bytes"] += size
        else:
            raise AuditError(f"unsupported source-tree entry: {path}")
    return {"sha256": digest.hexdigest(), **counts}


def verify_reference_tree_unchanged(reference: Path, candidate: Path) -> None:
    for source in sorted(reference.rglob("*")):
        relative = source.relative_to(reference)
        target = candidate / relative
        source_mode = stat.S_IMODE(source.lstat().st_mode)
        if source.is_symlink():
            if (
                not target.is_symlink()
                or stat.S_IMODE(target.lstat().st_mode) != source_mode
                or os.readlink(target) != os.readlink(source)
            ):
                raise AuditError(f"built source symlink drift: {relative}")
        elif source.is_dir():
            if (
                target.is_symlink()
                or not target.is_dir()
                or stat.S_IMODE(target.lstat().st_mode) != source_mode
            ):
                raise AuditError(f"built source directory drift: {relative}")
        elif source.is_file():
            if (
                target.is_symlink()
                or not target.is_file()
                or stat.S_IMODE(target.lstat().st_mode) != source_mode
                or target.stat().st_size != source.stat().st_size
                or sha256_file(target) != sha256_file(source)
            ):
                raise AuditError(f"built source file drift: {relative}")
        else:
            raise AuditError(f"unsupported reference source entry: {relative}")


def verify_reference_tree_exact(reference: Path, candidate: Path) -> None:
    """Require the post-build source to contain exactly the reviewed tree."""

    verify_reference_tree_unchanged(reference, candidate)
    if stat.S_IMODE(reference.stat().st_mode) != stat.S_IMODE(candidate.stat().st_mode):
        raise AuditError("built source root directory mode drift")
    reference_paths = {
        path.relative_to(reference).as_posix() for path in reference.rglob("*")
    }
    candidate_paths = {
        path.relative_to(candidate).as_posix() for path in candidate.rglob("*")
    }
    extras = sorted(candidate_paths - reference_paths)
    if extras:
        raise AuditError(f"built source has unapproved extra entries: {extras}")
    if tree_identity(reference) != tree_identity(candidate):
        raise AuditError("built source tree is not the exact reviewed build input")


def _tree_leaf_map(root: Path, *, prefix: str = "") -> dict[str, tuple[object, ...]]:
    result: dict[str, tuple[object, ...]] = {}
    for path in sorted(root.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(root).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            identity: tuple[object, ...] = ("symlink", mode, os.readlink(path))
        else:
            identity = ("file", mode, path.stat().st_size, sha256_file(path))
        if key in result:
            raise AuditError(f"duplicate derived build-input leaf: {key}")
        result[key] = identity
    return result


def verify_build_input_derivation(
    source_input: Path,
    nvidia_overlay: Path,
    build_input: Path,
    temp_root: Path,
) -> None:
    expected = _tree_leaf_map(source_input)
    overlay = _tree_leaf_map(
        nvidia_overlay,
        prefix="third_party/nvidia/backend",
    )
    for path, identity in overlay.items():
        if path in expected and expected[path] != identity:
            raise AuditError(f"reviewed NVIDIA overlay conflicts with source: {path}")
        expected[path] = identity
    observed = _tree_leaf_map(build_input)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(observed)
            if expected[path] != observed[path]
        )
        raise AuditError(
            "build input is not exact source plus reviewed NVIDIA overlay: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    with tempfile.TemporaryDirectory(
        prefix="triton-build-input-derivation-", dir=temp_root
    ) as temporary_directory:
        expected_root = Path(temporary_directory) / "expected"
        shutil.copytree(
            source_input,
            expected_root,
            symlinks=True,
            copy_function=os.link,
        )
        backend = expected_root / "third_party/nvidia/backend"
        shutil.copytree(
            nvidia_overlay,
            backend,
            dirs_exist_ok=True,
            symlinks=True,
            copy_function=os.link,
        )
        if tree_identity(expected_root) != tree_identity(build_input):
            raise AuditError(
                "build input directory/mode tree is not exact source plus "
                "reviewed NVIDIA overlay"
            )


def extract_source_archive_exact(archive: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise AuditError("source archive extraction destination is not fresh")
    destination.mkdir(mode=0o700)
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as source:
            members = []
            for member in source:
                name = member.name
                path = PurePosixPath(name)
                if (
                    not name
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.as_posix() != name
                    or name in seen
                ):
                    raise AuditError(f"source archive member is unsafe: {name!r}")
                seen.add(name)
                if len(seen) > 100_000:
                    raise AuditError("source archive exceeds member-count limit")
                if member.isdev() or member.isfifo():
                    raise AuditError(f"source archive has a special member: {name}")
                if not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                ):
                    raise AuditError(f"source archive member type is unsupported: {name}")
                if member.issym() or member.islnk():
                    target = PurePosixPath(member.linkname)
                    combined = (
                        target
                        if member.islnk()
                        else PurePosixPath(name).parent / target
                    )
                    depth = 0
                    for part in combined.parts:
                        if part in ("", "."):
                            continue
                        if part == "..":
                            depth -= 1
                            if depth < 0:
                                raise AuditError(
                                    f"source archive link escapes: {name}"
                                )
                        else:
                            depth += 1
                members.append(member)
            source.extractall(destination, members=members, filter="data")
    except (OSError, tarfile.TarError) as error:
        raise AuditError("source archive cannot be safely extracted") from error
    resolved_root = destination.resolve(strict=True)
    for path in destination.rglob("*"):
        if not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise AuditError(f"extracted source symlink escapes: {path}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_regular_file_once(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AuditError(f"cannot open {description} as a non-symlink: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise AuditError(f"{description} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def load_canonical_json(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_file_once(path, description)
    try:
        text = raw.decode("utf-8", errors="strict")
        document = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{description} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(document, dict):
        raise AuditError(f"{description} root must be an object")
    if text != canonical_json(document):
        raise AuditError(f"{description} is not canonical JSON")
    return document, raw


def validate_sha256(value: str, description: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise AuditError(f"{description} must be 64 lowercase hexadecimal characters")
    return value


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    return (allow_root and path == root) or root in path.parents


def require_workspace(workspace: Path) -> Path:
    if not workspace.is_absolute():
        raise AuditError("--workspace must be absolute")
    lexical = _absolute_lexical(workspace)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"workspace is absent: {lexical}") from error
    if lexical != resolved or not resolved.is_dir():
        raise AuditError("workspace must be a real, non-symlink directory")
    return resolved


def require_workspace_input(path: Path, workspace: Path, description: str) -> Path:
    if not path.is_absolute():
        raise AuditError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise AuditError(f"{description} must be a real workspace-owned path")
    if not resolved.is_file() or resolved.is_symlink():
        raise AuditError(f"{description} must be a regular non-symlink file")
    return resolved


def require_workspace_directory(path: Path, workspace: Path, description: str) -> Path:
    if not path.is_absolute():
        raise AuditError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise AuditError(f"{description} must be a real workspace-owned directory")
    if not resolved.is_dir() or resolved.is_symlink():
        raise AuditError(f"{description} must be a real directory")
    return resolved


def require_evidence_output(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise AuditError("evidence path must be absolute")
    lexical = _absolute_lexical(path)
    parent = lexical.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"evidence parent is absent: {parent}") from error
    if parent != resolved_parent or not _is_within(resolved_parent, workspace):
        raise AuditError("evidence parent must be a real workspace-owned directory")
    if not resolved_parent.is_dir():
        raise AuditError("evidence parent is not a directory")
    if lexical.exists() or lexical.is_symlink():
        raise AuditError(f"evidence already exists: {lexical}")
    return lexical


def require_absolute_tool(path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise AuditError(f"{description} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"{description} is absent: {path}") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AuditError(f"{description} must be an executable regular file")
    return resolved


def _safe_member_name(name: str, description: str = "ZIP member") -> PurePosixPath:
    if not name or len(name.encode("utf-8")) > 4096:
        raise AuditError(f"{description} name is empty or too long: {name!r}")
    if "\x00" in name or "\\" in name or unicodedata.normalize("NFC", name) != name:
        raise AuditError(f"{description} name is unsafe: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise AuditError(f"{description} name contains a control character: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.endswith("/")
        or path.as_posix() != name
        or any(part in ("", ".", "..") for part in path.parts)
        or (path.parts and re.match(r"^[A-Za-z]:", path.parts[0]))
    ):
        raise AuditError(f"{description} path is unsafe or non-canonical: {name!r}")
    return path


def _unix_mode(info: zipfile.ZipInfo) -> int:
    if info.create_system == 3:
        return (info.external_attr >> 16) & 0xFFFF
    return 0


def _validate_zip_info(info: zipfile.ZipInfo, limits: AuditLimits) -> dict[str, object]:
    path = _safe_member_name(info.filename)
    if info.flag_bits & 0x1:
        raise AuditError(f"encrypted ZIP member is forbidden: {info.filename}")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise AuditError(f"unsupported ZIP compression for {info.filename}")
    if info.file_size < 0 or info.file_size > limits.max_member_bytes:
        raise AuditError(f"ZIP member exceeds size limit: {info.filename}")
    if info.compress_size < 0:
        raise AuditError(f"negative compressed size: {info.filename}")
    if (
        info.file_size > 0
        and info.compress_size == 0
        or info.compress_size > 0
        and info.file_size > info.compress_size * limits.max_compression_ratio
    ):
        raise AuditError(f"ZIP member exceeds compression-ratio limit: {info.filename}")
    mode = _unix_mode(info)
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise AuditError(f"archive symlink is forbidden: {info.filename}")
    if file_type not in (0, stat.S_IFREG):
        raise AuditError(f"special archive member is forbidden: {info.filename}")
    if info.is_dir():
        raise AuditError(f"explicit directory ZIP member is forbidden: {info.filename}")
    return {
        "path": path.as_posix(),
        "compressed_bytes": info.compress_size,
        "compression": "stored" if info.compress_type == zipfile.ZIP_STORED else "deflated",
        "crc32": f"{info.CRC:08x}",
        "mode": f"{stat.S_IMODE(mode or 0o644):04o}",
        "size": info.file_size,
    }


def _hash_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with zf.open(info, "r") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                total += len(chunk)
                digest.update(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise AuditError(f"cannot read ZIP member exactly: {info.filename}") from error
    if total != info.file_size:
        raise AuditError(f"ZIP member size changed while reading: {info.filename}")
    return digest.hexdigest(), total


def _read_small_zip_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    maximum: int,
    description: str,
) -> bytes:
    if info.file_size > maximum:
        raise AuditError(f"{description} exceeds its size limit")
    try:
        value = zf.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise AuditError(f"cannot read {description}") from error
    if len(value) != info.file_size:
        raise AuditError(f"{description} has an inconsistent size")
    return value


def _single_header(message: object, name: str, description: str) -> str:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise AuditError(f"{description} must contain exactly one {name} header")
    return values[0].strip()


def _parse_filename_tags(filename: str, expected_version: str) -> set[str]:
    if not filename.endswith(".whl"):
        raise AuditError("artifact does not have a .whl filename")
    fields = filename[:-4].split("-")
    if len(fields) not in (5, 6):
        raise AuditError("wheel filename does not have five or six fields")
    distribution, version = fields[0], fields[1]
    if re.sub(r"[-_.]+", "-", distribution).lower() != "triton":
        raise AuditError("wheel filename distribution is not triton")
    if version != expected_version:
        raise AuditError("wheel filename version mismatch")
    if len(fields) == 6 and not re.fullmatch(r"[0-9][A-Za-z0-9_.]*", fields[2]):
        raise AuditError("wheel filename build tag is malformed")
    python_tag, abi_tag, platform_tag = fields[-3:]
    tags = {
        f"{python_value}-{abi_value}-{platform_value}"
        for python_value in python_tag.split(".")
        for abi_value in abi_tag.split(".")
        for platform_value in platform_tag.split(".")
    }
    if not tags:
        raise AuditError("wheel filename has no compatibility tag")
    for tag in tags:
        _validate_cp314_linux_tag(tag)
    return tags


def _validate_cp314_linux_tag(tag: str) -> None:
    if tag != "cp314-cp314-linux_x86_64":
        raise AuditError(
            f"wheel tag is not exact cp314/cp314/linux_x86_64: {tag!r}"
        )


def _parse_metadata(raw: bytes, expected_version: str) -> dict[str, object]:
    try:
        message = BytesParser(policy=policy.compat32).parsebytes(raw)
    except Exception as error:
        raise AuditError("METADATA cannot be parsed") from error
    metadata_version = _single_header(message, "Metadata-Version", "METADATA")
    name = _single_header(message, "Name", "METADATA")
    version = _single_header(message, "Version", "METADATA")
    if re.sub(r"[-_.]+", "-", name).lower() != "triton":
        raise AuditError("METADATA Name is not triton")
    if version != expected_version:
        raise AuditError("METADATA Version mismatch")
    return {
        "metadata_version": metadata_version,
        "name": name,
        "version": version,
    }


def _parse_wheel_metadata(raw: bytes, filename_tags: set[str]) -> dict[str, object]:
    try:
        message = BytesParser(policy=policy.compat32).parsebytes(raw)
    except Exception as error:
        raise AuditError("WHEEL metadata cannot be parsed") from error
    wheel_version = _single_header(message, "Wheel-Version", "WHEEL")
    purelib = _single_header(message, "Root-Is-Purelib", "WHEEL")
    if wheel_version != "1.0":
        raise AuditError("unsupported Wheel-Version")
    if purelib.lower() != "false":
        raise AuditError("Triton wheel must not be marked purelib")
    tag_values = message.get_all("Tag", [])
    if not tag_values or any(not isinstance(value, str) for value in tag_values):
        raise AuditError("WHEEL must contain compatibility tags")
    tags = {value.strip() for value in tag_values}
    if len(tags) != len(tag_values):
        raise AuditError("WHEEL compatibility tags are duplicated")
    for tag in tags:
        _validate_cp314_linux_tag(tag)
    if tags != filename_tags:
        raise AuditError("WHEEL and filename compatibility tags differ")
    generator_values = message.get_all("Generator", [])
    if len(generator_values) > 1:
        raise AuditError("WHEEL contains duplicate Generator headers")
    return {
        "generator": generator_values[0].strip() if generator_values else None,
        "root_is_purelib": False,
        "tags": sorted(tags),
        "wheel_version": wheel_version,
    }


def _record_digest(encoded: str, path: str) -> str:
    if not encoded.startswith("sha256="):
        raise AuditError(f"RECORD uses a non-SHA256 or absent digest: {path}")
    value = encoded.removeprefix("sha256=")
    if not value or "=" in value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise AuditError(f"RECORD SHA256 encoding is not canonical: {path}")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as error:
        raise AuditError(f"RECORD SHA256 cannot be decoded: {path}") from error
    if len(decoded) != 32:
        raise AuditError(f"RECORD SHA256 has the wrong length: {path}")
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if value != canonical:
        raise AuditError(f"RECORD SHA256 encoding is not canonical: {path}")
    return decoded.hex()


def _parse_record(
    raw: bytes,
    archive_identities: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    try:
        text = raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise AuditError("RECORD is not strict UTF-8 CSV") from error
    if not rows:
        raise AuditError("RECORD is empty")
    by_path: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise AuditError("every RECORD row must have exactly three fields")
        path = _safe_member_name(row[0], "RECORD")
        path_text = path.as_posix()
        if path_text in by_path:
            raise AuditError(f"duplicate RECORD path: {path_text}")
        by_path[path_text] = (row[1], row[2])
    if set(by_path) != set(archive_identities):
        missing = sorted(set(archive_identities) - set(by_path))
        extra = sorted(set(by_path) - set(archive_identities))
        raise AuditError(f"RECORD/archive membership mismatch: missing={missing}, extra={extra}")
    entries: list[dict[str, object]] = []
    for path in sorted(by_path):
        encoded_digest, encoded_size = by_path[path]
        actual = archive_identities[path]
        if path == RECORD_PATH:
            if encoded_digest or encoded_size:
                raise AuditError("RECORD's own row must have empty digest and size")
        else:
            expected_digest = _record_digest(encoded_digest, path)
            if expected_digest != actual["sha256"]:
                raise AuditError(f"RECORD SHA256 mismatch: {path}")
            if not re.fullmatch(r"0|[1-9][0-9]*", encoded_size):
                raise AuditError(f"RECORD size is not canonical decimal: {path}")
            if int(encoded_size) != actual["size"]:
                raise AuditError(f"RECORD size mismatch: {path}")
        entries.append(
            {
                "path": path,
                "sha256": actual["sha256"],
                "size": actual["size"],
            }
        )
    return entries


def _parse_module_version(raw: bytes, expected: str) -> str:
    try:
        source = raw.decode("utf-8", errors="strict")
        tree = ast.parse(source, filename=MODULE_PATH)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise AuditError("triton/__init__.py is not valid Python source") from error
    values: list[object] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            values.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__version__"
        ):
            values.append(statement.value)
    if len(values) != 1 or not isinstance(values[0], ast.Constant):
        raise AuditError("module must contain one static top-level __version__ assignment")
    value = values[0].value
    if not isinstance(value, str) or value != expected:
        raise AuditError("Triton module version mismatch")
    return value


def _require_unique_exact_basename(
    names: set[str], basename: str, expected_path: str
) -> None:
    matches = sorted(name for name in names if PurePosixPath(name).name == basename)
    if matches != [expected_path]:
        raise AuditError(
            f"wheel-owned {basename} must occur only at {expected_path}: {matches}"
        )


def inspect_wheel_static(
    zf: zipfile.ZipFile,
    wheel_name: str,
    expectations: AuditExpectations,
    limits: AuditLimits,
) -> tuple[dict[str, object], list[zipfile.ZipInfo]]:
    try:
        infos = zf.infolist()
    except zipfile.BadZipFile as error:
        raise AuditError("wheel central directory is invalid") from error
    if not infos or len(infos) > limits.max_members:
        raise AuditError("wheel member count is empty or exceeds the limit")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise AuditError("wheel contains duplicate ZIP members")
    if len(names) != len({unicodedata.normalize("NFC", name) for name in names}):
        raise AuditError("wheel contains Unicode-normalization-colliding members")

    total_expanded = 0
    archive_identities: dict[str, dict[str, object]] = {}
    info_by_name: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        identity = _validate_zip_info(info, limits)
        total_expanded += info.file_size
        if total_expanded > limits.max_expanded_bytes:
            raise AuditError("wheel expanded byte count exceeds the limit")
        digest, actual_size = _hash_zip_member(zf, info)
        identity["sha256"] = digest
        identity["size"] = actual_size
        archive_identities[info.filename] = identity
        info_by_name[info.filename] = info

    expected_paths = {METADATA_PATH, WHEEL_PATH, RECORD_PATH, MODULE_PATH}
    if not expected_paths.issubset(info_by_name):
        raise AuditError(f"wheel lacks required metadata/module paths: {sorted(expected_paths - set(info_by_name))}")
    dist_info_roots = {
        PurePosixPath(name).parts[0]
        for name in names
        if PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if dist_info_roots != {DIST_INFO}:
        raise AuditError(f"wheel has an unexpected dist-info root: {sorted(dist_info_roots)}")
    for name in names:
        lower = name.lower()
        basename = PurePosixPath(name).name.lower()
        if basename.endswith(".pth") or "__editable__" in lower or "editable_finder" in lower:
            raise AuditError(f"editable import artifact is forbidden: {name}")

    filename_tags = _parse_filename_tags(wheel_name, expectations.distribution_version)
    metadata_raw = _read_small_zip_member(
        zf, info_by_name[METADATA_PATH], limits.max_metadata_bytes, "METADATA"
    )
    wheel_raw = _read_small_zip_member(
        zf, info_by_name[WHEEL_PATH], limits.max_wheel_metadata_bytes, "WHEEL"
    )
    record_raw = _read_small_zip_member(
        zf, info_by_name[RECORD_PATH], limits.max_record_bytes, "RECORD"
    )
    module_raw = _read_small_zip_member(
        zf, info_by_name[MODULE_PATH], limits.max_metadata_bytes, MODULE_PATH
    )
    metadata = _parse_metadata(metadata_raw, expectations.distribution_version)
    wheel_metadata = _parse_wheel_metadata(wheel_raw, filename_tags)
    record_entries = _parse_record(record_raw, archive_identities)
    module_version = _parse_module_version(module_raw, expectations.module_version)

    required_tools = {FILECHECK_PATH: None}
    for tool, version in expectations.tool_versions().items():
        required_tools[f"{NVIDIA_BIN_PREFIX}/{tool}"] = version
    name_set = set(names)
    _require_unique_exact_basename(name_set, "FileCheck", FILECHECK_PATH)
    for tool in expectations.tool_versions():
        _require_unique_exact_basename(
            name_set, tool, f"{NVIDIA_BIN_PREFIX}/{tool}"
        )
    _require_unique_exact_basename(name_set, "libdevice.10.bc", LIBDEVICE_PATH)
    for path in (*required_tools, LIBDEVICE_PATH, *REQUIRED_HEADERS):
        if path not in info_by_name:
            raise AuditError(f"required wheel resource is absent: {path}")
    for path in required_tools:
        mode = _unix_mode(info_by_name[path])
        if not mode & 0o111:
            raise AuditError(f"wheel-owned tool is not executable: {path}")
        with zf.open(info_by_name[path], "r") as source:
            if source.read(4) != b"\x7fELF":
                raise AuditError(f"wheel-owned tool is not ELF: {path}")
    if archive_identities[LIBDEVICE_PATH]["sha256"] != expectations.libdevice_sha256:
        raise AuditError("libdevice SHA256 mismatch")
    cupti_libraries = sorted(
        name
        for name in names
        if name.startswith(f"{NVIDIA_LIB_PREFIX}/cupti/")
        and (
            PurePosixPath(name).name == "libcupti.so"
            or PurePosixPath(name).name.startswith("libcupti.so.")
        )
    )
    if not cupti_libraries:
        raise AuditError("wheel does not own a CUPTI shared library")

    return (
        {
            "archive": {
                "expanded_bytes": total_expanded,
                "members": [archive_identities[name] for name in sorted(names)],
                "members_count": len(names),
            },
            "distribution_metadata": metadata,
            "module_version": module_version,
            "record": {
                "entries": record_entries,
                "entries_count": len(record_entries),
                "path": RECORD_PATH,
                "sha256": archive_identities[RECORD_PATH]["sha256"],
                "size": archive_identities[RECORD_PATH]["size"],
            },
            "required_resources": {
                "cupti_libraries": cupti_libraries,
                "filecheck": FILECHECK_PATH,
                "headers": list(REQUIRED_HEADERS),
                "libdevice": {
                    "path": LIBDEVICE_PATH,
                    "sha256": archive_identities[LIBDEVICE_PATH]["sha256"],
                    "size": archive_identities[LIBDEVICE_PATH]["size"],
                },
                "nvidia_tools": {
                    tool: {
                        "expected_version": version,
                        "path": f"{NVIDIA_BIN_PREFIX}/{tool}",
                        "sha256": archive_identities[f"{NVIDIA_BIN_PREFIX}/{tool}"]["sha256"],
                        "size": archive_identities[f"{NVIDIA_BIN_PREFIX}/{tool}"]["size"],
                    }
                    for tool, version in expectations.tool_versions().items()
                },
            },
            "wheel_metadata": wheel_metadata,
        },
        infos,
    )


def extract_fresh(
    zf: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    destination: Path,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise AuditError(f"fresh extraction destination already exists: {destination}")
    destination.mkdir(mode=0o700)
    root = destination.resolve(strict=True)
    for info in sorted(infos, key=lambda item: item.filename):
        relative = _safe_member_name(info.filename)
        target = destination.joinpath(*relative.parts)
        lexical = _absolute_lexical(target)
        if not _is_within(lexical, root):
            raise AuditError(f"ZIP member escapes fresh extraction: {info.filename}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise AuditError(f"fresh extraction collision: {info.filename}")
        try:
            with zf.open(info, "r") as source, target.open("xb") as output:
                total = 0
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    total += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise AuditError(f"fresh extraction failed: {info.filename}") from error
        if total != info.file_size:
            raise AuditError(f"fresh extraction size mismatch: {info.filename}")
        mode = _unix_mode(info)
        os.chmod(target, 0o555 if mode & 0o111 else 0o444)
        file_stat = target.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise AuditError(f"fresh extraction produced a non-regular file: {info.filename}")
    # Directories remain owner-writable so TemporaryDirectory can clean this
    # disposable tree.  The probe sees the whole tree through a read-only
    # bubblewrap bind, and every extracted leaf itself is non-writable.


def _subprocess_limits(limits: AuditLimits) -> None:
    requested = {
        resource.RLIMIT_AS: limits.command_address_space_bytes,
        resource.RLIMIT_CORE: 0,
        resource.RLIMIT_CPU: max(1, limits.command_timeout_seconds),
        resource.RLIMIT_FSIZE: limits.max_command_output_bytes,
        resource.RLIMIT_NOFILE: 128,
        resource.RLIMIT_NPROC: 512,
    }
    for kind, value in requested.items():
        _, hard = resource.getrlimit(kind)
        bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (bounded, bounded))


def run_limited_command(
    argv: list[str],
    *,
    temp_root: Path,
    limits: AuditLimits,
    description: str,
) -> str:
    if not argv or not Path(argv[0]).is_absolute():
        raise AuditError(f"{description} does not use an absolute executable")
    environment = {
        "HOME": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        with tempfile.TemporaryFile(dir=temp_root) as output:
            result = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=limits.command_timeout_seconds,
                preexec_fn=lambda: _subprocess_limits(limits),
                start_new_session=True,
            )
            output.seek(0)
            raw = output.read(limits.max_command_output_bytes + 1)
    except subprocess.TimeoutExpired as error:
        raise AuditError(f"{description} timed out") from error
    except OSError as error:
        raise AuditError(f"{description} could not execute") from error
    if len(raw) > limits.max_command_output_bytes:
        raise AuditError(f"{description} output exceeded the limit")
    text = raw.decode("utf-8", errors="replace")
    if result.returncode != 0:
        excerpt = text[-1000:].replace("\n", " ")
        raise AuditError(f"{description} returned {result.returncode}: {excerpt}")
    if not text.strip():
        raise AuditError(f"{description} returned no output")
    return text


FORBIDDEN_NATIVE_RE = re.compile(
    r"(?:amdhip|libhip|libhsa|hsa-runtime|/opt/rocm|(?:^|[/_-])rocm(?:[/_.-]|$)|"
    r"gemsim|gem5|amd_comgr|libamd(?!64)|roctracer|rocprofiler)",
    flags=re.IGNORECASE,
)


def _check_native_text(value: str, description: str) -> None:
    if FORBIDDEN_NATIVE_RE.search(value):
        raise AuditError(f"forbidden AMD/HIP/HSA/ROCm/GemSim reference in {description}")


def _normalize_command_output(value: str, extraction_root: Path) -> str:
    normalized = value.replace(str(extraction_root), "$WHEEL_ROOT")
    normalized = re.sub(r"\(0x[0-9a-fA-F]+\)", "(<address>)", normalized)
    return normalized.rstrip("\n") + "\n"


def _parse_readelf(value: str, relative_path: str) -> dict[str, object]:
    fields: dict[str, str] = {}
    for name in ("Class", "Data", "Type", "Machine"):
        match = re.search(rf"^\s*{name}:\s*(.+?)\s*$", value, flags=re.MULTILINE)
        if match is None:
            raise AuditError(f"readelf lacks {name} for {relative_path}")
        fields[name.lower()] = match.group(1)
    build_ids = sorted(set(re.findall(r"Build ID:\s*([0-9A-Fa-f]+)", value)))
    if not build_ids:
        raise AuditError(f"readelf lacks a Build ID for {relative_path}")
    if any(not re.fullmatch(r"[0-9A-Fa-f]{8,}", build_id) for build_id in build_ids):
        raise AuditError(f"readelf returned a malformed Build ID for {relative_path}")
    needed = sorted(set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", value)))
    rpath = sorted(set(re.findall(r"\(RPATH\).*?\[([^]]*)\]", value)))
    runpath = sorted(set(re.findall(r"\(RUNPATH\).*?\[([^]]*)\]", value)))
    for item in (*needed, *rpath, *runpath):
        _check_native_text(item, f"readelf data for {relative_path}")
    return {
        "build_ids": [value.lower() for value in build_ids],
        "class": fields["class"],
        "data": fields["data"],
        "dt_needed": needed,
        "machine": fields["machine"],
        "rpath": rpath,
        "runpath": runpath,
        "type": fields["type"],
    }


def audit_elf_files(
    extraction_root: Path,
    tools: AuditToolPaths,
    temp_root: Path,
    limits: AuditLimits,
) -> list[dict[str, object]]:
    elf_paths: list[Path] = []
    for path in sorted(extraction_root.rglob("*")):
        if path.is_symlink():
            raise AuditError(f"fresh extraction contains a symlink: {path}")
        if not path.is_file():
            continue
        with path.open("rb") as source:
            if source.read(4) == b"\x7fELF":
                elf_paths.append(path)
    if not elf_paths:
        raise AuditError("wheel contains no ELF-magic native files")

    records: list[dict[str, object]] = []
    for path in elf_paths:
        relative = path.relative_to(extraction_root).as_posix()
        readelf_raw = run_limited_command(
            [str(tools.readelf), "-W", "-h", "-d", "-n", str(path)],
            temp_root=temp_root,
            limits=limits,
            description=f"readelf {relative}",
        )
        ldd_raw = run_limited_command(
            _bubblewrap_ldd_argv(
                tools.bwrap,
                tools.ldd,
                extraction_root,
                relative,
            ),
            temp_root=temp_root,
            limits=limits,
            description=f"ldd {relative}",
        )
        if re.search(r"\bnot found\b", ldd_raw, flags=re.IGNORECASE):
            raise AuditError(f"ldd has an unresolved dependency for {relative}")
        _check_native_text(ldd_raw, f"ldd output for {relative}")
        parsed = _parse_readelf(readelf_raw, relative)
        if parsed["rpath"] or parsed["runpath"]:
            raise AuditError(f"native ELF contains RPATH/RUNPATH: {relative}")
        for resolved_dependency in re.findall(
            r"(?:=>\s+)?(/[^\s(]+)", ldd_raw
        ):
            if not resolved_dependency.startswith(
                ("/wheel/", "/usr/lib/", "/lib/", "/lib64/")
            ):
                raise AuditError(
                    f"ldd resolved an external path for {relative}: "
                    f"{resolved_dependency}"
                )
        if (
            parsed["class"] != "ELF64"
            or parsed["machine"]
            not in ("X86-64", "Advanced Micro Devices X86-64")
            or "little endian" not in str(parsed["data"]).lower()
            or not str(parsed["type"]).startswith(("DYN", "EXEC"))
        ):
            raise AuditError(f"native ELF ABI mismatch for {relative}")
        normalized_readelf = _normalize_command_output(readelf_raw, extraction_root)
        normalized_ldd = _normalize_command_output(
            ldd_raw.replace("/wheel", str(extraction_root)), extraction_root
        )
        records.append(
            {
                "build_ids": parsed["build_ids"],
                "class": parsed["class"],
                "data": parsed["data"],
                "dt_needed": parsed["dt_needed"],
                "ldd": {
                    "output": normalized_ldd,
                    "output_sha256": sha256_bytes(normalized_ldd.encode("utf-8")),
                    "status": "resolved",
                },
                "machine": parsed["machine"],
                "path": relative,
                "readelf": {
                    "output": normalized_readelf,
                    "output_sha256": sha256_bytes(normalized_readelf.encode("utf-8")),
                },
                "rpath": parsed["rpath"],
                "runpath": parsed["runpath"],
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "type": parsed["type"],
            }
        )
    return records


def _bubblewrap_probe_argv(
    bwrap: Path, extraction_root: Path, relative_tool: str
) -> list[str]:
    argv = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(extraction_root),
        "/wheel",
        "--chdir",
        "/wheel",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        f"/wheel/{relative_tool}",
        "--version",
    ]
    return argv


def _bubblewrap_ldd_argv(
    bwrap: Path,
    ldd: Path,
    extraction_root: Path,
    relative_elf: str,
) -> list[str]:
    relative = _safe_member_name(relative_elf, "ELF path")
    return [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(extraction_root),
        "/wheel",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        str(ldd),
        f"/wheel/{relative.as_posix()}",
    ]


def probe_wheel_tools(
    extraction_root: Path,
    tools: AuditToolPaths,
    expectations: AuditExpectations,
    temp_root: Path,
    limits: AuditLimits,
) -> dict[str, object]:
    probes: dict[str, object] = {}
    expected = {"FileCheck": None, **expectations.tool_versions()}
    for name, expected_version in expected.items():
        relative = FILECHECK_PATH if name == "FileCheck" else f"{NVIDIA_BIN_PREFIX}/{name}"
        output = run_limited_command(
            _bubblewrap_probe_argv(tools.bwrap, extraction_root, relative),
            temp_root=temp_root,
            limits=limits,
            description=f"sandboxed {name} --version",
        )
        if name == "FileCheck":
            match = re.search(r"\bLLVM version\s+([^\s]+)", output)
            if match is None:
                raise AuditError("FileCheck did not report an LLVM version")
            observed_version = match.group(1)
        else:
            assert expected_version is not None
            release = ".".join(expected_version.split(".")[:2])
            version_match = re.search(
                rf"(?<![0-9.])V?{re.escape(expected_version)}(?![0-9.])", output
            )
            release_match = re.search(
                rf"\brelease\s+{re.escape(release)}(?:\s|,|$)",
                output,
                flags=re.IGNORECASE,
            )
            if version_match is None or release_match is None:
                raise AuditError(f"sandboxed {name} version mismatch")
            observed_version = expected_version
        normalized = _normalize_command_output(output, extraction_root)
        path = extraction_root.joinpath(*PurePosixPath(relative).parts)
        probes[name] = {
            "expected_version": expected_version,
            "observed_version": observed_version,
            "output": normalized,
            "output_sha256": sha256_bytes(normalized.encode("utf-8")),
            "path": relative,
            "sandbox": {
                "devices": "private-minimal-dev",
                "filesystem": "wheel-and-host-runtime-read-only",
                "network": "unshared",
            },
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return probes


def _validate_dependency_manifest(
    document: dict[str, object],
    raw: bytes,
    expected_digest: str,
    expectations: AuditExpectations,
) -> dict[str, object]:
    validate_sha256(expected_digest, "reviewed dependency manifest SHA256")
    actual_digest = sha256_bytes(raw)
    if actual_digest != expected_digest:
        raise AuditError("dependency manifest differs from its reviewed SHA256 anchor")
    required = {
        "schema_version": 1,
        "status": "materialized-unreviewed",
        "triton_commit": expectations.commit,
        "triton_tree": expectations.tree,
        "triton_llvm_commit": expectations.llvm_commit,
    }
    for field, expected in required.items():
        if document.get(field) != expected:
            raise AuditError(f"dependency manifest {field} mismatch")
    packages = document.get("packages")
    if not isinstance(packages, list) or not packages:
        raise AuditError("dependency manifest packages are absent")
    names: set[str] = set()
    for record in packages:
        if not isinstance(record, dict):
            raise AuditError("dependency manifest package record is not an object")
        name = record.get("name")
        digest = record.get("archive_sha256")
        size = record.get("archive_bytes")
        expanded = record.get("expanded_tree")
        if not isinstance(name, str) or not name or name in names:
            raise AuditError("dependency manifest package names are invalid/duplicated")
        names.add(name)
        if not isinstance(digest, str):
            raise AuditError(f"dependency archive SHA256 is absent: {name}")
        validate_sha256(digest, f"dependency archive SHA256 for {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise AuditError(f"dependency archive size is invalid: {name}")
        if not isinstance(expanded, dict) or not isinstance(expanded.get("sha256"), str):
            raise AuditError(f"dependency expanded-tree identity is absent: {name}")
        validate_sha256(expanded["sha256"], f"dependency expanded-tree SHA256 for {name}")
    build_inputs = document.get("build_inputs")
    if not isinstance(build_inputs, dict):
        raise AuditError("dependency manifest build_inputs are absent")
    overlay_path = build_inputs.get("nvidia_backend_overlay")
    overlay_tree = build_inputs.get("nvidia_backend_overlay_tree")
    if overlay_path != "nvidia-backend-overlay" or not isinstance(
        overlay_tree, dict
    ):
        raise AuditError("dependency NVIDIA overlay anchor is invalid")
    if not isinstance(overlay_tree.get("sha256"), str):
        raise AuditError("dependency NVIDIA overlay tree SHA256 is absent")
    validate_sha256(
        overlay_tree["sha256"], "dependency NVIDIA overlay tree SHA256"
    )
    return {
        "document": document,
        "reviewed_manifest_sha256": expected_digest,
        "sha256": actual_digest,
        "size": len(raw),
    }


def _validate_dependency_review(
    document: dict[str, object],
    raw: bytes,
    dependency_manifest: dict[str, object],
    expected_manifest_sha256: str,
    expectations: AuditExpectations,
) -> dict[str, object]:
    packages = dependency_manifest["packages"]
    assert isinstance(packages, list)
    expected = {
        "archives": [
            {
                "name": record["name"],
                "sha256": record["archive_sha256"],
            }
            for record in packages
        ],
        "manifest_sha256": expected_manifest_sha256,
        "schema_version": 1,
        "status": "reviewed",
        "triton_commit": expectations.commit,
        "triton_llvm_commit": expectations.llvm_commit,
        "triton_tree": expectations.tree,
    }
    if document != expected:
        raise AuditError("dependency review record differs from manifest/source anchor")
    return {
        "document": document,
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def _validate_source_provenance(
    document: dict[str, object],
    raw: bytes,
    expectations: AuditExpectations,
    *,
    source_archive: Path,
    source_input: Path,
    nvidia_overlay: Path,
    build_input: Path,
    built_source: Path,
    temp_root: Path,
) -> dict[str, object]:
    required = {
        "schema_version": 1,
        "kind": "triton-git-archive",
        "repository": TRITON_REPOSITORY,
        "commit": expectations.commit,
        "tree": expectations.tree,
        "module_version": expectations.module_version,
    }
    for field, expected in required.items():
        if document.get(field) != expected:
            raise AuditError(f"source provenance {field} mismatch")
    for field in (
        "archive_sha256",
        "extracted_tree_sha256",
        "build_input_tree_sha256",
    ):
        value = document.get(field)
        if not isinstance(value, str):
            raise AuditError(f"source provenance {field} is absent")
        validate_sha256(value, f"source provenance {field}")
    archive_sha256 = sha256_file(source_archive)
    if (
        archive_sha256 != expectations.source_archive_sha256
        or document["archive_sha256"] != archive_sha256
    ):
        raise AuditError("source archive differs from its frozen Git-archive anchor")
    source_input_identity = tree_identity(source_input)
    build_input_identity = tree_identity(build_input)
    if document["extracted_tree_sha256"] != source_input_identity["sha256"]:
        raise AuditError("source-input tree differs from source provenance")
    if document["build_input_tree_sha256"] != build_input_identity["sha256"]:
        raise AuditError("build-input tree differs from source provenance")
    with tempfile.TemporaryDirectory(
        prefix="triton-source-archive-audit-", dir=temp_root
    ) as temporary_directory:
        archive_extract = Path(temporary_directory) / "source"
        extract_source_archive_exact(source_archive, archive_extract)
        if tree_identity(archive_extract) != source_input_identity:
            raise AuditError(
                "source-input tree is not the exact frozen Git archive extraction"
            )
    verify_reference_tree_unchanged(source_input, build_input)
    verify_build_input_derivation(
        source_input, nvidia_overlay, build_input, temp_root
    )
    verify_reference_tree_exact(build_input, built_source)
    built_source_identity = tree_identity(built_source)
    return {
        "archive": {
            "path": source_archive.name,
            "sha256": archive_sha256,
            "size": source_archive.stat().st_size,
        },
        "build_input_tree": build_input_identity,
        "built_source_tree": built_source_identity,
        "document": document,
        "sha256": sha256_bytes(raw),
        "size": len(raw),
        "source_input_tree": source_input_identity,
    }


def _validate_producer_provenance(
    document: dict[str, object], raw: bytes, expected_identity: str
) -> dict[str, object]:
    validate_sha256(expected_identity, "expected producer identity SHA256")
    if document.get("identity_schema") != PRODUCER_IDENTITY_SCHEMA:
        raise AuditError("producer identity schema mismatch")
    if document.get("identity_scope") != PRODUCER_IDENTITY_SCOPE:
        raise AuditError("producer identity scope mismatch")
    if document.get("python_version") != "3.14.6":
        raise AuditError("producer Python version mismatch")
    if document.get("package_versions") != PRODUCER_PACKAGE_VERSIONS:
        raise AuditError("producer package versions mismatch")
    required_fields = {
        "distribution_set_sha256",
        "dynamic_libraries",
        "executables",
        "identity_schema",
        "identity_scope",
        "package_distributions",
        "package_versions",
        "python_version",
        "record_rewrite_count",
        "record_rewrite_policy_sha256",
        "selected_producer_identity_sha256",
    }
    if set(document) != required_fields:
        raise AuditError("producer provenance field set is incomplete")
    if (
        document.get("record_rewrite_count") != 6
        or not isinstance(document.get("package_distributions"), dict)
        or set(document["package_distributions"]) != set(PRODUCER_PACKAGE_VERSIONS)
        or not isinstance(document.get("executables"), list)
        or not document["executables"]
        or not isinstance(document.get("dynamic_libraries"), list)
        or not document["dynamic_libraries"]
    ):
        raise AuditError("producer provenance closure is incomplete")
    for collection_name in ("executables", "dynamic_libraries"):
        for record in document[collection_name]:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256")))
                is None
            ):
                raise AuditError(f"producer {collection_name} record is malformed")
    observed_identity = document.get("selected_producer_identity_sha256")
    if not isinstance(observed_identity, str):
        raise AuditError("producer selected identity SHA256 is absent")
    validate_sha256(observed_identity, "producer selected identity SHA256")
    unsigned = dict(document)
    del unsigned["selected_producer_identity_sha256"]
    if compact_json_sha256(unsigned) != observed_identity:
        raise AuditError("producer self-identity SHA256 mismatch")
    if observed_identity != expected_identity:
        raise AuditError("producer identity differs from its frozen anchor")
    return {
        "document": document,
        "selected_identity_sha256": observed_identity,
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def _validate_producer_projection(
    document: dict[str, object],
    raw: bytes,
    producer_site: Path,
    producer_bin: Path,
) -> dict[str, object]:
    site_identity = tree_identity(producer_site)
    expected_document = {
        **site_identity,
        "distributions": PRODUCER_SITE_IDENTITY["distributions"],
    }
    if document != expected_document or document != PRODUCER_SITE_IDENTITY:
        raise AuditError("producer-site projection differs from its identity")
    bin_leaves = _tree_leaf_map(producer_bin)
    if set(bin_leaves) != {"cmake", "lit", "ninja"}:
        raise AuditError("producer-bin projection has an unexpected command set")
    cmake = producer_bin / "cmake"
    if (
        cmake.is_symlink()
        or not cmake.is_file()
        or not os.access(cmake, os.X_OK)
        or cmake.read_bytes() != producer_cmake_wrapper(PRODUCER_CMAKE_PAYLOAD)
        or not PRODUCER_CMAKE_PAYLOAD.is_file()
        or PRODUCER_CMAKE_PAYLOAD.is_symlink()
    ):
        raise AuditError("producer-bin CMake is not the exact payload exec-wrapper")
    if sha256_file(cmake) != PRODUCER_CMAKE_WRAPPER_SHA256:
        raise AuditError("producer-bin CMake exec-wrapper hash mismatch")
    if sha256_file(PRODUCER_CMAKE_PAYLOAD) != PRODUCER_CMAKE_SHA256:
        raise AuditError("producer CMake payload mismatch")
    for name in ("lit", "ninja"):
        path = producer_bin / name
        if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
            raise AuditError(f"producer-bin {name} is not an executable regular file")
    if sha256_file(producer_bin / "ninja") != PRODUCER_NINJA_SHA256:
        raise AuditError("producer-bin Ninja payload mismatch")
    lit_lines = (producer_bin / "lit").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    if (
        len(lit_lines) != 6
        or not lit_lines[0].startswith("#!")
        or "/build-venv/bin/python" not in lit_lines[0]
        or lit_lines[1:] != [
            "import sys",
            "from lit.main import main",
            "if __name__ == '__main__':",
            "    sys.argv[0] = sys.argv[0].removesuffix('.exe')",
            "    sys.exit(main())",
        ]
    ):
        raise AuditError("producer-bin lit wrapper mismatch")
    return {
        "bin_tree": tree_identity(producer_bin),
        "document": document,
        "sha256": sha256_bytes(raw),
        "site_tree": site_identity,
        "size": len(raw),
    }


def _tool_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def validate_audit_tool_identity(tools: AuditToolPaths) -> dict[str, object]:
    observed: dict[str, object] = {}
    for name, path in (
        ("bwrap", tools.bwrap),
        ("ldd", tools.ldd),
        ("readelf", tools.readelf),
    ):
        if path != AUDIT_TOOL_PATHS[name].resolve(strict=True):
            raise AuditError(f"{name} path differs from frozen audit producer")
        identity = _tool_identity(path)
        if identity["sha256"] != AUDIT_TOOL_SHA256[name]:
            raise AuditError(f"{name} bytes differ from frozen audit producer")
        observed[name] = identity
    return observed


def _workspace_relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def publish_canonical_json_no_replace(
    output: Path, value: object, *, temp_root: Path
) -> str:
    encoded = canonical_json(value).encode("ascii")
    digest = sha256_bytes(encoded)
    if output.exists() or output.is_symlink():
        raise AuditError(f"evidence already exists: {output}")
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
            raise AuditError(f"evidence already exists: {output}") from error
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


def run_audit(
    request: AuditRequest,
    *,
    expectations: AuditExpectations = AuditExpectations(),
    limits: AuditLimits = AuditLimits(),
) -> tuple[dict[str, object], str]:
    workspace = require_workspace(request.workspace)
    wheel_path = require_workspace_input(request.wheel, workspace, "wheel")
    dependency_path = require_workspace_input(
        request.dependency_manifest, workspace, "dependency manifest"
    )
    if dependency_path.parent.name != request.reviewed_dependency_manifest_sha256:
        raise AuditError(
            "reviewed dependency directory name must equal manifest SHA256"
        )
    dependency_review_path = require_workspace_input(
        dependency_path.with_name("review.json"),
        workspace,
        "dependency review",
    )
    source_path = require_workspace_input(
        request.source_provenance, workspace, "source provenance"
    )
    source_archive = require_workspace_input(
        request.source_archive, workspace, "source archive"
    )
    source_input = require_workspace_directory(
        request.source_input, workspace, "source input"
    )
    build_input = require_workspace_directory(
        request.build_input, workspace, "build input"
    )
    built_source = require_workspace_directory(
        request.built_source, workspace, "built source"
    )
    producer_path = require_workspace_input(
        request.producer_provenance, workspace, "producer provenance"
    )
    producer_site_identity_path = require_workspace_input(
        request.producer_site_identity, workspace, "producer-site identity"
    )
    producer_site = require_workspace_directory(
        request.producer_site, workspace, "producer site"
    )
    producer_bin = require_workspace_directory(
        request.producer_bin, workspace, "producer bin"
    )
    temp_root = require_workspace_directory(request.temp_root, workspace, "temp root")
    evidence_path = require_evidence_output(request.evidence, workspace)
    tools = AuditToolPaths(
        readelf=require_absolute_tool(request.tools.readelf, "readelf"),
        ldd=require_absolute_tool(request.tools.ldd, "ldd"),
        bwrap=require_absolute_tool(request.tools.bwrap, "bubblewrap"),
    )
    audit_tool_identities = validate_audit_tool_identity(tools)
    if limits.command_timeout_seconds <= 0 or limits.command_timeout_seconds > 300:
        raise AuditError("command timeout must be between 1 and 300 seconds")
    if wheel_path.stat().st_size <= 0 or wheel_path.stat().st_size > limits.max_wheel_bytes:
        raise AuditError("wheel size is empty or exceeds the limit")

    dependency_document, dependency_raw = load_canonical_json(
        dependency_path, "dependency manifest"
    )
    dependency_review_document, dependency_review_raw = load_canonical_json(
        dependency_review_path, "dependency review"
    )
    source_document, source_raw = load_canonical_json(
        source_path, "source provenance"
    )
    producer_document, producer_raw = load_canonical_json(
        producer_path, "producer provenance"
    )
    producer_site_document, producer_site_raw = load_canonical_json(
        producer_site_identity_path, "producer-site identity"
    )
    dependency_provenance = _validate_dependency_manifest(
            dependency_document,
            dependency_raw,
            request.reviewed_dependency_manifest_sha256,
            expectations,
        )
    build_inputs = dependency_document["build_inputs"]
    assert isinstance(build_inputs, dict)
    nvidia_overlay = require_workspace_directory(
        dependency_path.parent / "nvidia-backend-overlay",
        workspace,
        "reviewed NVIDIA overlay",
    )
    overlay_identity = tree_identity(nvidia_overlay)
    if overlay_identity != build_inputs["nvidia_backend_overlay_tree"]:
        raise AuditError("reviewed NVIDIA overlay differs from manifest tree anchor")
    dependency_provenance["nvidia_backend_overlay"] = {
        "path": _workspace_relative(nvidia_overlay, workspace),
        "tree": overlay_identity,
    }
    dependency_provenance["review"] = _validate_dependency_review(
        dependency_review_document,
        dependency_review_raw,
        dependency_document,
        request.reviewed_dependency_manifest_sha256,
        expectations,
    )
    provenance = {
        "dependency_manifest": dependency_provenance,
        "producer": _validate_producer_provenance(
            producer_document,
            producer_raw,
            request.expected_producer_identity_sha256,
        ),
        "producer_projection": _validate_producer_projection(
            producer_site_document,
            producer_site_raw,
            producer_site,
            producer_bin,
        ),
        "source": _validate_source_provenance(
            source_document,
            source_raw,
            expectations,
            source_archive=source_archive,
            source_input=source_input,
            nvidia_overlay=nvidia_overlay,
            build_input=build_input,
            built_source=built_source,
            temp_root=temp_root,
        ),
    }

    wheel_sha256 = sha256_file(wheel_path)
    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            static_evidence, infos = inspect_wheel_static(
                zf, wheel_path.name, expectations, limits
            )
            with tempfile.TemporaryDirectory(
                prefix="triton-wheel-audit-", dir=temp_root
            ) as temporary_directory:
                extraction_root = Path(temporary_directory) / "fresh-extract"
                extract_fresh(zf, infos, extraction_root)
                native_manifest = audit_elf_files(
                    extraction_root, tools, temp_root, limits
                )
                tool_probes = probe_wheel_tools(
                    extraction_root, tools, expectations, temp_root, limits
                )
    except zipfile.BadZipFile as error:
        raise AuditError("wheel is not a valid ZIP archive") from error

    archive_elf_paths = sorted(
        record["path"] for record in native_manifest
    )
    if sorted(tool["path"] for tool in tool_probes.values()) != sorted(
        [FILECHECK_PATH]
        + [f"{NVIDIA_BIN_PREFIX}/{name}" for name in expectations.tool_versions()]
    ):
        raise AuditError("tool probe set is incomplete")

    evidence: dict[str, object] = {
        "acceptance": "accepted",
        "audit": "triton-workspace-wheel",
        "audit_tools": {
            "bubblewrap": audit_tool_identities["bwrap"],
            "ldd": audit_tool_identities["ldd"],
            "readelf": audit_tool_identities["readelf"],
        },
        "expectations": {
            "distribution_version": expectations.distribution_version,
            "libdevice_sha256": expectations.libdevice_sha256,
            "module_version": expectations.module_version,
            "nvidia_tool_versions": expectations.tool_versions(),
            "triton_commit": expectations.commit,
            "triton_llvm_commit": expectations.llvm_commit,
            "triton_tree": expectations.tree,
            "source_archive_sha256": expectations.source_archive_sha256,
        },
        "limits": {
            "command_timeout_seconds": limits.command_timeout_seconds,
            "max_command_output_bytes": limits.max_command_output_bytes,
            "max_expanded_bytes": limits.max_expanded_bytes,
            "max_member_bytes": limits.max_member_bytes,
            "max_members": limits.max_members,
            "max_wheel_bytes": limits.max_wheel_bytes,
        },
        "provenance": provenance,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "wheel": {
            **static_evidence,
            "elf_paths": archive_elf_paths,
            "filename": wheel_path.name,
            "native_manifest": native_manifest,
            "path": _workspace_relative(wheel_path, workspace),
            "sha256": wheel_sha256,
            "size": wheel_path.stat().st_size,
            "tool_probes": tool_probes,
        },
    }
    evidence_sha256 = publish_canonical_json_no_replace(
        evidence_path, evidence, temp_root=temp_root
    )
    return evidence, evidence_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument(
        "--reviewed-dependency-manifest-sha256", required=True
    )
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--build-input", type=Path, required=True)
    parser.add_argument("--built-source", type=Path, required=True)
    parser.add_argument("--producer-provenance", type=Path, required=True)
    parser.add_argument("--producer-site-identity", type=Path, required=True)
    parser.add_argument("--producer-site", type=Path, required=True)
    parser.add_argument("--producer-bin", type=Path, required=True)
    parser.add_argument("--expected-producer-identity-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--readelf", type=Path, default=Path("/usr/bin/readelf"))
    parser.add_argument("--ldd", type=Path, default=Path("/usr/bin/ldd"))
    parser.add_argument("--bwrap", type=Path, default=Path("/usr/bin/bwrap"))
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = AuditRequest(
        workspace=args.workspace,
        wheel=args.wheel,
        dependency_manifest=args.dependency_manifest,
        reviewed_dependency_manifest_sha256=args.reviewed_dependency_manifest_sha256,
        source_provenance=args.source_provenance,
        source_archive=args.source_archive,
        source_input=args.source_input,
        build_input=args.build_input,
        built_source=args.built_source,
        producer_provenance=args.producer_provenance,
        producer_site_identity=args.producer_site_identity,
        producer_site=args.producer_site,
        producer_bin=args.producer_bin,
        expected_producer_identity_sha256=args.expected_producer_identity_sha256,
        evidence=args.evidence,
        temp_root=args.temp_root,
        tools=AuditToolPaths(
            readelf=args.readelf,
            ldd=args.ldd,
            bwrap=args.bwrap,
        ),
    )
    limits = AuditLimits(command_timeout_seconds=args.timeout_seconds)
    try:
        _, evidence_sha256 = run_audit(request, limits=limits)
    except (AuditError, OSError, ValueError) as error:
        print(f"triton wheel audit failed: {error}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "evidence": str(request.evidence),
                "evidence_sha256": evidence_sha256,
                "status": "accepted",
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
