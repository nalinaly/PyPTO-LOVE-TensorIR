#!/usr/bin/env python3
"""Install and probe an already-audited Triton wheel in a fresh venv.

This tool is deliberately narrower than a package installer.  It accepts only
the exact wheel named and hashed by canonical ``audit_triton_wheel.py``
evidence, installs that wheel with a small fail-closed Wheel/RECORD installer,
and then runs two isolated Python probes.  The probes do not initialize CUDA:
SM120 is represented by an explicit Triton ``GPUTarget`` and is used only to
exercise the default NVIDIA tool selector.

The fresh interpreter is started with ``-I -B -S``.  Triton's new site-packages
directory is added before a separately supplied, workspace-owned Torch
site-packages directory.  Consequently Torch can exercise its Triton adapter
without processing any ambient ``.pth`` file or editable finder.  Successful
evidence is canonical JSON and is published atomically without replacement.
"""

from __future__ import annotations

import argparse
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
import stat
import subprocess
import sys
import tempfile
import unicodedata
import urllib.parse
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA_VERSION = 1
AUDIT_NAME = "triton-workspace-wheel"
PROBE_NAME = "triton-fresh-wheel"
TRITON_DISTRIBUTION_VERSION = "3.7.1+git5d6048aa"
TRITON_MODULE_VERSION = "3.7.1"
DIST_INFO = f"triton-{TRITON_DISTRIBUTION_VERSION}.dist-info"
RECORD_PATH = f"{DIST_INFO}/RECORD"
DIRECT_URL_PATH = f"{DIST_INFO}/direct_url.json"
PTXAS_BLACKWELL_PATH = "triton/backends/nvidia/bin/ptxas-blackwell"
TORCH_VERSION = "2.13.0+cu130"
TORCH_GIT_VERSION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TORCH_CUDA_VERSION = "13.0"
TORCH_TREE_SHA256 = (
    "1f77cf114e19ac071bbb3e552e98bd7d1b28f58484225d8929aa077d8ddc00d9"
)
TORCH_TREE_FILES = 15021
TORCH_TREE_BYTES = 1169343614
BASE_PYTHON_SHA256 = (
    "aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f"
)
BASE_PYTHON_VERSION = (3, 14, 6)


class ProbeError(RuntimeError):
    """A fresh-install or runtime provenance invariant was not proven."""


@dataclass(frozen=True, slots=True)
class ProbeLimits:
    max_wheel_bytes: int = 4 << 30
    max_member_bytes: int = 2 << 30
    max_expanded_bytes: int = 12 << 30
    max_members: int = 100_000
    max_output_bytes: int = 8 << 20
    address_space_bytes: int = 16 << 30
    timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class InstallScheme:
    prefix: Path
    purelib: Path
    platlib: Path
    scripts: Path
    headers: Path
    data: Path
    python_version: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    workspace: Path
    wheel: Path
    wheel_audit_evidence: Path
    expected_wheel_audit_evidence_sha256: str
    base_python: Path
    torch_site_packages: Path
    environment_lock: Path
    probe_prefix: Path
    evidence: Path


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
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProbeError(
            f"{description} must be 64 lowercase hexadecimal characters"
        )
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError(f"duplicate JSON key: {key!r}")
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
        raise ProbeError(
            f"cannot open {description} as a non-symlink: {path}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ProbeError(f"{description} is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_canonical_json(
    path: Path, description: str
) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_file_once(path, description)
    try:
        text = raw.decode("utf-8", errors="strict")
        document = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ProbeError(f"{description} root must be an object")
    if text != canonical_json(document):
        raise ProbeError(f"{description} is not canonical JSON")
    return document, raw


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    return (allow_root and path == root) or root in path.parents


def require_workspace(path: Path) -> Path:
    if not path.is_absolute():
        raise ProbeError("--workspace must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"workspace is absent: {lexical}") from error
    if lexical != resolved or not resolved.is_dir():
        raise ProbeError("workspace must be a real, non-symlink directory")
    return resolved


def require_workspace_file(path: Path, workspace: Path, description: str) -> Path:
    if not path.is_absolute():
        raise ProbeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise ProbeError(f"{description} must be a real workspace-owned path")
    if not resolved.is_file() or resolved.is_symlink():
        raise ProbeError(f"{description} must be a regular non-symlink file")
    return resolved


def require_workspace_directory(
    path: Path, workspace: Path, description: str
) -> Path:
    if not path.is_absolute():
        raise ProbeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"{description} is absent: {lexical}") from error
    if lexical != resolved or not _is_within(resolved, workspace):
        raise ProbeError(f"{description} must be a real workspace-owned directory")
    if not resolved.is_dir() or resolved.is_symlink():
        raise ProbeError(f"{description} must be a real directory")
    return resolved


def require_workspace_executable(
    path: Path, workspace: Path, description: str
) -> Path:
    """Accept an in-workspace executable symlink, but execute its fixed target."""

    if not path.is_absolute():
        raise ProbeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    if not _is_within(lexical, workspace):
        raise ProbeError(f"{description} link must be below the workspace")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"{description} is absent: {lexical}") from error
    if not _is_within(resolved, workspace):
        raise ProbeError(f"{description} target escaped the workspace")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ProbeError(f"{description} must resolve to an executable file")
    return resolved


def require_fresh_workspace_prefix(
    path: Path, workspace: Path, description: str = "probe prefix"
) -> Path:
    if not path.is_absolute():
        raise ProbeError(f"{description} path must be absolute")
    lexical = _absolute_lexical(path)
    if lexical == workspace or not _is_within(lexical, workspace):
        raise ProbeError(f"{description} must be a child of the workspace")
    parent = lexical.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"{description} parent is absent: {parent}") from error
    if parent != resolved_parent or not _is_within(parent, workspace, allow_root=True):
        raise ProbeError(f"{description} parent must be a real workspace directory")
    if not parent.is_dir():
        raise ProbeError(f"{description} parent is not a directory")
    if lexical.exists() or lexical.is_symlink():
        raise ProbeError(f"{description} is not fresh: {lexical}")
    return lexical


def require_evidence_output(path: Path, workspace: Path) -> Path:
    if not path.is_absolute():
        raise ProbeError("evidence path must be absolute")
    lexical = _absolute_lexical(path)
    parent = lexical.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"evidence parent is absent: {parent}") from error
    if parent != resolved_parent or not _is_within(parent, workspace, allow_root=True):
        raise ProbeError("evidence parent must be a real workspace directory")
    if lexical.exists() or lexical.is_symlink():
        raise ProbeError(f"evidence already exists: {lexical}")
    return lexical


def _safe_archive_path(name: str, description: str = "wheel member") -> PurePosixPath:
    if not name or len(name.encode("utf-8")) > 4096:
        raise ProbeError(f"{description} name is empty or too long")
    if (
        "\x00" in name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ProbeError(f"{description} name is unsafe: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.endswith("/")
        or path.as_posix() != name
        or any(part in ("", ".", "..") for part in path.parts)
        or (path.parts and re.match(r"^[A-Za-z]:", path.parts[0]))
    ):
        raise ProbeError(f"{description} path is unsafe: {name!r}")
    return path


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _single_metadata_header(message: object, name: str) -> str:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ProbeError(f"installed METADATA must contain exactly one {name}")
    return values[0].strip()


def validate_audit_anchor(
    document: dict[str, object],
    raw: bytes,
    *,
    expected_evidence_sha256: str,
    wheel_path: Path,
    workspace: Path,
    limits: ProbeLimits,
) -> dict[str, object]:
    expected_evidence_sha256 = validate_sha256(
        expected_evidence_sha256, "expected wheel-audit evidence SHA256"
    )
    actual_evidence_sha256 = sha256_bytes(raw)
    if actual_evidence_sha256 != expected_evidence_sha256:
        raise ProbeError("wheel-audit evidence differs from its SHA256 anchor")
    if document.get("schema_version") != 1:
        raise ProbeError("unsupported wheel-audit evidence schema")
    if document.get("audit") != AUDIT_NAME or document.get("acceptance") != "accepted":
        raise ProbeError("wheel-audit evidence is not an accepted Triton wheel audit")

    expectations = document.get("expectations")
    if not isinstance(expectations, dict):
        raise ProbeError("wheel-audit expectations are absent")
    if expectations.get("distribution_version") != TRITON_DISTRIBUTION_VERSION:
        raise ProbeError("audited Triton distribution version mismatch")
    if expectations.get("module_version") != TRITON_MODULE_VERSION:
        raise ProbeError("audited Triton module version mismatch")

    wheel = document.get("wheel")
    if not isinstance(wheel, dict):
        raise ProbeError("wheel-audit wheel record is absent")
    recorded_relative = wheel.get("path")
    if not isinstance(recorded_relative, str):
        raise ProbeError("wheel-audit wheel path is absent")
    relative = _safe_archive_path(recorded_relative, "workspace-relative wheel")
    recorded_path = workspace.joinpath(*relative.parts)
    if recorded_path != wheel_path:
        raise ProbeError("requested wheel is not the wheel named by audit evidence")
    if wheel.get("filename") != wheel_path.name:
        raise ProbeError("wheel filename differs from audit evidence")
    recorded_sha256 = validate_sha256(wheel.get("sha256"), "audited wheel SHA256")
    recorded_size = wheel.get("size")
    if not isinstance(recorded_size, int) or isinstance(recorded_size, bool):
        raise ProbeError("audited wheel size is invalid")
    actual_size = wheel_path.stat().st_size
    if actual_size <= 0 or actual_size > limits.max_wheel_bytes:
        raise ProbeError("wheel is empty or exceeds the probe size limit")
    if actual_size != recorded_size or sha256_file(wheel_path) != recorded_sha256:
        raise ProbeError("requested wheel bytes differ from accepted audit evidence")

    metadata = wheel.get("distribution_metadata")
    if not isinstance(metadata, dict):
        raise ProbeError("audited distribution metadata is absent")
    if (
        not isinstance(metadata.get("name"), str)
        or _normalized_distribution_name(metadata["name"]) != "triton"
        or metadata.get("version") != TRITON_DISTRIBUTION_VERSION
    ):
        raise ProbeError("audited distribution identity is not exact Triton")
    if wheel.get("module_version") != TRITON_MODULE_VERSION:
        raise ProbeError("audited module version is not exact")
    wheel_metadata = wheel.get("wheel_metadata")
    if not isinstance(wheel_metadata, dict) or wheel_metadata.get(
        "root_is_purelib"
    ) is not False:
        raise ProbeError("audited Triton wheel must use platlib")

    archive = wheel.get("archive")
    if not isinstance(archive, dict):
        raise ProbeError("audited archive inventory is absent")
    members = archive.get("members")
    if not isinstance(members, list) or not members:
        raise ProbeError("audited archive member inventory is absent")
    if archive.get("members_count") != len(members) or len(members) > limits.max_members:
        raise ProbeError("audited archive member count is invalid")
    member_map: dict[str, dict[str, object]] = {}
    expanded = 0
    for record in members:
        if not isinstance(record, dict):
            raise ProbeError("audited archive member is not an object")
        path = record.get("path")
        if not isinstance(path, str):
            raise ProbeError("audited archive member path is absent")
        safe_path = _safe_archive_path(path).as_posix()
        if safe_path in member_map:
            raise ProbeError(f"duplicate audited archive member: {safe_path}")
        digest = validate_sha256(record.get("sha256"), f"SHA256 for {safe_path}")
        size = record.get("size")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > limits.max_member_bytes
        ):
            raise ProbeError(f"invalid audited size for {safe_path}")
        expanded += size
        if expanded > limits.max_expanded_bytes:
            raise ProbeError("audited expanded wheel size exceeds the probe limit")
        member_map[safe_path] = {**record, "path": safe_path, "sha256": digest}
    if archive.get("expanded_bytes") != expanded:
        raise ProbeError("audited expanded byte count is inconsistent")
    for required in (
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        RECORD_PATH,
        "triton/__init__.py",
        PTXAS_BLACKWELL_PATH,
    ):
        if required not in member_map:
            raise ProbeError(f"audited wheel lacks required member: {required}")
    if DIRECT_URL_PATH in member_map:
        raise ProbeError("wheel must not pre-own generated direct_url.json")

    record = wheel.get("record")
    if not isinstance(record, dict) or record.get("path") != RECORD_PATH:
        raise ProbeError("audited wheel RECORD inventory is absent")
    entries = record.get("entries")
    if not isinstance(entries, list) or record.get("entries_count") != len(entries):
        raise ProbeError("audited wheel RECORD count is invalid")
    record_map: dict[str, tuple[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ProbeError("audited wheel RECORD entry is invalid")
        path = _safe_archive_path(entry["path"], "audited RECORD").as_posix()
        digest = validate_sha256(entry.get("sha256"), f"RECORD SHA256 for {path}")
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProbeError(f"audited RECORD size is invalid: {path}")
        if path in record_map:
            raise ProbeError(f"duplicate audited RECORD entry: {path}")
        record_map[path] = (digest, size)
    expected_record_map = {
        path: (record["sha256"], record["size"])
        for path, record in member_map.items()
    }
    if record_map != expected_record_map:
        raise ProbeError("audited RECORD and archive inventories differ")

    resources = wheel.get("required_resources")
    if not isinstance(resources, dict):
        raise ProbeError("audited required-resource inventory is absent")
    nvidia_tools = resources.get("nvidia_tools")
    if not isinstance(nvidia_tools, dict):
        raise ProbeError("audited NVIDIA tool inventory is absent")
    ptxas = nvidia_tools.get("ptxas-blackwell")
    if (
        not isinstance(ptxas, dict)
        or ptxas.get("path") != PTXAS_BLACKWELL_PATH
        or ptxas.get("expected_version") != "13.1.80"
        or ptxas.get("sha256") != member_map[PTXAS_BLACKWELL_PATH]["sha256"]
        or ptxas.get("size") != member_map[PTXAS_BLACKWELL_PATH]["size"]
    ):
        raise ProbeError("audited ptxas-blackwell identity is incomplete")

    libtriton_members = sorted(
        path
        for path in member_map
        if path.startswith("triton/_C/")
        and PurePosixPath(path).name.startswith("libtriton.")
    )
    if not libtriton_members:
        raise ProbeError("audited wheel contains no triton/_C/libtriton module")
    elf_paths = wheel.get("elf_paths")
    if not isinstance(elf_paths, list) or not set(libtriton_members).issubset(
        set(elf_paths)
    ):
        raise ProbeError("audited ELF inventory omits libtriton")

    return {
        "audit_evidence_sha256": actual_evidence_sha256,
        "audit_evidence_size": len(raw),
        "libtriton_members": libtriton_members,
        "member_map": member_map,
        "ptxas_blackwell": dict(ptxas),
        "wheel_sha256": recorded_sha256,
        "wheel_size": recorded_size,
    }


def _subprocess_limits(limits: ProbeLimits) -> None:
    requested = {
        resource.RLIMIT_AS: limits.address_space_bytes,
        resource.RLIMIT_CORE: 0,
        resource.RLIMIT_CPU: max(1, limits.timeout_seconds),
        resource.RLIMIT_FSIZE: limits.max_output_bytes,
        resource.RLIMIT_NOFILE: 256,
        resource.RLIMIT_NPROC: 512,
    }
    for kind, value in requested.items():
        _, hard = resource.getrlimit(kind)
        bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (bounded, bounded))


def clean_environment(prefix: Path) -> dict[str, str]:
    home = prefix / ".probe-home"
    cache = prefix / ".probe-cache"
    temporary = prefix / ".probe-tmp"
    for path in (home, cache, temporary, cache / "triton"):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def run_command(
    argv: list[str],
    *,
    environment: dict[str, str],
    limits: ProbeLimits,
    description: str,
) -> tuple[bytes, bytes]:
    if not argv or not Path(argv[0]).is_absolute():
        raise ProbeError(f"{description} does not use an absolute executable")
    temporary_root_value = environment.get("TMPDIR")
    if not temporary_root_value:
        raise ProbeError(f"{description} environment has no workspace TMPDIR")
    temporary_root = Path(temporary_root_value)
    if (
        not temporary_root.is_absolute()
        or temporary_root.is_symlink()
        or not temporary_root.is_dir()
    ):
        raise ProbeError(f"{description} TMPDIR is not a real directory")
    try:
        with tempfile.TemporaryFile(dir=temporary_root) as stdout, tempfile.TemporaryFile(
            dir=temporary_root
        ) as stderr:
            result = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                timeout=limits.timeout_seconds,
                preexec_fn=lambda: _subprocess_limits(limits),
                start_new_session=True,
            )
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(limits.max_output_bytes + 1)
            err = stderr.read(limits.max_output_bytes + 1)
    except subprocess.TimeoutExpired as error:
        raise ProbeError(f"{description} timed out") from error
    except OSError as error:
        raise ProbeError(f"{description} could not execute") from error
    if len(out) > limits.max_output_bytes or len(err) > limits.max_output_bytes:
        raise ProbeError(f"{description} output exceeded the limit")
    if result.returncode != 0:
        excerpt = (out + b"\n" + err)[-2000:].decode("utf-8", errors="replace")
        raise ProbeError(
            f"{description} returned {result.returncode}: "
            f"{excerpt.replace(chr(10), ' ')}"
        )
    return out, err


def run_json_command(
    argv: list[str],
    *,
    environment: dict[str, str],
    limits: ProbeLimits,
    description: str,
) -> dict[str, object]:
    stdout, stderr = run_command(
        argv, environment=environment, limits=limits, description=description
    )
    if stderr:
        raise ProbeError(f"{description} produced unexpected stderr")
    try:
        value = json.loads(
            stdout.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"{description} did not return one JSON document") from error
    if not isinstance(value, dict):
        raise ProbeError(f"{description} JSON root is not an object")
    return value


def create_fresh_venv(
    base_python: Path,
    prefix: Path,
    *,
    limits: ProbeLimits,
) -> Path:
    try:
        prefix.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ProbeError(f"probe prefix ceased to be fresh: {prefix}") from error
    environment = clean_environment(prefix)
    stdout, stderr = run_command(
        [
            str(base_python),
            "-I",
            "-B",
            "-m",
            "venv",
            "--without-pip",
            str(prefix),
        ],
        environment=environment,
        limits=limits,
        description="fresh venv creation",
    )
    if stdout or stderr:
        raise ProbeError("fresh venv creation produced unexpected output")
    if not prefix.is_dir() or prefix.is_symlink():
        raise ProbeError("venv command did not create the exact fresh prefix")
    python = prefix / "bin" / "python"
    if not python.exists() or not os.access(python, os.X_OK):
        raise ProbeError("fresh venv has no executable bin/python")
    try:
        resolved_python = python.resolve(strict=True)
    except OSError as error:
        raise ProbeError("fresh venv interpreter link is invalid") from error
    if resolved_python != base_python:
        raise ProbeError("fresh venv interpreter does not resolve to the audited base Python")
    return python


SCHEME_PROGRAM = r"""
import json
import pathlib
import sys
import sysconfig

value = {
    "executable": sys.executable,
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
    "prefix": sys.prefix,
    "version": list(sys.version_info[:3]),
}
print(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def query_install_scheme(
    python: Path, prefix: Path, *, limits: ProbeLimits
) -> InstallScheme:
    report = run_json_command(
        [str(python), "-I", "-B", "-S", "-c", SCHEME_PROGRAM],
        environment=clean_environment(prefix),
        limits=limits,
        description="fresh venv sysconfig probe",
    )
    try:
        observed_prefix = Path(report["prefix"])
        version_value = report["version"]
        paths = report["paths"]
        executable = Path(report["executable"])
    except (KeyError, TypeError) as error:
        raise ProbeError("fresh venv sysconfig report is incomplete") from error
    if observed_prefix.resolve(strict=True) != prefix:
        raise ProbeError("fresh venv sys.prefix differs from requested prefix")
    if _absolute_lexical(executable) != python:
        raise ProbeError("fresh venv sys.executable differs from bin/python")
    if (
        not isinstance(version_value, list)
        or len(version_value) != 3
        or any(not isinstance(item, int) for item in version_value)
    ):
        raise ProbeError("fresh venv Python version report is invalid")
    version = tuple(version_value)
    if version != BASE_PYTHON_VERSION:
        raise ProbeError(f"fresh venv is not pinned CPython 3.14.6: {version}")
    if not isinstance(paths, dict):
        raise ProbeError("fresh venv sysconfig paths are invalid")
    resolved_paths: dict[str, Path] = {}
    for name in ("data", "headers", "platlib", "purelib", "scripts"):
        value = paths.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ProbeError(f"fresh venv sysconfig {name} path is invalid")
        path = _absolute_lexical(Path(value))
        if not _is_within(path, prefix, allow_root=True):
            raise ProbeError(f"fresh venv sysconfig {name} escaped the prefix")
        resolved_paths[name] = path
    return InstallScheme(
        prefix=prefix,
        purelib=resolved_paths["purelib"],
        platlib=resolved_paths["platlib"],
        scripts=resolved_paths["scripts"],
        headers=resolved_paths["headers"],
        data=resolved_paths["data"],
        python_version=version,  # type: ignore[arg-type]
    )


def _zip_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0xFFFF if info.create_system == 3 else 0


def _validate_zip_info(info: zipfile.ZipInfo, limits: ProbeLimits) -> None:
    _safe_archive_path(info.filename)
    if info.is_dir():
        raise ProbeError(f"directory wheel member is forbidden: {info.filename}")
    if info.flag_bits & 1:
        raise ProbeError(f"encrypted wheel member is forbidden: {info.filename}")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ProbeError(f"unsupported wheel compression: {info.filename}")
    if info.file_size < 0 or info.file_size > limits.max_member_bytes:
        raise ProbeError(f"wheel member exceeds size limit: {info.filename}")
    file_type = stat.S_IFMT(_zip_mode(info))
    if file_type not in (0, stat.S_IFREG):
        raise ProbeError(f"non-regular wheel member is forbidden: {info.filename}")


def _data_root() -> str:
    return DIST_INFO.removesuffix(".dist-info") + ".data"


def wheel_member_destination(
    archive_path: str, scheme: InstallScheme
) -> Path:
    path = _safe_archive_path(archive_path)
    first = path.parts[0]
    if first == _data_root():
        if len(path.parts) < 3:
            raise ProbeError(f"wheel .data member lacks a payload path: {archive_path}")
        category = path.parts[1]
        roots = {
            "data": scheme.data,
            "headers": scheme.headers,
            "platlib": scheme.platlib,
            "purelib": scheme.purelib,
            "scripts": scheme.scripts,
        }
        if category not in roots:
            raise ProbeError(f"unsupported wheel .data category: {category}")
        destination = roots[category].joinpath(*path.parts[2:])
    else:
        if first.endswith(".data"):
            raise ProbeError(f"unexpected wheel .data root: {first}")
        destination = scheme.platlib.joinpath(*path.parts)
    lexical = _absolute_lexical(destination)
    if not _is_within(lexical, scheme.prefix):
        raise ProbeError(f"installed wheel member escapes fresh prefix: {archive_path}")
    return lexical


def _open_exclusive_regular(path: Path, mode: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, mode)


def _write_exclusive(path: Path, value: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor = _open_exclusive_regular(path, mode)
    try:
        total = 0
        while total < len(value):
            total += os.write(descriptor, value[total:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_digest(digest: str) -> str:
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _record_relative(path: Path, platlib: Path) -> str:
    return PurePosixPath(os.path.relpath(path, platlib)).as_posix()


def direct_url_document(wheel_path: Path, wheel_sha256: str) -> dict[str, object]:
    return {
        "archive_info": {
            "hash": f"sha256={wheel_sha256}",
            "hashes": {"sha256": wheel_sha256},
        },
        "url": wheel_path.as_uri(),
    }


def install_audited_wheel(
    wheel_path: Path,
    scheme: InstallScheme,
    anchor: dict[str, object],
    *,
    limits: ProbeLimits,
) -> dict[str, object]:
    member_map = anchor["member_map"]
    if not isinstance(member_map, dict):
        raise ProbeError("internal audited member map is invalid")
    for root in (
        scheme.purelib,
        scheme.platlib,
        scheme.scripts,
        scheme.headers,
        scheme.data,
    ):
        root.mkdir(mode=0o755, parents=True, exist_ok=True)

    installed: dict[str, dict[str, object]] = {}
    destinations: set[Path] = set()
    expanded = 0
    try:
        with zipfile.ZipFile(wheel_path, "r") as wheel:
            infos = wheel.infolist()
            if len(infos) != len(member_map):
                raise ProbeError("wheel member count changed after its accepted audit")
            if len({info.filename for info in infos}) != len(infos):
                raise ProbeError("wheel contains duplicate members")
            if set(info.filename for info in infos) != set(member_map):
                raise ProbeError("wheel membership changed after its accepted audit")
            for info in sorted(infos, key=lambda value: value.filename):
                _validate_zip_info(info, limits)
                expected = member_map[info.filename]
                if not isinstance(expected, dict):
                    raise ProbeError("internal audited member record is invalid")
                if info.file_size != expected["size"]:
                    raise ProbeError(f"wheel member size drift: {info.filename}")
                expanded += info.file_size
                if expanded > limits.max_expanded_bytes:
                    raise ProbeError("wheel expanded size exceeds the install limit")
                destination = wheel_member_destination(info.filename, scheme)
                if destination in destinations:
                    raise ProbeError(f"wheel install path collision: {info.filename}")
                destinations.add(destination)
                if info.filename == RECORD_PATH:
                    continue
                if destination.exists() or destination.is_symlink():
                    raise ProbeError(f"fresh install destination already exists: {destination}")
                destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                mode = 0o755 if _zip_mode(info) & 0o111 else 0o644
                descriptor = _open_exclusive_regular(destination, mode)
                digest = hashlib.sha256()
                size = 0
                try:
                    with wheel.open(info, "r") as source:
                        for chunk in iter(lambda: source.read(1 << 20), b""):
                            size += len(chunk)
                            digest.update(chunk)
                            written = 0
                            while written < len(chunk):
                                written += os.write(descriptor, chunk[written:])
                    os.fsync(descriptor)
                except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                    raise ProbeError(f"cannot install wheel member: {info.filename}") from error
                finally:
                    os.close(descriptor)
                observed_digest = digest.hexdigest()
                if size != expected["size"] or observed_digest != expected["sha256"]:
                    raise ProbeError(f"installed wheel member differs: {info.filename}")
                installed[info.filename] = {
                    "archive_path": info.filename,
                    "installed_path": _record_relative(destination, scheme.platlib),
                    "sha256": observed_digest,
                    "size": size,
                }
    except zipfile.BadZipFile as error:
        raise ProbeError("accepted wheel is no longer a valid ZIP archive") from error

    record_destination = wheel_member_destination(RECORD_PATH, scheme)
    direct_url_destination = wheel_member_destination(DIRECT_URL_PATH, scheme)
    if direct_url_destination.exists() or direct_url_destination.is_symlink():
        raise ProbeError("generated direct_url.json destination already exists")
    direct_document = direct_url_document(
        wheel_path, anchor["wheel_sha256"]  # type: ignore[arg-type]
    )
    direct_bytes = canonical_json(direct_document).encode("ascii")
    _write_exclusive(direct_url_destination, direct_bytes)
    direct_digest = sha256_bytes(direct_bytes)

    record_rows: list[tuple[str, str, str]] = []
    for record in installed.values():
        record_rows.append(
            (
                str(record["installed_path"]),
                _record_digest(str(record["sha256"])),
                str(record["size"]),
            )
        )
    direct_record_path = _record_relative(direct_url_destination, scheme.platlib)
    record_rows.append(
        (direct_record_path, _record_digest(direct_digest), str(len(direct_bytes)))
    )
    installed_record_path = _record_relative(record_destination, scheme.platlib)
    record_rows.sort(key=lambda row: row[0])
    record_rows.append((installed_record_path, "", ""))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(record_rows)
    record_bytes = output.getvalue().encode("utf-8")
    if record_destination.exists() or record_destination.is_symlink():
        raise ProbeError("generated RECORD destination already exists")
    _write_exclusive(record_destination, record_bytes)

    return {
        "archive_members": [installed[path] for path in sorted(installed)],
        "archive_record": {
            "path": RECORD_PATH,
            "sha256": anchor["member_map"][RECORD_PATH]["sha256"],  # type: ignore[index]
            "size": anchor["member_map"][RECORD_PATH]["size"],  # type: ignore[index]
        },
        "direct_url": {
            "document": direct_document,
            "path": direct_record_path,
            "sha256": direct_digest,
            "size": len(direct_bytes),
        },
        "installed_record": {
            "entries_count": len(record_rows),
            "path": installed_record_path,
            "sha256": sha256_bytes(record_bytes),
            "size": len(record_bytes),
        },
    }


def _decode_record_digest(value: str, path: str) -> str:
    if not value.startswith("sha256="):
        raise ProbeError(f"installed RECORD uses a non-SHA256 digest: {path}")
    encoded = value.removeprefix("sha256=")
    if not encoded or "=" in encoded or re.fullmatch(r"[A-Za-z0-9_-]+", encoded) is None:
        raise ProbeError(f"installed RECORD SHA256 is not canonical: {path}")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as error:
        raise ProbeError(f"installed RECORD SHA256 cannot be decoded: {path}") from error
    if len(decoded) != 32:
        raise ProbeError(f"installed RECORD SHA256 has wrong length: {path}")
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != encoded:
        raise ProbeError(f"installed RECORD SHA256 is not canonical: {path}")
    return decoded.hex()


def _installed_record_target(value: str, scheme: InstallScheme) -> Path:
    if not value or "\\" in value or "\x00" in value:
        raise ProbeError(f"installed RECORD path is unsafe: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in ("", ".") for part in path.parts
    ):
        raise ProbeError(f"installed RECORD path is non-canonical: {value!r}")
    target = _absolute_lexical(scheme.platlib.joinpath(*path.parts))
    if not _is_within(target, scheme.prefix):
        raise ProbeError(f"installed RECORD path escapes fresh prefix: {value}")
    return target


def _editable_artifacts(scheme: InstallScheme) -> list[str]:
    artifacts: list[str] = []
    roots = {scheme.platlib, scheme.purelib}
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            lower = path.name.lower()
            if (
                lower.endswith((".pth", ".egg-link"))
                or "__editable__" in lower
                or "editable_finder" in lower
            ):
                artifacts.append(_record_relative(path, scheme.platlib))
    return sorted(set(artifacts))


def verify_installed_wheel(
    scheme: InstallScheme,
    wheel_path: Path,
    anchor: dict[str, object],
    installation: dict[str, object],
) -> dict[str, object]:
    editable = _editable_artifacts(scheme)
    if editable:
        raise ProbeError(f"fresh probe contains editable/pth artifacts: {editable}")

    dist_infos = sorted(
        path
        for root in {scheme.platlib, scheme.purelib}
        for path in root.glob("triton-*.dist-info")
    )
    expected_dist_info = scheme.platlib / DIST_INFO
    if dist_infos != [expected_dist_info]:
        raise ProbeError(f"fresh probe does not own exactly one Triton distribution: {dist_infos}")
    if expected_dist_info.is_symlink():
        raise ProbeError("installed Triton dist-info is a symlink")

    metadata_raw = _read_regular_file_once(expected_dist_info / "METADATA", "installed METADATA")
    message = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
    if _normalized_distribution_name(_single_metadata_header(message, "Name")) != "triton":
        raise ProbeError("installed distribution name is not Triton")
    if _single_metadata_header(message, "Version") != TRITON_DISTRIBUTION_VERSION:
        raise ProbeError("installed distribution version mismatch")

    direct_path = expected_dist_info / "direct_url.json"
    direct_document, direct_raw = load_canonical_json(direct_path, "installed direct_url.json")
    expected_direct = direct_url_document(
        wheel_path, anchor["wheel_sha256"]  # type: ignore[arg-type]
    )
    if direct_document != expected_direct:
        raise ProbeError("installed direct_url.json is not the audited workspace wheel")
    parsed = urllib.parse.urlparse(str(direct_document.get("url", "")))
    if parsed.scheme != "file" or parsed.netloc:
        raise ProbeError("installed direct_url.json is not a local archive URL")
    direct_target = Path(urllib.parse.unquote(parsed.path))
    if direct_target.resolve(strict=True) != wheel_path:
        raise ProbeError("installed direct_url.json resolves to another wheel")

    record_path = expected_dist_info / "RECORD"
    record_raw = _read_regular_file_once(record_path, "installed RECORD")
    installed_record = installation.get("installed_record")
    if (
        not isinstance(installed_record, dict)
        or installed_record.get("sha256") != sha256_bytes(record_raw)
        or installed_record.get("size") != len(record_raw)
    ):
        raise ProbeError("installed RECORD changed after the safe wheel install")
    try:
        record_text = record_raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ProbeError("installed RECORD is not strict UTF-8 CSV") from error
    records: dict[str, tuple[str, str, Path]] = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in records:
            raise ProbeError("installed RECORD has malformed/duplicate rows")
        records[row[0]] = (row[1], row[2], _installed_record_target(row[0], scheme))

    archive_members = installation.get("archive_members")
    if not isinstance(archive_members, list):
        raise ProbeError("internal installation manifest is absent")
    expected: dict[str, tuple[str, int]] = {}
    for member in archive_members:
        if not isinstance(member, dict):
            raise ProbeError("internal installed member record is invalid")
        expected[str(member["installed_path"])] = (
            str(member["sha256"]),
            int(member["size"]),
        )
    direct_relative = _record_relative(direct_path, scheme.platlib)
    expected[direct_relative] = (sha256_bytes(direct_raw), len(direct_raw))
    record_relative = _record_relative(record_path, scheme.platlib)
    expected_paths = set(expected) | {record_relative}
    if set(records) != expected_paths:
        missing = sorted(expected_paths - set(records))
        extra = sorted(set(records) - expected_paths)
        raise ProbeError(f"installed RECORD ownership mismatch: missing={missing}, extra={extra}")

    verified_entries: list[dict[str, object]] = []
    for relative in sorted(records):
        encoded_digest, encoded_size, target = records[relative]
        if target.is_symlink() or not target.is_file():
            raise ProbeError(f"installed RECORD target is not a regular file: {relative}")
        if relative == record_relative:
            if encoded_digest or encoded_size:
                raise ProbeError("installed RECORD's own row must have empty hash/size")
            digest = sha256_file(target)
            size = target.stat().st_size
        else:
            digest = _decode_record_digest(encoded_digest, relative)
            if re.fullmatch(r"0|[1-9][0-9]*", encoded_size) is None:
                raise ProbeError(f"installed RECORD size is not canonical: {relative}")
            size = int(encoded_size)
            actual_digest = sha256_file(target)
            actual_size = target.stat().st_size
            if (digest, size) != expected[relative] or (
                actual_digest,
                actual_size,
            ) != expected[relative]:
                raise ProbeError(f"installed RECORD byte ownership mismatch: {relative}")
        verified_entries.append({"path": relative, "sha256": digest, "size": size})

    owned_paths = set(records)
    for root in (scheme.platlib / "triton", expected_dist_info):
        if not root.is_dir() or root.is_symlink():
            raise ProbeError(f"installed Triton tree is absent or a symlink: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ProbeError(f"installed Triton tree contains a symlink: {path}")
            if path.is_file():
                relative = _record_relative(path, scheme.platlib)
                if relative not in owned_paths:
                    raise ProbeError(f"installed Triton file is not RECORD-owned: {relative}")

    return {
        "direct_url": direct_document,
        "editable_artifacts": [],
        "entries": verified_entries,
        "entries_count": len(verified_entries),
        "record_sha256": sha256_bytes(record_raw),
        "record_size": len(record_raw),
    }


RUNTIME_PROBE_PROGRAM = r"""
import importlib.metadata
import json
import pathlib
import sys

probe_site = pathlib.Path(sys.argv[1]).resolve(strict=True)
torch_runtime_view = pathlib.Path(sys.argv[2]).resolve(strict=True)
prefix = pathlib.Path(sys.argv[3]).resolve(strict=True)
torch_source_site = pathlib.Path(sys.argv[4]).resolve(strict=True)

def below(path, root):
    resolved = pathlib.Path(path).resolve(strict=True)
    return resolved == root or root in resolved.parents

def carrier_modules(value):
    return sorted({
        item for item in (
            getattr(value, "__module__", None),
            getattr(type(value), "__module__", None),
        ) if isinstance(item, str) and item
    })

if pathlib.Path(sys.prefix).resolve(strict=True) != prefix:
    raise RuntimeError("runtime sys.prefix escaped fresh prefix")
if "" in sys.path:
    raise RuntimeError("isolated runtime retained an empty import path")
sys.path.insert(0, str(probe_site))

# Resolve and initialize Triton before exposing the Torch environment.  This
# prevents the old environment's Triton dist-info/editable carrier from
# participating even as inert metadata.
triton_distributions = [
    dist for dist in importlib.metadata.distributions(
        path=[str(probe_site), str(torch_runtime_view)]
    )
    if (dist.metadata.get("Name") or "").lower().replace("_", "-") == "triton"
]
if len(triton_distributions) != 1:
    raise RuntimeError("probe site does not expose exactly one Triton distribution")
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

# The fresh-prefix projection contains no .pth/editable/Triton carrier.  -S
# also guarantees that merely adding it never invokes site processing.
sys.path.append(str(torch_runtime_view))
import torch
from torch.utils._triton import get_triton_version, has_triton_package
from torch._inductor.runtime import triton_compat
from torch._inductor.codecache import triton_key as torch_triton_key

compat_key = triton_compat.triton_key()
torch_key = torch_triton_key()
torch_file = pathlib.Path(torch.__file__).resolve(strict=True)
if not below(torch_file, torch_source_site):
    raise RuntimeError("Torch import escaped the explicit source site-packages")
compat_symbols = {
    name: getattr(triton_compat, name, None) is not None
    for name in ("Config", "CompiledKernel", "GPUTarget", "JITFunction", "tl", "triton_key")
}

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
            raise RuntimeError(f"loaded Triton module has no owned path: {name}")
        resolved = [str(libtriton_owner)]
        native_submodules.append(name)
    if any(not below(value, prefix) for value in resolved):
        raise RuntimeError(f"loaded Triton module escaped fresh prefix: {name}={resolved}")
    module_paths[name] = resolved

mapped_libtriton = []
for line in pathlib.Path("/proc/self/maps").read_text(errors="strict").splitlines():
    if "libtriton" not in line.lower():
        continue
    fields = line.split(maxsplit=5)
    if len(fields) != 6 or not fields[5].startswith("/"):
        raise RuntimeError(f"unrecognized libtriton map: {line}")
    mapped = fields[5].removesuffix(" (deleted)")
    resolved = pathlib.Path(mapped).resolve(strict=True)
    if not below(resolved, prefix):
        raise RuntimeError(f"mapped libtriton escaped fresh prefix: {resolved}")
    mapped_libtriton.append(str(resolved))
mapped_libtriton = sorted(set(mapped_libtriton))
if not mapped_libtriton:
    raise RuntimeError("libtriton was imported but is absent from /proc/self/maps")

carriers = {}
for carrier_name, values in (
    ("meta_path", sys.meta_path),
    ("path_hooks", sys.path_hooks),
    ("path_importer_cache", sys.path_importer_cache.values()),
):
    found = sorted({
        module_name
        for value in values
        for module_name in carrier_modules(value)
        if "editable" in module_name.lower()
    })
    if found:
        raise RuntimeError(f"editable importer carrier is active: {carrier_name}={found}")
    carriers[carrier_name] = found
loaded_editable = sorted(
    name for name in sys.modules
    if name.startswith(("_editable", "__editable")) or "editable_finder" in name.lower()
)
if loaded_editable:
    raise RuntimeError(f"editable finder modules are loaded: {loaded_editable}")

report = {
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
    "editable": {
        "carriers": carriers,
        "loaded_modules": loaded_editable,
    },
    "keys": {
        "direct": direct_key,
        "torch_compat": compat_key,
        "torch_inductor": torch_key,
    },
    "libtriton_maps": mapped_libtriton,
    "module_paths": module_paths,
    "native_submodules": native_submodules,
    "module_version": triton.__version__,
    "ptxas_blackwell": {
        "path": str(pathlib.Path(selected_ptxas.path).resolve(strict=True)),
        "reported_release": selected_ptxas.version,
    },
    "sys_path": list(sys.path),
    "torch": {
        "cuda": torch.version.cuda,
        "file": str(torch_file),
        "git_version": torch.version.git_version,
        "hip": torch.version.hip,
        "has_triton_package": has_triton_package(),
        "hip": torch.version.hip,
        "triton_compat_has_triton": triton_compat.HAS_TRITON,
        "triton_compat_symbols": compat_symbols,
        "triton_version": list(get_triton_version()),
        "version": str(torch.__version__),
    },
}
print(json.dumps(report, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def _normalize_probe_path(path: str, prefix: Path, torch_site: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    if _is_within(resolved, prefix, allow_root=True):
        relative = resolved.relative_to(prefix).as_posix()
        return "$PROBE_PREFIX" if relative == "." else f"$PROBE_PREFIX/{relative}"
    if _is_within(resolved, torch_site, allow_root=True):
        relative = resolved.relative_to(torch_site).as_posix()
        return "$TORCH_SITE_PACKAGES" if relative == "." else f"$TORCH_SITE_PACKAGES/{relative}"
    return str(resolved)


def validate_runtime_report(
    report: dict[str, object],
    *,
    scheme: InstallScheme,
    torch_site: Path,
    anchor: dict[str, object],
) -> dict[str, object]:
    distribution = report.get("distribution")
    if not isinstance(distribution, dict):
        raise ProbeError("runtime distribution report is absent")
    expected_dist_info = scheme.platlib / DIST_INFO
    if (
        distribution.get("name") != "triton"
        or distribution.get("version") != TRITON_DISTRIBUTION_VERSION
        or not isinstance(distribution.get("dist_info"), str)
        or Path(distribution["dist_info"]).resolve(strict=True) != expected_dist_info
    ):
        raise ProbeError("runtime Triton distribution identity/path mismatch")
    if report.get("module_version") != TRITON_MODULE_VERSION:
        raise ProbeError("runtime Triton module version mismatch")

    backend = report.get("backend")
    if not isinstance(backend, dict) or backend != {
        "arch": "sm120",
        "class": "triton.backends.nvidia.compiler.CUDABackend",
        "target": ["cuda", 120, 32],
    }:
        raise ProbeError("runtime did not construct the exact SM120 NVIDIA backend")

    ptxas = report.get("ptxas_blackwell")
    expected_ptxas = wheel_member_destination(PTXAS_BLACKWELL_PATH, scheme)
    if (
        not isinstance(ptxas, dict)
        or not isinstance(ptxas.get("path"), str)
        or Path(ptxas["path"]).resolve(strict=True) != expected_ptxas
        or ptxas.get("reported_release") != "13.1"
        or sha256_file(expected_ptxas) != anchor["ptxas_blackwell"]["sha256"]  # type: ignore[index]
    ):
        raise ProbeError("SM120 did not select audited wheel-owned ptxas-blackwell")

    torch = report.get("torch")
    if not isinstance(torch, dict):
        raise ProbeError("Torch Triton compatibility report is absent")
    symbols = torch.get("triton_compat_symbols")
    if (
        torch.get("has_triton_package") is not True
        or torch.get("triton_compat_has_triton") is not True
        or torch.get("triton_version") != [3, 7]
        or not isinstance(symbols, dict)
        or not symbols
        or any(value is not True for value in symbols.values())
        or torch.get("version") != TORCH_VERSION
        or torch.get("git_version") != TORCH_GIT_VERSION
        or torch.get("cuda") != TORCH_CUDA_VERSION
        or torch.get("hip") is not None
    ):
        raise ProbeError("Torch Triton compatibility checks did not all pass")
    torch_file = torch.get("file")
    if (
        not isinstance(torch_file, str)
        or not _is_within(Path(torch_file).resolve(strict=True), torch_site)
    ):
        raise ProbeError("Torch import did not come from supplied site-packages")
    torch_dist_infos = sorted(torch_site.glob("torch-*.dist-info"))
    if len(torch_dist_infos) != 1:
        raise ProbeError("supplied Torch distribution changed during runtime probe")
    torch_metadata = BytesParser(policy=policy.compat32).parsebytes(
        _read_regular_file_once(torch_dist_infos[0] / "METADATA", "Torch METADATA")
    )
    expected_torch_version = _single_metadata_header(torch_metadata, "Version")
    if (
        torch.get("version") != expected_torch_version
        or torch.get("hip") is not None
        or not isinstance(torch.get("git_version"), str)
        or not torch.get("git_version")
    ):
        raise ProbeError("runtime Torch identity is inconsistent or non-NVIDIA")

    keys = report.get("keys")
    if not isinstance(keys, dict):
        raise ProbeError("Triton key report is absent")
    key_values = [keys.get(name) for name in ("direct", "torch_compat", "torch_inductor")]
    if any(not isinstance(value, str) or not value for value in key_values):
        raise ProbeError("a Triton key is empty or non-string")
    if len(set(key_values)) != 1:
        raise ProbeError("Torch and Triton key implementations disagree")

    module_paths = report.get("module_paths")
    if not isinstance(module_paths, dict) or "triton._C.libtriton" not in module_paths:
        raise ProbeError("runtime Triton module inventory is incomplete")
    normalized_modules: dict[str, list[str]] = {}
    native_submodules = report.get("native_submodules")
    if (
        not isinstance(native_submodules, list)
        or len(native_submodules) != len(set(native_submodules))
        or any(
            not isinstance(name, str)
            or not name.startswith("triton._C.libtriton.")
            for name in native_submodules
        )
    ):
        raise ProbeError("runtime native libtriton submodule inventory is malformed")
    for name, values in module_paths.items():
        if (
            not isinstance(name, str)
            or (name != "triton" and not name.startswith("triton."))
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
        ):
            raise ProbeError("runtime Triton module inventory is malformed")
        resolved = [Path(value).resolve(strict=True) for value in values]
        if any(not _is_within(value, scheme.prefix) for value in resolved):
            raise ProbeError(f"runtime Triton module escaped fresh prefix: {name}")
        normalized_modules[name] = [
            _normalize_probe_path(str(value), scheme.prefix, torch_site)
            for value in resolved
        ]
    libtriton_owner_paths = module_paths["triton._C.libtriton"]
    if any(
        name not in module_paths or module_paths[name] != libtriton_owner_paths
        for name in native_submodules
    ):
        raise ProbeError("native submodule is not attributed to wheel-owned libtriton")

    maps = report.get("libtriton_maps")
    if not isinstance(maps, list) or not maps or any(not isinstance(value, str) for value in maps):
        raise ProbeError("runtime libtriton map inventory is absent")
    normalized_maps: list[str] = []
    expected_libtriton_paths = {
        wheel_member_destination(path, scheme)
        for path in anchor["libtriton_members"]  # type: ignore[union-attr]
    }
    for value in maps:
        resolved = Path(value).resolve(strict=True)
        if not _is_within(resolved, scheme.prefix) or resolved not in expected_libtriton_paths:
            raise ProbeError(f"mapped libtriton is not wheel-owned: {resolved}")
        normalized_maps.append(
            _normalize_probe_path(str(resolved), scheme.prefix, torch_site)
        )

    editable = report.get("editable")
    if not isinstance(editable, dict):
        raise ProbeError("runtime editable-carrier report is absent")
    carriers = editable.get("carriers")
    if (
        editable.get("loaded_modules") != []
        or not isinstance(carriers, dict)
        or set(carriers) != {"meta_path", "path_hooks", "path_importer_cache"}
        or any(value != [] for value in carriers.values())
    ):
        raise ProbeError("runtime contains an editable finder carrier")

    normalized = dict(report)
    normalized["distribution"] = {
        **distribution,
        "dist_info": _normalize_probe_path(
            str(distribution["dist_info"]), scheme.prefix, torch_site
        ),
    }
    normalized["module_paths"] = normalized_modules
    normalized["libtriton_maps"] = sorted(set(normalized_maps))
    normalized["ptxas_blackwell"] = {
        **ptxas,
        "path": _normalize_probe_path(str(ptxas["path"]), scheme.prefix, torch_site),
        "audited_full_version": anchor["ptxas_blackwell"][  # type: ignore[index]
            "expected_version"
        ],
        "sha256": anchor["ptxas_blackwell"]["sha256"],  # type: ignore[index]
    }
    normalized["torch"] = {
        **torch,
        "file": _normalize_probe_path(str(torch_file), scheme.prefix, torch_site),
    }
    sys_path = report.get("sys_path")
    if not isinstance(sys_path, list) or any(not isinstance(value, str) for value in sys_path):
        raise ProbeError("runtime sys.path report is malformed")
    normalized["sys_path"] = [
        _normalize_probe_path(value, scheme.prefix, torch_site)
        if value and Path(value).exists()
        else value
        for value in sys_path
    ]
    return normalized


def run_runtime_probe(
    python: Path,
    scheme: InstallScheme,
    torch_site: Path,
    torch_runtime_view: Path,
    anchor: dict[str, object],
    *,
    limits: ProbeLimits,
) -> dict[str, object]:
    report = run_json_command(
        [
            str(python),
            "-I",
            "-B",
            "-S",
            "-c",
            RUNTIME_PROBE_PROGRAM,
            str(scheme.platlib),
            str(torch_runtime_view),
            str(scheme.prefix),
            str(torch_site),
        ],
        environment=clean_environment(scheme.prefix),
        limits=limits,
        description="isolated Triton/Torch runtime probe",
    )
    return validate_runtime_report(
        report, scheme=scheme, torch_site=torch_site, anchor=anchor
    )


def publish_canonical_json_no_replace(output: Path, value: object) -> str:
    encoded = canonical_json(value).encode("ascii")
    digest = sha256_bytes(encoded)
    if output.exists() or output.is_symlink():
        raise ProbeError(f"evidence already exists: {output}")
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
            raise ProbeError(f"evidence already exists: {output}") from error
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


def _workspace_relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _path_identity(path: Path, workspace: Path) -> dict[str, object]:
    raw = _read_regular_file_once(path, "identity input")
    return {
        "path": _workspace_relative(path, workspace),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def torch_tree_identity(
    torch_site: Path, torch_root: Path, dist_info: Path
) -> dict[str, object]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    files = sorted(
        path
        for root in (torch_root, dist_info)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
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
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
        byte_count += size
    return {
        "torch_tree_sha256": digest.hexdigest(),
        "torch_tree_files": file_count,
        "torch_tree_bytes": byte_count,
    }


def torch_tree_structure_identity(
    torch_site: Path, torch_root: Path, dist_info: Path
) -> dict[str, object]:
    """Bind tree roots, directories, modes, file sizes, and symlink targets."""

    digest = hashlib.sha256()
    counts = {"directories": 0, "files": 0, "symlinks": 0}
    paths = sorted(
        path
        for root in (torch_root, dist_info)
        for path in (root, *root.rglob("*"))
    )
    for path in paths:
        relative = path.relative_to(torch_site).as_posix().encode("utf-8")
        mode = stat.S_IMODE(path.lstat().st_mode)
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
            digest.update(b"F")
            digest.update(path.stat().st_size.to_bytes(8, "little"))
            counts["files"] += 1
        else:
            raise ProbeError(f"unsupported Torch-tree entry: {path}")
    return {"sha256": digest.hexdigest(), **counts}


def validate_torch_site(
    torch_site: Path,
    environment_lock: dict[str, object],
    *,
    workspace: Path,
) -> dict[str, object]:
    torch_init = torch_site / "torch" / "__init__.py"
    if not torch_init.is_file() or torch_init.is_symlink():
        raise ProbeError("supplied Torch site-packages has no regular torch package")
    dist_infos = sorted(torch_site.glob("torch-*.dist-info"))
    if len(dist_infos) != 1 or not dist_infos[0].is_dir() or dist_infos[0].is_symlink():
        raise ProbeError("supplied Torch site-packages must contain one Torch dist-info")
    metadata_raw = _read_regular_file_once(dist_infos[0] / "METADATA", "Torch METADATA")
    message = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
    if _normalized_distribution_name(_single_metadata_header(message, "Name")) != "torch":
        raise ProbeError("supplied Torch distribution name is invalid")
    expected_lock = {
        "cuda": TORCH_CUDA_VERSION,
        "hip": None,
        "torch": TORCH_VERSION,
        "torch_git": TORCH_GIT_VERSION,
        "torch_tree_bytes": TORCH_TREE_BYTES,
        "torch_tree_files": TORCH_TREE_FILES,
        "torch_tree_sha256": TORCH_TREE_SHA256,
    }
    for field, expected in expected_lock.items():
        if environment_lock.get(field) != expected:
            raise ProbeError(f"ENVIRONMENT.lock {field} mismatch")
    prefix_value = environment_lock.get("destination_prefix")
    if not isinstance(prefix_value, str):
        raise ProbeError("ENVIRONMENT.lock destination_prefix is absent")
    prefix = (workspace / prefix_value).resolve(strict=True)
    if not _is_within(torch_site, prefix):
        raise ProbeError("Torch site-packages differs from ENVIRONMENT.lock prefix")
    identity = torch_tree_identity(torch_site, torch_site / "torch", dist_infos[0])
    if identity != {
        "torch_tree_sha256": TORCH_TREE_SHA256,
        "torch_tree_files": TORCH_TREE_FILES,
        "torch_tree_bytes": TORCH_TREE_BYTES,
    }:
        raise ProbeError("complete Torch package tree differs from frozen identity")
    if _single_metadata_header(message, "Version") != TORCH_VERSION:
        raise ProbeError("Torch METADATA version mismatch")
    return {
        **identity,
        "dist_info_name": dist_infos[0].name,
        "metadata_sha256": sha256_bytes(metadata_raw),
        "module_sha256": sha256_file(torch_init),
        "version": TORCH_VERSION,
    }


def validate_torch_runtime_identity(
    torch_site: Path,
    environment_lock: dict[str, object],
    *,
    workspace: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    torch_root = torch_site / "torch"
    dist_infos = sorted(torch_site.glob("torch-*.dist-info"))
    if len(dist_infos) != 1:
        raise ProbeError("supplied Torch site-packages changed during identity capture")
    structure_before = torch_tree_structure_identity(
        torch_site, torch_root, dist_infos[0]
    )
    frozen_identity = validate_torch_site(
        torch_site, environment_lock, workspace=workspace
    )
    structure_after = torch_tree_structure_identity(
        torch_site, torch_root, dist_infos[0]
    )
    if structure_before != structure_after:
        raise ProbeError("complete Torch tree changed while its identity was captured")
    return frozen_identity, {
        "frozen_identity": frozen_identity,
        "structure_identity": structure_after,
    }


def _runtime_view_exclusion_reason(path: Path) -> str | None:
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
        lower == "triton"
        or lower == "triton.py"
        or re.fullmatch(r"triton[-_.].*\.(?:dist|egg)-info", lower) is not None
        or re.fullmatch(r"triton[-_.].*\.egg-info", lower) is not None
    ):
        return "ambient-triton-carrier"
    if path.is_dir() and path.name.endswith(".dist-info"):
        direct_url = path / "direct_url.json"
        if direct_url.is_file():
            try:
                value = json.loads(direct_url.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return "invalid-direct-url-metadata"
            if not isinstance(value, dict):
                return "invalid-direct-url-metadata"
            directory_info = value.get("dir_info")
            if directory_info is not None and not isinstance(directory_info, dict):
                return "invalid-direct-url-metadata"
            if isinstance(directory_info, dict) and directory_info.get("editable") is True:
                return "editable-direct-url-metadata"
    return None


def create_torch_runtime_view(
    torch_site: Path, prefix: Path
) -> tuple[Path, dict[str, object]]:
    """Expose Torch dependencies without exposing ambient site processing.

    A site-packages directory is indivisible on ``sys.path``: adding the
    original directory would also expose its editable Triton dist-info even
    under ``-S``.  The view is a fresh-prefix symlink projection that omits all
    site startup/editable carriers and every ambient Triton carrier.  Module
    provenance still resolves to the explicit workspace-owned Torch site and
    is checked separately by the runtime probe.
    """

    view = prefix / ".torch-runtime-view"
    try:
        view.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ProbeError("Torch runtime view destination is not fresh") from error
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    for source in sorted(torch_site.iterdir(), key=lambda path: path.name):
        reason = _runtime_view_exclusion_reason(source)
        if reason is not None:
            excluded.append({"name": source.name, "reason": reason})
            continue
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ProbeError(f"Torch runtime dependency is a broken link: {source}") from error
        if not _is_within(resolved, torch_site):
            raise ProbeError(f"Torch runtime dependency escaped site-packages: {source}")
        destination = view / source.name
        destination.symlink_to(source, target_is_directory=source.is_dir())
        included.append(source.name)
    if "torch" not in included or not any(
        re.fullmatch(r"torch-.*\.dist-info", name, flags=re.IGNORECASE)
        for name in included
    ):
        raise ProbeError("Torch runtime view omitted the Torch package/distribution")
    if any(_runtime_view_exclusion_reason(view / name) is not None for name in included):
        raise ProbeError("Torch runtime view retained a forbidden site carrier")
    projection_identity = torch_runtime_view_identity(view, torch_site)
    return view, {
        "excluded_entries": excluded,
        "included_entries": included,
        "included_entries_count": len(included),
        "included_entries_sha256": sha256_bytes(
            canonical_json(included).encode("ascii")
        ),
        **projection_identity,
    }


def torch_runtime_view_identity(
    view: Path, torch_site: Path
) -> dict[str, object]:
    """Bind every link in the filtered Torch dependency projection."""

    if not view.is_dir() or view.is_symlink():
        raise ProbeError("Torch runtime view is absent or indirect")
    entries: list[dict[str, object]] = []
    for path in sorted(view.iterdir(), key=lambda item: item.name):
        if not path.is_symlink():
            raise ProbeError(
                f"Torch runtime view entry is not a projection symlink: {path.name}"
            )
        expected_source = torch_site / path.name
        if Path(os.readlink(path)) != expected_source:
            raise ProbeError(
                f"Torch runtime view entry changed projection source: {path.name}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ProbeError(
                f"Torch runtime view entry is a broken link: {path.name}"
            ) from error
        if not _is_within(resolved, torch_site):
            raise ProbeError(
                f"Torch runtime view entry escaped site-packages: {path.name}"
            )
        entries.append(
            {
                "mode": f"{stat.S_IMODE(path.lstat().st_mode):04o}",
                "name": path.name,
                "target": resolved.relative_to(torch_site).as_posix(),
            }
        )
    if not entries:
        raise ProbeError("Torch runtime view projection is empty")
    return {
        "projection_entries": entries,
        "projection_entries_count": len(entries),
        "projection_entries_sha256": sha256_bytes(
            canonical_json(entries).encode("ascii")
        ),
        "projection_root_mode": f"{stat.S_IMODE(view.stat().st_mode):04o}",
    }


def _environment_lock_identity(
    path: Path, workspace: Path
) -> tuple[dict[str, object], dict[str, object]]:
    document, raw = load_canonical_json(path, "environment lock")
    return document, {
        "path": _workspace_relative(path, workspace),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def capture_runtime_input_identity(
    *,
    base_python: Path,
    environment_lock_path: Path,
    torch_site: Path,
    torch_runtime_view: Path,
    workspace: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Recompute every mutable input exposed to the runtime subprocesses."""

    environment_lock, environment_identity = _environment_lock_identity(
        environment_lock_path, workspace
    )
    _, torch_runtime_identity = validate_torch_runtime_identity(
        torch_site, environment_lock, workspace=workspace
    )
    return environment_lock, {
        "base_python": _path_identity(base_python, workspace),
        "environment_lock": environment_identity,
        "torch_runtime_view": torch_runtime_view_identity(
            torch_runtime_view, torch_site
        ),
        "torch_site_packages": torch_runtime_identity,
    }


def require_runtime_input_identity_unchanged(
    before: dict[str, object],
    *,
    base_python: Path,
    environment_lock_path: Path,
    torch_site: Path,
    torch_runtime_view: Path,
    workspace: Path,
) -> dict[str, object]:
    _, after = capture_runtime_input_identity(
        base_python=base_python,
        environment_lock_path=environment_lock_path,
        torch_site=torch_site,
        torch_runtime_view=torch_runtime_view,
        workspace=workspace,
    )
    for name in (
        "base_python",
        "environment_lock",
        "torch_site_packages",
        "torch_runtime_view",
    ):
        if after[name] != before[name]:
            raise ProbeError(f"runtime probes changed {name.replace('_', ' ')} identity")
    return after


def run_probe(
    request: ProbeRequest,
    *,
    limits: ProbeLimits = ProbeLimits(),
) -> tuple[dict[str, object], str]:
    if limits.timeout_seconds <= 0 or limits.timeout_seconds > 600:
        raise ProbeError("probe timeout must be between 1 and 600 seconds")
    workspace = require_workspace(request.workspace)
    wheel_path = require_workspace_file(request.wheel, workspace, "wheel")
    audit_path = require_workspace_file(
        request.wheel_audit_evidence, workspace, "wheel-audit evidence"
    )
    base_python = require_workspace_executable(
        request.base_python, workspace, "base Python"
    )
    torch_site = require_workspace_directory(
        request.torch_site_packages, workspace, "Torch site-packages"
    )
    environment_lock_path = require_workspace_file(
        request.environment_lock, workspace, "environment lock"
    )
    prefix = require_fresh_workspace_prefix(request.probe_prefix, workspace)
    evidence_path = require_evidence_output(request.evidence, workspace)
    environment_lock, environment_lock_identity = _environment_lock_identity(
        environment_lock_path, workspace
    )
    expected_python_value = environment_lock.get("python_executable")
    if not isinstance(expected_python_value, str):
        raise ProbeError("ENVIRONMENT.lock python_executable is absent")
    expected_python = Path(expected_python_value).resolve(strict=True)
    if (
        base_python != expected_python
        or sha256_file(base_python) != BASE_PYTHON_SHA256
        or environment_lock.get("python_abi") != "cp314"
        or environment_lock.get("python_implementation") != "CPython"
        or not str(environment_lock.get("python", "")).startswith("3.14.6 ")
    ):
        raise ProbeError("base Python differs from frozen environment identity")
    torch_identity, torch_runtime_identity = validate_torch_runtime_identity(
        torch_site, environment_lock, workspace=workspace
    )
    base_python_identity = _path_identity(base_python, workspace)

    audit_document, audit_raw = load_canonical_json(
        audit_path, "wheel-audit evidence"
    )
    anchor = validate_audit_anchor(
        audit_document,
        audit_raw,
        expected_evidence_sha256=request.expected_wheel_audit_evidence_sha256,
        wheel_path=wheel_path,
        workspace=workspace,
        limits=limits,
    )

    python = create_fresh_venv(base_python, prefix, limits=limits)
    scheme = query_install_scheme(python, prefix, limits=limits)
    installation = install_audited_wheel(
        wheel_path, scheme, anchor, limits=limits
    )
    installed_verification_before = verify_installed_wheel(
        scheme, wheel_path, anchor, installation
    )
    torch_runtime_view, torch_runtime_view_evidence = create_torch_runtime_view(
        torch_site, prefix
    )
    runtime_input_identity_before = {
        "base_python": base_python_identity,
        "environment_lock": environment_lock_identity,
        "torch_runtime_view": torch_runtime_view_identity(
            torch_runtime_view, torch_site
        ),
        "torch_site_packages": torch_runtime_identity,
    }

    process_reports = [
        run_runtime_probe(
            python,
            scheme,
            torch_site,
            torch_runtime_view,
            anchor,
            limits=limits,
        )
        for _ in range(2)
    ]
    keys = [report["keys"]["torch_inductor"] for report in process_reports]  # type: ignore[index]
    if len(keys) != 2 or keys[0] != keys[1]:
        raise ProbeError("Triton key is not stable in two independent processes")
    if process_reports[0] != process_reports[1]:
        raise ProbeError("independent Triton runtime provenance reports differ")
    runtime_input_identity_after = require_runtime_input_identity_unchanged(
        runtime_input_identity_before,
        base_python=base_python,
        environment_lock_path=environment_lock_path,
        torch_site=torch_site,
        torch_runtime_view=torch_runtime_view,
        workspace=workspace,
    )
    installed_verification_after = verify_installed_wheel(
        scheme, wheel_path, anchor, installation
    )
    if installed_verification_before != installed_verification_after:
        raise ProbeError("runtime probes changed the installed Triton tree")
    if sha256_file(wheel_path) != anchor["wheel_sha256"]:
        raise ProbeError("wheel changed while the fresh probe was running")
    if sha256_file(audit_path) != anchor["audit_evidence_sha256"]:
        raise ProbeError("wheel-audit evidence changed while probing")

    evidence: dict[str, object] = {
        "acceptance": "accepted",
        "inputs": {
            "base_python": base_python_identity,
            "torch_site_packages": {
                **torch_identity,
                "path": _workspace_relative(torch_site, workspace),
            },
            "environment_lock": {
                **environment_lock_identity,
            },
            "wheel": {
                "audit_evidence": {
                    "path": _workspace_relative(audit_path, workspace),
                    "sha256": anchor["audit_evidence_sha256"],
                    "size": anchor["audit_evidence_size"],
                },
                "filename": wheel_path.name,
                "path": _workspace_relative(wheel_path, workspace),
                "sha256": anchor["wheel_sha256"],
                "size": anchor["wheel_size"],
            },
        },
        "installation": {
            **installation,
            "fresh_prefix": True,
            "method": "stdlib-safe-wheel-installer",
            "prefix": _workspace_relative(prefix, workspace),
            "python": _workspace_relative(python, workspace),
            "python_version": list(scheme.python_version),
            "record_verification": installed_verification_after,
            "scheme": {
                name: _workspace_relative(getattr(scheme, name), workspace)
                for name in ("data", "headers", "platlib", "purelib", "scripts")
            },
            "torch_runtime_view": {
                **torch_runtime_view_evidence,
                "path": _workspace_relative(torch_runtime_view, workspace),
            },
        },
        "probe": PROBE_NAME,
        "runtime": {
            "gpu_execution": False,
            "input_integrity": {
                "after": runtime_input_identity_after,
                "before": runtime_input_identity_before,
                "stable": True,
            },
            "processes": process_reports,
            "processes_count": 2,
            "triton_key": {
                "sha256": sha256_bytes(keys[0].encode("utf-8")),
                "stable_across_processes": True,
                "value": keys[0],
            },
        },
        "schema_version": EVIDENCE_SCHEMA_VERSION,
    }
    evidence_sha256 = publish_canonical_json_no_replace(evidence_path, evidence)
    return evidence, evidence_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-audit-evidence", type=Path, required=True)
    parser.add_argument(
        "--expected-wheel-audit-evidence-sha256", required=True
    )
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--torch-site-packages", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--probe-prefix", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = ProbeRequest(
        workspace=args.workspace,
        wheel=args.wheel,
        wheel_audit_evidence=args.wheel_audit_evidence,
        expected_wheel_audit_evidence_sha256=(
            args.expected_wheel_audit_evidence_sha256
        ),
        base_python=args.base_python,
        torch_site_packages=args.torch_site_packages,
        environment_lock=args.environment_lock,
        probe_prefix=args.probe_prefix,
        evidence=args.evidence,
    )
    try:
        _, evidence_sha256 = run_probe(
            request,
            limits=ProbeLimits(timeout_seconds=args.timeout_seconds),
        )
    except (ProbeError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Triton wheel probe failed: {error}", file=sys.stderr)
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
