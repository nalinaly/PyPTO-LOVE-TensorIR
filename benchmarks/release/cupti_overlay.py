"""Hash-locked cupti-python overlay for formal CUPTI collection."""

from __future__ import annotations

from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

from .workload import ReleaseContractError, atomic_json, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_LOCK = ROOT / "environment/python-artifacts.json"
DEFAULT_WHEELHOUSE = ROOT / "caches/release-wheelhouse"
TOOLKIT_ROOT = Path("/usr/local/cuda-13.3")
IDENTITY_NAME = ".identity.json"


def _contract() -> tuple[dict[str, object], dict[str, object]]:
    lock = read_json(ARTIFACT_LOCK)
    artifacts = lock.get("artifacts")
    overlay = lock.get("profiling_overlay")
    if (
        lock.get("schema") != 1
        or lock.get("release") != "qwen35-sm120-v1"
        or not isinstance(artifacts, dict)
        or not isinstance(overlay, dict)
    ):
        raise ReleaseContractError("CUPTI overlay artifact lock is invalid")
    if set(overlay) != {
        "artifact",
        "destination",
        "distribution",
        "libcupti_relative",
        "python_abi",
        "toolkit",
        "toolkit_root_label",
        "version",
    }:
        raise ReleaseContractError("CUPTI overlay contract has an unknown schema")
    if (
        overlay.get("artifact") != "cupti_python"
        or overlay.get("distribution") != "cupti-python"
        or overlay.get("python_abi") != "cp314"
        or overlay.get("toolkit") != "13.3.73"
        or overlay.get("toolkit_root_label") != "cuda-13.3"
    ):
        raise ReleaseContractError("CUPTI overlay identity differs")
    artifact = artifacts.get("cupti_python")
    if not isinstance(artifact, dict) or set(artifact) != {
        "bytes",
        "filename",
        "purpose",
        "sha256",
        "url",
        "version",
    }:
        raise ReleaseContractError("cupti-python artifact metadata is invalid")
    if (
        artifact.get("version") != overlay.get("version")
        or artifact.get("purpose") != "profiling-overlay"
        or type(artifact.get("bytes")) is not int
        or int(artifact["bytes"]) <= 0
        or type(artifact.get("sha256")) is not str
        or len(str(artifact["sha256"])) != 64
        or not str(artifact.get("url", "")).startswith("https://")
    ):
        raise ReleaseContractError("cupti-python artifact identity differs")
    return artifact, overlay


def _destination(overlay: dict[str, object]) -> Path:
    relative = Path(str(overlay["destination"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseContractError("CUPTI overlay destination is not relative")
    destination = (ROOT / relative).resolve()
    if ROOT.resolve() not in destination.parents:
        raise ReleaseContractError("CUPTI overlay destination escaped workspace")
    return destination


def _tree_identity(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.name == IDENTITY_NAME:
            continue
        if path.is_symlink():
            raise ReleaseContractError(f"CUPTI overlay contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
        count += 1
        total += len(content)
    if count == 0:
        raise ReleaseContractError("CUPTI overlay is empty")
    return {
        "file_count": count,
        "bytes": total,
        "content_tree_sha256": digest.hexdigest(),
    }


def _validate_extracted(root: Path, version: str) -> dict[str, object]:
    module = root / "cupti/cupti.cpython-314-x86_64-linux-gnu.so"
    metadata_path = root / f"cupti_python-{version}.dist-info/METADATA"
    license_path = root / f"cupti_python-{version}.dist-info/licenses/LICENSE"
    for path in (module, metadata_path, license_path):
        if path.is_symlink() or not path.is_file():
            raise ReleaseContractError(f"CUPTI overlay file is missing: {path}")
    metadata = BytesParser().parsebytes(metadata_path.read_bytes())
    requirements = metadata.get_all("Requires-Dist", [])
    if (
        metadata.get("Name") != "cupti-python"
        or metadata.get("Version") != version
        or not any(
            value.startswith("nvidia-cuda-cupti~=13.2") for value in requirements
        )
    ):
        raise ReleaseContractError("cupti-python wheel metadata differs")
    return _tree_identity(root)


def _wheel_path(
    artifact: dict[str, object],
    wheelhouse: Path,
    *,
    allow_download: bool,
) -> Path:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    path = wheelhouse / str(artifact["filename"])
    if not path.is_file() and allow_download:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".cupti-python-download-", dir=wheelhouse
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            urllib.request.urlretrieve(str(artifact["url"]), temporary)
            if (
                temporary.stat().st_size != artifact["bytes"]
                or sha256_file(temporary) != artifact["sha256"]
            ):
                raise ReleaseContractError(
                    "downloaded cupti-python wheel identity differs"
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != artifact["bytes"]
        or sha256_file(path) != artifact["sha256"]
    ):
        raise ReleaseContractError(f"locked cupti-python wheel is missing: {path}")
    return path


def validate_overlay() -> dict[str, object]:
    artifact, overlay = _contract()
    destination = _destination(overlay)
    identity_path = destination / IDENTITY_NAME
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or not identity_path.is_file()
    ):
        raise ReleaseContractError("CUPTI overlay has not been materialized")
    identity = read_json(identity_path)
    observed = _validate_extracted(destination, str(overlay["version"]))
    expected = {
        "schema": 1,
        "kind": "cupti-python-profiling-overlay",
        "artifact_sha256": artifact["sha256"],
        "filename": artifact["filename"],
        "version": overlay["version"],
        **observed,
    }
    if identity != expected:
        raise ReleaseContractError("CUPTI overlay identity file differs")
    library = (TOOLKIT_ROOT / str(overlay["libcupti_relative"])).resolve(strict=True)
    if TOOLKIT_ROOT.resolve() not in library.parents or not library.is_file():
        raise ReleaseContractError("CUPTI library escaped the locked CUDA toolkit")
    return {
        "artifact_lock": ARTIFACT_LOCK.relative_to(ROOT).as_posix(),
        "artifact_lock_sha256": sha256_file(ARTIFACT_LOCK),
        "destination": destination.relative_to(ROOT).as_posix(),
        "identity": identity,
        "identity_path": identity_path.relative_to(ROOT).as_posix(),
        "identity_sha256": sha256_file(identity_path),
        "libcupti_path": str(library),
        "libcupti_sha256": sha256_file(library),
    }


def materialize_overlay(
    wheelhouse: Path | None = None, *, allow_download: bool = False
) -> dict[str, object]:
    artifact, overlay = _contract()
    destination = _destination(overlay)
    if destination.exists():
        return validate_overlay()
    wheel = _wheel_path(
        artifact,
        (wheelhouse or DEFAULT_WHEELHOUSE).resolve(),
        allow_download=allow_download,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cupti-overlay-", dir=destination.parent))
    try:
        with zipfile.ZipFile(wheel) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or not path.parts or ".." in path.parts:
                    raise ReleaseContractError("cupti-python wheel path is unsafe")
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ReleaseContractError("cupti-python wheel contains a symlink")
                if path.parts[0] not in {
                    "cupti",
                    f"cupti_python-{overlay['version']}.dist-info",
                }:
                    continue
                archive.extract(item, temporary)
        observed = _validate_extracted(temporary, str(overlay["version"]))
        identity = {
            "schema": 1,
            "kind": "cupti-python-profiling-overlay",
            "artifact_sha256": artifact["sha256"],
            "filename": artifact["filename"],
            "version": overlay["version"],
            **observed,
        }
        atomic_json(temporary / IDENTITY_NAME, identity)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_overlay()


def activate_overlay() -> dict[str, object]:
    record = validate_overlay()
    destination = str((ROOT / str(record["destination"])).resolve(strict=True))
    if destination not in sys.path:
        sys.path.insert(0, destination)
    os.environ["TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"] = str(record["libcupti_path"])
    module = importlib.import_module("cupti.cupti")
    module_path = Path(str(module.__file__)).resolve(strict=True)
    overlay_root = Path(destination)
    if overlay_root not in module_path.parents:
        raise ReleaseContractError("cupti-python import escaped the locked overlay")
    if importlib.metadata.version("cupti-python") != "13.2.0":
        raise ReleaseContractError("cupti-python overlay version differs")
    record["module_path"] = module_path.relative_to(ROOT).as_posix()
    record["module_sha256"] = sha256_file(module_path)
    return record
