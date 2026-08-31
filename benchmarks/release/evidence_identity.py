"""Fail-closed identities shared by all formal release measurements.

The release report must bind the bytes that were measured, not merely file
sizes or human-readable version strings.  This module deliberately avoids any
CUDA runtime call: ``nvidia-smi`` supplies the immutable device identity and
the PyPTO compiler build-info query is documented to be device independent.
It is therefore safe to call before CUPTI collection starts.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from .workload import (
    ReleaseContractError,
    canonical_json_sha256,
    read_json,
    sha256_file,
)


IDENTITY_SCHEMA_VERSION = 1
REQUIRED_CANDIDATE_DISTRIBUTIONS = (
    "pypto",
    "pypto-framework-plugins",
    "pypto-kernels",
)
COMPILER_FIELDS = (
    "compiled",
    "compiler_factory_available",
    "pypto_revision",
    "tensor_ir_revision",
    "cuda_tile_revision",
    "llvm_revision",
    "cuda_toolkit_root",
    "cuda_toolkit_version",
    "tileiras_real_path",
    "tileiras_version",
    "tileiras_sha256",
    "sm120_target",
)


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ReleaseContractError(f"release input escaped the workspace: {resolved}")
    return resolved.relative_to(resolved_root).as_posix()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseContractError(f"{label} is not a lowercase SHA-256")
    return value


def _hash_file_set(
    base: Path,
    paths: Iterable[Path],
    sha_cache: dict[Path, str] | None = None,
) -> dict[str, object]:
    """Hash names and complete contents of a deterministic regular-file set."""

    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(paths, key=lambda item: item.relative_to(base).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ReleaseContractError(f"identity input is not a regular file: {path}")
        relative = path.relative_to(base).as_posix()
        content_sha256 = (
            sha_cache.get(path) if sha_cache is not None else None
        ) or sha256_file(path)
        if sha_cache is not None:
            sha_cache[path] = content_sha256
        size = path.stat().st_size
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(content_sha256))
        count += 1
        total += size
    if count == 0:
        raise ReleaseContractError(f"identity file set is empty: {base}")
    return {
        "file_count": count,
        "bytes": total,
        "content_tree_sha256": digest.hexdigest(),
    }


def collect_model_identity(
    root: Path, model_path: Path, *, model_name: str | None = None
) -> dict[str, object]:
    """Verify the exact manifest and SHA-256 of every model file."""

    root = root.resolve()
    manifest_path = (root / "models/MANIFEST.json").resolve(strict=True)
    manifest = read_json(manifest_path)
    models = manifest.get("models")
    if manifest.get("schema") != 1 or not isinstance(models, dict):
        raise ReleaseContractError("model MANIFEST has an unknown schema")
    resolved_model = model_path.resolve(strict=True)
    if model_name is None:
        matches = []
        for name, candidate in models.items():
            if (
                not isinstance(candidate, dict)
                or type(candidate.get("destination")) is not str
            ):
                continue
            destination = (root / str(candidate["destination"])).resolve()
            if destination == resolved_model:
                matches.append(str(name))
        if len(matches) != 1:
            raise ReleaseContractError(
                "model path must match exactly one MANIFEST destination: "
                f"{resolved_model}"
            )
        model_name = matches[0]
    record = models.get(model_name)
    if not isinstance(record, dict):
        raise ReleaseContractError(f"model MANIFEST has no {model_name} record")
    expected_destination = root / str(record.get("destination"))
    if resolved_model != expected_destination.resolve(strict=True):
        raise ReleaseContractError(
            f"model path does not match MANIFEST destination: {resolved_model}"
        )
    expected_files = record.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ReleaseContractError("model MANIFEST file set is empty")
    actual_names = {
        path.relative_to(resolved_model).as_posix()
        for path in resolved_model.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_names != set(expected_files):
        raise ReleaseContractError(
            "model file set differs from MANIFEST: "
            f"missing={sorted(set(expected_files) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected_files))}"
        )
    verified = []
    for name in sorted(expected_files):
        if (
            type(name) is not str
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ReleaseContractError(f"invalid model MANIFEST path: {name!r}")
        expected = expected_files[name]
        if not isinstance(expected, dict):
            raise ReleaseContractError(f"invalid model MANIFEST entry: {name}")
        raw_path = resolved_model / name
        if raw_path.is_symlink():
            raise ReleaseContractError(
                f"model input must not be a symbolic link: {raw_path}"
            )
        path = raw_path.resolve(strict=True)
        if resolved_model not in path.parents:
            raise ReleaseContractError(f"model file escaped its destination: {path}")
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ReleaseContractError(
                f"model input must be an ordinary single-link file: {path}"
            )
        expected_size = expected.get("bytes")
        expected_sha256 = _require_sha256(
            expected.get("sha256"), f"model MANIFEST {name} SHA-256"
        )
        actual_sha256 = sha256_file(path)
        if type(expected_size) is not int or path.stat().st_size != expected_size:
            raise ReleaseContractError(f"model file size differs: {path}")
        if actual_sha256 != expected_sha256:
            raise ReleaseContractError(f"model file SHA-256 differs: {path}")
        verified.append(
            {"path": name, "bytes": expected_size, "sha256": actual_sha256}
        )
    identity = {
        "name": model_name,
        "repository_id": record.get("repository_id"),
        "revision": record.get("revision"),
        "destination": _relative(root, resolved_model),
        "manifest": {
            "path": _relative(root, manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "files": verified,
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _environment_lock(root: Path, prefix_relative: str) -> dict[str, object]:
    path = (root / prefix_relative / ".identity-lock.json").resolve(strict=True)
    payload = read_json(path)
    required = (
        "release",
        "python_abi",
        "torch",
        "torch_git",
        "cuda",
        "hip",
        "torch_tree_sha256",
        "distributions_sha256",
    )
    if payload.get("release") != "qwen35-sm120-v1" or any(
        key not in payload for key in required
    ):
        raise ReleaseContractError(f"formal environment lock is incomplete: {path}")
    _require_sha256(payload["torch_tree_sha256"], f"{prefix_relative} Torch tree")
    _require_sha256(
        payload["distributions_sha256"], f"{prefix_relative} distributions"
    )
    selected = {key: payload[key] for key in required}
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "identity": selected,
    }


def collect_environment_locks(root: Path) -> dict[str, object]:
    runtime_path = (root / "benchmarks/release/runtime.json").resolve(strict=True)
    runtime = read_json(runtime_path)
    profiles = runtime.get("profiles")
    if runtime.get("schema") != 1 or not isinstance(profiles, dict):
        raise ReleaseContractError("release runtime manifest has an unknown schema")
    result: dict[str, object] = {}
    for profile in ("pypto", "baseline"):
        record = profiles.get(profile)
        if not isinstance(record, dict) or type(record.get("prefix")) is not str:
            raise ReleaseContractError(f"runtime profile is incomplete: {profile}")
        result[profile] = _environment_lock(root, str(record["prefix"]))
    result["manifest"] = {
        "path": _relative(root, runtime_path),
        "sha256": sha256_file(runtime_path),
    }
    return result


def _site_packages(prefix: Path) -> Path:
    candidates = sorted((prefix / "lib").glob("python*/site-packages"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise ReleaseContractError(
            f"expected one site-packages directory below {prefix}, got {candidates}"
        )
    return candidates[0].resolve(strict=True)


def _distribution_identity(
    prefix: Path,
    site_packages: Path,
    normalized_name: str,
    sha_cache: dict[Path, str],
) -> dict[str, object]:
    spelling = normalized_name.replace("-", "_")
    dist_infos = sorted(site_packages.glob(f"{spelling}-*.dist-info"))
    if len(dist_infos) != 1:
        raise ReleaseContractError(
            f"expected one {normalized_name} dist-info, got {dist_infos}"
        )
    dist_info = dist_infos[0]
    metadata_path = dist_info / "METADATA"
    record_path = dist_info / "RECORD"
    if not metadata_path.is_file() or not record_path.is_file():
        raise ReleaseContractError(f"distribution metadata is incomplete: {dist_info}")
    version = None
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
            break
    if not version:
        raise ReleaseContractError(f"distribution has no version: {metadata_path}")
    files: set[Path] = {
        path
        for path in dist_info.rglob("*")
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
    }
    with record_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0].endswith(".pyc") or "__pycache__" in Path(row[0]).parts:
                continue
            raw_path = site_packages / row[0]
            if raw_path.is_symlink():
                raise ReleaseContractError(
                    f"distribution RECORD points to a symbolic link: {row[0]}"
                )
            path = raw_path.resolve()
            if path != prefix and prefix not in path.parents:
                raise ReleaseContractError(
                    f"distribution RECORD escaped the formal prefix: {row[0]}"
                )
            if path.is_file():
                files.add(path)
    hashed = _hash_file_set(prefix, files, sha_cache)
    return {
        "version": version,
        "dist_info": dist_info.relative_to(site_packages).as_posix(),
        **hashed,
    }


def _candidate_package_identity(root: Path) -> dict[str, object]:
    prefix = (root / "envs/pypto-release").resolve(strict=True)
    site_packages = _site_packages(prefix)
    package = (site_packages / "pypto").resolve(strict=True)
    dso_candidates = sorted(package.glob("pypto_core*.so"))
    if len(dso_candidates) != 1:
        raise ReleaseContractError(f"installed PyPTO DSO is ambiguous: {dso_candidates}")
    dso = dso_candidates[0]
    dso_sha256 = sha256_file(dso)
    sha_cache = {dso: dso_sha256}
    distributions = {
        name: _distribution_identity(prefix, site_packages, name, sha_cache)
        for name in REQUIRED_CANDIDATE_DISTRIBUTIONS
    }
    return {
        "prefix": _relative(root, prefix),
        "dso": {
            "path": _relative(root, dso),
            "bytes": dso.stat().st_size,
            "sha256": dso_sha256,
        },
        "distributions": distributions,
    }


def _git_identity(root: Path, repository: Path) -> dict[str, object]:
    values = {}
    for name, arguments in (
        ("commit", ("rev-parse", "HEAD")),
        ("tree", ("rev-parse", "HEAD^{tree}")),
        ("status", ("status", "--porcelain=v1", "--untracked-files=all")),
    ):
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise ReleaseContractError(
                completed.stderr.strip() or f"cannot inspect {repository}"
            )
        values[name] = completed.stdout.strip()
    if values["status"]:
        raise ReleaseContractError(f"release source checkout is dirty: {repository}")
    return {
        "path": _relative(root, repository),
        "commit": values["commit"],
        "tree": values["tree"],
        "clean": True,
    }


def _source_identity(root: Path) -> dict[str, object]:
    lock_path = (root / "vendor/source-lock.json").resolve(strict=True)
    lock = read_json(lock_path)
    repositories = lock.get("repositories")
    if lock.get("schema") != 1 or not isinstance(repositories, dict):
        raise ReleaseContractError("source release lock has an unknown schema")
    result = {
        "lock": {"path": _relative(root, lock_path), "sha256": sha256_file(lock_path)}
    }
    materialization = lock.get("materialization")
    if not isinstance(materialization, dict):
        raise ReleaseContractError("source materialization record is missing")
    for name in ("pypto", "tensor_ir", "sglang"):
        record = repositories.get(name)
        relative = materialization.get(name)
        if not isinstance(record, dict) or type(relative) is not str:
            raise ReleaseContractError(f"source identity is missing: {name}")
        repository = (root / ".sources" / relative).resolve(strict=True)
        observed = _git_identity(root, repository)
        expected_commit = record.get("head_commit")
        expected_tree = record.get("head_tree")
        if (
            observed["commit"] != expected_commit
            or observed["tree"] != expected_tree
        ):
            raise ReleaseContractError(f"{name} checkout differs from source lock")
        result[name] = observed
        if name == "sglang":
            version = record.get("version")
            if type(version) is not str or not version:
                raise ReleaseContractError("SGLang release version is missing")
            result[name]["version"] = version.removeprefix("v")
    submodules = lock.get("pypto_submodules")
    if type(submodules) is not list:
        raise ReleaseContractError("PyPTO submodule identities are missing")
    by_path = {
        item.get("path"): item for item in submodules if isinstance(item, dict)
    }
    for name, relative in (
        ("cuda_tile", "3rdparty/nvidia/cuda-tile"),
        ("llvm", "3rdparty/nvidia/llvm-project"),
    ):
        record = by_path.get(relative)
        if not isinstance(record, dict):
            raise ReleaseContractError(f"source identity is missing: {name}")
        repository = (root / ".sources/pypto" / relative).resolve(strict=True)
        observed = _git_identity(root, repository)
        if (
            observed["commit"] != record.get("commit")
            or observed["tree"] != record.get("tree")
        ):
            raise ReleaseContractError(f"{name} checkout differs from source lock")
        result[name] = observed
    return result


def _gpu_identity() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    timeout_error: subprocess.TimeoutExpired | None = None
    completed = None
    for timeout_seconds in (10, 30):
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
            break
        except subprocess.TimeoutExpired as error:
            timeout_error = error
    if completed is None:
        raise ReleaseContractError(
            "nvidia-smi identity query timed out after bounded retries: "
            f"{timeout_error}"
        ) from timeout_error
    if completed.returncode != 0:
        raise ReleaseContractError(
            completed.stderr.strip() or "nvidia-smi identity query failed"
        )
    rows = [row for row in csv.reader(completed.stdout.splitlines()) if row]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ReleaseContractError(
            f"release requires exactly one NVIDIA GPU, got {completed.stdout!r}"
        )
    name, uuid, capability, memory_mib, driver = (item.strip() for item in rows[0])
    if not uuid or capability != "12.0" or not memory_mib.isdigit() or not driver:
        raise ReleaseContractError("NVIDIA identity is incomplete or not SM120")
    return {
        "name": name,
        "uuid": uuid,
        "compute_capability": capability,
        "total_memory_mib": int(memory_mib),
        "driver": driver,
    }


def _compiler_identity() -> dict[str, object]:
    try:
        from pypto import compiler

        info = compiler.get_nvidia_backend_build_info()
    except (ImportError, AttributeError, RuntimeError) as error:
        raise ReleaseContractError(f"cannot query PyPTO compiler identity: {error}") from error
    result = {field: getattr(info, field) for field in COMPILER_FIELDS}
    if result["compiled"] is not True or result["compiler_factory_available"] is not True:
        raise ReleaseContractError("installed PyPTO NVIDIA compiler is unavailable")
    for field in COMPILER_FIELDS[2:]:
        if type(result[field]) is not str or not result[field]:
            raise ReleaseContractError(f"PyPTO compiler identity is empty: {field}")
    _require_sha256(result["tileiras_sha256"], "tileiras SHA-256")
    toolkit_root = Path(result["cuda_toolkit_root"]).resolve(strict=True)
    tileiras = Path(result["tileiras_real_path"]).resolve(strict=True)
    if not toolkit_root.is_dir() or not tileiras.is_file():
        raise ReleaseContractError("PyPTO compiler toolkit paths are unavailable")
    if sha256_file(tileiras) != result["tileiras_sha256"]:
        raise ReleaseContractError("live tileiras bytes differ from compiler identity")
    return result


def _runtime_identity(root: Path, lane: str, sources: dict[str, object]) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise ReleaseContractError(f"Torch is unavailable: {error}") from error
    selected_prefix = (
        root / ("envs/pypto-release" if lane == "pypto" else "envs/sglang-baseline")
    ).resolve(strict=True)
    if Path(sys.prefix).resolve() != selected_prefix:
        raise ReleaseContractError(
            f"selected runtime prefix differs: {sys.prefix} != {selected_prefix}"
        )
    torch_file = Path(torch.__file__).resolve(strict=True)
    if selected_prefix not in torch_file.parents:
        raise ReleaseContractError("Torch import escaped the selected formal prefix")
    sglang_source = sources["sglang"]
    if not isinstance(sglang_source, dict):
        raise ReleaseContractError("locked SGLang source identity is unavailable")
    sglang_module = (
        root / str(sglang_source["path"]) / "python/sglang/__init__.py"
    ).resolve(strict=True)
    if str(sglang_module.parent.parent) not in {
        str(Path(value).resolve()) for value in sys.path if value
    }:
        raise ReleaseContractError("locked SGLang source is absent from sys.path")
    return {
        "profile": lane,
        "prefix": _relative(root, selected_prefix),
        "torch": {
            "version": str(torch.__version__),
            "git": str(torch.version.git_version),
            "cuda_toolkit": torch.version.cuda,
            "hip": torch.version.hip,
            "module": _relative(root, torch_file),
        },
        "sglang": {
            "version": sglang_source["version"],
            "module": _relative(root, sglang_module),
            "module_sha256": sha256_file(sglang_module),
            "source_commit": sglang_source["commit"],
            "source_tree": sglang_source["tree"],
        },
    }


def collect_run_identity(root: Path, lane: str, model_path: Path) -> dict[str, object]:
    """Collect and verify one complete, cross-run-comparable evidence identity."""

    if lane not in {"pypto", "baseline"}:
        raise ReleaseContractError(f"unknown evidence identity lane: {lane}")
    root = root.resolve()
    locks = collect_environment_locks(root)
    sources = _source_identity(root)
    compiler = _compiler_identity() if lane == "pypto" else None
    identity: dict[str, object] = {
        "schema": IDENTITY_SCHEMA_VERSION,
        "kind": "qwen35-release-evidence-identity",
        "model": collect_model_identity(root, model_path),
        "environment_locks": locks,
        "selected_environment_lock": lane,
        "candidate_packages": _candidate_package_identity(root),
        "sources": sources,
        "runtime": _runtime_identity(root, lane, sources),
        "compiler": compiler,
        "gpu": _gpu_identity(),
    }
    expected_compiler = {
        "pypto_revision": sources["pypto"]["commit"],
        "tensor_ir_revision": sources["tensor_ir"]["commit"],
        "cuda_tile_revision": sources["cuda_tile"]["commit"],
        "llvm_revision": sources["llvm"]["commit"],
    }
    identity["expected_compiler"] = expected_compiler
    if compiler is not None and any(
        compiler[key] != expected for key, expected in expected_compiler.items()
    ):
        raise ReleaseContractError("PyPTO compiler revisions differ from source lock")
    torch_identity = locks[lane]["identity"]
    runtime_torch = identity["runtime"]["torch"]
    for lock_key, runtime_key in (
        ("torch", "version"),
        ("torch_git", "git"),
        ("cuda", "cuda_toolkit"),
        ("hip", "hip"),
    ):
        if torch_identity[lock_key] != runtime_torch[runtime_key]:
            raise ReleaseContractError(
                f"selected Torch differs from identity lock: {lock_key}"
            )
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def comparable_identity(identity: dict[str, object]) -> dict[str, object]:
    """Return only immutable fields that must agree across every release run."""

    required = {
        "schema",
        "kind",
        "model",
        "environment_locks",
        "candidate_packages",
        "sources",
        "expected_compiler",
        "gpu",
    }
    if identity.get("schema") != IDENTITY_SCHEMA_VERSION or not required.issubset(
        identity
    ):
        raise ReleaseContractError("run evidence identity is incomplete")
    model = identity.get("model")
    if not isinstance(model, dict):
        raise ReleaseContractError("model evidence identity is missing")
    model_digest = model.get("identity_sha256")
    unsigned_model = dict(model)
    unsigned_model.pop("identity_sha256", None)
    if model_digest != canonical_json_sha256(unsigned_model):
        raise ReleaseContractError("model evidence identity digest differs")
    manifest = model.get("manifest")
    files = model.get("files")
    if (
        not isinstance(manifest, dict)
        or type(files) is not list
        or not files
        or len({item.get("path") for item in files if isinstance(item, dict)})
        != len(files)
    ):
        raise ReleaseContractError("model per-file evidence is incomplete")
    _require_sha256(manifest.get("sha256"), "model MANIFEST SHA-256")
    for item in files:
        if (
            not isinstance(item, dict)
            or type(item.get("path")) is not str
            or not item["path"]
            or Path(item["path"]).is_absolute()
            or type(item.get("bytes")) is not int
            or int(item["bytes"]) < 0
        ):
            raise ReleaseContractError("model per-file evidence has an invalid entry")
        _require_sha256(item.get("sha256"), f"model file {item.get('path')} SHA-256")
    locks = identity.get("environment_locks")
    if not isinstance(locks, dict) or not {"pypto", "baseline", "manifest"}.issubset(
        locks
    ):
        raise ReleaseContractError("candidate/baseline environment locks are missing")
    for profile in ("pypto", "baseline"):
        record = locks[profile]
        if not isinstance(record, dict) or not isinstance(record.get("identity"), dict):
            raise ReleaseContractError(f"{profile} environment identity is incomplete")
        _require_sha256(record.get("sha256"), f"{profile} identity-lock SHA-256")
        for field in ("torch_tree_sha256", "distributions_sha256"):
            _require_sha256(record["identity"].get(field), f"{profile} {field}")
    if not isinstance(locks["manifest"], dict):
        raise ReleaseContractError("runtime manifest identity is incomplete")
    _require_sha256(locks["manifest"].get("sha256"), "runtime manifest SHA-256")
    packages = identity.get("candidate_packages")
    if not isinstance(packages, dict) or not isinstance(packages.get("dso"), dict):
        raise ReleaseContractError("candidate package identity is incomplete")
    _require_sha256(packages["dso"].get("sha256"), "PyPTO DSO SHA-256")
    if (
        type(packages["dso"].get("path")) is not str
        or Path(packages["dso"]["path"]).is_absolute()
        or type(packages["dso"].get("bytes")) is not int
        or int(packages["dso"]["bytes"]) <= 0
    ):
        raise ReleaseContractError("PyPTO DSO path/size identity is incomplete")
    distributions = packages.get("distributions")
    if not isinstance(distributions, dict) or set(distributions) != set(
        REQUIRED_CANDIDATE_DISTRIBUTIONS
    ):
        raise ReleaseContractError("candidate distribution identities are incomplete")
    for name, record in distributions.items():
        if not isinstance(record, dict) or type(record.get("file_count")) is not int:
            raise ReleaseContractError(f"candidate distribution is incomplete: {name}")
        _require_sha256(
            record.get("content_tree_sha256"), f"{name} content tree SHA-256"
        )
    sources = identity.get("sources")
    if not isinstance(sources, dict):
        raise ReleaseContractError("source identities are missing")
    for name in ("pypto", "tensor_ir", "sglang", "cuda_tile", "llvm"):
        record = sources.get(name)
        if not isinstance(record, dict) or record.get("clean") is not True:
            raise ReleaseContractError(f"source identity is incomplete: {name}")
        for field in ("commit", "tree"):
            value = record.get(field)
            if (
                type(value) is not str
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ReleaseContractError(f"source {name} {field} is invalid")
    expected = identity.get("expected_compiler")
    expected_values = {
        "pypto_revision": sources["pypto"]["commit"],
        "tensor_ir_revision": sources["tensor_ir"]["commit"],
        "cuda_tile_revision": sources["cuda_tile"]["commit"],
        "llvm_revision": sources["llvm"]["commit"],
    }
    if expected != expected_values:
        raise ReleaseContractError("expected compiler identity differs from sources")
    gpu = identity.get("gpu")
    if (
        not isinstance(gpu, dict)
        or type(gpu.get("uuid")) is not str
        or not gpu["uuid"]
        or gpu.get("compute_capability") != "12.0"
        or type(gpu.get("driver")) is not str
        or type(gpu.get("total_memory_mib")) is not int
    ):
        raise ReleaseContractError("GPU UUID/driver/SM identity is incomplete")
    compiler = identity.get("compiler")
    if compiler is not None:
        if not isinstance(compiler, dict):
            raise ReleaseContractError("compiler identity has an invalid type")
        if set(compiler) != set(COMPILER_FIELDS):
            raise ReleaseContractError("compiler identity fields are incomplete")
        _require_sha256(compiler.get("tileiras_sha256"), "tileiras SHA-256")
        if any(compiler.get(key) != value for key, value in expected_values.items()):
            raise ReleaseContractError("compiler identity differs from source identity")
    return {key: identity[key] for key in sorted(required)}
