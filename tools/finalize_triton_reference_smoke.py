#!/usr/bin/env python3
"""Bind a provisional Triton GPU smoke to an exclusive completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]
SMOKE = "reference-only-triton-sm120"
PROVISIONAL_STATUS = "gpu-execution-complete-awaiting-run-finalization"
PROBE = "triton-fresh-wheel"
EXPECTED_TORCH_VERSION = "2.13.0+cu130"
EXPECTED_TORCH_CUDA = "13.0"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"
EXPECTED_PTXAS_FULL_VERSION = "13.1.80"
EXPECTED_PTXAS_RELEASE = "13.1"
PTXAS_BLACKWELL_RELATIVE = Path(
    "triton/backends/nvidia/bin/ptxas-blackwell"
)
VECTOR_ELEMENTS = 65_537
BLOCK_SIZE = 256


class FinalizeError(RuntimeError):
    """Exclusive-run evidence could not be proven."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FinalizeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"{description} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizeError(f"{description} is not strict JSON") from error
    if not isinstance(value, dict) or text != canonical_json(value):
        raise FinalizeError(f"{description} is not a canonical JSON object")
    return value, raw


def require_workspace(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = lexical.resolve(strict=True)
    if lexical != resolved or resolved != ROOT:
        raise FinalizeError("workspace must be the real configured workspace")
    return resolved


def require_workspace_file(path: Path, workspace: Path, description: str) -> Path:
    if not path.is_absolute():
        raise FinalizeError(f"{description} path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = lexical.resolve(strict=True)
    if lexical != resolved or workspace not in resolved.parents:
        raise FinalizeError(f"{description} must be workspace-owned")
    if resolved.is_symlink() or not resolved.is_file():
        raise FinalizeError(f"{description} must be a regular file")
    return resolved


def publish_no_replace(path: Path, value: object) -> str:
    if not path.is_absolute():
        raise FinalizeError("final evidence path must be absolute")
    parent = path.parent.resolve(strict=True)
    if ROOT not in parent.parents and parent != ROOT:
        raise FinalizeError("final evidence parent must be workspace-owned")
    if path.exists() or path.is_symlink():
        raise FinalizeError("final evidence already exists")
    encoded = canonical_json(value).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FinalizeError("final evidence already exists") from error
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FinalizeError(f"{description} SHA256 is invalid")
    return value


def _count(value: object, description: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise FinalizeError(f"{description} is invalid")
    return value


def _relative_path(
    value: object,
    workspace: Path,
    description: str,
    *,
    directory: bool,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or Path(value).is_absolute()
        or Path(value).as_posix() != value
        or any(part in ("", ".", "..") for part in Path(value).parts)
    ):
        raise FinalizeError(f"{description} path is not canonical workspace-relative")
    lexical = workspace.joinpath(*Path(value).parts)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise FinalizeError(f"{description} is absent") from error
    if lexical != resolved or workspace not in resolved.parents:
        raise FinalizeError(f"{description} escaped the workspace or is indirect")
    if directory:
        if not resolved.is_dir() or resolved.is_symlink():
            raise FinalizeError(f"{description} is not a real directory")
    elif not resolved.is_file() or resolved.is_symlink():
        raise FinalizeError(f"{description} is not a regular file")
    return resolved


def _file_identity(
    value: object,
    workspace: Path,
    description: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size"}:
        raise FinalizeError(f"{description} identity is malformed")
    path = _relative_path(
        value.get("path"), workspace, description, directory=False
    )
    if expected_path is not None and path != expected_path:
        raise FinalizeError(f"{description} path is not the expected input")
    digest = _sha256(value.get("sha256"), description)
    size = _count(value.get("size"), f"{description} size")
    if size != path.stat().st_size or digest != sha256_file(path):
        raise FinalizeError(f"{description} bytes differ from provisional identity")
    return path


def _probe_token(path: Path, prefix: Path) -> str:
    return "$PROBE_PREFIX/" + path.relative_to(prefix).as_posix()


def _expand_probe_token(value: object, prefix: Path, description: str) -> Path:
    marker = "$PROBE_PREFIX/"
    if not isinstance(value, str) or not value.startswith(marker):
        raise FinalizeError(f"{description} is not normalized below $PROBE_PREFIX")
    relative = value.removeprefix(marker)
    if (
        not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or Path(relative).as_posix() != relative
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise FinalizeError(f"{description} probe path is unsafe")
    lexical = prefix.joinpath(*Path(relative).parts)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise FinalizeError(f"{description} is absent") from error
    if prefix not in resolved.parents:
        raise FinalizeError(f"{description} escaped the probe prefix")
    return resolved


def _validate_torch_input(value: object, workspace: Path) -> tuple[dict[str, object], Path]:
    fields = {
        "dist_info_name",
        "metadata_sha256",
        "module_sha256",
        "path",
        "torch_tree_bytes",
        "torch_tree_files",
        "torch_tree_sha256",
        "version",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise FinalizeError("provisional Torch-site identity is malformed")
    path = _relative_path(
        value.get("path"), workspace, "provisional Torch site", directory=True
    )
    if (
        not isinstance(value.get("dist_info_name"), str)
        or not value["dist_info_name"]
        or value.get("version") != EXPECTED_TORCH_VERSION
    ):
        raise FinalizeError("provisional Torch distribution identity is invalid")
    for name in ("metadata_sha256", "module_sha256", "torch_tree_sha256"):
        _sha256(value.get(name), f"provisional Torch {name}")
    _count(value.get("torch_tree_bytes"), "provisional Torch tree bytes", positive=True)
    _count(value.get("torch_tree_files"), "provisional Torch tree files", positive=True)
    return value, path


def _validate_probe_bound_inputs(
    inputs: dict[str, object], workspace: Path
) -> dict[str, object]:
    required_inputs = {
        "base_python",
        "environment_lock",
        "probe_evidence",
        "probe_prefix",
        "probe_site",
        "runner",
        "torch_runtime_view",
        "torch_site_packages",
        "wheel",
    }
    if set(inputs) != required_inputs:
        raise FinalizeError("provisional smoke input set is incomplete")

    runner_path = workspace / "benchmarks/operators/triton_reference_sm120.py"
    _file_identity(
        inputs.get("runner"), workspace, "provisional smoke runner", expected_path=runner_path
    )
    _file_identity(inputs.get("base_python"), workspace, "provisional base Python")
    environment_path = _file_identity(
        inputs.get("environment_lock"), workspace, "provisional environment lock"
    )
    load_canonical(environment_path, "provisional environment lock")
    probe_path = _file_identity(
        inputs.get("probe_evidence"), workspace, "provisional probe evidence"
    )

    wheel = inputs.get("wheel")
    if not isinstance(wheel, dict) or set(wheel) != {
        "audit_evidence",
        "filename",
        "path",
        "sha256",
        "size",
    }:
        raise FinalizeError("provisional wheel identity is malformed")
    wheel_file = _file_identity(
        {name: wheel[name] for name in ("path", "sha256", "size")},
        workspace,
        "provisional wheel",
    )
    if wheel.get("filename") != wheel_file.name:
        raise FinalizeError("provisional wheel filename is inconsistent")
    _file_identity(
        wheel.get("audit_evidence"), workspace, "provisional wheel audit evidence"
    )

    torch_input, torch_site = _validate_torch_input(
        inputs.get("torch_site_packages"), workspace
    )
    prefix = _relative_path(
        inputs.get("probe_prefix"), workspace, "provisional probe prefix", directory=True
    )
    probe_site = _relative_path(
        inputs.get("probe_site"), workspace, "provisional probe site", directory=True
    )
    runtime_view = _relative_path(
        inputs.get("torch_runtime_view"),
        workspace,
        "provisional Torch runtime view",
        directory=True,
    )
    if prefix not in probe_site.parents or prefix not in runtime_view.parents:
        raise FinalizeError("provisional probe site/view escaped the probe prefix")

    probe_document, _ = load_canonical(probe_path, "provisional probe evidence")
    if (
        probe_document.get("acceptance") != "accepted"
        or probe_document.get("probe") != PROBE
        or probe_document.get("schema_version") != 1
    ):
        raise FinalizeError("provisional probe evidence is not accepted schema 1")
    probe_inputs = probe_document.get("inputs")
    if not isinstance(probe_inputs, dict) or any(
        inputs.get(name) != probe_inputs.get(name)
        for name in ("base_python", "environment_lock", "torch_site_packages", "wheel")
    ):
        raise FinalizeError("provisional inputs differ from accepted probe identities")
    installation = probe_document.get("installation")
    if not isinstance(installation, dict):
        raise FinalizeError("accepted probe installation is absent")
    scheme = installation.get("scheme")
    view = installation.get("torch_runtime_view")
    if (
        installation.get("prefix") != inputs.get("probe_prefix")
        or not isinstance(scheme, dict)
        or scheme.get("platlib") != inputs.get("probe_site")
        or not isinstance(view, dict)
        or view.get("path") != inputs.get("torch_runtime_view")
    ):
        raise FinalizeError("provisional probe paths differ from accepted probe")
    probe_runtime = probe_document.get("runtime")
    processes = probe_runtime.get("processes") if isinstance(probe_runtime, dict) else None
    if (
        not isinstance(processes, list)
        or len(processes) != 2
        or processes[0] != processes[1]
        or not isinstance(processes[0], dict)
    ):
        raise FinalizeError("accepted probe runtime anchor is malformed")
    probe_process = processes[0]
    probe_torch = probe_process.get("torch")
    if (
        not isinstance(probe_torch, dict)
        or probe_torch.get("version") != EXPECTED_TORCH_VERSION
        or probe_torch.get("cuda") != EXPECTED_TORCH_CUDA
        or probe_torch.get("hip") is not None
        or not isinstance(probe_torch.get("git_version"), str)
        or not probe_torch.get("git_version")
    ):
        raise FinalizeError("accepted probe Torch runtime identity is malformed")
    return {
        "prefix": prefix,
        "probe_process": probe_process,
        "probe_site": probe_site,
        "runtime_view": runtime_view,
        "torch_input": torch_input,
        "torch_site": torch_site,
    }


def _validate_cache(cache: object, workspace: Path) -> None:
    fields = {
        "artifacts",
        "artifacts_count",
        "artifacts_sha256",
        "cubin_count",
        "fresh_before_run",
        "path",
        "total_bytes",
    }
    if not isinstance(cache, dict) or set(cache) != fields:
        raise FinalizeError("provisional smoke cache schema is malformed")
    if cache.get("fresh_before_run") is not True:
        raise FinalizeError("provisional smoke cache was not fresh")
    cache_root = _relative_path(
        cache.get("path"), workspace, "provisional smoke cache", directory=True
    )
    artifacts = cache.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise FinalizeError("provisional smoke cache artifacts are absent")
    actual: list[dict[str, object]] = []
    for path in sorted(cache_root.rglob("*")):
        if path.is_symlink():
            raise FinalizeError("provisional smoke cache contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise FinalizeError("provisional smoke cache contains a special entry")
        actual.append(
            {
                "path": path.relative_to(cache_root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    actual.sort(key=lambda record: str(record["path"]))
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise FinalizeError("provisional smoke cache artifact is malformed")
        _sha256(record.get("sha256"), "provisional smoke cache artifact")
        _count(record.get("size"), "provisional smoke cache artifact size")
    cubin_count = sum(
        str(record["path"]).casefold().endswith(".cubin") for record in actual
    )
    if (
        artifacts != actual
        or cache.get("artifacts_count") != len(actual)
        or cache.get("cubin_count") != cubin_count
        or cubin_count < 1
        or any(
            int(record["size"]) <= 0
            for record in actual
            if str(record["path"]).casefold().endswith(".cubin")
        )
        or cache.get("total_bytes") != sum(int(record["size"]) for record in actual)
        or cache.get("artifacts_sha256")
        != hashlib.sha256(canonical_json(actual).encode("ascii")).hexdigest()
    ):
        raise FinalizeError("provisional smoke cache aggregate identity is invalid")


def _validate_runtime_provenance(
    provenance: object,
    *,
    inputs: dict[str, object],
    context: dict[str, object],
) -> None:
    fields = {
        "editable",
        "libtriton_maps",
        "module_paths",
        "python",
        "sys_path",
        "torch_file",
        "torch_runtime",
    }
    if not isinstance(provenance, dict) or set(provenance) != fields:
        raise FinalizeError("provisional smoke runtime provenance schema is malformed")
    editable = provenance.get("editable")
    if editable != {
        "carriers": {
            "meta_path": [],
            "path_hooks": [],
            "path_importer_cache": [],
        },
        "loaded_modules": [],
    }:
        raise FinalizeError("provisional smoke retained an editable carrier")

    prefix = context["prefix"]
    probe_site = context["probe_site"]
    assert isinstance(prefix, Path) and isinstance(probe_site, Path)
    module_paths = provenance.get("module_paths")
    maps = provenance.get("libtriton_maps")
    if not isinstance(module_paths, dict) or not module_paths or not isinstance(maps, list):
        raise FinalizeError("provisional smoke Triton path provenance is absent")
    if maps != sorted(set(maps)) or not maps:
        raise FinalizeError("provisional smoke libtriton map inventory is malformed")
    for value in maps:
        resolved = _expand_probe_token(
            value, prefix, "provisional smoke libtriton map"
        )
        if probe_site not in resolved.parents:
            raise FinalizeError("provisional smoke libtriton map is not wheel-site owned")
    if not {"triton", "triton._C.libtriton"}.issubset(module_paths):
        raise FinalizeError("provisional smoke core Triton module provenance is absent")
    for name, values in module_paths.items():
        if (
            not isinstance(name, str)
            or (name != "triton" and not name.startswith("triton."))
            or not isinstance(values, list)
            or not values
            or values != sorted(set(values))
        ):
            raise FinalizeError("provisional smoke module provenance is malformed")
        for value in values:
            resolved = _expand_probe_token(
                value, prefix, f"provisional smoke module {name}"
            )
            if resolved != probe_site and probe_site not in resolved.parents:
                raise FinalizeError(
                    f"provisional smoke module is not wheel-site owned: {name}"
                )
    if module_paths.get("triton._C.libtriton") != maps:
        raise FinalizeError("provisional smoke libtriton ownership is incomplete")
    probe_process = context["probe_process"]
    assert isinstance(probe_process, dict)
    if maps != probe_process.get("libtriton_maps"):
        raise FinalizeError("provisional smoke libtriton maps differ from accepted probe")

    base = inputs.get("base_python")
    python = provenance.get("python")
    if (
        not isinstance(base, dict)
        or not isinstance(python, dict)
        or set(python) != {
            "executable",
            "resolved_executable",
            "resolved_sha256",
            "resolved_size",
        }
        or python.get("executable") != "$PROBE_PREFIX/bin/python"
        or python.get("resolved_executable") != base.get("path")
        or python.get("resolved_sha256") != base.get("sha256")
        or python.get("resolved_size") != base.get("size")
    ):
        raise FinalizeError("provisional smoke Python runtime provenance is invalid")

    probe_torch = probe_process.get("torch")
    torch_runtime = provenance.get("torch_runtime")
    if (
        not isinstance(probe_torch, dict)
        or not isinstance(torch_runtime, dict)
        or set(torch_runtime) != {"cuda", "file", "git_version", "hip", "version"}
        or any(
            torch_runtime.get(name) != probe_torch.get(name)
            for name in ("cuda", "file", "git_version", "hip", "version")
        )
        or provenance.get("torch_file") != torch_runtime.get("file")
    ):
        raise FinalizeError("provisional smoke Torch runtime provenance is invalid")
    torch_file = provenance.get("torch_file")
    if not isinstance(torch_file, str) or not torch_file.startswith(
        "$TORCH_SITE_PACKAGES/"
    ):
        raise FinalizeError("provisional smoke Torch path is not source-site normalized")

    sys_path = provenance.get("sys_path")
    runtime_view = context["runtime_view"]
    assert isinstance(probe_site, Path) and isinstance(runtime_view, Path)
    expected_probe_site = _probe_token(probe_site, prefix)
    expected_view = _probe_token(runtime_view, prefix)
    if (
        not isinstance(sys_path, list)
        or not sys_path
        or any(not isinstance(value, str) or not value for value in sys_path)
        or sys_path[0] != expected_probe_site
        or sys_path[-1] != expected_view
        or sys_path.count(expected_probe_site) != 1
        or sys_path.count(expected_view) != 1
        or any(
            ("site-packages" in value.casefold() or "dist-packages" in value.casefold())
            and not value.startswith(("$PROBE_PREFIX/", "$TORCH_SITE_PACKAGES/"))
            for value in sys_path
        )
    ):
        raise FinalizeError("provisional smoke sys.path provenance is invalid")


def _validate_integrity(
    integrity: object,
    *,
    inputs: dict[str, object],
    probe_process: dict[str, object],
) -> None:
    if (
        not isinstance(integrity, dict)
        or set(integrity) != {"after", "before", "stable"}
        or integrity.get("stable") is not True
        or not isinstance(integrity.get("before"), dict)
        or integrity.get("after") != integrity.get("before")
    ):
        raise FinalizeError("provisional smoke before/after integrity is invalid")
    snapshot = integrity["before"]
    assert isinstance(snapshot, dict)
    if set(snapshot) != {
        "environment_lock",
        "installed_triton",
        "linked_inputs",
        "torch_tree",
    }:
        raise FinalizeError("provisional smoke integrity snapshot is incomplete")

    linked = snapshot.get("linked_inputs")
    wheel = inputs.get("wheel")
    if (
        not isinstance(wheel, dict)
        or not isinstance(linked, dict)
        or set(linked) != {"base_python", "probe_evidence", "wheel"}
        or linked.get("base_python") != inputs.get("base_python")
        or linked.get("probe_evidence") != inputs.get("probe_evidence")
        or linked.get("wheel")
        != {
            "audit_evidence_sha256": wheel["audit_evidence"]["sha256"],
            "sha256": wheel["sha256"],
        }
    ):
        raise FinalizeError("provisional smoke linked-input integrity is invalid")

    probe_torch = probe_process.get("torch")
    environment = snapshot.get("environment_lock")
    environment_input = inputs.get("environment_lock")
    if (
        not isinstance(probe_torch, dict)
        or not isinstance(environment_input, dict)
        or not isinstance(environment, dict)
        or set(environment) != {"cuda", "hip", "path", "sha256", "size", "torch", "torch_git"}
        or {name: environment.get(name) for name in ("path", "sha256", "size")}
        != environment_input
        or environment.get("cuda") != probe_torch.get("cuda")
        or environment.get("hip") != probe_torch.get("hip")
        or environment.get("torch") != probe_torch.get("version")
        or environment.get("torch_git") != probe_torch.get("git_version")
    ):
        raise FinalizeError("provisional smoke environment integrity is invalid")

    torch_tree = snapshot.get("torch_tree")
    torch_input = inputs.get("torch_site_packages")
    expected_torch_fields = {
        "cuda",
        "dist_info_name",
        "git_version",
        "hip",
        "metadata_sha256",
        "module_sha256",
        "path",
        "torch_tree_bytes",
        "torch_tree_files",
        "torch_tree_sha256",
        "version",
    }
    if (
        not isinstance(torch_tree, dict)
        or set(torch_tree) != expected_torch_fields
        or not isinstance(torch_input, dict)
        or any(torch_tree.get(name) != torch_input.get(name) for name in torch_input)
        or torch_tree.get("cuda") != probe_torch.get("cuda")
        or torch_tree.get("git_version") != probe_torch.get("git_version")
        or torch_tree.get("hip") != probe_torch.get("hip")
    ):
        raise FinalizeError("provisional smoke Torch-tree integrity is invalid")

    installed = snapshot.get("installed_triton")
    installed_fields = {
        "native_bytes",
        "native_entries_count",
        "native_entries_sha256",
        "package_entries_count",
        "package_entries_sha256",
        "record_entries_count",
        "record_entries_sha256",
    }
    if not isinstance(installed, dict) or set(installed) != installed_fields:
        raise FinalizeError("provisional smoke installed-Triton integrity is malformed")
    native_count = _count(
        installed.get("native_entries_count"), "installed native entry count", positive=True
    )
    package_count = _count(
        installed.get("package_entries_count"), "installed package entry count", positive=True
    )
    record_count = _count(
        installed.get("record_entries_count"), "installed RECORD entry count", positive=True
    )
    _count(installed.get("native_bytes"), "installed native bytes", positive=True)
    for name in (
        "native_entries_sha256",
        "package_entries_sha256",
        "record_entries_sha256",
    ):
        _sha256(installed.get(name), f"installed Triton {name}")
    if native_count > record_count or package_count > record_count:
        raise FinalizeError("provisional smoke installed-Triton counts are inconsistent")


def validate_provisional_semantics(
    provisional: dict[str, object], workspace: Path
) -> dict[str, object]:
    inputs = provisional.get("inputs")
    runtime = provisional.get("runtime")
    run_context = provisional.get("run_context")
    if not isinstance(inputs, dict) or not isinstance(runtime, dict):
        raise FinalizeError("provisional smoke inputs/runtime are absent")
    context = _validate_probe_bound_inputs(inputs, workspace)
    expected_runtime_fields = {
        "compiled_cache",
        "correctness",
        "device",
        "gpu_execution",
        "integrity",
        "provenance",
        "ptxas_blackwell",
        "synchronization",
        "target",
    }
    if set(runtime) != expected_runtime_fields:
        raise FinalizeError("provisional smoke runtime schema is incomplete")
    if runtime.get("gpu_execution") is not True:
        raise FinalizeError("provisional smoke did not record GPU execution")
    correctness = runtime.get("correctness")
    if correctness != {
        "block_size": BLOCK_SIZE,
        "comparison": "torch.equal",
        "dtype": "float32",
        "equal": True,
        "kernel": "masked-vector-add",
        "n_elements": VECTOR_ELEMENTS,
        "reference_provider": "torch",
    }:
        raise FinalizeError("provisional smoke correctness is invalid")
    device = runtime.get("device")
    if not isinstance(device, dict) or device != {
        "compute_capability": [12, 0],
        "index": 0,
        "name": EXPECTED_DEVICE_NAME,
    }:
        raise FinalizeError("provisional smoke device identity is invalid")
    if runtime.get("target") != {"arch": 120, "backend": "cuda", "warp_size": 32}:
        raise FinalizeError("provisional smoke target identity is invalid")
    if runtime.get("synchronization") != {
        "after_comparison": True,
        "after_kernel": True,
        "before_launch": True,
        "error": None,
    }:
        raise FinalizeError("provisional smoke synchronization is invalid")

    probe_process = context["probe_process"]
    prefix = context["prefix"]
    probe_site = context["probe_site"]
    assert isinstance(probe_process, dict)
    assert isinstance(prefix, Path) and isinstance(probe_site, Path)
    expected_probe_ptxas = probe_process.get("ptxas_blackwell")
    ptxas = runtime.get("ptxas_blackwell")
    if (
        not isinstance(ptxas, dict)
        or set(ptxas) != {
            "audited_full_version",
            "path",
            "reported_release",
            "sha256",
            "wheel_owned",
        }
        or ptxas.get("wheel_owned") is not True
        or ptxas.get("audited_full_version") != EXPECTED_PTXAS_FULL_VERSION
        or ptxas.get("reported_release") != EXPECTED_PTXAS_RELEASE
        or not isinstance(expected_probe_ptxas, dict)
        or any(
            ptxas.get(name) != expected_probe_ptxas.get(name)
            for name in ("audited_full_version", "path", "reported_release", "sha256")
        )
    ):
        raise FinalizeError("provisional smoke ptxas identity is invalid")
    expected_ptxas_path = probe_site / PTXAS_BLACKWELL_RELATIVE
    if (
        ptxas.get("path") != _probe_token(expected_ptxas_path, prefix)
        or _expand_probe_token(ptxas.get("path"), prefix, "provisional ptxas")
        != expected_ptxas_path
        or _sha256(ptxas.get("sha256"), "provisional ptxas")
        != sha256_file(expected_ptxas_path)
    ):
        raise FinalizeError("provisional smoke ptxas path/SHA is not wheel-owned")

    _validate_cache(runtime.get("compiled_cache"), workspace)
    _validate_runtime_provenance(
        runtime.get("provenance"), inputs=inputs, context=context
    )
    _validate_integrity(
        runtime.get("integrity"), inputs=inputs, probe_process=probe_process
    )
    if not isinstance(run_context, dict) or set(run_context) != {
        "mode",
        "pgid",
        "pid",
        "preflight",
        "provisional_evidence_path",
        "run_id",
    }:
        raise FinalizeError("provisional smoke run context is absent")
    _count(run_context.get("pgid"), "provisional smoke process group", positive=True)
    _count(run_context.get("pid"), "provisional smoke process id", positive=True)
    preflight = run_context.get("preflight")
    if not isinstance(preflight, dict) or set(preflight) != {"path", "sha256", "size"}:
        raise FinalizeError("provisional smoke preflight identity is malformed")
    _sha256(preflight.get("sha256"), "provisional smoke preflight")
    _count(preflight.get("size"), "provisional smoke preflight size", positive=True)
    return run_context


def finalize(
    *,
    workspace: Path,
    provisional_path: Path,
    expected_provisional_sha256: str,
    run_id: str,
    output: Path,
) -> tuple[dict[str, object], str]:
    workspace = require_workspace(workspace)
    if re.fullmatch(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}", run_id) is None:
        raise FinalizeError("run-id format is invalid")
    provisional_path = require_workspace_file(
        provisional_path, workspace, "provisional smoke evidence"
    )
    if re.fullmatch(r"[0-9a-f]{64}", expected_provisional_sha256) is None:
        raise FinalizeError("expected provisional SHA256 is invalid")
    provisional, provisional_raw = load_canonical(
        provisional_path, "provisional smoke evidence"
    )
    actual_provisional_sha256 = hashlib.sha256(provisional_raw).hexdigest()
    if actual_provisional_sha256 != expected_provisional_sha256:
        raise FinalizeError("provisional smoke differs from its external anchor")
    if (
        provisional.get("schema_version") != 1
        or provisional.get("smoke") != SMOKE
        or provisional.get("acceptance") != PROVISIONAL_STATUS
    ):
        raise FinalizeError("provisional smoke is not finalizable")
    if set(provisional) != {
        "acceptance",
        "inputs",
        "run_context",
        "runtime",
        "schema_version",
        "scope",
        "smoke",
    }:
        raise FinalizeError("provisional smoke top-level schema is malformed")
    scope = provisional.get("scope")
    if not isinstance(scope, dict) or scope != {
        "coverage_result": False,
        "performance_result": False,
        "provider": "triton",
        "pypto_kernel": False,
        "reference_only": True,
    }:
        raise FinalizeError("provisional smoke scope is invalid")
    run_context = validate_provisional_semantics(provisional, workspace)

    run_dir = workspace / "runs" / run_id
    process_path = require_workspace_file(
        run_dir / "process.json", workspace, "run process metadata"
    )
    preflight_path = require_workspace_file(
        run_dir / "preflight.json", workspace, "run preflight"
    )
    process, process_raw = load_canonical(process_path, "run process metadata")
    preflight, preflight_raw = load_canonical(preflight_path, "run preflight")
    process_preflight = process.get("preflight")
    if not isinstance(process_preflight, dict):
        raise FinalizeError("process metadata has no preflight anchor")
    if (
        process_preflight.get("path") != str(preflight_path)
        or process_preflight.get("sha256")
        != hashlib.sha256(preflight_raw).hexdigest()
    ):
        raise FinalizeError("process/preflight digest join failed")
    if (
        process.get("run_id") != run_id
        or process.get("mode") != "gpu-benchmark"
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("gpu_benchmark_abort") is not None
        or process.get("coexistence_pauses", []) != []
    ):
        raise FinalizeError("GPU smoke run did not exit cleanly and exclusively")
    coexistence = process.get("coexistence")
    if not isinstance(coexistence, dict) or coexistence.get("requested") is not False:
        raise FinalizeError("GPU smoke run incorrectly requested coexistence")
    if (
        preflight.get("ok") is not True
        or preflight.get("mode") != "gpu-benchmark"
        or preflight.get("nvidia_compute_pids") != []
        or preflight.get("protected_heavy_processes") != []
        or preflight.get("protected_cpu_only_coexistence_requested") is not False
    ):
        raise FinalizeError("GPU smoke preflight was not exclusive")
    if (
        run_context.get("run_id") != run_id
        or run_context.get("mode") != "gpu-benchmark"
        or run_context.get("pgid") != process.get("pgid")
        or run_context.get("provisional_evidence_path")
        != _relative(provisional_path, workspace)
        or not isinstance(run_context.get("pid"), int)
        or isinstance(run_context.get("pid"), bool)
        or run_context.get("preflight")
        != {
            "path": _relative(preflight_path, workspace),
            "sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "size": len(preflight_raw),
        }
    ):
        raise FinalizeError("provisional smoke run context does not join this run")
    command = process.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise FinalizeError("GPU smoke command record is invalid")
    runner = str(workspace / "benchmarks/operators/triton_reference_sm120.py")
    if not any(runner in item for item in command) or not any(
        str(provisional_path) in item for item in command
    ):
        raise FinalizeError("GPU smoke command does not bind runner/evidence")

    final = dict(provisional)
    final["acceptance"] = "accepted"
    final["exclusive_run"] = {
        "coexistence_pauses": [],
        "gpu_benchmark_abort": None,
        "preflight": {
            "document_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "mode": "gpu-benchmark",
            "ok": True,
            "path": _relative(preflight_path, workspace),
        },
        "process": {
            "command_sha256": hashlib.sha256(
                canonical_json(command).encode("ascii")
            ).hexdigest(),
            "document_sha256": hashlib.sha256(process_raw).hexdigest(),
            "mode": "gpu-benchmark",
            "path": _relative(process_path, workspace),
            "return_code": 0,
            "status": "exited",
        },
        "run_id": run_id,
        "finalizer": {
            "path": _relative(Path(__file__).resolve(), workspace),
            "sha256": sha256_file(Path(__file__).resolve()),
            "size": Path(__file__).resolve().stat().st_size,
        },
    }
    final["provisional_evidence"] = {
        "path": _relative(provisional_path, workspace),
        "sha256": actual_provisional_sha256,
        "size": len(provisional_raw),
    }
    output = Path(os.path.abspath(os.fspath(output)))
    digest = publish_no_replace(output, final)
    return final, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--provisional-evidence", type=Path, required=True)
    parser.add_argument("--expected-provisional-evidence-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final-evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, digest = finalize(
            workspace=args.workspace,
            provisional_path=args.provisional_evidence,
            expected_provisional_sha256=args.expected_provisional_evidence_sha256,
            run_id=args.run_id,
            output=args.final_evidence,
        )
    except (FinalizeError, OSError, ValueError) as error:
        print(f"Triton smoke finalization failed: {error}", file=sys.stderr)
        return 1
    print(canonical_json({"evidence": str(args.final_evidence), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
