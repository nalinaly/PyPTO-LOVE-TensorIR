"""Exact framework identity checks.

PyTorch exposes its source commit directly. SGLang wheels expose a version but
not necessarily a Git SHA, so the workspace runner also supplies
``PYPTO_SGLANG_SOURCE_ROOT``; when present, the plugin verifies its clean HEAD.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any

from .errors import FrameworkCompatibilityError


EXPECTED_TORCH_VERSION = "2.13.0"
EXPECTED_TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
EXPECTED_SGLANG_VERSION = "0.5.18"
EXPECTED_SGLANG_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
EXPECTED_TORCH_CUDA = "13.0"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _required_root(
    explicit: str | os.PathLike[str] | None, environment_name: str
) -> pathlib.Path:
    value = explicit or os.environ.get(environment_name)
    if not value:
        raise FrameworkCompatibilityError(
            f"{environment_name} is required to bind imported code to the locked workspace"
        )
    return pathlib.Path(value).resolve()


def _require_import_below(module: Any, expected_root: pathlib.Path, name: str) -> None:
    module_file = pathlib.Path(str(module.__file__)).resolve()
    if not module_file.is_relative_to(expected_root):
        raise FrameworkCompatibilityError(
            f"imported {name} from {module_file}, expected code below {expected_root}"
        )


def assert_torch_compatible(
    torch_module: Any | None = None,
    *,
    environment_root: str | os.PathLike[str] | None = None,
    workspace_root: str | os.PathLike[str] | None = None,
) -> None:
    if torch_module is None:
        import torch as torch_module
    version = str(torch_module.__version__)
    commit = str(torch_module.version.git_version)
    if (
        _base_version(version) != EXPECTED_TORCH_VERSION
        or commit != EXPECTED_TORCH_COMMIT
    ):
        raise FrameworkCompatibilityError(
            "PyPTO TorchInductor plugin requires "
            f"torch {EXPECTED_TORCH_VERSION} at {EXPECTED_TORCH_COMMIT}; "
            f"found {version} at {commit}."
        )
    if (
        torch_module.version.hip is not None
        or torch_module.version.cuda != EXPECTED_TORCH_CUDA
    ):
        raise FrameworkCompatibilityError(
            "PyPTO requires the pinned NVIDIA CUDA build; "
            f"found cuda={torch_module.version.cuda!r}, hip={torch_module.version.hip!r}."
        )
    if not torch_module.cuda.is_available():
        raise FrameworkCompatibilityError("the pinned PyTorch build cannot access CUDA")
    capability = tuple(torch_module.cuda.get_device_capability(0))
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise FrameworkCompatibilityError(
            f"PyPTO requires compute capability {EXPECTED_COMPUTE_CAPABILITY}, got {capability}"
        )
    root = _required_root(environment_root, "PYPTO_ENV_PREFIX")
    _require_import_below(torch_module, root, "torch")
    workspace = _required_root(workspace_root, "PYPTO_WORKSPACE_ROOT")
    try:
        manifest = json.loads((workspace / "ENVIRONMENT.lock").read_text())
    except (OSError, ValueError) as exc:
        raise FrameworkCompatibilityError(
            f"cannot read locked environment manifest under {workspace}: {exc}"
        ) from exc
    tree_digest = manifest.get("torch_tree_sha256")
    if manifest.get("status") != "cloned" or not isinstance(tree_digest, str) or len(
        tree_digest
    ) != 64:
        raise FrameworkCompatibilityError(
            "ENVIRONMENT.lock lacks a frozen PyTorch tree digest"
        )
    imported_file = str(pathlib.Path(str(torch_module.__file__)).resolve())
    if manifest.get("torch_file") != imported_file:
        raise FrameworkCompatibilityError(
            f"imported torch file {imported_file} does not match ENVIRONMENT.lock"
        )


def _git_output(source_root: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def assert_sglang_compatible(
    *,
    installed_version: str | None = None,
    source_root: str | os.PathLike[str] | None = None,
    sglang_module: Any | None = None,
) -> None:
    if sglang_module is None:
        import sglang as sglang_module
    version = installed_version or str(sglang_module.__version__)
    if _base_version(version) != EXPECTED_SGLANG_VERSION:
        raise FrameworkCompatibilityError(
            f"PyPTO SGLang plugin requires {EXPECTED_SGLANG_VERSION}; found {version}."
        )
    root = _required_root(source_root, "PYPTO_SGLANG_SOURCE_ROOT")
    _require_import_below(sglang_module, root / "python" / "sglang", "sglang")
    try:
        commit = _git_output(root, "rev-parse", "HEAD")
        dirty = _git_output(root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrameworkCompatibilityError(
            f"cannot verify SGLang source root {root}: {exc}"
        ) from exc
    if commit != EXPECTED_SGLANG_COMMIT or dirty:
        raise FrameworkCompatibilityError(
            "PyPTO SGLang plugin requires clean commit "
            f"{EXPECTED_SGLANG_COMMIT}; found commit={commit}, dirty={bool(dirty)}."
        )
