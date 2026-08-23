"""Exact framework identity checks.

PyTorch exposes its source commit directly. SGLang wheels expose a version but
not necessarily a Git SHA, so the workspace runner also supplies
``PYPTO_SGLANG_SOURCE_ROOT``; when present, the plugin verifies its clean HEAD.
"""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import subprocess
from typing import Any

from .errors import FrameworkCompatibilityError


EXPECTED_TORCH_VERSION = "2.13.0"
EXPECTED_TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
EXPECTED_SGLANG_VERSION = "0.5.18"
EXPECTED_SGLANG_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def assert_torch_compatible(torch_module: Any | None = None) -> None:
    if torch_module is None:
        import torch as torch_module
    version = str(torch_module.__version__)
    commit = str(torch_module.version.git_version)
    if _base_version(version) != EXPECTED_TORCH_VERSION or commit != EXPECTED_TORCH_COMMIT:
        raise FrameworkCompatibilityError(
            "PyPTO TorchInductor plugin requires "
            f"torch {EXPECTED_TORCH_VERSION} at {EXPECTED_TORCH_COMMIT}; "
            f"found {version} at {commit}."
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
) -> None:
    version = installed_version or importlib.metadata.version("sglang")
    if _base_version(version) != EXPECTED_SGLANG_VERSION:
        raise FrameworkCompatibilityError(
            f"PyPTO SGLang plugin requires {EXPECTED_SGLANG_VERSION}; found {version}."
        )
    root_value = source_root or os.environ.get("PYPTO_SGLANG_SOURCE_ROOT")
    if root_value is None:
        raise FrameworkCompatibilityError(
            "PYPTO_SGLANG_SOURCE_ROOT is required so the plugin can verify the exact clean SGLang checkout."
        )
    root = pathlib.Path(root_value).resolve()
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

