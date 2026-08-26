#!/usr/bin/env python3
"""Versioned admission contract layered on the frozen fused-pointwise v1 case set."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT_RELATIVE_PATH = Path("tools/_pypto_fused_pointwise_sm120_contract.py")
BASE_CONTRACT_SIZE = 20_186
BASE_CONTRACT_SHA256 = (
    "7c812ccd3d9a76f2e5a258cf53fd029df776a67dfaf42c631363332fb9f8811c"
)
BASE_RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120.py")
BASE_RUNNER_SIZE = 66_999
BASE_RUNNER_SHA256 = "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15"


def _load_exact(name: str, path: Path, size: int, sha256: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"exact v2 contract dependency is noncanonical: {path}")
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise RuntimeError(f"exact v2 contract dependency differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = sha256
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_BASE = _load_exact(
    "_pypto_fused_pointwise_sm120_contract_v1_base",
    ROOT / BASE_CONTRACT_RELATIVE_PATH,
    BASE_CONTRACT_SIZE,
    BASE_CONTRACT_SHA256,
)

SMOKE_SCHEMA_VERSION = 2
SMOKE_NAME = "pypto-fused-pointwise-sm120-v2"
GPU_SMOKE_POLICY_VERSION = 2
GPU_SMOKE_AUTHORIZATION = (
    "user-authorized-protected-zero-nvidia-gpu-smoke-host-floor-22gib-v2"
)
PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB = 32 * 1024 * 1024
OWNED_RUN_ABORT_MEMORY_FLOOR_KIB = 16 * 1024 * 1024
GPU_FREE_MEMORY_FLOOR_MIB = 4 * 1024

RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120_v2.py")
RUNNER_SIZE = 25_051
RUNNER_SHA256 = "8eaed4a6280187ea8fbf99bebc1e9674748b7f66e40031ab3fa2147b9191b103"
PREFLIGHT_ADAPTER_RELATIVE_PATH = Path("tools/preflight_gpu_smoke_v2.py")
CONTROLLER_RELATIVE_PATH = Path("tools/run_pypto_fused_pointwise_sm120_v2_isolated.py")
FINALIZER_RELATIVE_PATH = Path("tools/finalize_pypto_fused_pointwise_sm120_v2.py")
CONTROL_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py"
)
REPLAY_DIRECTORY_NAME = "pypto-fused-pointwise-sm120-v2"
PROVISIONAL_NAME = "provisional.json"
FINAL_REPORT_DIRECTORY = Path("reports/data")


def fixed_child_command(workspace: Path) -> list[str]:
    """Return the only direct child accepted by the v2 admission lane."""

    root = workspace.resolve()
    return [
        str(root / "envs/pypto-nvidia/bin/python"),
        "-I",
        "-B",
        "-S",
        str(root / RUNNER_RELATIVE_PATH),
    ]


def replay_directory(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / "runs" / run_id / REPLAY_DIRECTORY_NAME


def provisional_path(workspace: Path, run_id: str) -> Path:
    return replay_directory(workspace, run_id) / PROVISIONAL_NAME


def final_report_path(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / FINAL_REPORT_DIRECTORY / f"{SMOKE_NAME}-{run_id}.json"


def __getattr__(name: str) -> Any:
    """Delegate the unchanged numerical/compiler contract to frozen v1."""

    return getattr(_BASE, name)
