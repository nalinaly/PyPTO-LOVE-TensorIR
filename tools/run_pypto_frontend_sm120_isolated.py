#!/usr/bin/env python3
"""Versioned controller for the frontend vector-add SM120 smoke.

This adapter supplies the v1 contract to the accepted v4 preflight,
run-isolation, watchdog, and stop primitives without changing those blobs.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ControllerError(RuntimeError):
    """The versioned controller cannot safely delegate to the v4 primitive."""


def load_exact(name: str, path: pathlib.Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ControllerError(f"cannot load exact controller dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def delegated_argv(
    contract: ModuleType,
    *,
    run_id_file: pathlib.Path,
    allow_protected_zero_nvidia_gpu_smoke: bool,
) -> list[str]:
    arguments = [
        "run_isolated.py",
        "--mode",
        "gpu-smoke",
        "--exact-pypto-nvidia-smoke",
        "--timeout-seconds",
        str(contract.GPU_SMOKE_TIMEOUT_SECONDS),
        "--minimum-free-disk-gib",
        str(contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB),
        "--environment",
        "pypto-nvidia",
        "--framework-profile",
        "pypto",
        "--environment-lock-mode",
        "shared",
        "--run-id-file",
        str(run_id_file),
    ]
    if allow_protected_zero_nvidia_gpu_smoke:
        arguments.append("--allow-protected-zero-nvidia-gpu-smoke")
    arguments.extend(["--", *contract.fixed_child_command(ROOT)])
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-protected-zero-nvidia-gpu-smoke",
        action="store_true",
        help=(
            "apply the accepted v4 protected CPU-lane policy only when that lane "
            "has zero NVIDIA mappings and compute PIDs"
        ),
    )
    parser.add_argument("--run-id-file", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("frontend GPU-smoke controller requires Python -E -B -S")

    contract = load_exact(
        "_pypto_frontend_vector_add_sm120_contract",
        ROOT / "tools/_pypto_frontend_vector_add_sm120_contract.py",
    )
    control = load_exact(
        "_pypto_frontend_sm120_control_manifest",
        ROOT / "tools/_pypto_frontend_sm120_control_manifest.py",
    )

    # The accepted primitives import the v4 module names.  Provide the new
    # immutable contract objects under those private import keys, then execute
    # the primitive unchanged.
    sys.modules["_pypto_nvidia_executable_sm120_contract"] = contract
    sys.modules["_pypto_nvidia_sm120_control_manifest"] = control
    preflight = load_exact("preflight", ROOT / "tools/preflight.py")
    stop_run = load_exact("stop_run", ROOT / "tools/stop_run.py")
    run_isolated = load_exact("run_isolated", ROOT / "tools/run_isolated.py")
    if (
        run_isolated.preflight_tool is not preflight
        or run_isolated.stop_run is not stop_run
        or run_isolated.nvidia_smoke_contract is not contract
        or run_isolated.nvidia_smoke_control is not control
    ):
        raise ControllerError("accepted v4 primitive dependency injection differs")
    sys.argv = delegated_argv(
        contract,
        run_id_file=args.run_id_file,
        allow_protected_zero_nvidia_gpu_smoke=(
            args.allow_protected_zero_nvidia_gpu_smoke
        ),
    )
    return int(run_isolated.main())


if __name__ == "__main__":
    raise SystemExit(main())
