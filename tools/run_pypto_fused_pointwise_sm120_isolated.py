#!/usr/bin/env python3
"""Versioned controller for the fused-pointwise SM120 smoke.

This adapter supplies the v1 contract to the accepted v4 preflight,
run-isolation, watchdog, and stop primitives without changing those blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ControllerError(RuntimeError):
    """The versioned controller cannot safely delegate to the v4 primitive."""


def load_exact(name: str, path: pathlib.Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControllerError(f"exact controller source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
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
        parser.error("fused-pointwise GPU-smoke controller requires Python -E -B -S")

    control = load_exact(
        "_pypto_fused_pointwise_sm120_control_manifest",
        ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
    )
    control.reject_control_bytecode_cache(ROOT)
    contract = load_exact(
        "_pypto_fused_pointwise_sm120_contract",
        ROOT / "tools/_pypto_fused_pointwise_sm120_contract.py",
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
