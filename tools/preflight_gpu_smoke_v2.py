#!/usr/bin/env python3
"""Observation-only v2 GPU-smoke preflight with a 22 GiB protected floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PREFLIGHT_RELATIVE_PATH = Path("tools/preflight.py")
BASE_PREFLIGHT_SIZE = 27_684
BASE_PREFLIGHT_SHA256 = (
    "0b9884f8dbd34337a85f62c351b1e19dda3a8b84ec9a88c835d8701af053e3d1"
)
CONTRACT_RELATIVE_PATH = Path("tools/_pypto_fused_pointwise_sm120_contract_v2.py")
PREFLIGHT_POLICY_VERSION = 3
COEXISTENCE_POLICY_VERSION = 1
GPU_SMOKE_POLICY_VERSION = 2
PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB = 32 * 1024 * 1024
GPU_SMOKE_FREE_MEMORY_FLOOR_MIB = 4 * 1024


class PreflightV2Error(RuntimeError):
    """The exact v2 preflight dependency or policy is invalid."""


def _load_exact(
    name: str,
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise PreflightV2Error(f"exact v2 preflight source is noncanonical: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_size is not None and len(raw) != expected_size:
        raise PreflightV2Error(f"exact v2 preflight source size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise PreflightV2Error(f"exact v2 preflight source hash differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = digest
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = _load_exact(
    "_pypto_fused_pointwise_sm120_contract_v2",
    ROOT / CONTRACT_RELATIVE_PATH,
)
sys.modules["_pypto_nvidia_executable_sm120_contract"] = contract
base = _load_exact(
    "_pypto_gpu_smoke_preflight_v1_base",
    ROOT / BASE_PREFLIGHT_RELATIVE_PATH,
    expected_size=BASE_PREFLIGHT_SIZE,
    expected_sha256=BASE_PREFLIGHT_SHA256,
)

ProcessInfo = base.ProcessInfo
PROTECTED_ROOTS = base.PROTECTED_ROOTS
FORBIDDEN_ENV_NAMES = base.FORBIDDEN_ENV_NAMES
FORBIDDEN_ENV_PREFIXES = base.FORBIDDEN_ENV_PREFIXES
FORBIDDEN_DSO_MARKERS = base.FORBIDDEN_DSO_MARKERS
NVIDIA_PROCESS_MAP_MARKERS = base.NVIDIA_PROCESS_MAP_MARKERS
belongs_to_roots = base.belongs_to_roots
is_heavy_command = base.is_heavy_command
process_table = base.process_table
mem_available_kib = base.mem_available_kib
nvidia_identity = base.nvidia_identity
nvidia_compute_pids = base.nvidia_compute_pids
protected_nvidia_runtime_mappings = base.protected_nvidia_runtime_mappings
static_torch_identity = base.static_torch_identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def policy_document() -> dict[str, object]:
    adapter = Path(__file__).resolve(strict=True)
    return {
        "schema_version": GPU_SMOKE_POLICY_VERSION,
        "protected_memory_floor_kib": PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB,
        "exclusive_memory_floor_kib": EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB,
        "owned_run_abort_memory_floor_kib": contract.OWNED_RUN_ABORT_MEMORY_FLOOR_KIB,
        "gpu_free_memory_floor_mib": GPU_SMOKE_FREE_MEMORY_FLOOR_MIB,
        "adapter": {
            "path": adapter.relative_to(ROOT).as_posix(),
            "bytes": adapter.stat().st_size,
            "sha256": sha256_file(adapter),
        },
        "base_preflight": {
            "path": BASE_PREFLIGHT_RELATIVE_PATH.as_posix(),
            "bytes": BASE_PREFLIGHT_SIZE,
            "sha256": BASE_PREFLIGHT_SHA256,
        },
    }


def gpu_smoke_policy_failures(
    *,
    protected_authorized: bool,
    protected_heavy: list[Any],
    available_kib: int,
    protected_nvidia_compute_pids: list[int],
    protected_nvidia_runtime_pids: list[int],
    unreadable_protected_maps: list[int],
) -> list[str]:
    failures: list[str] = []
    if protected_authorized:
        if protected_nvidia_compute_pids:
            failures.append(
                "protected workload has active NVIDIA compute processes: "
                f"{protected_nvidia_compute_pids}"
            )
        if protected_nvidia_runtime_pids:
            failures.append(
                "protected workload has NVIDIA runtime mappings: "
                f"{protected_nvidia_runtime_pids}"
            )
        if unreadable_protected_maps:
            failures.append(
                "cannot prove protected workload has no NVIDIA runtime mappings: "
                f"{unreadable_protected_maps}"
            )
        memory_floor_kib = PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
    else:
        if protected_heavy:
            failures.append(
                "protected zcode/gem5/SGLang heavy workload is active; pause this project"
            )
        memory_floor_kib = EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
    if available_kib < memory_floor_kib:
        failures.append(
            f"MemAvailable {available_kib / 1024 / 1024:.1f} GiB is below "
            f"{memory_floor_kib / 1024 / 1024:.0f} GiB safety floor"
        )
    return failures


def build_report(*, protected_authorized: bool) -> dict[str, object]:
    cwd = Path.cwd().resolve()
    failures: list[str] = []
    if ROOT not in (cwd, *cwd.parents):
        failures.append(f"cwd is outside workspace: {cwd}")
    if any(Path(root) in (cwd, *cwd.parents) for root in PROTECTED_ROOTS):
        failures.append(f"cwd is inside protected scope: {cwd}")
    leaked_env = sorted(
        name
        for name in os.environ
        if name in FORBIDDEN_ENV_NAMES or name.startswith(FORBIDDEN_ENV_PREFIXES)
    )
    if leaked_env:
        failures.append(f"forbidden AMD/simulator environment variables: {leaked_env}")

    all_processes, protected, workspace = process_table()
    protected_heavy = [proc for proc in protected if is_heavy_command(proc.command)]
    available_kib = mem_available_kib()
    try:
        gpu = nvidia_identity()
    except Exception as error:
        gpu = {"error": f"{type(error).__name__}: {error}"}
        failures.append("nvidia-smi identity check failed")
    if gpu.get("compute_capability") != "12.0":
        failures.append(f"expected SM120 GPU, got {gpu}")
    torch = static_torch_identity()
    if torch.get("static_identity_error"):
        failures.append(str(torch["static_identity_error"]))
    if torch.get("hip") is not None:
        failures.append(f"PyTorch reports HIP runtime: {torch.get('hip')}")
    if torch.get("forbidden_dsos"):
        failures.append(f"forbidden DSOs loaded: {torch['forbidden_dsos']}")

    compute_pids: set[int] = set()
    nvidia_compute_audit_ok = False
    protected_compute: list[int] = []
    protected_runtime: list[int] = []
    unreadable_maps: list[int] = []
    try:
        compute_pids = nvidia_compute_pids()
        nvidia_compute_audit_ok = True
    except Exception as error:
        failures.append(
            f"cannot audit NVIDIA compute processes: {type(error).__name__}: {error}"
        )
    if protected:
        protected_pid_set = {process.pid for process in protected}
        protected_compute = sorted(compute_pids & protected_pid_set)
        protected_runtime, unreadable_maps = protected_nvidia_runtime_mappings(
            protected
        )
    failures.extend(
        gpu_smoke_policy_failures(
            protected_authorized=protected_authorized,
            protected_heavy=protected_heavy,
            available_kib=available_kib,
            protected_nvidia_compute_pids=protected_compute,
            protected_nvidia_runtime_pids=protected_runtime,
            unreadable_protected_maps=unreadable_maps,
        )
    )
    external_compute = sorted(compute_pids - {os.getpid()})
    if external_compute:
        failures.append(
            f"external NVIDIA compute processes are active: {external_compute}"
        )
    try:
        free_memory_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    except (KeyError, TypeError, ValueError):
        failures.append("cannot derive NVIDIA free memory for GPU smoke")
    else:
        if free_memory_mib < GPU_SMOKE_FREE_MEMORY_FLOOR_MIB:
            failures.append(
                f"NVIDIA free memory {free_memory_mib} MiB is below "
                f"{GPU_SMOKE_FREE_MEMORY_FLOOR_MIB} MiB smoke floor"
            )

    expected_waiver = (
        protected_authorized
        and bool(protected)
        and nvidia_compute_audit_ok
        and not protected_compute
        and not protected_runtime
        and not unreadable_maps
        and not failures
    )
    return {
        "policy_version": PREFLIGHT_POLICY_VERSION,
        "coexistence_policy_version": COEXISTENCE_POLICY_VERSION,
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "mode": "gpu-smoke",
        "protected_cpu_only_coexistence_requested": False,
        "protected_zero_nvidia_gpu_smoke_requested": protected_authorized,
        "protected_activity_waiver_applied": False,
        "protected_gpu_smoke_waiver_applied": expected_waiver,
        "ok": not failures,
        "failures": failures,
        "mem_available_kib": available_kib,
        "memory_floor_kib": (
            PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
            if protected_authorized
            else EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
        ),
        "gpu_smoke_policy_version": GPU_SMOKE_POLICY_VERSION,
        "gpu_smoke_free_memory_floor_mib": GPU_SMOKE_FREE_MEMORY_FLOOR_MIB,
        "gpu_smoke_admission_policy": policy_document(),
        "gpu": gpu,
        "torch": torch,
        "protected_processes": [asdict(proc) for proc in protected],
        "protected_heavy_processes": [asdict(proc) for proc in protected_heavy],
        "nvidia_compute_pids": sorted(compute_pids),
        "nvidia_compute_audit_ok": nvidia_compute_audit_ok,
        "protected_nvidia_compute_pids": protected_compute,
        "protected_nvidia_runtime_mapping_pids": protected_runtime,
        "unreadable_protected_maps": unreadable_maps,
        "workspace_processes": [asdict(proc) for proc in workspace],
        "policy": "observation-only; no external process is ever signalled",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu-smoke",), required=True)
    parser.add_argument("--json", action="store_true", dest="json_only")
    parser.add_argument("--allow-protected-zero-nvidia-gpu-smoke", action="store_true")
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("v2 GPU-smoke preflight requires Python -E -B -S")
    report = build_report(
        protected_authorized=args.allow_protected_zero_nvidia_gpu_smoke
    )
    print(json.dumps(report, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if report["ok"] is True else 75


def __getattr__(name: str) -> Any:
    return getattr(base, name)


if __name__ == "__main__":
    raise SystemExit(main())
