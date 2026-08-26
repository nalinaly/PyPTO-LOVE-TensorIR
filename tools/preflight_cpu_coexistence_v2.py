#!/usr/bin/env python3
"""Observation-only preflight for CPU coexistence policy v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


class PreflightError(RuntimeError):
    """An exact dependency or CPU-v2 admission invariant differs."""


def load_source(name: str, path: Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise PreflightError(f"CPU-v2 source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = load_source(
    "_pypto_cpu_coexistence_v2_contract_for_preflight",
    ROOT / "tools/_pypto_cpu_coexistence_v2_contract.py",
)
base_nvidia_contract = contract.load_exact(
    "_pypto_nvidia_executable_sm120_contract",
    ROOT / contract.BASE_NVIDIA_CONTRACT_RELATIVE_PATH,
    contract.BASE_NVIDIA_CONTRACT_SIZE,
    contract.BASE_NVIDIA_CONTRACT_SHA256,
)
base = contract.load_exact(
    "_pypto_cpu_v2_base_preflight",
    ROOT / contract.BASE_PREFLIGHT_RELATIVE_PATH,
    contract.BASE_PREFLIGHT_SIZE,
    contract.BASE_PREFLIGHT_SHA256,
)
if base.nvidia_smoke_contract is not base_nvidia_contract:
    raise PreflightError("CPU-v2 base preflight dependency identity differs")

ProcessInfo = base.ProcessInfo
process_table = base.process_table
is_heavy_command = base.is_heavy_command
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
        **contract.policy_document(),
        "adapter": {
            "path": adapter.relative_to(ROOT).as_posix(),
            "bytes": adapter.stat().st_size,
            "sha256": sha256_file(adapter),
        },
    }


def validate_report(value: object, *, description: str) -> dict[str, object]:
    expected = {
        "schema_version",
        "kind",
        "mode",
        "workspace",
        "cwd",
        "ok",
        "failures",
        "mem_available_kib",
        "launch_memory_floor_kib",
        "resume_memory_floor_kib",
        "pause_memory_floor_kib",
        "protected_cpu_only_coexistence_requested",
        "protected_activity_waiver_applied",
        "gpu",
        "torch",
        "protected_processes",
        "protected_heavy_processes",
        "workspace_processes",
        "nvidia_compute_audit_ok",
        "nvidia_compute_pids",
        "protected_nvidia_compute_pids",
        "protected_nvidia_runtime_mapping_pids",
        "unreadable_protected_maps",
        "policy",
        "admission_policy",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise PreflightError(f"{description} key set differs")
    failures = value.get("failures")
    memory = value.get("mem_available_kib")
    protected_heavy = value.get("protected_heavy_processes")
    pid_fields = (
        "nvidia_compute_pids",
        "protected_nvidia_compute_pids",
        "protected_nvidia_runtime_mapping_pids",
        "unreadable_protected_maps",
    )
    process_fields = (
        "protected_processes",
        "protected_heavy_processes",
        "workspace_processes",
    )
    process_keys = {"pid", "ppid", "start_ticks", "rss_kib", "command", "cwd"}
    try:
        reported_cwd = Path(str(value.get("cwd", ""))).resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"{description} cwd is noncanonical") from error
    if (
        value.get("schema_version") != contract.SCHEMA_VERSION
        or value.get("kind") != contract.POLICY_KIND
        or value.get("mode") != contract.MODE
        or value.get("workspace") != str(ROOT)
        or reported_cwd != Path(str(value.get("cwd")))
        or ROOT not in (reported_cwd, *reported_cwd.parents)
        or any(
            Path(root) in (reported_cwd, *reported_cwd.parents)
            for root in base.PROTECTED_ROOTS
        )
        or value.get("protected_cpu_only_coexistence_requested") is not True
        or value.get("launch_memory_floor_kib")
        != contract.LAUNCH_MEMORY_FLOOR_KIB
        or value.get("resume_memory_floor_kib")
        != contract.RESUME_MEMORY_FLOOR_KIB
        or value.get("pause_memory_floor_kib")
        != contract.PAUSE_MEMORY_FLOOR_KIB
        or value.get("admission_policy") != policy_document()
        or value.get("policy")
        != "observation-only; no external process is ever signalled"
        or not isinstance(failures, list)
        or any(not isinstance(item, str) or not item for item in failures)
        or not isinstance(value.get("ok"), bool)
        or value.get("ok") != (not failures)
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or memory < 0
        or not isinstance(value.get("nvidia_compute_audit_ok"), bool)
        or any(
            not isinstance(value.get(name), list)
            or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                for pid in value[name]
            )
            for name in pid_fields
        )
        or any(
            not isinstance(value.get(name), list)
            or any(
                not isinstance(record, dict) or set(record) != process_keys
                for record in value[name]
            )
            for name in process_fields
        )
        or not isinstance(value.get("gpu"), dict)
        or not isinstance(value.get("torch"), dict)
        or not isinstance(value.get("protected_activity_waiver_applied"), bool)
        or value.get("protected_activity_waiver_applied")
        != (bool(protected_heavy) and value.get("ok") is True)
    ):
        raise PreflightError(f"{description} policy identity differs")
    if value.get("ok") is True and (
        memory < contract.LAUNCH_MEMORY_FLOOR_KIB
        or value.get("nvidia_compute_audit_ok") is not True
        or value.get("protected_nvidia_compute_pids")
        or value.get("protected_nvidia_runtime_mapping_pids")
        or value.get("unreadable_protected_maps")
    ):
        raise PreflightError(f"{description} accepts an unsafe observation")
    return value


def build_report() -> dict[str, object]:
    cwd = Path.cwd().resolve()
    failures: list[str] = []
    if ROOT not in (cwd, *cwd.parents):
        failures.append(f"cwd is outside workspace: {cwd}")
    if any(Path(root) in (cwd, *cwd.parents) for root in base.PROTECTED_ROOTS):
        failures.append(f"cwd is inside protected scope: {cwd}")
    leaked_env = sorted(
        name
        for name in os.environ
        if name in base.FORBIDDEN_ENV_NAMES
        or name.startswith(base.FORBIDDEN_ENV_PREFIXES)
    )
    if leaked_env:
        failures.append(f"forbidden AMD/simulator environment variables: {leaked_env}")

    all_processes, protected, workspace = process_table()
    protected_heavy = [item for item in protected if is_heavy_command(item.command)]
    available_kib = mem_available_kib()
    if available_kib < contract.LAUNCH_MEMORY_FLOOR_KIB:
        failures.append(
            f"MemAvailable {available_kib / 1024 / 1024:.1f} GiB is below "
            "22 GiB CPU-v2 launch floor"
        )

    try:
        gpu = nvidia_identity()
    except Exception as error:
        gpu = {"error": f"{type(error).__name__}: {error}"}
        failures.append("nvidia-smi identity check failed")
    if (
        gpu.get("name") != contract.EXPECTED_DEVICE_NAME
        or gpu.get("compute_capability") != contract.EXPECTED_COMPUTE_CAPABILITY
        or gpu.get("driver") != contract.EXPECTED_DRIVER_RELEASE
    ):
        failures.append(f"expected SM120 GPU identity, got {gpu}")

    torch = static_torch_identity()
    if torch.get("static_identity_error"):
        failures.append(str(torch["static_identity_error"]))
    if torch.get("hip") is not None:
        failures.append(f"PyTorch static identity reports HIP: {torch.get('hip')}")
    if torch.get("forbidden_dsos"):
        failures.append(f"forbidden DSOs loaded: {torch['forbidden_dsos']}")

    compute_pids: set[int] = set()
    nvidia_compute_audit_ok = False
    protected_compute: list[int] = []
    protected_runtime: list[int] = []
    unreadable: list[int] = []
    try:
        compute_pids = nvidia_compute_pids()
        nvidia_compute_audit_ok = True
    except Exception as error:
        failures.append(
            "cannot audit NVIDIA compute processes: "
            f"{type(error).__name__}: {error}"
        )
    if protected:
        protected_ids = {item.pid for item in protected}
        protected_compute = sorted(compute_pids & protected_ids)
        try:
            protected_runtime, unreadable = protected_nvidia_runtime_mappings(protected)
        except Exception as error:
            failures.append(
                "cannot audit protected NVIDIA runtime mappings: "
                f"{type(error).__name__}: {error}"
            )
    if protected_compute:
        failures.append(
            f"protected workload has active NVIDIA compute: {protected_compute}"
        )
    if protected_runtime:
        failures.append(
            f"protected workload has NVIDIA runtime mappings: {protected_runtime}"
        )
    if unreadable:
        failures.append(f"protected process maps are unreadable: {unreadable}")

    waiver = bool(protected_heavy) and not failures and nvidia_compute_audit_ok
    report = {
        "schema_version": contract.SCHEMA_VERSION,
        "kind": contract.POLICY_KIND,
        "mode": contract.MODE,
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "ok": not failures,
        "failures": failures,
        "mem_available_kib": available_kib,
        "launch_memory_floor_kib": contract.LAUNCH_MEMORY_FLOOR_KIB,
        "resume_memory_floor_kib": contract.RESUME_MEMORY_FLOOR_KIB,
        "pause_memory_floor_kib": contract.PAUSE_MEMORY_FLOOR_KIB,
        "protected_cpu_only_coexistence_requested": True,
        "protected_activity_waiver_applied": waiver,
        "gpu": gpu,
        "torch": torch,
        "protected_processes": [asdict(item) for item in protected],
        "protected_heavy_processes": [asdict(item) for item in protected_heavy],
        "workspace_processes": [asdict(item) for item in workspace],
        "nvidia_compute_audit_ok": nvidia_compute_audit_ok,
        "nvidia_compute_pids": sorted(compute_pids),
        "protected_nvidia_compute_pids": protected_compute,
        "protected_nvidia_runtime_mapping_pids": protected_runtime,
        "unreadable_protected_maps": unreadable,
        "policy": "observation-only; no external process is ever signalled",
        "admission_policy": policy_document(),
    }
    return validate_report(report, description="CPU-v2 preflight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(contract.MODE,), required=True)
    parser.add_argument("--json", action="store_true", dest="json_only")
    args = parser.parse_args()
    if not (
        sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        parser.error("CPU-v2 preflight requires Python -E -B -S")
    report = build_report()
    print(json.dumps(report, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if report["ok"] is True else 75


if __name__ == "__main__":
    raise SystemExit(main())
