#!/usr/bin/env python3
"""Fail-closed workspace, NVIDIA, and protected-workload preflight.

This tool is observation-only. It never sends signals or changes external
processes. The default remains fail-closed. An explicit user-authorized
coexistence mode permits bounded non-benchmark heavy work only when a successful
NVIDIA compute-process query has no protected PID and memory is above its floor.
Loaded NVIDIA-library mappings remain diagnostic because loading is not active
compute.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import asdict, dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTECTED_ROOTS = (
    "/home/zhaosiying/amdgpu-sim",
    "/home/zhaosiying/zcode-lane",
    "/home/zhaosiying/amdgpu-sim-agentenv",
)
HEAVY_MARKERS = (
    "gem5.opt",
    "sglang::scheduler",
    "sglang::schedul",
    "scons",
    "ninja",
    "run_model_lane.sh",
    "run_engine_lane.sh",
    "examples/sglang/qwen35_inference.py",
    "sglang.launch_server",
    "python -m sglang",
    "vllm.entrypoints",
    "python -m vllm",
    "qwen35_inference.py",
)
FORBIDDEN_ENV_PREFIXES = ("HSA_", "ROCR_", "GEMSIM_", "AMDGPU_SIM_")
FORBIDDEN_ENV_NAMES = {
    "ROCM_PATH",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "PYTORCH_ROCM_ARCH",
    "TORCHDYNAMO_SUPPRESS_ERRORS",
}
FORBIDDEN_DSO_MARKERS = (
    "libamdhip64",
    "libhsa-runtime64",
    "self-amdgpu-runtime",
    "gemsim",
)
NVIDIA_PROCESS_MAP_MARKERS = (
    "libcuda.so",
    "libcudart.so",
    "libtorch_cuda",
    "libnvidia-",
)
STANDARD_HEAVY_MEMORY_FLOOR_KIB = 32 * 1024 * 1024
COEXISTENCE_MEMORY_FLOOR_KIB = 24 * 1024 * 1024
PREFLIGHT_POLICY_VERSION = 2
COEXISTENCE_POLICY_VERSION = 1


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    start_ticks: int
    rss_kib: int
    command: str
    cwd: str


def belongs_to_roots(command: str, cwd: str, roots: tuple[str, ...]) -> bool:
    return any(
        root in command
        or cwd == root
        or (cwd and pathlib.Path(root) in pathlib.Path(cwd).parents)
        for root in roots
    )


def is_heavy_command(command: str) -> bool:
    lowered = command.lower()
    return any(marker.lower() in lowered for marker in HEAVY_MARKERS)


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(errors="replace")
    except (OSError, UnicodeError):
        return ""


def process_table() -> tuple[list[ProcessInfo], list[ProcessInfo], list[ProcessInfo]]:
    all_processes: list[ProcessInfo] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        raw = b""
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        if not command:
            continue
        try:
            cwd = str((entry / "cwd").resolve(strict=True))
        except OSError:
            cwd = ""
        rss_kib = 0
        for line in read_text(entry / "status").splitlines():
            if line.startswith("VmRSS:"):
                try:
                    rss_kib = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
                break
        ppid = 0
        start_ticks = 0
        try:
            stat_tail = (entry / "stat").read_text().rpartition(")")[2].split()
            ppid = int(stat_tail[1])
            start_ticks = int(stat_tail[19])
        except (OSError, IndexError, ValueError):
            pass
        info = ProcessInfo(
            pid=pid,
            ppid=ppid,
            start_ticks=start_ticks,
            rss_kib=rss_kib,
            command=command,
            cwd=cwd,
        )
        all_processes.append(info)
    directly_protected = [
        process
        for process in all_processes
        if belongs_to_roots(process.command, process.cwd, PROTECTED_ROOTS)
    ]
    directly_workspace = [
        process
        for process in all_processes
        if belongs_to_roots(process.command, process.cwd, (str(ROOT),))
    ]
    protected = process_descendant_closure(all_processes, directly_protected)
    workspace = process_descendant_closure(all_processes, directly_workspace)
    return all_processes, protected, workspace


def process_descendant_closure(
    all_processes: list[ProcessInfo], seeds: list[ProcessInfo]
) -> list[ProcessInfo]:
    selected_pids = {process.pid for process in seeds}
    changed = True
    while changed:
        changed = False
        for process in all_processes:
            if process.pid in selected_pids or process.ppid not in selected_pids:
                continue
            selected_pids.add(process.pid)
            changed = True
    return [process for process in all_processes if process.pid in selected_pids]


def mem_available_kib() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def nvidia_identity() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd, check=True, text=True, capture_output=True, timeout=10
    )
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return dict(zip(("name", "compute_capability", "memory_mib", "used_mib", "driver"), fields))


def nvidia_compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=5,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit():
            raise RuntimeError(f"unexpected NVIDIA compute PID: {value!r}")
        pids.add(int(value))
    return pids


def protected_nvidia_runtime_mappings(
    processes: list[ProcessInfo],
) -> tuple[list[int], list[int]]:
    observed: set[int] = set()
    unreadable: list[int] = []
    for process in processes:
        stat_path = pathlib.Path(f"/proc/{process.pid}/stat")
        try:
            stat_tail = stat_path.read_text().rpartition(")")[2].split()
            if int(stat_tail[19]) != process.start_ticks:
                unreadable.append(process.pid)
                continue
            mappings = pathlib.Path(f"/proc/{process.pid}/maps").read_text(
                errors="replace"
            ).lower()
        except (OSError, IndexError, ValueError):
            if pathlib.Path(f"/proc/{process.pid}").exists():
                unreadable.append(process.pid)
            continue
        if any(marker in mappings for marker in NVIDIA_PROCESS_MAP_MARKERS):
            observed.add(process.pid)
    return sorted(observed), sorted(unreadable)


def protected_lane_processes(
    all_processes: list[ProcessInfo], protected_heavy: list[ProcessInfo]
) -> list[ProcessInfo]:
    by_pid = {process.pid: process for process in all_processes}
    selected_pids = {process.pid for process in protected_heavy}
    for process in protected_heavy:
        parent_pid = process.ppid
        while parent_pid in by_pid:
            parent = by_pid[parent_pid]
            if not belongs_to_roots(parent.command, parent.cwd, PROTECTED_ROOTS):
                break
            selected_pids.add(parent.pid)
            parent_pid = parent.ppid
    changed = True
    while changed:
        changed = False
        for process in all_processes:
            if process.pid in selected_pids or process.ppid not in selected_pids:
                continue
            selected_pids.add(process.pid)
            changed = True
    return [process for process in all_processes if process.pid in selected_pids]


def heavy_policy_failures(
    *,
    mode: str,
    coexistence_authorized: bool,
    protected_heavy: list[ProcessInfo],
    available_kib: int,
    protected_nvidia_compute_pids: list[int],
) -> list[str]:
    if mode == "light":
        return []
    failures: list[str] = []
    if coexistence_authorized:
        if mode != "heavy":
            failures.append(
                "protected coexistence is only valid for non-benchmark heavy work"
            )
        if protected_nvidia_compute_pids:
            failures.append(
                "protected workload has active NVIDIA compute processes: "
                f"{protected_nvidia_compute_pids}"
            )
        memory_floor_kib = COEXISTENCE_MEMORY_FLOOR_KIB
    else:
        if protected_heavy:
            failures.append(
                "protected zcode/gem5/SGLang heavy workload is active; pause this project"
            )
        memory_floor_kib = STANDARD_HEAVY_MEMORY_FLOOR_KIB
    if available_kib < memory_floor_kib:
        failures.append(
            f"MemAvailable {available_kib / 1024 / 1024:.1f} GiB is below "
            f"{memory_floor_kib / 1024 / 1024:.0f} GiB safety floor"
        )
    return failures


def torch_identity() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment bootstrap gate
        return {"import_error": f"{type(exc).__name__}: {exc}"}
    identity: dict[str, object] = {
        "version": torch.__version__,
        "git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        identity["device"] = torch.cuda.get_device_name(0)
        identity["capability"] = list(torch.cuda.get_device_capability(0))
    maps = read_text(pathlib.Path("/proc/self/maps")).lower()
    identity["forbidden_dsos"] = sorted(
        marker for marker in FORBIDDEN_DSO_MARKERS if marker in maps
    )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("light", "heavy", "gpu-benchmark"), default="light")
    parser.add_argument("--json", action="store_true", dest="json_only")
    parser.add_argument(
        "--allow-protected-cpu-only-coexistence",
        action="store_true",
        help=(
            "permit explicitly authorized, non-benchmark heavy work beside a "
            "protected lane with no active NVIDIA compute PID"
        ),
    )
    args = parser.parse_args()
    if args.allow_protected_cpu_only_coexistence and args.mode != "heavy":
        parser.error("--allow-protected-cpu-only-coexistence requires --mode heavy")

    cwd = pathlib.Path.cwd().resolve()
    failures: list[str] = []
    if ROOT not in (cwd, *cwd.parents):
        failures.append(f"cwd is outside workspace: {cwd}")
    if any(pathlib.Path(root) in (cwd, *cwd.parents) for root in PROTECTED_ROOTS):
        failures.append(f"cwd is inside protected scope: {cwd}")

    leaked_env = sorted(
        name
        for name in os.environ
        if name in FORBIDDEN_ENV_NAMES or name.startswith(FORBIDDEN_ENV_PREFIXES)
    )
    if leaked_env:
        failures.append(f"forbidden AMD/simulator environment variables: {leaked_env}")

    all_processes, protected, workspace = process_table()
    protected_heavy = [
        proc for proc in protected if is_heavy_command(proc.command)
    ]
    available_kib = mem_available_kib()
    try:
        gpu = nvidia_identity()
    except Exception as exc:
        gpu = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("nvidia-smi identity check failed")
    if gpu.get("compute_capability") != "12.0":
        failures.append(f"expected SM120 GPU, got {gpu}")

    torch = torch_identity()
    if torch.get("hip") is not None:
        failures.append(f"PyTorch reports HIP runtime: {torch.get('hip')}")
    if torch.get("forbidden_dsos"):
        failures.append(f"forbidden DSOs loaded: {torch['forbidden_dsos']}")

    compute_pids: set[int] = set()
    nvidia_compute_audit_ok = args.mode == "light"
    protected_nvidia_compute_pids: list[int] = []
    protected_nvidia_runtime_pids: list[int] = []
    unreadable_protected_maps: list[int] = []
    if args.mode != "light":
        try:
            compute_pids = nvidia_compute_pids()
            nvidia_compute_audit_ok = True
        except Exception as exc:
            if (
                args.allow_protected_cpu_only_coexistence
                or args.mode == "gpu-benchmark"
            ):
                failures.append(
                    f"cannot audit NVIDIA compute processes: {type(exc).__name__}: {exc}"
                )
        if args.allow_protected_cpu_only_coexistence and protected_heavy:
            protected_lane_pids = {process.pid for process in protected}
            protected_nvidia_compute_pids = sorted(
                compute_pids & protected_lane_pids
            )
            protected_nvidia_runtime_pids, unreadable_protected_maps = (
                protected_nvidia_runtime_mappings(protected)
            )
        failures.extend(
            heavy_policy_failures(
                mode=args.mode,
                coexistence_authorized=args.allow_protected_cpu_only_coexistence,
                protected_heavy=protected_heavy,
                available_kib=available_kib,
                protected_nvidia_compute_pids=protected_nvidia_compute_pids,
            )
        )
        if args.mode == "gpu-benchmark":
            external_compute_pids = sorted(compute_pids - {os.getpid()})
            if external_compute_pids:
                failures.append(
                    f"external NVIDIA compute processes are active: "
                    f"{external_compute_pids}"
                )

    report = {
        "policy_version": PREFLIGHT_POLICY_VERSION,
        "coexistence_policy_version": COEXISTENCE_POLICY_VERSION,
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "mode": args.mode,
        "protected_cpu_only_coexistence_requested": (
            args.allow_protected_cpu_only_coexistence
        ),
        "protected_activity_waiver_applied": (
            args.allow_protected_cpu_only_coexistence
            and bool(protected_heavy)
            and nvidia_compute_audit_ok
            and not protected_nvidia_compute_pids
            and not failures
        ),
        "ok": not failures,
        "failures": failures,
        "mem_available_kib": available_kib,
        "memory_floor_kib": (
            0
            if args.mode == "light"
            else (
                COEXISTENCE_MEMORY_FLOOR_KIB
                if args.allow_protected_cpu_only_coexistence
                else STANDARD_HEAVY_MEMORY_FLOOR_KIB
            )
        ),
        "gpu": gpu,
        "torch": torch,
        "protected_processes": [asdict(proc) for proc in protected],
        "protected_heavy_processes": [asdict(proc) for proc in protected_heavy],
        "nvidia_compute_pids": sorted(compute_pids),
        "nvidia_compute_audit_ok": nvidia_compute_audit_ok,
        "protected_nvidia_compute_pids": protected_nvidia_compute_pids,
        "protected_nvidia_runtime_mapping_pids": protected_nvidia_runtime_pids,
        "unreadable_protected_maps": unreadable_protected_maps,
        "workspace_processes": [asdict(proc) for proc in workspace],
        "policy": "observation-only; no external process is ever signalled",
    }
    print(json.dumps(report, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if not failures else 75


if __name__ == "__main__":
    sys.exit(main())
