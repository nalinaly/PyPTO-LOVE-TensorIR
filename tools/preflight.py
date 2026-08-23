#!/usr/bin/env python3
"""Fail-closed workspace, NVIDIA, and protected-workload preflight.

This tool is observation-only. It never sends signals or changes external
processes. Heavy work exits 75 while protected zcode/gem5/SGLang workloads are
active or host memory is below the safety floor.
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


@dataclass
class ProcessInfo:
    pid: int
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


def process_table() -> tuple[list[ProcessInfo], list[ProcessInfo]]:
    protected: list[ProcessInfo] = []
    workspace: list[ProcessInfo] = []
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
        info = ProcessInfo(pid=pid, rss_kib=rss_kib, command=command, cwd=cwd)
        if belongs_to_roots(command, cwd, PROTECTED_ROOTS):
            protected.append(info)
        if belongs_to_roots(command, cwd, (str(ROOT),)):
            workspace.append(info)
    return protected, workspace


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
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return dict(zip(("name", "compute_capability", "memory_mib", "used_mib", "driver"), fields))


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
    args = parser.parse_args()

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

    protected, workspace = process_table()
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

    if args.mode != "light":
        if protected_heavy:
            failures.append(
                "protected zcode/gem5/SGLang heavy workload is active; pause this project"
            )
        if available_kib < 32 * 1024 * 1024:
            failures.append(
                f"MemAvailable {available_kib / 1024 / 1024:.1f} GiB is below 32 GiB safety floor"
            )

    report = {
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "mode": args.mode,
        "ok": not failures,
        "failures": failures,
        "mem_available_kib": available_kib,
        "gpu": gpu,
        "torch": torch,
        "protected_processes": [asdict(proc) for proc in protected],
        "protected_heavy_processes": [asdict(proc) for proc in protected_heavy],
        "workspace_processes": [asdict(proc) for proc in workspace],
        "policy": "observation-only; no external process is ever signalled",
    }
    print(json.dumps(report, indent=None if args.json_only else 2, sort_keys=True))
    return 0 if not failures else 75


if __name__ == "__main__":
    sys.exit(main())
