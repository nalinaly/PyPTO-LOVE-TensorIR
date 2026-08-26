#!/usr/bin/env python3
"""Fail-closed workspace, NVIDIA, and protected-workload preflight.

This tool is observation-only. It never sends signals or changes external
processes. The default remains fail-closed. An explicit user-authorized
coexistence mode permits bounded non-benchmark heavy work only when a successful
NVIDIA compute-process query has no protected PID and memory is above its floor.
A separate fixed-command correctness-smoke policy may coexist with the protected
CPU lane only while both NVIDIA runtime mappings and compute PIDs remain absent
from that lane. Performance benchmarks remain fully exclusive.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass

import _pypto_nvidia_executable_sm120_contract as nvidia_smoke_contract


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
COEXISTENCE_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
GPU_SMOKE_MEMORY_FLOOR_KIB = 24 * 1024 * 1024
GPU_SMOKE_FREE_MEMORY_FLOOR_MIB = 4 * 1024
PREFLIGHT_POLICY_VERSION = 3
COEXISTENCE_POLICY_VERSION = 2
GPU_SMOKE_POLICY_VERSION = 1


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


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_identity() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=10)
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return dict(
        zip(("name", "compute_capability", "memory_mib", "used_mib", "driver"), fields)
    )


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
            mappings = (
                pathlib.Path(f"/proc/{process.pid}/maps")
                .read_text(errors="replace")
                .lower()
            )
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
    gpu_smoke_coexistence_authorized: bool = False,
    protected_heavy: list[ProcessInfo],
    available_kib: int,
    protected_nvidia_compute_pids: list[int],
    protected_nvidia_runtime_pids: list[int] | None = None,
    unreadable_protected_maps: list[int] | None = None,
) -> list[str]:
    if mode == "light":
        return []
    protected_nvidia_runtime_pids = protected_nvidia_runtime_pids or []
    unreadable_protected_maps = unreadable_protected_maps or []
    failures: list[str] = []
    if gpu_smoke_coexistence_authorized:
        if mode != "gpu-smoke":
            failures.append(
                "protected zero-NVIDIA GPU-smoke coexistence requires gpu-smoke mode"
            )
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
        memory_floor_kib = GPU_SMOKE_MEMORY_FLOOR_KIB
    elif coexistence_authorized:
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


def static_torch_identity() -> dict[str, object]:
    """Authenticate selected Python/Torch/libcudart without importing code."""

    lock_path = ROOT / "ENVIRONMENT.lock"
    try:
        raw = lock_path.read_bytes()
        locked = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return {"static_identity_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(locked, dict):
        return {"static_identity_error": "ENVIRONMENT.lock is not an object"}
    expected_lock_fields = {
        "schema": 1,
        "destination_prefix": "envs/pypto-nvidia",
        "python_abi": "cp314",
        "torch": nvidia_smoke_contract.EXPECTED_TORCH_VERSION,
        "torch_git": nvidia_smoke_contract.EXPECTED_TORCH_GIT,
        "cuda": nvidia_smoke_contract.EXPECTED_TORCH_CUDA,
        "hip": None,
    }
    mismatches = {
        name: {"expected": value, "actual": locked.get(name)}
        for name, value in expected_lock_fields.items()
        if locked.get(name) != value
    }
    if (
        hashlib.sha256(raw).hexdigest() != nvidia_smoke_contract.ENVIRONMENT_LOCK_SHA256
        or mismatches
    ):
        return {
            "static_identity_error": (
                "ENVIRONMENT.lock differs from the fixed GPU-smoke contract: "
                f"{mismatches}"
            )
        }
    prefix = ROOT / "envs" / "pypto-nvidia"
    runtime = ROOT / nvidia_smoke_contract.CUDA_RUNTIME_RELATIVE_PATH
    python = ROOT / nvidia_smoke_contract.PYTHON_REAL_RELATIVE_PATH
    try:
        runtime_resolved = runtime.resolve(strict=True)
        python_resolved = python.resolve(strict=True)
        if (
            runtime.absolute() != runtime_resolved
            or python.absolute() != python_resolved
        ):
            raise RuntimeError("selected Python or libcudart contains a symlinked path")
        runtime_stat = runtime.stat()
        python_stat = python.stat()
        if not stat.S_ISREG(runtime_stat.st_mode) or not stat.S_ISREG(
            python_stat.st_mode
        ):
            raise RuntimeError("selected Python or libcudart is not a regular file")
        if runtime.is_symlink() or python.is_symlink():
            raise RuntimeError("selected Python or libcudart is a symlink")
        if (
            stat.S_IMODE(runtime_stat.st_mode) != 0o644
            or runtime_stat.st_nlink != 1
            or runtime_stat.st_uid != os.getuid()
        ):
            raise RuntimeError("libcudart ownership, mode, or link count is invalid")
        if (
            runtime_stat.st_size != nvidia_smoke_contract.CUDA_RUNTIME_SIZE
            or sha256_file(runtime) != nvidia_smoke_contract.CUDA_RUNTIME_SHA256
        ):
            raise RuntimeError("libcudart bytes differ from the fixed contract")
        if (
            python_stat.st_size != nvidia_smoke_contract.PYTHON_SIZE
            or sha256_file(python) != nvidia_smoke_contract.PYTHON_SHA256
        ):
            raise RuntimeError("selected Python bytes differ from the fixed contract")
        locked_python = pathlib.Path(str(locked.get("python_executable", "")))
        if locked_python != python_resolved:
            raise RuntimeError(
                "ENVIRONMENT.lock Python path differs from selected Python"
            )
        site_candidates = sorted((prefix / "lib").glob("python*/site-packages"))
        if len(site_candidates) != 1:
            raise RuntimeError("selected prefix does not have one site-packages")
        site = site_candidates[0]
        distributions = sorted(
            site.glob(
                f"nvidia_cuda_runtime-{nvidia_smoke_contract.CUDA_RUNTIME_VERSION}.dist-info"
            )
        )
        if len(distributions) != 1:
            raise RuntimeError(
                "selected prefix does not have one CUDA Runtime dist-info"
            )
        metadata = (distributions[0] / "METADATA").read_text(encoding="utf-8")
        if (
            f"Name: {nvidia_smoke_contract.CUDA_RUNTIME_DISTRIBUTION}\n" not in metadata
            or f"Version: {nvidia_smoke_contract.CUDA_RUNTIME_VERSION}\n"
            not in metadata
        ):
            raise RuntimeError("CUDA Runtime distribution metadata differs")
        record = (distributions[0] / "RECORD").read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO(record)))
        relative_runtime = runtime.relative_to(site).as_posix()
        matches = [row for row in rows if row and row[0] == relative_runtime]
        expected_record_digest = (
            base64.urlsafe_b64encode(
                bytes.fromhex(nvidia_smoke_contract.CUDA_RUNTIME_SHA256)
            )
            .decode("ascii")
            .rstrip("=")
        )
        if matches != [
            [
                relative_runtime,
                f"sha256={expected_record_digest}",
                str(nvidia_smoke_contract.CUDA_RUNTIME_SIZE),
            ]
        ]:
            raise RuntimeError("CUDA Runtime RECORD ownership differs")
        command_python = prefix / "bin" / "python"
        if (
            not command_python.is_symlink()
            or command_python.resolve(strict=True) != python_resolved
        ):
            raise RuntimeError("fixed command Python symlink differs")
        runtime_candidates = sorted(
            (prefix / "lib").glob("python*/site-packages/**/libcudart.so*")
        )
        if runtime_candidates != [runtime]:
            raise RuntimeError("selected prefix has an unexpected libcudart set")
    except (OSError, RuntimeError, UnicodeError) as exc:
        return {"static_identity_error": f"{type(exc).__name__}: {exc}"}
    try:
        maps = (
            pathlib.Path("/proc/self/maps")
            .read_text(encoding="utf-8", errors="strict")
            .lower()
        )
    except (OSError, UnicodeError) as exc:
        return {
            "static_identity_error": (
                "cannot audit GPU-smoke preflight NVIDIA mappings: "
                f"{type(exc).__name__}: {exc}"
            )
        }
    nvidia_mappings = sorted(
        marker for marker in NVIDIA_PROCESS_MAP_MARKERS if marker in maps
    )
    if nvidia_mappings:
        return {
            "static_identity_error": (
                "GPU-smoke preflight process already has NVIDIA runtime mappings: "
                f"{nvidia_mappings}"
            )
        }
    try:
        if runtime_resolved != runtime or not runtime.is_file():
            raise RuntimeError("libcudart provider is not a regular non-symlink file")
    except (OSError, RuntimeError) as exc:
        return {"static_identity_error": f"{type(exc).__name__}: {exc}"}
    return {
        "source": "static ENVIRONMENT.lock and selected-prefix file audit",
        "environment_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "version": locked.get("torch"),
        "git_version": locked.get("torch_git"),
        "cuda": locked.get("cuda"),
        "hip": locked.get("hip"),
        "python_executable": locked.get("python_executable"),
        "libcudart_path": str(runtime_resolved),
        "libcudart_size": runtime_stat.st_size,
        "libcudart_sha256": nvidia_smoke_contract.CUDA_RUNTIME_SHA256,
        "libcudart_record_owned": True,
        "nvidia_runtime_mappings": nvidia_mappings,
        "cuda_initialized": False,
        "forbidden_dsos": sorted(
            marker for marker in FORBIDDEN_DSO_MARKERS if marker in maps
        ),
    }


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
    parser.add_argument(
        "--mode",
        choices=("light", "heavy", "gpu-smoke", "gpu-benchmark"),
        default="light",
    )
    parser.add_argument("--json", action="store_true", dest="json_only")
    parser.add_argument(
        "--allow-protected-cpu-only-coexistence",
        action="store_true",
        help=(
            "permit explicitly authorized, non-benchmark heavy work beside a "
            "protected lane with no active NVIDIA compute PID"
        ),
    )
    parser.add_argument(
        "--allow-protected-zero-nvidia-gpu-smoke",
        action="store_true",
        help=(
            "permit only the fixed correctness-smoke lane beside a protected "
            "CPU workload with no NVIDIA runtime mapping or compute PID"
        ),
    )
    args = parser.parse_args()
    if args.allow_protected_cpu_only_coexistence and args.mode != "heavy":
        parser.error("--allow-protected-cpu-only-coexistence requires --mode heavy")
    if args.allow_protected_zero_nvidia_gpu_smoke and args.mode != "gpu-smoke":
        parser.error(
            "--allow-protected-zero-nvidia-gpu-smoke requires --mode gpu-smoke"
        )
    if (
        args.allow_protected_cpu_only_coexistence
        and args.allow_protected_zero_nvidia_gpu_smoke
    ):
        parser.error("protected coexistence policies are mutually exclusive")

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
    protected_heavy = [proc for proc in protected if is_heavy_command(proc.command)]
    available_kib = mem_available_kib()
    try:
        gpu = nvidia_identity()
    except Exception as exc:
        gpu = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("nvidia-smi identity check failed")
    if gpu.get("compute_capability") != "12.0":
        failures.append(f"expected SM120 GPU, got {gpu}")

    torch = static_torch_identity() if args.mode == "gpu-smoke" else torch_identity()
    if torch.get("static_identity_error"):
        failures.append(str(torch["static_identity_error"]))
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
            failures.append(
                f"cannot audit NVIDIA compute processes: {type(exc).__name__}: {exc}"
            )
        if (
            args.mode == "gpu-smoke"
            or args.allow_protected_cpu_only_coexistence
            or args.allow_protected_zero_nvidia_gpu_smoke
        ) and protected:
            protected_lane_pids = {process.pid for process in protected}
            protected_nvidia_compute_pids = sorted(compute_pids & protected_lane_pids)
            protected_nvidia_runtime_pids, unreadable_protected_maps = (
                protected_nvidia_runtime_mappings(protected)
            )
        failures.extend(
            heavy_policy_failures(
                mode=args.mode,
                coexistence_authorized=args.allow_protected_cpu_only_coexistence,
                gpu_smoke_coexistence_authorized=(
                    args.allow_protected_zero_nvidia_gpu_smoke
                ),
                protected_heavy=protected_heavy,
                available_kib=available_kib,
                protected_nvidia_compute_pids=protected_nvidia_compute_pids,
                protected_nvidia_runtime_pids=protected_nvidia_runtime_pids,
                unreadable_protected_maps=unreadable_protected_maps,
            )
        )
        if args.mode in {"gpu-smoke", "gpu-benchmark"}:
            external_compute_pids = sorted(compute_pids - {os.getpid()})
            if external_compute_pids:
                failures.append(
                    f"external NVIDIA compute processes are active: "
                    f"{external_compute_pids}"
                )
        if args.mode == "gpu-smoke":
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

    report = {
        "policy_version": PREFLIGHT_POLICY_VERSION,
        "coexistence_policy_version": COEXISTENCE_POLICY_VERSION,
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "mode": args.mode,
        "protected_cpu_only_coexistence_requested": (
            args.allow_protected_cpu_only_coexistence
        ),
        "protected_zero_nvidia_gpu_smoke_requested": (
            args.allow_protected_zero_nvidia_gpu_smoke
        ),
        "protected_activity_waiver_applied": (
            args.allow_protected_cpu_only_coexistence
            and bool(protected_heavy)
            and nvidia_compute_audit_ok
            and not protected_nvidia_compute_pids
            and not failures
        ),
        "protected_gpu_smoke_waiver_applied": (
            args.allow_protected_zero_nvidia_gpu_smoke
            and bool(protected)
            and nvidia_compute_audit_ok
            and not protected_nvidia_compute_pids
            and not protected_nvidia_runtime_pids
            and not unreadable_protected_maps
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
                else (
                    GPU_SMOKE_MEMORY_FLOOR_KIB
                    if args.allow_protected_zero_nvidia_gpu_smoke
                    else STANDARD_HEAVY_MEMORY_FLOOR_KIB
                )
            )
        ),
        "gpu_smoke_policy_version": GPU_SMOKE_POLICY_VERSION,
        "gpu_smoke_free_memory_floor_mib": (
            GPU_SMOKE_FREE_MEMORY_FLOOR_MIB if args.mode == "gpu-smoke" else 0
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
