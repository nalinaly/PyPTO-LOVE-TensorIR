#!/usr/bin/env python3
"""Pure contract for the additive CPU-only coexistence policy v2."""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
POLICY_KIND = "pypto-cpu-only-coexistence-v2"
MODE = "cpu-heavy"
LAUNCH_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
RESUME_MEMORY_FLOOR_KIB = 22 * 1024 * 1024
PAUSE_MEMORY_FLOOR_KIB = 16 * 1024 * 1024
POLL_SECONDS = 5
DEFAULT_TIMEOUT_SECONDS = 3_600
DEFAULT_MINIMUM_FREE_DISK_GIB = 64
START_GATE_SCHEMA_VERSION = 1
START_GATE_TIMEOUT_SECONDS = 60
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"
EXPECTED_COMPUTE_CAPABILITY = "12.0"
EXPECTED_DRIVER_RELEASE = "610.74"

BASE_NVIDIA_CONTRACT_RELATIVE_PATH = Path(
    "tools/_pypto_nvidia_executable_sm120_contract.py"
)
BASE_NVIDIA_CONTRACT_SIZE = 4_783
BASE_NVIDIA_CONTRACT_SHA256 = (
    "fa477d91933df765e9163bf3081ed6d41f323bb49285106dcdbd4bee554113bf"
)
BASE_NVIDIA_CONTROL_RELATIVE_PATH = Path(
    "tools/_pypto_nvidia_sm120_control_manifest.py"
)
BASE_NVIDIA_CONTROL_SIZE = 7_491
BASE_NVIDIA_CONTROL_SHA256 = (
    "bfa0e5c66ffad9435c0c31dc82ed7581d6bee608f1487e3fb2932cabbb2b597a"
)
BASE_NVIDIA_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_nvidia_executable_sm120_v4.json"
)
BASE_NVIDIA_MANIFEST_SIZE = 1_569
BASE_NVIDIA_MANIFEST_SHA256 = (
    "a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf"
)
BASE_PREFLIGHT_RELATIVE_PATH = Path("tools/preflight.py")
BASE_PREFLIGHT_SIZE = 27_684
BASE_PREFLIGHT_SHA256 = (
    "0b9884f8dbd34337a85f62c351b1e19dda3a8b84ec9a88c835d8701af053e3d1"
)
BASE_ISOLATION_RELATIVE_PATH = Path("tools/run_isolated.py")
BASE_ISOLATION_SIZE = 76_558
BASE_ISOLATION_SHA256 = (
    "978686ac09743a98233c9616d23b04e57d3a257bd643d5db3b8a71eaac7465c8"
)
BASE_STOP_RELATIVE_PATH = Path("tools/stop_run.py")
BASE_STOP_SIZE = 7_885
BASE_STOP_SHA256 = (
    "879a2e3863671531a548c71d788d56298500eab989bd1420d2c7ae01717ddfe4"
)

CONTRACT_RELATIVE_PATH = Path("tools/_pypto_cpu_coexistence_v2_contract.py")
PREFLIGHT_RELATIVE_PATH = Path("tools/preflight_cpu_coexistence_v2.py")
CONTROLLER_RELATIVE_PATH = Path("tools/run_pypto_cpu_coexistence_v2_isolated.py")
CONTROL_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_cpu_coexistence_v2_control_manifest.py"
)
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_cpu_coexistence_v2.json")
CONTROL_PATHS = (
    CONTRACT_RELATIVE_PATH.as_posix(),
    PREFLIGHT_RELATIVE_PATH.as_posix(),
    CONTROLLER_RELATIVE_PATH.as_posix(),
    CONTROL_VALIDATOR_RELATIVE_PATH.as_posix(),
)

RUN_ID_PATTERN = re.compile(r"pypto-cpu-v2-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_COMMAND_MARKERS = (
    "/home/zhaosiying/amdgpu-sim",
    "/home/zhaosiying/zcode-lane",
    "/home/zhaosiying/amdgpu-sim-agentenv",
    "amdgpu-sim",
    "zcode-lane",
    "gpu-benchmark",
    "gpu-smoke",
    "gpu_benchmark",
    "gpu_smoke",
    "nvidia-smi",
    "nvidia_smi",
    "vllm",
    "sglang",
    "torchrun",
    "torch.distributed.run",
    "deepspeed",
    "ray start",
    "run_model_lane.sh",
    "run_engine_lane.sh",
    "qwen35_inference.py",
    "pypto_plugins.bootstrap",
)
FORBIDDEN_EXECUTABLE_NAMES = {
    "awk",
    "bash",
    "busybox",
    "chrt",
    "dash",
    "deno",
    "deepspeed",
    "env",
    "expect",
    "find",
    "fish",
    "gawk",
    "gdb",
    "ionice",
    "kill",
    "killall",
    "lua",
    "luajit",
    "mawk",
    "nice",
    "node",
    "nohup",
    "perl",
    "php",
    "pkill",
    "pgrep",
    "ray",
    "ruby",
    "screen",
    "script",
    "sed",
    "setsid",
    "sh",
    "sglang",
    "sglang_router",
    "stdbuf",
    "strace",
    "stty",
    "su",
    "sudo",
    "taskset",
    "tclsh",
    "timeout",
    "tmux",
    "vllm",
    "watch",
    "wish",
    "xargs",
    "zsh",
}
INTERPRETER_BASENAME_PREFIXES = ("python", "pypy")
FORBIDDEN_MODULE_PREFIXES = (
    "deepspeed",
    "ray",
    "sglang",
    "torch.distributed",
    "vllm",
)
PROTECTED_ROOT_NAMES = (
    "/home/zhaosiying/amdgpu-sim",
    "/home/zhaosiying/zcode-lane",
    "/home/zhaosiying/amdgpu-sim-agentenv",
)


class ContractError(RuntimeError):
    """The v2 policy contract or an exact immutable dependency differs."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_exact(
    name: str, path: Path, size: int, digest: str
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ContractError(f"exact CPU-v2 dependency is noncanonical: {path}")
    raw = path.read_bytes()
    if len(raw) != size or sha256_bytes(raw) != digest:
        raise ContractError(f"exact CPU-v2 dependency differs: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        if (
            getattr(existing, "__file__", None) != str(path)
            or getattr(existing, "__exact_source_bytes__", None) != size
            or getattr(existing, "__exact_source_sha256__", None) != digest
        ):
            raise ContractError(f"CPU-v2 dependency name is already occupied: {name}")
        return existing
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


def resolved_executable(command_entry: str) -> Path | None:
    """Resolve argv[0] through symlinks and PATH without executing anything."""
    candidate = Path(command_entry)
    try:
        if candidate.is_absolute() or "/" in command_entry:
            return candidate.resolve(strict=True)
        found = shutil.which(command_entry)
        if found is None:
            return None
        return Path(found).resolve(strict=True)
    except OSError:
        return None


def validate_command(command: object) -> list[str]:
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str)
            or not item
            or any(marker in item for marker in ("\0", "\n", "\r"))
            for item in command
        )
    ):
        raise ContractError("CPU-v2 child command is malformed")
    executable = Path(command[0]).name.lower()
    resolved_entry = resolved_executable(command[0])
    executable_names = {executable}
    if resolved_entry is not None:
        executable_names.add(resolved_entry.name.lower())
    if executable_names & FORBIDDEN_EXECUTABLE_NAMES or "=" in command[0]:
        raise ContractError("CPU-v2 child command uses a forbidden wrapper or signal tool")
    if any(
        name.startswith(INTERPRETER_BASENAME_PREFIXES) for name in executable_names
    ):
        inline_code = False
        trailing_module_flag = False
        past_options = False
        positional_target = False
        skip_option_value = False
        module_candidates: list[str] = []
        for index, token in enumerate(command[1:], start=1):
            if skip_option_value:
                skip_option_value = False
                continue
            if past_options:
                if token == "-":
                    inline_code = True
                    continue
                positional_target = True
                continue
            if token == "--":
                past_options = True
                continue
            if token == "--check-hash-based-pycs":
                skip_option_value = True
                continue
            if token.startswith("--"):
                continue
            if token == "-":
                inline_code = True
                continue
            body = token.partition("=")[0] if "=" in token else token
            if body.startswith("-c"):
                inline_code = True
                continue
            if len(body) > 1 and body.startswith("-"):
                cluster = body[1:]
                position = 0
                while position < len(cluster):
                    character = cluster[position]
                    if character == "-":
                        inline_code = True
                        break
                    if character == "c":
                        inline_code = True
                        break
                    if character == "m":
                        glued = cluster[position + 1 :]
                        if glued:
                            module_candidates.append(glued)
                        elif index + 1 < len(command):
                            module_candidates.append(command[index + 1])
                        else:
                            trailing_module_flag = True
                        break
                    if character in ("W", "X"):
                        if not cluster[position + 1 :]:
                            skip_option_value = True
                        break
                    position += 1
                continue
            positional_target = True
        if inline_code:
            raise ContractError("CPU-v2 Python child must use a file or module, not -c")
        if trailing_module_flag:
            raise ContractError("CPU-v2 Python child has a trailing -m without a module")
        if not module_candidates and not positional_target:
            raise ContractError("CPU-v2 Python child must name a file or module")
        for module in module_candidates:
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", module
            ):
                raise ContractError(
                    "CPU-v2 Python -m operand is not a plain module name"
                )
            if module.lower().startswith(FORBIDDEN_MODULE_PREFIXES):
                raise ContractError(
                    "CPU-v2 child command launches a forbidden framework module: "
                    + module
                )
    joined = " ".join(command).lower()
    compact = "".join(command).lower()
    if resolved_entry is not None:
        joined += " " + str(resolved_entry).lower()
        compact += str(resolved_entry).lower()
    flattened = re.sub(r"[^0-9a-z]", "", compact)
    for marker in FORBIDDEN_COMMAND_MARKERS:
        squashed = marker.replace(" ", "")
        flattened_marker = re.sub(r"[^0-9a-z]", "", marker)
        if (
            marker in joined
            or (squashed and squashed in compact)
            or (flattened_marker and flattened_marker in flattened)
        ):
            raise ContractError(
                "CPU-v2 child command is not a non-framework CPU command: "
                + marker
            )
    protected_roots = tuple(Path(value) for value in PROTECTED_ROOT_NAMES)
    resolved_paths: list[Path] = []
    if resolved_entry is not None:
        resolved_paths.append(resolved_entry)
    for item in command:
        candidate = item
        while candidate.startswith("-"):
            candidate = candidate[1:]
        if "=" in candidate:
            candidate = candidate.partition("=")[2]
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = ROOT / candidate_path
        try:
            resolved_paths.append(candidate_path.resolve(strict=True))
        except OSError:
            continue
    for resolved_path in resolved_paths:
        if any(
            root in (resolved_path, *resolved_path.parents)
            for root in protected_roots
        ):
            raise ContractError("CPU-v2 child resolves into a protected path")
    return list(command)


def policy_document() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": POLICY_KIND,
        "mode": MODE,
        "launch_memory_floor_kib": LAUNCH_MEMORY_FLOOR_KIB,
        "resume_memory_floor_kib": RESUME_MEMORY_FLOOR_KIB,
        "pause_memory_floor_kib": PAUSE_MEMORY_FLOOR_KIB,
        "poll_seconds": POLL_SECONDS,
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "void",
        "framework_launch": False,
        "external_process_signals": False,
        "owned_group_signals_require_revalidation": True,
        "start_gate_schema_version": START_GATE_SCHEMA_VERSION,
        "start_gate_timeout_seconds": START_GATE_TIMEOUT_SECONDS,
        "base_dependencies": {
            "preflight": {
                "path": BASE_PREFLIGHT_RELATIVE_PATH.as_posix(),
                "bytes": BASE_PREFLIGHT_SIZE,
                "sha256": BASE_PREFLIGHT_SHA256,
            },
            "run_isolated": {
                "path": BASE_ISOLATION_RELATIVE_PATH.as_posix(),
                "bytes": BASE_ISOLATION_SIZE,
                "sha256": BASE_ISOLATION_SHA256,
            },
            "stop_run": {
                "path": BASE_STOP_RELATIVE_PATH.as_posix(),
                "bytes": BASE_STOP_SIZE,
                "sha256": BASE_STOP_SHA256,
            },
            "nvidia_contract": {
                "path": BASE_NVIDIA_CONTRACT_RELATIVE_PATH.as_posix(),
                "bytes": BASE_NVIDIA_CONTRACT_SIZE,
                "sha256": BASE_NVIDIA_CONTRACT_SHA256,
            },
            "nvidia_control": {
                "path": BASE_NVIDIA_CONTROL_RELATIVE_PATH.as_posix(),
                "bytes": BASE_NVIDIA_CONTROL_SIZE,
                "sha256": BASE_NVIDIA_CONTROL_SHA256,
            },
            "nvidia_manifest": {
                "path": BASE_NVIDIA_MANIFEST_RELATIVE_PATH.as_posix(),
                "bytes": BASE_NVIDIA_MANIFEST_SIZE,
                "sha256": BASE_NVIDIA_MANIFEST_SHA256,
            },
        },
    }
