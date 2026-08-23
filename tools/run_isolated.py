#!/usr/bin/env python3
"""Launch one workspace-owned command in an isolated NVIDIA process group."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import secrets
import signal
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PREFIX = ROOT / "envs" / "pypto-nvidia"
FORBIDDEN_ENV_PREFIXES = ("HSA_", "ROCR_", "GEMSIM_", "AMDGPU_SIM_")
FORBIDDEN_ENV_NAMES = {
    "ROCM_PATH",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "PYTORCH_ROCM_ARCH",
}


def process_start_ticks(pid: int) -> int:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def atomic_json(path: pathlib.Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def isolated_environment(run_id: str, run_dir: pathlib.Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in FORBIDDEN_ENV_NAMES
        and not name.startswith(FORBIDDEN_ENV_PREFIXES)
    }
    executable_path = [str(pathlib.Path("/usr/local/cuda-13.3/bin"))]
    library_path = ["/usr/lib/wsl/lib", "/usr/local/cuda-13.3/lib64"]
    if ENV_PREFIX.is_dir():
        executable_path.insert(0, str(ENV_PREFIX / "bin"))
        library_path.insert(0, str(ENV_PREFIX / "lib"))
    executable_path.append(environment.get("PATH", "/usr/bin:/bin"))

    cache_root = ROOT / "caches"
    paths = {
        "TMPDIR": run_dir / "tmp",
        "HF_HOME": cache_root / "huggingface",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor" / run_id,
        "TRITON_CACHE_DIR": cache_root / "triton" / run_id,
        "CUDA_CACHE_PATH": cache_root / "cuda" / run_id,
        "PYPTO_CACHE_DIR": cache_root / "pypto",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    environment.update({name: str(path) for name, path in paths.items()})
    environment.update(
        {
            "PATH": os.pathsep.join(executable_path),
            "LD_LIBRARY_PATH": os.pathsep.join(library_path),
            "CUDA_HOME": "/usr/local/cuda-13.3",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(ROOT / "projects" / "pypto-framework-plugins" / "src"),
                    str(ROOT / "projects" / "pypto-kernels" / "src"),
                    str(ROOT / "projects" / "pypto" / "python"),
                    str(ROOT / "upstream" / "sglang" / "python"),
                )
            ),
            "PIP_REQUIRE_VIRTUALENV": "true",
            "PYPTO_ENV_PREFIX": str(ENV_PREFIX),
            "PYPTO_RUN_ID": run_id,
            "PYPTO_WORKSPACE_ROOT": str(ROOT),
            "PYPTO_SGLANG_SOURCE_ROOT": str(ROOT / "upstream" / "sglang"),
            "SGLANG_PLUGINS": "pypto",
        }
    )
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("light", "heavy", "gpu-benchmark"),
        default="light",
    )
    parser.add_argument(
        "--require-framework-plugins",
        action="store_true",
        help="verify installed Torch and SGLang PyPTO entry points before launch",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    preflight = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "preflight.py"), "--mode", args.mode],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode != 0:
        return preflight.returncode

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"pypto-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    environment = isolated_environment(run_id, run_dir)
    command_requires_plugins = args.mode == "gpu-benchmark" or any(
        marker in item.lower()
        for item in command
        for marker in ("sglang", "acceptance")
    )
    if args.mode == "gpu-benchmark":
        python = ENV_PREFIX / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(
                f"project environment is missing; cannot verify runtime: {python}"
            )
        subprocess.run(
            [str(python), str(ROOT / "tools" / "environment_identity.py"), "--verify"],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    if args.require_framework_plugins or command_requires_plugins:
        python = ENV_PREFIX / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(
                f"project environment is missing; cannot verify plugins: {python}"
            )
        subprocess.run(
            [str(python), "-m", "pypto_plugins.bootstrap"],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        start_new_session=True,
    )
    metadata = {
        "schema": 1,
        "run_id": run_id,
        "workspace": str(ROOT),
        "mode": args.mode,
        "command": command,
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": process_start_ticks(process.pid),
        "started_at": timestamp,
        "status": "running",
    }
    metadata_path = run_dir / "process.json"
    atomic_json(metadata_path, metadata)
    print(f"PYPTO_RUN_ID={run_id} PID={process.pid} PGID={metadata['pgid']}", flush=True)

    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        # This is the exact process group created above, never a discovered or
        # name-matched external process group.
        os.killpg(int(metadata["pgid"]), signal.SIGTERM)
        return_code = process.wait()

    metadata["status"] = "exited"
    metadata["return_code"] = return_code
    metadata["finished_at"] = datetime.datetime.now(datetime.UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    atomic_json(metadata_path, metadata)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
