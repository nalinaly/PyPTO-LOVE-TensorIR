#!/usr/bin/env python3
"""Verify the selected interpreter's NVIDIA runtime before a child launch."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN_DSO_MARKERS = (
    "libamdhip64",
    "libhsa-runtime64",
    "self-amdgpu-runtime",
    "gemsim",
)
EXPECTED_SGLANG_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
EXPECTED_SGLANG_VERSION = "0.5.18"


def is_below(path: pathlib.Path, root: pathlib.Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def git_output(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, required=True)
    parser.add_argument("--lock", type=pathlib.Path, required=True)
    parser.add_argument("--profile", choices=("pypto", "baseline"), required=True)
    parser.add_argument("--framework", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix.resolve()
    locked = json.loads(args.lock.resolve().read_text())
    failures: list[str] = []
    if pathlib.Path(sys.prefix).resolve() != prefix:
        failures.append(f"sys.prefix={sys.prefix} does not match {prefix}")
    if not is_below(pathlib.Path(sys.executable), prefix):
        failures.append(f"interpreter escaped selected prefix: {sys.executable}")

    import torch

    torch_file = pathlib.Path(torch.__file__).resolve()
    if not is_below(torch_file, prefix):
        failures.append(f"Torch import escaped selected prefix: {torch_file}")
    runtime = {
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "torch_file": str(torch_file),
    }
    for name, value in runtime.items():
        if locked.get(name) != value:
            failures.append(
                f"runtime {name} mismatch: locked={locked.get(name)!r}, actual={value!r}"
            )
    if torch.version.hip is not None:
        failures.append(f"selected Torch reports HIP {torch.version.hip}")
    if not torch.cuda.is_available():
        failures.append("selected Torch cannot access CUDA")
        gpu: dict[str, object] = {}
    else:
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
        if gpu["capability"] != [12, 0]:
            failures.append(f"selected device is not SM120: {gpu}")

    if args.framework:
        import sglang

        sglang_root = ROOT / "upstream" / "sglang"
        sglang_file = pathlib.Path(sglang.__file__).resolve()
        if not is_below(sglang_file, sglang_root / "python" / "sglang"):
            failures.append(f"SGLang import escaped pinned checkout: {sglang_file}")
        if str(sglang.__version__).split("+", 1)[0] != EXPECTED_SGLANG_VERSION:
            failures.append(f"unexpected SGLang version: {sglang.__version__}")
        try:
            commit = git_output(sglang_root, "rev-parse", "HEAD")
            dirty = git_output(sglang_root, "status", "--porcelain")
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"cannot verify SGLang checkout: {error}")
        else:
            if commit != EXPECTED_SGLANG_COMMIT or dirty:
                failures.append(
                    f"SGLang checkout mismatch: commit={commit}, dirty={bool(dirty)}"
                )
        if args.profile == "baseline":
            leaked = sorted(
                name
                for name in ("pypto", "pypto_kernels", "pypto_plugins")
                if name in sys.modules
            )
            if leaked:
                failures.append(f"baseline imported candidate modules: {leaked}")

    maps = pathlib.Path("/proc/self/maps").read_text(errors="replace").lower()
    forbidden_dsos = sorted(
        marker for marker in FORBIDDEN_DSO_MARKERS if marker in maps
    )
    if forbidden_dsos:
        failures.append(f"forbidden AMD/simulator DSOs loaded: {forbidden_dsos}")

    report = {
        "status": "pass" if not failures else "fail",
        "prefix": str(prefix),
        "profile": args.profile,
        "framework": args.framework,
        "runtime": runtime,
        "gpu": gpu,
        "forbidden_dsos": forbidden_dsos,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 75


if __name__ == "__main__":
    sys.exit(main())
