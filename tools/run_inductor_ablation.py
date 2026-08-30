#!/usr/bin/env python3
"""Run one performance-only SwiGLU ablation under the GPU controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.controllers import (  # noqa: E402
    invoke_controlled,
    isolated_command,
    pypto_gpu_command,
)
from benchmarks.release.inductor_ablation import run  # noqa: E402
from benchmarks.release.workload import ReleaseContractError, atomic_json, require_run_directory  # noqa: E402


PHASE_ROWS = {"prefill": 19, "decode": 1}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("eager", "inductor-nv", "pypto"), required=True)
    value.add_argument("--phase", choices=tuple(PHASE_ROWS), required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--warmup-calls", type=int, default=20)
    value.add_argument("--timed-calls", type=int, default=100)
    value.add_argument("--timeout-seconds", type=int, default=3600)
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    rows = PHASE_ROWS[args.phase]
    if args._worker:
        run_id, run_dir = require_run_directory(ROOT)
        output = args.output.resolve()
        if ROOT not in output.parents or output == ROOT:
            raise ReleaseContractError("ablation output must be inside the workspace")
        result = run(
            args.mode,
            rows,
            12_288,
            args.warmup_calls,
            args.timed_calls,
        )
        result["phase"] = args.phase
        result["run_id"] = run_id
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    worker_args = (
        "--_worker",
        "--mode",
        args.mode,
        "--phase",
        args.phase,
        "--output",
        str(args.output.resolve()),
        "--warmup-calls",
        str(args.warmup_calls),
        "--timed-calls",
        str(args.timed_calls),
    )
    if args.mode == "pypto":
        factory = lambda pointer: pypto_gpu_command(
            ROOT,
            Path(__file__),
            worker_args,
            pointer,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        factory = lambda pointer: isolated_command(
            ROOT,
            Path(__file__),
            worker_args,
            pointer,
            framework_profile="baseline",
            timeout_seconds=args.timeout_seconds,
        )
    controlled = invoke_controlled(factory, root=ROOT)
    payload = {
        "mode": args.mode,
        "phase": args.phase,
        "run_id": controlled.run_id,
        "return_code": controlled.return_code,
        "output": str(args.output.resolve()),
        "status": "complete" if controlled.return_code == 0 else "failed",
        "command": list(controlled.command),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return controlled.return_code


if __name__ == "__main__":
    raise SystemExit(main())
