#!/usr/bin/env python3
"""Run a full-model eager control with the matched SGLang providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.controllers import invoke_controlled, isolated_command  # noqa: E402
from benchmarks.release.performance_runtime import run  # noqa: E402
from benchmarks.release.workload import ReleaseContractError, require_run_directory  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--timeout-seconds", type=int, default=14400)
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    worker_args = (
        "--_worker",
        "--model-path",
        str(args.model_path.resolve()),
    )
    if args._worker:
        run_id, run_dir = require_run_directory(ROOT)
        return run(
            "sglang-matched",
            args.model_path,
            run_id,
            run_dir,
            "zero-offload",
            compile_enabled=False,
        )
    controlled = invoke_controlled(
        lambda pointer: isolated_command(
            ROOT,
            Path(__file__),
            worker_args,
            pointer,
            framework_profile="baseline",
            timeout_seconds=args.timeout_seconds,
        ),
        root=ROOT,
    )
    report = (
        ROOT / "runs" / controlled.run_id / "qwen35-9b-performance-sglang-matched.json"
        if controlled.run_id
        else None
    )
    payload = {
        "mode": "eager-control",
        "run_id": controlled.run_id,
        "return_code": controlled.return_code,
        "report": str(report) if report else None,
        "status": "complete" if controlled.return_code == 0 else "failed",
        "command": list(controlled.command),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return controlled.return_code


if __name__ == "__main__":
    raise SystemExit(main())
