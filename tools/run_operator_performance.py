#!/usr/bin/env python3
"""Run the independent performance-only Qwen3.5 SwiGLU A/B matrix."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import secrets
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.controllers import (  # noqa: E402
    invoke_controlled,
    isolated_command,
    pypto_gpu_command,
)
from benchmarks.release.operator_performance_runtime import (  # noqa: E402
    OPERATOR_LANES,
    OPERATOR_SCHEDULE,
    run,
    summarize_fresh_starts,
)
from benchmarks.release.workload import (  # noqa: E402
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    read_json,
    require_run_directory,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    selection = value.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lane", choices=OPERATOR_LANES)
    selection.add_argument("--matrix", action="store_true")
    value.add_argument(
        "--model-path", type=Path, default=ROOT / "models/Qwen3.5-9B"
    )
    value.add_argument("--timeout-seconds", type=int, default=3600)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args._worker:
        if args.lane is None:
            raise ReleaseContractError("worker requires one operator lane")
        run_id, run_dir = require_run_directory(ROOT)
        return run(args.lane, args.model_path, run_id, run_dir)

    def launch(lane: str):
        worker_args = (
            "--_worker",
            "--lane",
            lane,
            "--model-path",
            str(args.model_path.resolve()),
        )
        if lane == "pypto":

            def factory(pointer):
                return pypto_gpu_command(
                    ROOT,
                    Path(__file__),
                    worker_args,
                    pointer,
                    timeout_seconds=args.timeout_seconds,
                )
        else:

            def factory(pointer):
                return isolated_command(
                    ROOT,
                    Path(__file__),
                    worker_args,
                    pointer,
                    framework_profile="baseline",
                    timeout_seconds=args.timeout_seconds,
                )

        return invoke_controlled(factory, root=ROOT, dry_run=args.dry_run)

    lanes = OPERATOR_SCHEDULE if args.matrix else (args.lane,)
    records = []
    for index, lane in enumerate(lanes):
        controlled = launch(lane)
        report = (
            ROOT / "runs" / controlled.run_id / f"qwen35-swiglu-performance-{lane}.json"
            if controlled.run_id is not None
            else None
        )
        records.append(
            {
                "schedule_index": index,
                "lane": lane,
                "run_id": controlled.run_id,
                "return_code": controlled.return_code,
                "command": list(controlled.command),
                "report": None if report is None else str(report),
            }
        )
        if controlled.return_code != 0:
            break
    complete = len(records) == len(lanes) and all(
        item["return_code"] == 0 for item in records
    )
    payload: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": (
            "qwen35-swiglu-operator-performance-matrix-control"
            if args.matrix
            else "qwen35-swiglu-operator-performance-control"
        ),
        "schedule": list(lanes),
        "runs": records,
        "status": "planned" if args.dry_run else "complete" if complete else "failed",
    }
    if not args.dry_run and args.matrix:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            ROOT
            / "runs"
            / f"release-operator-ab-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}"
        )
        if complete:
            grouped = {
                lane: [
                    read_json(Path(item["report"]).resolve(strict=True))
                    for item in records
                    if item["lane"] == lane
                ]
                for lane in OPERATOR_LANES
            }
            summary = summarize_fresh_starts(grouped)
            summary_path = directory / "aggregation.json"
            atomic_json(summary_path, summary)
            payload["aggregation"] = str(summary_path)
        control_path = directory / "summary.json"
        atomic_json(control_path, payload)
        payload["summary_path"] = str(control_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if payload["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
