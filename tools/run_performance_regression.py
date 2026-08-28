#!/usr/bin/env python3
"""Run one fixed 19+64, concurrency-one release timing lane."""

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
from benchmarks.release.performance_runtime import run  # noqa: E402
from benchmarks.release.workload import (  # noqa: E402
    LANES,
    PERFORMANCE_SCHEDULE,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    require_run_directory,
)


MATRIX_SCHEDULE = PERFORMANCE_SCHEDULE


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    selection = value.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lane", choices=LANES)
    selection.add_argument("--matrix", action="store_true")
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument(
        "--optimized-memory-mode",
        choices=("zero-offload", "matched"),
        default="zero-offload",
    )
    value.add_argument("--timeout-seconds", type=int, default=14400)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args._worker:
        if args.lane is None:
            raise ReleaseContractError("worker requires one lane")
        run_id, run_dir = require_run_directory(ROOT)
        return run(
            args.lane,
            args.model_path,
            run_id,
            run_dir,
            args.optimized_memory_mode,
        )

    def launch(lane: str):
        worker_args = (
            "--_worker",
            "--lane",
            lane,
            "--model-path",
            str(args.model_path.resolve()),
            "--optimized-memory-mode",
            args.optimized_memory_mode,
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

    if args.matrix:
        records = []
        for index, lane in enumerate(MATRIX_SCHEDULE):
            controlled = launch(lane)
            records.append(
                {
                    "schedule_index": index,
                    "lane": lane,
                    "run_id": controlled.run_id,
                    "return_code": controlled.return_code,
                    "command": list(controlled.command),
                    "report": (
                        str(
                            ROOT
                            / "runs"
                            / controlled.run_id
                            / f"qwen35-9b-performance-{lane}.json"
                        )
                        if controlled.run_id is not None
                        else None
                    ),
                }
            )
            if controlled.return_code != 0:
                break
        complete = len(records) == len(MATRIX_SCHEDULE) and all(
            item["return_code"] == 0 for item in records
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-9b-performance-matrix-control",
            "schedule": list(MATRIX_SCHEDULE),
            "starts_per_lane": 4,
            "requests_per_start": 10,
            "optimized_memory_mode": args.optimized_memory_mode,
            "runs": records,
            "status": (
                "planned" if args.dry_run else "complete" if complete else "failed"
            ),
        }
        if not args.dry_run:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            path = (
                ROOT
                / "runs"
                / (
                    f"release-performance-matrix-{timestamp}-{os.getpid()}-"
                    f"{secrets.token_hex(3)}"
                )
                / "summary.json"
            )
            atomic_json(path, payload)
            payload["summary_path"] = str(path)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if payload["status"] in {"planned", "complete"} else 1

    controlled = launch(args.lane)
    payload = {
        "lane": args.lane,
        "run_id": controlled.run_id,
        "return_code": controlled.return_code,
        "command": list(controlled.command),
        "report": (
            str(
                ROOT
                / "runs"
                / controlled.run_id
                / f"qwen35-9b-performance-{args.lane}.json"
            )
            if controlled.run_id is not None
            else None
        ),
        "status": (
            "planned"
            if args.dry_run
            else "complete"
            if controlled.return_code == 0
            else "failed"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return controlled.return_code


if __name__ == "__main__":
    raise SystemExit(main())
