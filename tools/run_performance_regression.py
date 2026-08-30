#!/usr/bin/env python3
"""Run the fixed 31+64 non-thinking chat-template release timing lane."""

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
from benchmarks.release.performance_runtime import (  # noqa: E402
    run,
    summarize_fresh_starts,
)
from benchmarks.release.workload import (  # noqa: E402
    LANES,
    PAIR_PERFORMANCE_SCHEDULE,
    PERFORMANCE_SCHEDULE,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    read_json,
    require_run_directory,
)
from tools.summarize_qwen_performance_pair import (  # noqa: E402
    load as load_pair_report,
    summarize_records as summarize_pair_records,
)


MATRIX_SCHEDULE = PERFORMANCE_SCHEDULE


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    selection = value.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lane", choices=LANES)
    selection.add_argument("--pair-matrix", action="store_true")
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

    if args.matrix or args.pair_matrix:
        schedule = PAIR_PERFORMANCE_SCHEDULE if args.pair_matrix else MATRIX_SCHEDULE
        records = []
        for index, lane in enumerate(schedule):
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
        complete = len(records) == len(schedule) and all(
            item["return_code"] == 0 for item in records
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "kind": (
                "qwen35-9b-performance-pair-matrix-control"
                if args.pair_matrix
                else "qwen35-9b-performance-matrix-control"
            ),
            "schedule": list(schedule),
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
            directory = (
                ROOT
                / "runs"
                / (
                    f"release-performance-"
                    f"{'pair-' if args.pair_matrix else ''}matrix-"
                    f"{timestamp}-{os.getpid()}-"
                    f"{secrets.token_hex(3)}"
                )
            )
            if complete and not args.pair_matrix:
                grouped = {
                    lane: [
                        read_json(Path(item["report"]).resolve(strict=True))
                        for item in records
                        if item["lane"] == lane
                    ]
                    for lane in LANES
                }
                aggregation = summarize_fresh_starts(grouped)
                aggregation_path = directory / "aggregation.json"
                atomic_json(aggregation_path, aggregation)
                payload["aggregation"] = str(aggregation_path)
                if aggregation["status"] != "complete":
                    payload["status"] = "failed"
            elif complete:
                aggregation = summarize_pair_records(
                    [
                        load_pair_report(Path(item["report"]), "pypto")
                        for item in records
                        if item["lane"] == "pypto"
                    ],
                    [
                        load_pair_report(Path(item["report"]), "sglang-matched")
                        for item in records
                        if item["lane"] == "sglang-matched"
                    ],
                )
                aggregation_path = directory / "aggregation.json"
                atomic_json(aggregation_path, aggregation)
                payload["aggregation"] = str(aggregation_path)
            path = directory / "summary.json"
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
