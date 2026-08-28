#!/usr/bin/env python3
"""Collect one logical Qwen profile or reconcile three profile lanes."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from benchmarks.release.profile_runtime import reconcile, run  # noqa: E402
from benchmarks.release.workload import (  # noqa: E402
    LANES,
    PERFORMANCE_SCHEDULE,
    PROFILE_SCHEDULE,
    ReleaseContractError,
    atomic_json,
    read_json,
    require_path_below_runs,
    require_run_directory,
)


def _selectors(values: list[str], label: str) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = defaultdict(list)
    for raw in values:
        lane, separator, path = raw.partition("=")
        if separator != "=" or lane not in LANES:
            raise ReleaseContractError(f"invalid {label} selector: {raw}")
        selected[lane].append(Path(path).resolve(strict=True))
    return dict(selected)


def _measurement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--optimized-memory-mode",
        choices=("zero-offload", "matched"),
        default="zero-offload",
    )
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")


def _performance_from_matrix(path: Path) -> dict[str, list[dict[str, object]]]:
    summary = read_json(path.resolve(strict=True))
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-performance-matrix-control"
        or len(summary.get("runs", [])) != 12
        or [item.get("lane") for item in summary.get("runs", [])]
        != list(PERFORMANCE_SCHEDULE)
    ):
        raise ReleaseContractError("performance matrix is not complete")
    reports: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in summary["runs"]:
        reports[str(item["lane"])].append(read_json(Path(item["report"])))
    return dict(reports)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--lane", choices=LANES, required=True)
    _measurement_arguments(collect)
    collect.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    matrix = commands.add_parser("matrix")
    _measurement_arguments(matrix)
    matrix_performance = matrix.add_mutually_exclusive_group()
    matrix_performance.add_argument("--performance", action="append", default=[])
    matrix_performance.add_argument("--performance-matrix", type=Path)
    merge = commands.add_parser("reconcile")
    merge.add_argument("--profile", action="append", required=True)
    merge_performance = merge.add_mutually_exclusive_group()
    merge_performance.add_argument("--performance", action="append", default=[])
    merge_performance.add_argument("--performance-matrix", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "reconcile":
        paths = _selectors(args.profile, "profile")
        profiles = {
            lane: [read_json(path) for path in lane_paths]
            for lane, lane_paths in paths.items()
        }
        performance_paths = _selectors(args.performance, "performance")
        performance = (
            _performance_from_matrix(args.performance_matrix)
            if args.performance_matrix is not None
            else {
                lane: [read_json(path) for path in lane_paths]
                for lane, lane_paths in performance_paths.items()
            }
            if performance_paths
            else None
        )
        output = require_path_below_runs(ROOT, args.output)
        payload = reconcile(profiles, performance)
        payload["inputs"] = {
            "profiles": {
                lane: [str(path) for path in lane_paths]
                for lane, lane_paths in paths.items()
            },
            "performance": {
                lane: [str(path) for path in lane_paths]
                for lane, lane_paths in performance_paths.items()
            },
            "performance_matrix": (
                None
                if args.performance_matrix is None
                else str(args.performance_matrix.resolve(strict=True))
            ),
        }
        atomic_json(output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args.command == "collect" and args._worker:
        run_id, run_dir = require_run_directory(ROOT)
        return run(
            args.lane,
            args.model_path,
            run_id,
            run_dir,
            ROOT,
            args.optimized_memory_mode,
        )

    def launch(lane: str):
        worker_args = (
            "collect",
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

    if args.command == "matrix":
        records = []
        grouped_paths: dict[str, list[Path]] = defaultdict(list)
        for index, lane in enumerate(PROFILE_SCHEDULE):
            controlled = launch(lane)
            report = (
                ROOT / "runs" / controlled.run_id / f"qwen35-9b-profile-{lane}.json"
                if controlled.run_id is not None
                else None
            )
            if report is not None:
                grouped_paths[lane].append(report)
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
        complete = len(records) == len(PROFILE_SCHEDULE) and all(
            item["return_code"] == 0 for item in records
        )
        payload = {
            "schema": 1,
            "kind": "qwen35-9b-profile-matrix-control",
            "schedule": list(PROFILE_SCHEDULE),
            "starts_per_lane": 3,
            "profile_requests_per_start": 5,
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
                    f"release-profile-matrix-{timestamp}-{os.getpid()}-"
                    f"{secrets.token_hex(3)}"
                )
            )
            profiles = {
                lane: [read_json(path) for path in lane_paths]
                for lane, lane_paths in grouped_paths.items()
            }
            performance_paths = _selectors(args.performance, "performance")
            performance = (
                _performance_from_matrix(args.performance_matrix)
                if args.performance_matrix is not None
                else {
                    lane: [read_json(path) for path in lane_paths]
                    for lane, lane_paths in performance_paths.items()
                }
                if performance_paths
                else None
            )
            if complete:
                reconciliation = reconcile(profiles, performance)
                reconciliation_path = directory / "reconciliation.json"
                atomic_json(reconciliation_path, reconciliation)
                payload["reconciliation"] = str(reconciliation_path)
            summary_path = directory / "summary.json"
            atomic_json(summary_path, payload)
            payload["summary_path"] = str(summary_path)
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
                / f"qwen35-9b-profile-{args.lane}.json"
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
