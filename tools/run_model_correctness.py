#!/usr/bin/env python3
"""Create a 64-step stock reference or audit three PyPTO fresh starts."""

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
from benchmarks.release.correctness_runtime import (  # noqa: E402
    run_candidate,
    run_reference,
)
from benchmarks.release.workload import (  # noqa: E402
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    require_run_directory,
)


FRESH_STARTS = 3


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("reference", "candidate"))
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--reference-report", type=Path)
    value.add_argument("--timeout-seconds", type=int, default=14400)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def _worker(args: argparse.Namespace) -> int:
    run_id, run_dir = require_run_directory(ROOT)
    if args.mode == "reference":
        return run_reference(args.model_path, run_id, run_dir)
    if args.reference_report is None:
        raise ReleaseContractError("candidate worker requires a reference report")
    return run_candidate(args.model_path, args.reference_report, run_id, run_dir)


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args.mode == "reference" and args.reference_report is not None:
        raise ReleaseContractError("reference mode does not accept --reference-report")
    if args.mode == "candidate" and args.reference_report is None:
        raise ReleaseContractError("candidate mode requires --reference-report")
    if args._worker:
        return _worker(args)

    model_path = args.model_path.resolve()
    worker = Path(__file__).resolve()
    count = 1 if args.mode == "reference" else FRESH_STARTS
    records = []
    for index in range(count):
        worker_args = [
            args.mode,
            "--_worker",
            "--model-path",
            str(model_path),
        ]
        if args.reference_report is not None:
            worker_args.extend(
                ("--reference-report", str(args.reference_report.resolve()))
            )
        if args.mode == "reference":

            def factory(pointer):
                return isolated_command(
                    ROOT,
                    worker,
                    tuple(worker_args),
                    pointer,
                    framework_profile="baseline",
                    timeout_seconds=args.timeout_seconds,
                )
        else:

            def factory(pointer):
                return pypto_gpu_command(
                    ROOT,
                    worker,
                    tuple(worker_args),
                    pointer,
                    timeout_seconds=args.timeout_seconds,
                )

        controlled = invoke_controlled(factory, root=ROOT, dry_run=args.dry_run)
        report_name = (
            "qwen35-9b-reference.json"
            if args.mode == "reference"
            else "qwen35-9b-correctness.json"
        )
        records.append(
            {
                "fresh_start_index": index,
                "run_id": controlled.run_id,
                "return_code": controlled.return_code,
                "command": list(controlled.command),
                "report": (
                    str(ROOT / "runs" / controlled.run_id / report_name)
                    if controlled.run_id is not None
                    else None
                ),
            }
        )
        if controlled.return_code != 0:
            break
    complete = len(records) == count and all(
        item["return_code"] == 0 for item in records
    )
    summary = {
        "schema": SCHEMA_VERSION,
        "kind": f"qwen35-9b-{args.mode}-control",
        "fresh_starts": count,
        "requests_per_candidate_start": 10 if args.mode == "candidate" else 1,
        "dry_run": args.dry_run,
        "runs": records,
        "status": "planned" if args.dry_run else "complete" if complete else "failed",
    }
    if not args.dry_run:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = (
            ROOT
            / "runs"
            / (f"release-correctness-{timestamp}-{os.getpid()}-{secrets.token_hex(3)}")
            / "summary.json"
        )
        atomic_json(path, summary)
        summary["summary_path"] = str(path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
