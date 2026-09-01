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
    Qwen35ModelSpec,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    require_run_directory,
    resolve_qwen35_model_spec,
)


FRESH_STARTS = 3


def _report_name(model_spec: Qwen35ModelSpec, mode: str) -> str:
    suffix = "reference" if mode == "reference" else "correctness"
    return f"{model_spec.report_stem}-{suffix}.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("reference", "candidate", "all"))
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--reference-report", type=Path)
    value.add_argument("--semantic-oracle", type=Path)
    value.add_argument("--timeout-seconds", type=int, default=14400)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return value


def _worker(args: argparse.Namespace) -> int:
    run_id, run_dir = require_run_directory(ROOT)
    if args.mode == "reference":
        if args.semantic_oracle is None:
            raise ReleaseContractError("reference worker requires --semantic-oracle")
        return run_reference(
            args.model_path, run_id, run_dir, args.semantic_oracle
        )
    if args.mode != "candidate":
        raise ReleaseContractError("all mode is control-only")
    if args.reference_report is None:
        raise ReleaseContractError("candidate worker requires a reference report")
    return run_candidate(args.model_path, args.reference_report, run_id, run_dir)


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ReleaseContractError("timeout must be positive")
    if args.mode in {"reference", "all"} and args.reference_report is not None:
        raise ReleaseContractError(f"{args.mode} mode does not accept --reference-report")
    if args.mode in {"reference", "all"} and args.semantic_oracle is None:
        raise ReleaseContractError(
            f"{args.mode} mode requires --semantic-oracle"
        )
    if args.mode == "candidate" and args.semantic_oracle is not None:
        raise ReleaseContractError("candidate mode does not accept --semantic-oracle")
    if args.mode == "candidate" and args.reference_report is None:
        raise ReleaseContractError("candidate mode requires --reference-report")
    if args._worker:
        return _worker(args)

    model_path = args.model_path.resolve(strict=True)
    model_spec = resolve_qwen35_model_spec(ROOT, model_path)
    worker = Path(__file__).resolve()
    records: list[dict[str, object]] = []

    def execute(mode: str, index: int, reference_report: Path | None):
        worker_args = [
            mode,
            "--_worker",
            "--model-path",
            str(model_path),
        ]
        if reference_report is not None:
            worker_args.extend(
                ("--reference-report", str(reference_report.resolve()))
            )
        if mode == "reference" and args.semantic_oracle is not None:
            worker_args.extend(
                ("--semantic-oracle", str(args.semantic_oracle.resolve()))
            )
        if mode == "reference":

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
                    # The zero-offload 9B candidate legitimately fills the GPU
                    # up to its static fraction; a fixed 4 GiB free floor
                    # aborts the run mid-request. Use the completion-only
                    # policy already authorized for tight-memory lanes.
                    gpu_free_floor_mib=0,
                )

        controlled = invoke_controlled(factory, root=ROOT, dry_run=args.dry_run)
        report_name = _report_name(model_spec, mode)
        report = (
            ROOT / "runs" / controlled.run_id / report_name
            if controlled.run_id is not None
            else None
        )
        record = {
            "phase": mode,
            "fresh_start_index": index,
            "run_id": controlled.run_id,
            "return_code": controlled.return_code,
            "command": list(controlled.command),
            "report": str(report) if report is not None else None,
        }
        records.append(record)
        return record, report

    if args.mode == "all":
        reference_record, reference_path = execute("reference", 0, None)
        if args.dry_run:
            reference_path = ROOT / "runs/DRY-RUN" / _report_name(
                model_spec, "reference"
            )
        if reference_record["return_code"] == 0:
            for index in range(FRESH_STARTS):
                candidate_record, _unused = execute(
                    "candidate", index, reference_path
                )
                if candidate_record["return_code"] != 0:
                    break
        count = 1 + FRESH_STARTS
    else:
        count = 1 if args.mode == "reference" else FRESH_STARTS
        for index in range(count):
            record, _report = execute(
                args.mode,
                index,
                args.reference_report,
            )
            if record["return_code"] != 0:
                break
    complete = len(records) == count and all(
        item["return_code"] == 0 for item in records
    )
    summary = {
        "schema": SCHEMA_VERSION,
        "kind": f"{model_spec.report_stem}-{args.mode}-control",
        "model_spec": model_spec.record(),
        "fresh_starts": count,
        "requests_per_candidate_start": (
            10 if args.mode in {"candidate", "all"} else 1
        ),
        "dry_run": args.dry_run,
        "runs": records,
        "status": "planned" if args.dry_run else "complete" if complete else "failed",
    }
    if not args.dry_run:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = (
            ROOT
            / "runs"
            / (
                f"release-correctness-{model_spec.report_stem}-{timestamp}-"
                f"{os.getpid()}-{secrets.token_hex(3)}"
            )
            / "summary.json"
        )
        atomic_json(path, summary)
        summary["summary_path"] = str(path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
