#!/usr/bin/env python3
"""Aggregate accepted strict CUPTI request profiles by PyPTO source node."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability-report",
        action="append",
        required=True,
        help="MODEL_SIZE=PATH; supply exactly three accepted reports per model",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: dict[str, list[Path]] = defaultdict(list)
    for raw in args.stability_report:
        model_size, separator, path = raw.partition("=")
        if separator != "=" or model_size not in {"0.8B", "9B"}:
            raise ValueError(f"invalid stability report selector: {raw}")
        reports[model_size].append(Path(path).resolve(strict=True))
    if set(reports) != {"0.8B", "9B"} or any(
        len(paths) != 3 for paths in reports.values()
    ):
        raise ValueError("profiling requires exactly three reports for each model")

    output: dict[str, object] = {
        "kind": "qwen35-final-strict-cupti-profile-summary",
        "models": {},
        "schema": 1,
    }
    for model_size, paths in sorted(reports.items()):
        source_totals: dict[tuple[str, str, str], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        excluded_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        request_gpu_ms: list[float] = []
        compiler_revisions: set[str] = set()
        kernels_revisions: set[str] = set()
        run_ids: list[str] = []
        request_count = 0
        for report_path in paths:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                not report.get("all_passed")
                or report.get("status") != "complete"
                or len(report.get("requests", [])) != 10
            ):
                raise ValueError(f"stability report is not accepted: {report_path}")
            run_ids.append(str(report["run_id"]))
            for request in report["requests"]:
                coverage_path = Path(request["coverage_report"]).resolve(strict=True)
                coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
                if (
                    not coverage.get("strict_policy_passed")
                    or coverage.get("fallbacks")
                    or coverage.get("violations")
                ):
                    raise ValueError(f"coverage report is not strict: {coverage_path}")
                registry = {
                    item["artifact_id"]: item
                    for item in coverage["artifact_registry"]
                }
                for item in registry.values():
                    compiler_revisions.add(str(item["compiler_revision"]))
                    kernels_revisions.add(str(item["kernels_revision"]))
                for event in coverage["events"]:
                    if (
                        event.get("disposition") != "covered"
                        or event.get("activity") != "compute"
                    ):
                        continue
                    provenance = event.get("provenance") or {}
                    artifact = registry.get(provenance.get("artifact_id"))
                    if artifact is None:
                        raise ValueError(
                            f"covered event lacks an artifact: {coverage_path}"
                        )
                    key = (
                        str(artifact["source_node"]),
                        str(event["provider"]),
                        str(event["kernel_name"]),
                    )
                    source_totals[key][0] += int(event["call_count"])
                    source_totals[key][1] += int(event["gpu_time_ns"])
                for event in coverage["excluded"]:
                    key = (str(event["activity"]), str(event["provider"]))
                    excluded_totals[key][0] += int(event["call_count"])
                    excluded_totals[key][1] += int(event["gpu_time_ns"])
                compute = coverage["model_forward_compute"]
                request_gpu_ms.append(
                    int(compute["gpu_time_ns"]["total"]) / 1e6
                )
                request_count += 1

        if len(compiler_revisions) != 1 or len(kernels_revisions) != 1:
            raise ValueError(f"{model_size} profile spans multiple revisions")
        total_gpu_ns = sum(item[1] for item in source_totals.values())
        model_profile = {
            "accepted_run_ids": run_ids,
            "request_count": request_count,
            "compiler_revision": next(iter(compiler_revisions)),
            "kernels_revision": next(iter(kernels_revisions)),
            "compute_gpu_ms_per_request": {
                "min": min(request_gpu_ms),
                "median": statistics.median(request_gpu_ms),
                "p90_nearest_rank": _percentile(request_gpu_ms, 0.9),
                "max": max(request_gpu_ms),
                "mean": statistics.fmean(request_gpu_ms),
            },
            "compute": [
                {
                    "source_node": key[0],
                    "provider": key[1],
                    "kernel_name": key[2],
                    "calls_total": totals[0],
                    "calls_per_request": totals[0] / request_count,
                    "gpu_time_ms_total": totals[1] / 1e6,
                    "gpu_time_ms_per_request": totals[1] / request_count / 1e6,
                    "gpu_time_share": totals[1] / total_gpu_ns,
                }
                for key, totals in sorted(
                    source_totals.items(), key=lambda item: item[1][1], reverse=True
                )
            ],
            "excluded_runtime": [
                {
                    "activity": key[0],
                    "provider": key[1],
                    "calls_total": totals[0],
                    "calls_per_request": totals[0] / request_count,
                    "gpu_time_ms_total": totals[1] / 1e6,
                    "gpu_time_ms_per_request": totals[1] / request_count / 1e6,
                }
                for key, totals in sorted(excluded_totals.items())
            ],
        }
        output["models"][model_size] = model_profile

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
