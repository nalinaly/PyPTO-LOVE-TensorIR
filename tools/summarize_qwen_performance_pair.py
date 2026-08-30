#!/usr/bin/env python3
"""Summarize the accepted PyPTO/matched Qwen performance starts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.lanes import matched_lane_comparability
from benchmarks.release.performance_runtime import (
    GPU_FREE_FLOOR_BYTES,
    validate_resource_identity,
)
from benchmarks.release.workload import (
    ReleaseContractError,
    atomic_json,
    bootstrap_median_comparison_ci,
    fresh_start_summary,
    sha256_file,
    workload_record,
)

LANES = ("pypto", "sglang-matched")
METRICS = (
    "e2e_ms",
    "ttft_ms",
    "tpot_ms",
    "itl_ms",
    "output_tokens_per_second",
    "decode_tokens_per_second",
    "total_tokens_per_second",
)


def validate_resources(
    resources: object, resolved: Path
) -> tuple[dict[str, object], dict[str, object]]:
    identity = validate_resource_identity(resources, str(resolved))
    assert isinstance(resources, dict)
    summary = resources["summary"]
    assert isinstance(summary, dict)
    return summary, identity


def load(path: Path, lane: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("kind") != "qwen35-9b-performance-only"
        or payload.get("lane") != lane
    ):
        raise ReleaseContractError(f"unaccepted {lane} report: {resolved}")
    if payload.get("workload") != workload_record():
        raise ReleaseContractError(f"workload drift in {resolved}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ReleaseContractError(f"missing metrics in {resolved}")
    for metric in METRICS:
        if not isinstance(metrics.get(metric), dict):
            raise ReleaseContractError(f"missing {metric} in {resolved}")
    resources = payload.get("resources")
    summary, identity = validate_resources(resources, resolved)
    return {
        "lane": lane,
        "run_id": payload["run_id"],
        "report": resolved.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(resolved),
        "metrics": {
            metric: float(metrics[metric]["p50"]) for metric in METRICS
        },
        "cold_engine_start_ms": float(payload["cold_engine_start_ms"]),
        "first_compile_trigger_request_ms": float(
            payload["first_compile_trigger_request_ms"]
        ),
        "memory_qualification": payload.get("memory_qualification"),
        "compilation": payload.get("compilation"),
        "resources": summary,
        "gpu_identity": identity,
        "requested_server_config": portable(payload.get("requested_server_config")),
        "resolved_backends": portable(payload.get("resolved_backends")),
    }


def portable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): portable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [portable(child) for child in value]
    if isinstance(value, str):
        try:
            return Path(value).resolve(strict=False).relative_to(ROOT).as_posix()
        except ValueError:
            return value
    return value


def summarize_lane(records: list[dict[str, object]], lane: str) -> dict[str, object]:
    if len(records) != 4:
        raise ReleaseContractError(f"{lane} requires exactly four fresh starts")
    requested = [record.get("requested_server_config") for record in records]
    resolved = [record.get("resolved_backends") for record in records]
    if any(value != requested[0] for value in requested[1:]):
        raise ReleaseContractError(f"{lane} requested configuration drifted")
    if any(value != resolved[0] for value in resolved[1:]):
        raise ReleaseContractError(f"{lane} resolved configuration drifted")
    return {
        "fresh_starts": len(records),
        "starts": records,
        "metrics": {
            metric: fresh_start_summary(
                (float(record["metrics"][metric]) for record in records),
                salt=f"qwen35-pair:{lane}:{metric}",
            )
            for metric in METRICS
        },
        "cold_engine_start_ms": fresh_start_summary(
            (float(record["cold_engine_start_ms"]) for record in records),
            salt=f"qwen35-pair:{lane}:cold-engine",
        ),
        "first_compile_trigger_request_ms": fresh_start_summary(
            (float(record["first_compile_trigger_request_ms"]) for record in records),
            salt=f"qwen35-pair:{lane}:compile-trigger",
        ),
        "resources": {
            "minimum_gpu_memory_free_bytes": min(
                int(record["resources"]["minimum_gpu_memory_free_bytes"])
                for record in records
            ),
            "peak_gpu_memory_used_bytes": max(
                int(record["resources"]["peak_gpu_memory_used_bytes"])
                for record in records
            ),
            "thermal_throttle_observed": any(
                bool(record["resources"]["thermal_throttle_observed"])
                for record in records
            ),
        },
        "memory_qualifications": [
            record["memory_qualification"] for record in records
        ],
    }


def validate_pair_comparability(
    pypto: list[dict[str, object]], matched: list[dict[str, object]]
) -> dict[str, object]:
    comparability = matched_lane_comparability(
        pypto[0]["requested_server_config"],
        matched[0]["requested_server_config"],
        pypto[0]["resolved_backends"],
        matched[0]["resolved_backends"],
    )
    if comparability["matched_claim_allowed"] is not True:
        fields = [
            mismatch.get("field")
            for mismatch in comparability.get("control_mismatches", [])
        ]
        raise ReleaseContractError(
            f"PyPTO/matched timing controls differ: {fields}"
        )
    return comparability


def summarize_records(
    pypto: list[dict[str, object]], matched: list[dict[str, object]]
) -> dict[str, object]:
    gpu_ids = []
    for record in pypto + matched:
        identity = record.get("gpu_identity")
        if isinstance(identity, dict):
            gpu_ids.append(identity)
    if gpu_ids and any(identity != gpu_ids[0] for identity in gpu_ids[1:]):
        raise ReleaseContractError("reports span different GPU identities")
    lanes = {
        lane: summarize_lane(records, lane)
        for lane, records in (("pypto", pypto), ("sglang-matched", matched))
    }
    comparability = validate_pair_comparability(pypto, matched)
    candidate_values = [record["metrics"]["output_tokens_per_second"] for record in pypto]
    matched_values = [
        record["metrics"]["output_tokens_per_second"] for record in matched
    ]
    candidate_rate = lanes["pypto"]["metrics"]["output_tokens_per_second"]
    matched_rate = lanes["sglang-matched"]["metrics"]["output_tokens_per_second"]
    ratio = float(candidate_rate["p50"]) / float(matched_rate["p50"])
    interval = bootstrap_median_comparison_ci(
        candidate_values,
        matched_values,
        operation="ratio",
        salt="qwen35-pair:pypto-vs-matched:output-rate",
    )
    return {
        "schema": 1,
        "kind": "qwen35-9b-performance-pair-summary",
        "status": "complete",
        "performance_only": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "summary_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "summary_script_sha256": sha256_file(Path(__file__).resolve()),
            "source_lock_sha256": sha256_file(ROOT / "vendor/source-lock.json"),
        },
        "workload": workload_record(),
        "lanes": lanes,
        "comparison": {
            "throughput_metric": "output_tokens_per_second",
            "pypto_percent_of_matched": ratio * 100.0,
            "pypto_throughput_change_vs_matched_percent": (ratio - 1.0) * 100.0,
            "median_ratio_bootstrap_95ci_percent": {
                **interval,
                "lower": float(interval["lower"]) * 100.0,
                "upper": float(interval["upper"]) * 100.0,
            },
            "latency_metrics": {
                metric: {
                    "pypto_percent_of_matched": (
                        float(lanes["pypto"]["metrics"][metric]["p50"])
                        / float(lanes["sglang-matched"]["metrics"][metric]["p50"])
                        * 100.0
                    )
                }
                for metric in ("e2e_ms", "ttft_ms", "tpot_ms")
            },
        },
        "optimized_lane": {
            "status": "blocked-by-resource-qualification",
            "claim": None,
            "reason": (
                "The exact fixed workload could not keep the optimized stock "
                "capture configuration above the controller GPU free-memory floor."
            ),
        },
        "methodology": {
            "within_start_estimator": "p50 across ten measured requests",
            "headline_estimator": "median across four fresh process starts",
            "uncertainty": "95% percentile bootstrap over fresh starts",
            "pooling_requests_across_starts": False,
        },
        "matched_comparability": comparability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-report", action="append", type=Path, required=True)
    parser.add_argument(
        "--matched-report", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pypto_report) != 4 or len(args.matched_report) != 4:
        raise ReleaseContractError("pass four reports for each lane")
    result = summarize_records(
        [load(path, "pypto") for path in args.pypto_report],
        [load(path, "sglang-matched") for path in args.matched_report],
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
