#!/usr/bin/env python3
"""Compare one full-model eager control with a resource-valid matched subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import sha256_file, workload_record  # noqa: E402


METRICS = ("e2e_ms", "ttft_ms", "tpot_ms", "output_tokens_per_second")
GPU_FREE_FLOOR_BYTES = 4 * 1024**3
HOST_FREE_FLOOR_KIB = 12 * 1024**2


def validate_eager_resources(resources: object) -> dict[str, object]:
    if not isinstance(resources, dict) or resources.get("nvml_error") is not None:
        raise SystemExit("eager control lacks valid NVML telemetry")
    summary = resources.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("eager control lacks a resource summary")
    if (
        type(summary.get("minimum_gpu_memory_free_bytes")) is not int
        or int(summary["minimum_gpu_memory_free_bytes"]) < GPU_FREE_FLOOR_BYTES
        or type(summary.get("minimum_mem_available_kib")) is not int
        or int(summary["minimum_mem_available_kib"]) < HOST_FREE_FLOOR_KIB
        or type(summary.get("sample_count")) is not int
        or int(summary["sample_count"]) <= 0
        or summary.get("thermal_throttle_observed") is not False
    ):
        raise SystemExit("eager control is not resource-qualified")
    return {
        "accepted": True,
        "gpu_free_floor_bytes": GPU_FREE_FLOOR_BYTES,
        "host_free_floor_kib": HOST_FREE_FLOOR_KIB,
        "minimum_gpu_memory_free_bytes": summary["minimum_gpu_memory_free_bytes"],
        "minimum_mem_available_kib": summary["minimum_mem_available_kib"],
        "sample_count": summary["sample_count"],
        "thermal_throttle_observed": False,
    }


def validate_matched_subset(summary: object) -> dict[str, object]:
    if not isinstance(summary, dict):
        raise SystemExit("matched summary is not an object")
    status = summary.get("status")
    invalidation = summary.get("acceptance")
    if status == "complete":
        source_boundary = "source pair is complete"
    elif (
        status == "invalidated-resource-floor"
        and isinstance(invalidation, dict)
        and invalidation.get("accepted") is False
        and invalidation.get("affected_lane") == "pypto"
    ):
        source_boundary = (
            "source pair is invalidated by the PyPTO resource floor; only the "
            "independently resource-valid matched subset is consumed"
        )
    else:
        raise SystemExit("matched summary has no consumable matched subset")
    lane = summary.get("lanes", {}).get("sglang-matched", {})
    starts = lane.get("starts", []) if isinstance(lane, dict) else []
    if len(starts) != 4:
        raise SystemExit("matched subset does not contain four starts")
    for start in starts:
        resources = start.get("resources") if isinstance(start, dict) else None
        if (
            not isinstance(resources, dict)
            or type(resources.get("minimum_gpu_memory_free_bytes")) is not int
            or int(resources["minimum_gpu_memory_free_bytes"])
            < GPU_FREE_FLOOR_BYTES
            or resources.get("thermal_throttle_observed") is not False
        ):
            raise SystemExit("matched subset is not resource-qualified")
    return {
        "source_pair_status": status,
        "source_pair_accepted": status == "complete",
        "matched_subset_resource_accepted": True,
        "matched_starts": 4,
        "gpu_free_floor_bytes": GPU_FREE_FLOOR_BYTES,
        "evidence_boundary": source_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-report", type=Path, required=True)
    parser.add_argument("--matched-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eager_path = args.eager_report.resolve(strict=True)
    matched_path = args.matched_summary.resolve(strict=True)
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    if (
        eager.get("status") != "complete"
        or eager.get("comparison_mode") != "eager-control"
        or eager.get("workload") != workload_record()
    ):
        raise SystemExit("eager report is not the expected timing-only control")
    eager_resource_boundary = validate_eager_resources(eager.get("resources"))
    matched_boundary = validate_matched_subset(matched)
    matched_metrics = matched["lanes"]["sglang-matched"]["metrics"]
    eager_metrics = eager["metrics"]
    result = {
        "schema": 1,
        "kind": "qwen35-9b-eager-compile-ablation",
        "status": "complete",
        "performance_only": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workload": workload_record(),
        "eager_control": {
            "report": eager_path.relative_to(ROOT).as_posix(),
            "report_sha256": sha256_file(eager_path),
            "comparison_mode": eager["comparison_mode"],
            "resource_boundary": eager_resource_boundary,
            "metrics_p50": {
                metric: float(eager_metrics[metric]["p50"]) for metric in METRICS
            },
            "cold_engine_start_ms": float(eager["cold_engine_start_ms"]),
            "first_compile_trigger_request_ms": float(
                eager["first_compile_trigger_request_ms"]
            ),
        },
        "matched_compile_requested": {
            "summary": matched_path.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256_file(matched_path),
            "metrics_p50": {
                metric: float(matched_metrics[metric]["p50"]) for metric in METRICS
            },
            "cold_engine_start_ms": float(
                matched["lanes"]["sglang-matched"]["cold_engine_start_ms"]["p50"]
            ),
            "first_compile_trigger_request_ms": float(
                matched["lanes"]["sglang-matched"][
                    "first_compile_trigger_request_ms"
                ]["p50"]
            ),
            "backend_invocation_observed": False,
            "effective_compile": False,
            "source_pair_boundary": matched_boundary,
        },
        "observed_difference": {
            metric: {
                "eager_to_matched_ratio_percent": (
                    float(matched_metrics[metric]["p50"])
                    / float(eager_metrics[metric]["p50"])
                    * 100.0
                )
            }
            for metric in METRICS
        },
        "interpretation": {
            "whole_model_torch_compile_speedup_percent": None,
            "reason": (
                "The matched compile-request lane disables CUDA graphs in this "
                "pinned SGLang configuration, so its global CompilerInterface "
                "and Inductor backend were not invoked. The pair is a workload "
                "control, not a causal compile-speedup measurement."
            ),
            "causal_compile_measurement": "not_available",
            "operator_causal_measurement": (
                "Use qwen35-9b-inductor-ablation-current.json for the supported "
                "SwiGLU eager versus official-NV/PyPTO comparison."
            ),
        },
    }
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
