#!/usr/bin/env python3
"""Compare one full-model eager control with the accepted matched timing pair."""

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
    if matched.get("status") != "complete":
        raise SystemExit("matched summary is not complete")
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
