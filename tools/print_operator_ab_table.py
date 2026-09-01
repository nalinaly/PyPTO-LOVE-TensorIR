#!/usr/bin/env python3
"""Print the newest operator A/B aggregation as a compact per-case table.

Reads the most recent ``runs/release-operator-ab-*/aggregation.json`` plus the
newest per-lane reports and prints one row per aligned operator case. Used for
live-run terminal evidence; the frozen release numbers live in
``state/evidence/qwen35-9b-operator-performance-breakdown-current.json``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def newest(pattern: str) -> Path:
    matches = sorted(Path(p) for p in glob.glob(str(ROOT / pattern)))
    if not matches:
        raise SystemExit(f"no file matches {pattern}")
    return matches[-1]


def main() -> int:
    aggregation = newest("runs/release-operator-ab-*/aggregation.json")
    summary = json.loads(aggregation.read_text(encoding="utf-8"))
    pypto_report = newest("runs/*/qwen35-9b-operator-performance-pypto.json")
    stock_report = newest(
        "runs/*/qwen35-9b-operator-performance-sglang-matched.json"
    )
    pypto = {
        case["name"]: case["latency_ms_per_call"]["p50"]
        for case in json.loads(pypto_report.read_text(encoding="utf-8"))["cases"]
    }
    stock = {
        case["name"]: case["latency_ms_per_call"]["p50"]
        for case in json.loads(stock_report.read_text(encoding="utf-8"))["cases"]
    }
    starts = (
        summary.get("lanes", {}).get("pypto", {}).get("fresh_starts", "n/a")
    )
    print(f"aggregation: {aggregation.relative_to(ROOT)}")
    print(f"status: {summary['status']} | fresh starts per lane: {starts}")
    print("case | pypto p50 ms/call | stock p50 ms/call | pypto vs stock")
    for name, entry in sorted(summary["comparisons"].items()):
        interval = entry["median_ratio_bootstrap_95ci_percent"]
        print(
            f"{name} | {pypto[name]:.6f} | {stock[name]:.6f} | "
            f"{entry['pypto_latency_percent_of_stock']:.1f}% CI "
            f"[{interval['lower']:.1f}, {interval['upper']:.1f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
