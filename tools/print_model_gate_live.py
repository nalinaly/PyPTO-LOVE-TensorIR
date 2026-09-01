#!/usr/bin/env python3
"""Print the newest live Qwen3.5 model-correctness run as a compact summary.

Reads the most recent ``runs/*/qwen35-9b-correctness.json`` candidate report
and its sibling coverage report, and prints the pass status, coverage audit,
fixed prompt, and generated output. Used for live-run terminal evidence; the
frozen release numbers live in ``state/evidence/qwen35-9b-model-gate-current.json``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    candidates = sorted(
        Path(p) for p in glob.glob(str(ROOT / "runs/*/qwen35-9b-correctness.json"))
    )
    if not candidates:
        raise SystemExit("no qwen35-9b-correctness.json report found")
    report_path = candidates[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    coverage_files = sorted(report_path.parent.glob("coverage-request-*.json"))
    if coverage_files:
        coverage = json.loads(coverage_files[-1].read_text(encoding="utf-8"))
        calls = coverage["model_forward_compute"]["calls"]
        coverage_line = (
            f"coverage: {calls['covered']}/{calls['total']} | "
            f"strict: {coverage['strict_policy_passed']} | "
            f"violations: {len(coverage['violations'])} | "
            f"fallbacks: {len(coverage['fallbacks'])}"
        )
    else:
        coverage_line = "coverage: (no coverage report in run directory)"
    print(f"candidate report: {report_path.relative_to(ROOT)}")
    print(
        f"all_passed: {report['all_passed']} | requests: {report['request_count']} "
        f"| stable_output: {report['stable_output']} | "
        f"teacher-forced: {report.get('teacher_forced_request_count', 'n/a')}"
    )
    print(coverage_line)
    print("prompt: 为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？")
    output = str(report.get("output_text", ""))
    shown = output if len(output) <= 100 else output[:100] + "…"
    print(f"output: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
