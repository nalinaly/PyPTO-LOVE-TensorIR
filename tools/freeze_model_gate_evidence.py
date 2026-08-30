#!/usr/bin/env python3
"""Freeze compact model correctness/coverage evidence without changing reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON object required: {resolved}")
    value["_path"] = resolved
    return value


def candidate_record(value: dict[str, object]) -> dict[str, object]:
    requests = value.get("requests")
    first = requests[0] if isinstance(requests, list) and requests else {}
    coverage = first.get("coverage") if isinstance(first, dict) else {}
    execution = first.get("compilation_execution") if isinstance(first, dict) else {}
    path = value["_path"]
    return {
        "run_id": value.get("run_id"),
        "status": value.get("status"),
        "all_passed": value.get("all_passed"),
        "request_count": value.get("request_count"),
        "stable_output": value.get("stable_output"),
        "output_sequence_sha256": first.get("output_sequence_sha256")
        if isinstance(first, dict)
        else None,
        "output_text_prefix": str(value.get("output_text") or "")[:240],
        "coverage": {
            "total_calls": coverage.get("total_calls"),
            "covered_calls": coverage.get("covered_calls"),
            "total_gpu_time_ns": coverage.get("total_gpu_time_ns"),
            "covered_gpu_time_ns": coverage.get("covered_gpu_time_ns"),
            "violation_count": coverage.get("violation_count"),
            "strict_policy_passed": coverage.get("strict_policy_passed"),
        },
        "compilation_execution": {
            "handwritten_compute_calls": execution.get("handwritten_compute_calls"),
            "inductor_compute_calls": execution.get("inductor_compute_calls"),
            "unknown_artifacts": execution.get("unknown_artifacts"),
        },
        "package_tree_sha256": (
            value.get("evidence_identity", {})
            .get("candidate_packages", {})
            .get("distributions", {})
            .get("pypto-kernels", {})
            .get("content_tree_sha256")
        ),
        "pypto_revision": (
            value.get("evidence_identity", {})
            .get("compiler", {})
            .get("pypto_revision")
        ),
        "report": path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = read(args.reference.resolve(strict=True))
    candidates = [candidate_record(read(path)) for path in args.candidate]
    accepted = [
        item
        for item in candidates
        if item["status"] == "complete" and item["all_passed"] is True
    ]
    coverage = [item["coverage"] for item in accepted]
    complete_coverage = bool(coverage) and all(
        item["total_calls"] == item["covered_calls"]
        and item["violation_count"] == 0
        and item["strict_policy_passed"] is True
        for item in coverage
    )
    output = args.output.resolve()
    result = {
        "schema": 1,
        "kind": "qwen35-model-gate-evidence",
        "model": args.model,
        "status": (
            "complete"
            if len(accepted) == 3 and complete_coverage
            else "partial"
            if accepted and complete_coverage
            else "failed"
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_lock_sha256": sha256(ROOT / "vendor/source-lock.json"),
        "reference": {
            "report": reference["_path"].relative_to(ROOT).as_posix(),
            "report_sha256": sha256(reference["_path"]),
            "status": reference.get("status"),
            "output_token_count": reference.get("output_token_count"),
            "output_text_prefix": str(reference.get("output_text") or "")[:240],
        },
        "candidate_start_count": len(candidates),
        "accepted_candidate_start_count": len(accepted),
        "candidates": candidates,
        "coverage_contract": {
            "all_accepted_traces_closed_world": complete_coverage,
            "evaluation": "teacher-forced-reference-prefixes",
            "unknown_or_fallback_allowed": False,
        },
        "unresolved": (
            []
            if len(accepted) == 3 and complete_coverage
            else ["three current-wheel candidate starts are required"]
        ),
    }
    for item in candidates:
        item.pop("_path", None)
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
