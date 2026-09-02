#!/usr/bin/env python3
"""Validate and print the accepted Qwen3.5-9B model-gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
EXPECTED_CALLS = 27_808
EXPECTED_HANDWRITTEN = 25_760
EXPECTED_INDUCTOR = 2_048


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"model-gate evidence rejected: {message}")


def display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def truncate_display(value: str, width: int) -> str:
    output = []
    used = 0
    for character in value.replace("\n", " "):
        increment = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        if used + increment > width - 3:
            break
        output.append(character)
        used += increment
    return "".join(output).rstrip() + "..."


def validate(path: Path) -> dict[str, object]:
    evidence_path = path.resolve(strict=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require(evidence.get("schema") == 1, "unknown sidecar schema")
    require(evidence.get("kind") == "qwen35-model-gate-evidence", "wrong kind")
    require(evidence.get("model") == "Qwen3.5-9B", "wrong model")
    require(evidence.get("status") == "complete", "sidecar is not complete")
    require(evidence.get("unresolved") == [], "sidecar has unresolved entries")
    require(evidence.get("accepted_candidate_start_count") == 3, "start count")
    require(
        evidence.get("source_lock_sha256") == sha256(ROOT / "vendor/source-lock.json"),
        "source lock changed",
    )
    reference = evidence.get("reference")
    require(isinstance(reference, dict), "reference is absent")
    reference_report = (ROOT / str(reference.get("report", ""))).resolve(strict=True)
    require(ROOT in reference_report.parents, "reference escapes the repository")
    require(reference.get("report_sha256") == sha256(reference_report), "reference hash")
    reference_raw = json.loads(reference_report.read_text(encoding="utf-8"))
    require(reference_raw.get("status") == "complete", "reference status")
    reference_text = str(reference.get("output_text_prefix"))
    candidates = evidence.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 3, "candidates")
    output_hashes: set[str] = set()
    output_texts: set[str] = set()
    for candidate in candidates:
        require(isinstance(candidate, dict), "candidate is not an object")
        report = (ROOT / str(candidate.get("report", ""))).resolve(strict=True)
        require(ROOT in report.parents, "report escapes the repository")
        require(candidate.get("report_sha256") == sha256(report), "report hash")
        require(candidate.get("status") == "complete", "candidate status")
        require(candidate.get("all_passed") is True, "candidate checks")
        require(candidate.get("stable_output") is True, "candidate output drift")
        require(candidate.get("request_count") == 10, "candidate request count")
        raw = json.loads(report.read_text(encoding="utf-8"))
        require(raw.get("status") == "complete", "raw report status")
        require(raw.get("all_passed") is True, "raw report checks")
        require(raw.get("stable_output") is True, "raw report output drift")
        require(raw.get("run_id") == candidate.get("run_id"), "run id drift")
        workload = raw.get("workload", {})
        require(workload.get("prompt") == EXPECTED_PROMPT, "prompt drift")
        require(workload.get("prompt_tokens") == 31, "prompt token count")
        require(workload.get("output_tokens") == 64, "output token count")
        engine = raw.get("engine", {})
        require(engine.get("all_complete") is True, "Engine requests incomplete")
        require(engine.get("stable_output") is True, "Engine output drift")
        require(len(engine.get("requests", [])) == 10, "Engine request count")
        requests = raw.get("requests")
        require(isinstance(requests, list) and len(requests) == 1, "strict trace count")
        trace = requests[0]
        coverage = trace.get("coverage", {})
        compilation = trace.get("compilation_execution", {})
        require(coverage.get("strict_policy_passed") is True, "coverage policy")
        require(trace.get("passed") is True, "strict trace failed")
        require(trace.get("exact_output_sequence") is True, "token sequence mismatch")
        require(coverage.get("violation_count") == 0, "coverage violations")
        require(coverage.get("fallback_event_groups", 0) == 0, "fallback events")
        require(coverage.get("covered_calls") == EXPECTED_CALLS, "covered calls")
        require(coverage.get("total_calls") == EXPECTED_CALLS, "total calls")
        require(
            coverage.get("covered_gpu_time_ns") == coverage.get("total_gpu_time_ns"),
            "covered GPU time",
        )
        require(
            compilation.get("handwritten_compute_calls") == EXPECTED_HANDWRITTEN,
            "handwritten partition",
        )
        require(
            compilation.get("inductor_compute_calls") == EXPECTED_INDUCTOR,
            "Inductor partition",
        )
        require(compilation.get("unknown_artifacts") == [], "unknown artifacts")
        require(
            int(compilation.get("handwritten_compute_calls", -1))
            + int(compilation.get("inductor_compute_calls", -1))
            == EXPECTED_CALLS,
            "provider partition sum",
        )
        require(
            trace.get("output_sequence_sha256")
            == candidate.get("output_sequence_sha256"),
            "trace output hash",
        )
        require(str(raw.get("output_text")) == reference_text, "reference text drift")
        output_hashes.add(str(candidate.get("output_sequence_sha256")))
        output_texts.add(str(raw.get("output_text")))
    require(len(output_hashes) == 1, "output token sequence differs across starts")
    require(len(output_texts) == 1, "decoded output differs across starts")
    return {
        "candidates": candidates,
        "output_sequence_sha256": output_hashes.pop(),
        "output_text": output_texts.pop(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "state/evidence/qwen35-9b-model-gate-current.json",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = validate(args.evidence)
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    print("accepted Qwen3.5-9B evidence replay (not a live rerun)")
    print("3 starts x 10 Engine requests | stable 64-token output")
    print(f"run={candidates[0]['run_id']}")
    print("coverage=27808/27808 strict=true fallback=0")
    print("providers: handwritten=25760 Inductor=2048 unknown=0")
    if args.compact:
        print(f"prompt: {EXPECTED_PROMPT}")
        print(f"output: {truncate_display(str(result['output_text']), 58)}")
    else:
        print(f"output_sha256={result['output_sequence_sha256']}")
        print(f"prompt: {EXPECTED_PROMPT}")
        print(f"output: {result['output_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
