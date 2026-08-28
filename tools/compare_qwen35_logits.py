#!/usr/bin/env python3
"""Compare saved Qwen logits against one frozen parity policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    policy_bytes = args.policy.read_bytes()
    policy = json.loads(policy_bytes)
    thresholds = policy["candidate_requirements"]
    reference = torch.load(
        args.reference_logits, map_location="cpu", weights_only=True
    ).float().contiguous()
    candidate = torch.load(
        args.candidate_logits, map_location="cpu", weights_only=True
    ).float().contiguous()
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            f"logit shapes differ: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )
    difference = (candidate - reference).abs()
    relative_floor = float(thresholds["max_relative_error_reference_floor"])
    relative_mask = reference.abs() >= relative_floor
    if not bool(relative_mask.any()):
        raise ValueError("reference-relative mask is empty")
    relative = difference[relative_mask] / reference.abs()[relative_mask]
    cosine = torch.nn.functional.cosine_similarity(
        candidate.double(), reference.double(), dim=1
    )
    reference_top_values, reference_top_ids = torch.topk(reference[0], 5)
    candidate_top_values, candidate_top_ids = torch.topk(candidate[0], 5)
    candidate_margin = float(candidate_top_values[0] - candidate_top_values[1])
    top5_overlap = len(
        set(reference_top_ids.tolist()) & set(candidate_top_ids.tolist())
    )
    metrics = {
        "candidate_top1_margin": candidate_margin,
        "cosine_similarity": float(cosine.min()),
        "max_abs_error": float(difference.max()),
        "max_relative_error": float(relative.max()),
        "mean_abs_error": float(difference.mean()),
        "reference_top5_logits": reference_top_values.tolist(),
        "reference_top5_token_ids": reference_top_ids.tolist(),
        "candidate_top5_logits": candidate_top_values.tolist(),
        "candidate_top5_token_ids": candidate_top_ids.tolist(),
        "top5_token_overlap": top5_overlap,
    }
    checks = {
        "candidate_top1_margin": candidate_margin
        >= float(thresholds["minimum_candidate_top1_margin"]),
        "cosine_similarity": metrics["cosine_similarity"]
        >= float(thresholds["cosine_similarity_min"]),
        "exact_greedy_token_ids": candidate_top_ids[:1].tolist()
        == policy["reference"]["next_token_ids"],
        "max_abs_error": metrics["max_abs_error"]
        <= float(thresholds["max_abs_error_max"]),
        "max_relative_error": metrics["max_relative_error"]
        <= float(thresholds["max_relative_error_max"]),
        "mean_abs_error": metrics["mean_abs_error"]
        <= float(thresholds["mean_abs_error_max"]),
        "top5_token_overlap": top5_overlap
        >= int(thresholds["top5_token_overlap_min"]),
    }
    payload = {
        "candidate_logits": str(args.candidate_logits.resolve()),
        "checks": checks,
        "kind": "qwen35-logits-parity-result",
        "metrics": metrics,
        "passed": all(checks.values()),
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "reference_logits": str(args.reference_logits.resolve()),
        "schema": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
