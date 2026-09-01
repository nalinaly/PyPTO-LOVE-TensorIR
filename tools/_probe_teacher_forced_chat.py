"""Temporary teacher-forced PyPTO logits comparison."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.correctness_runtime import (  # noqa: E402
    _load_runner,
    _shutdown_runner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    model_path = ROOT / "models/Qwen3.5-0.8B"
    reference = torch.load(
        args.reference_logits.resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    ).float()
    (
        torch_mod,
        one_batch,
        runner,
        _requested,
        _resolved,
        _compat,
        workload,
        _resolution,
    ) = _load_runner("pypto", model_path)
    batch = None
    try:
        ids = workload["prompt_token_ids"]
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(1, len(ids), [ids])
        next_ids, logits, batch = runner.extend(reqs)
        runner.synchronize()
        observed = [logits.detach().float().cpu().contiguous()]
        reference_ids = reference.argmax(-1).tolist()
        for step in range(1, len(reference_ids)):
            forced = torch_mod.tensor([reference_ids[step - 1]], device="cuda", dtype=torch.int64)
            _next, logits = runner.decode(forced, batch)
            runner.synchronize()
            observed.append(logits.detach().float().cpu().contiguous())
        candidate = torch_mod.cat(observed, dim=0)
        delta = (candidate - reference).abs()
        cosine = torch_mod.nn.functional.cosine_similarity(candidate, reference, dim=-1)
        mismatches = []
        for step, row in enumerate(candidate):
            ref_top = torch_mod.topk(reference[step], 5)
            cand_top = torch_mod.topk(row, 5)
            if int(ref_top.indices[0]) != int(cand_top.indices[0]):
                mismatches.append(
                    {
                        "step": step,
                        "reference_top_ids": ref_top.indices.tolist(),
                        "candidate_top_ids": cand_top.indices.tolist(),
                        "reference_top_logits": ref_top.values.tolist(),
                        "candidate_top_logits": cand_top.values.tolist(),
                    }
                )
        payload = {
                    "shape": list(candidate.shape),
                    "global_max_abs": float(delta.max()),
                    "global_mean_abs": float(delta.mean()),
                    "minimum_step_cosine": float(cosine.min()),
                    "top1_mismatch_count": len(mismatches),
                    "top1_mismatches": mismatches,
                }
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if batch is not None:
            runner.cleanup(batch)
        runner.clear()
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
