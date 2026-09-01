"""Temporary 0.8B PyPTO chat logits differential."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _load_runner,
    _shutdown_runner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    # Set this before importing/loading any SGLang model code so the plugin's
    # atexit differential writer observes the path in the worker process.
    import os

    os.environ["PYPTO_DIFFERENTIAL_REPORT"] = str(
        args.output.resolve().with_name("operator-differential-records.json")
    )

    model_path = ROOT / "models/Qwen3.5-0.8B"
    reference = torch.load(
        args.reference_logits.resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    ).float()
    torch_mod, one_batch, runner, _requested, _resolved, _compat, workload, _resolution = (
        _load_runner("pypto", model_path)
    )
    try:
        output_ids, candidate, _windows = _generate(
            torch_mod,
            one_batch,
            runner,
            prompt_token_ids=workload["prompt_token_ids"],
        )
        candidate = candidate.float().cpu()
        difference = (candidate - reference).abs()
        mismatch = next(
            (
                index
                for index, (expected, observed) in enumerate(
                    zip(reference.argmax(-1).tolist(), output_ids, strict=True)
                )
                if expected != observed
            ),
            None,
        )
        step = 55 if mismatch is None else mismatch
        reference_top = torch.topk(reference[step], 8)
        candidate_top = torch.topk(candidate[step], 8)
        cosine = torch.nn.functional.cosine_similarity(candidate, reference, dim=-1)
        payload = {
                    "first_mismatch": mismatch,
                    "reference_output_ids": reference.argmax(-1).tolist(),
                    "candidate_output_ids": output_ids,
                    "global_max_abs": float(difference.max()),
                    "global_mean_abs": float(difference.mean()),
                    "minimum_step_cosine": float(cosine.min()),
                    "step": step,
                    "step_max_abs": float(difference[step].max()),
                    "step_mean_abs": float(difference[step].mean()),
                    "step_cosine": float(cosine[step]),
                    "reference_top_ids": reference_top.indices.tolist(),
                    "reference_top_logits": reference_top.values.tolist(),
                    "candidate_top_ids": candidate_top.indices.tolist(),
                    "candidate_top_logits": candidate_top.values.tolist(),
                }
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
