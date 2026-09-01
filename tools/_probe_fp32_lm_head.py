"""Temporary shared FP32 LM-head correctness probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import correctness_runtime  # noqa: E402
from benchmarks.release import lanes  # noqa: E402
from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _load_runner,
    _shutdown_runner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("sglang-matched", "pypto"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = lanes.server_kwargs

    def kwargs(*values, **options):
        result = dict(original(*values, **options))
        result["enable_fp32_lm_head"] = True
        return result

    correctness_runtime.server_kwargs = kwargs
    model_path = ROOT / "models/Qwen3.5-0.8B"
    (
        torch,
        one_batch,
        runner,
        requested,
        resolved,
        _compatibility,
        workload,
        _resolution,
    ) = _load_runner(args.lane, model_path)
    try:
        output_ids, logits, _windows = _generate(
            torch, one_batch, runner, prompt_token_ids=workload["prompt_token_ids"]
        )
        logits_path = args.output.with_suffix(".pt")
        torch.save(logits.float().cpu(), logits_path)
        payload = {
            "lane": args.lane,
            "requested": requested,
            "resolved": resolved,
            "output_ids": output_ids,
            "logits_path": str(logits_path),
        }
        args.output.write_text(json.dumps(payload), encoding="utf-8")
        print(
            json.dumps(
                {"lane": args.lane, "output_prefix": output_ids[:64]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    finally:
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
