"""Temporary 9B semantic probe with CPU offload disabled."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import lanes  # noqa: E402
from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _load_runner,
    _shutdown_runner,
)


def main() -> int:
    lanes.memory_qualification = lambda *_args, **_kwargs: {
        "name": "diagnostic-zero-offload",
        "cpu_offload_gb": 0,
        "mem_fraction_static": 0.78,
    }
    model_path = ROOT / "models/Qwen3.5-9B"
    (
        torch,
        one_batch,
        runner,
        requested,
        resolved,
        _compatibility,
        workload,
        _resolution,
    ) = _load_runner("sglang-matched", model_path)
    try:
        output_ids, logits, _windows = _generate(
            torch, one_batch, runner, prompt_token_ids=workload["prompt_token_ids"]
        )
        values, indices = torch.topk(logits[0].float(), 8)
        print(
            json.dumps(
                {
                    "requested": requested,
                    "resolved": resolved,
                    "first_top_ids": [int(value) for value in indices],
                    "first_top_values": [float(value) for value in values],
                    "output_prefix": output_ids[:16],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
