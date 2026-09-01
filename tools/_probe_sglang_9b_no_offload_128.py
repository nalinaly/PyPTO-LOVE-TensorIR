"""Temporary 9B probe: no CPU offload with a 128-token cache envelope."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import lanes  # noqa: E402
from benchmarks.release import correctness_runtime  # noqa: E402
from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _load_runner,
    _shutdown_runner,
)


def _server_kwargs(*args, **kwargs):
    values = dict(lanes.server_kwargs(*args, **kwargs))
    values.update(
        {
            "cpu_offload_gb": 0,
            "context_length": 128,
            "max_total_tokens": 128,
            "max_prefill_tokens": 128,
            "mem_fraction_static": 0.76,
        }
    )
    return values


def main() -> int:
    correctness_runtime.server_kwargs = _server_kwargs
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
