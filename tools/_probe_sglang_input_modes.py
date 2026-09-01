"""Temporary raw/chat SGLang first-step comparison."""

from __future__ import annotations

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
from benchmarks.release.workload import RAW_PROMPT_TOKEN_IDS  # noqa: E402


def main() -> int:
    model_path = ROOT / "models/Qwen3.5-0.8B"
    torch, one_batch, runner, _requested, _resolved, _compat, workload, _resolution = (
        _load_runner("sglang-matched", model_path)
    )
    try:
        records = []
        for name, ids in (
            ("raw", list(RAW_PROMPT_TOKEN_IDS)),
            ("chat", list(workload["prompt_token_ids"])),
        ):
            output_ids, logits, _windows = _generate(
                torch, one_batch, runner, prompt_token_ids=ids
            )
            first = logits[0].float().cpu()
            values, indices = torch.topk(first, 8)
            records.append(
                {
                    "name": name,
                    "input_token_count": len(ids),
                    "input_token_ids": ids,
                    "first_top_ids": [int(value) for value in indices],
                    "first_top_values": [float(value) for value in values],
                    "output_prefix": output_ids[:16],
                }
            )
        print(json.dumps({"records": records}, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
