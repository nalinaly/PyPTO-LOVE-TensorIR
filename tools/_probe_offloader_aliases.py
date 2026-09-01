"""Temporary 0.8B offloader alias and chat-prefix probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import correctness_runtime, lanes  # noqa: E402
from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _load_runner,
    _shutdown_runner,
)


def _server_kwargs(*args, **kwargs):
    values = dict(lanes.server_kwargs(*args, **kwargs))
    values.update({"cpu_offload_gb": 1, "mem_fraction_static": 0.78})
    return values


def main() -> int:
    correctness_runtime.server_kwargs = _server_kwargs
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
    ) = _load_runner("sglang-matched", model_path)
    try:
        model = runner.torch_runner.model
        layer = model.model.layers[0]
        state = layer.state_dict()
        aliases = {}
        for key in ("linear_attn.A_log", "linear_attn.attn.A_log"):
            value = state.get(key)
            aliases[key] = {
                "device": str(value.device) if value is not None else None,
                "data_ptr": int(value.data_ptr()) if value is not None else None,
            }
        gdn = layer.linear_attn
        conv = gdn.conv1d.weight
        view = gdn.attn.conv_weights
        aliases["conv1d.weight"] = {
            "device": str(conv.device),
            "data_ptr": int(conv.data_ptr()),
        }
        aliases["attn.conv_weights"] = {
            "device": str(view.device),
            "data_ptr": int(view.data_ptr()),
            "same_storage": bool(view.data_ptr() == conv.data_ptr()),
        }
        output_ids, logits, _windows = _generate(
            torch, one_batch, runner, prompt_token_ids=workload["prompt_token_ids"]
        )
        values, indices = torch.topk(logits[0].float(), 8)
        print(
            json.dumps(
                {
                    "requested": requested,
                    "resolved": resolved,
                    "aliases": aliases,
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
