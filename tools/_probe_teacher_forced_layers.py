"""Temporary teacher-forced decoder-layer snapshot probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.correctness_runtime import (  # noqa: E402
    _load_runner,
    _shutdown_runner,
)


def _append_snapshot(torch, snapshots, name, output) -> None:
    values = output if isinstance(output, (tuple, list)) else (output,)
    captured = []
    for value in values:
        if type(value) is torch.Tensor:
            captured.append(value.detach().to(device="cpu", dtype=torch.bfloat16))
    snapshots[name].append(tuple(captured))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("pypto", "sglang-matched"), required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    reference = torch.load(
        args.reference_logits.resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    ).float()
    model_path = ROOT / "models/Qwen3.5-0.8B"
    (
        torch_mod,
        one_batch,
        runner,
        requested,
        resolved,
        _compatibility,
        workload,
        _resolution,
    ) = _load_runner(args.lane, model_path)
    torch_runner = getattr(runner, "torch_runner", runner)
    model = torch_runner.model.model
    snapshots = {f"layer.{index}": [] for index in range(len(model.layers))}
    snapshots["final_norm"] = []
    handles = []
    for index, layer in enumerate(model.layers):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: _append_snapshot(
                    torch_mod, snapshots, f"layer.{index}", output
                )
            )
        )
    handles.append(
        model.norm.register_forward_hook(
            lambda _module, _inputs, output: _append_snapshot(
                torch_mod, snapshots, "final_norm", output
            )
        )
    )

    batch = None
    try:
        prompt_ids = workload["prompt_token_ids"]
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(prompt_ids), [prompt_ids]
        )
        _next_ids, logits, batch = runner.extend(reqs)
        runner.synchronize()
        observed_logits = [logits.detach().float().cpu().contiguous()]
        reference_ids = reference.argmax(-1).tolist()
        for step in range(1, len(reference_ids)):
            forced = torch_mod.tensor(
                [reference_ids[step - 1]], device="cuda", dtype=torch.int64
            )
            _next_ids, logits = runner.decode(forced, batch)
            runner.synchronize()
            observed_logits.append(logits.detach().float().cpu().contiguous())
        payload = {
            "lane": args.lane,
            "requested": requested,
            "resolved": resolved,
            "reference_ids": reference_ids,
            "logits": torch_mod.cat(observed_logits, dim=0),
            "snapshots": snapshots,
        }
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        torch_mod.save(payload, args.output.resolve())
        print(
            {
                "lane": args.lane,
                "layers": len(model.layers),
                "steps": len(reference_ids),
                "output": str(args.output.resolve()),
            },
            flush=True,
        )
        return 0
    finally:
        for handle in handles:
            handle.remove()
        if batch is not None:
            runner.cleanup(batch)
        runner.clear()
        _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
