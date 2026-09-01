"""Temporary one-request teacher-forced CUPTI diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.cupti_overlay import activate_overlay  # noqa: E402
from benchmarks.release.correctness_runtime import (  # noqa: E402
    _generate,
    _generate_teacher_forced,
    _load_runner,
    _shutdown_runner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_report.resolve(strict=True).read_text())
    expected_ids = [int(value) for value in reference["output_token_ids"]]

    overlay = activate_overlay()
    import torch
    from torch.profiler import _cupti_monitor as monitor_api

    if torch.cuda.is_initialized():
        raise RuntimeError("CUPTI diagnostic must start before CUDA initialization")
    monitor = monitor_api.start_collection(args.output.resolve().parent / "cupti-probe")
    runner = None
    try:
        (
            torch_mod,
            one_batch,
            runner,
            _requested,
            _resolved,
            _compatibility,
            workload,
            _resolution,
        ) = _load_runner("pypto", ROOT / "models/Qwen3.5-0.8B")
        _generate(
            torch_mod,
            one_batch,
            runner,
            prompt_token_ids=workload["prompt_token_ids"],
        )
        torch_mod.cuda.synchronize()
        sampled, _logits, windows = _generate_teacher_forced(
            torch_mod,
            one_batch,
            runner,
            expected_ids,
            monitor,
            prompt_token_ids=workload["prompt_token_ids"],
        )
        stats = monitor_api.stop_collection()
        monitor = None
        payload = {
            "overlay": overlay,
            "sampled_ids": sampled,
            "stats": stats,
            "windows": windows,
        }
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            {"window_count": len(windows), "output": str(args.output.resolve())},
            flush=True,
        )
        return 0
    finally:
        if monitor is not None:
            monitor_api.stop_collection()
        if runner is not None:
            _shutdown_runner()


if __name__ == "__main__":
    raise SystemExit(main())
