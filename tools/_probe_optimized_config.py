#!/usr/bin/env python3
"""Temporary optimized-lane memory probe; never part of release evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import lanes, performance_runtime
from benchmarks.release.workload import require_run_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cpu-offload-gb", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    args = parser.parse_args()
    import torch

    torch._inductor.config.compile_threads = 1
    original_memory = lanes.memory_qualification
    original_server = lanes.server_kwargs

    def patched_memory(lane, mode="zero-offload", model_path=None):
        value = original_memory(lane, mode, model_path)
        if lane == "sglang-optimized":
            value = dict(value)
            value.update(
                {
                    "name": "probe-optimized",
                    "cpu_offload_gb": args.cpu_offload_gb,
                    "mem_fraction_static": args.mem_fraction_static,
                }
            )
        return value

    def patched_server(lane, model_path, mode="zero-offload"):
        value = original_server(lane, model_path, mode)
        if lane == "sglang-optimized":
            value = dict(value)
            value.update(
                {
                    "cpu_offload_gb": args.cpu_offload_gb,
                    "mem_fraction_static": args.mem_fraction_static,
                    "cuda_graph_bs_decode": [1],
                    "cuda_graph_bs_prefill": [32],
                    "cuda_graph_backend_prefill": "disabled",
                }
            )
        return value

    lanes.memory_qualification = patched_memory
    lanes.server_kwargs = patched_server
    performance_runtime.memory_qualification = patched_memory
    performance_runtime.server_kwargs = patched_server
    run_id, run_dir = require_run_directory(ROOT)
    return performance_runtime.run(
        "sglang-optimized",
        args.model_path,
        run_id,
        run_dir,
        "matched",
    )


if __name__ == "__main__":
    raise SystemExit(main())
