#!/usr/bin/env python3
"""Sweep schedule tile sizes for the PyPTO linear kernels on live hardware.

For each candidate shape this compiles the real ``linear_kernel`` /
``linear_to_float_kernel`` graph with each candidate schedule tile (the tile
is part of the compile identity, so every candidate is a fresh artifact),
sanity-checks numerics against ``torch.nn.functional.linear``, and times the
kernel with CUDA events using the same methodology as the operator A/B
harness (20 warmup calls, 30 timed batches of ``calls_per_batch`` launches).

This is an exploration tool: it does not touch the frozen release lanes and
writes its raw results under ``runs/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]

from pypto_kernels import _boot  # noqa: E402
from pypto_kernels.linear import (  # noqa: E402
    linear_kernel,
    linear_to_float_kernel,
)

import math  # noqa: E402


def build_program(rows, in_features, out_features, output_dtype):
    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    dtype = torch.bfloat16 if output_dtype == "bfloat16" else torch.float32
    out = torch.empty((rows, out_features), dtype=dtype, device="meta")
    kernel = (
        linear_kernel if output_dtype == "bfloat16" else linear_to_float_kernel
    )
    return kernel.specialize(x, weight, out)


def compile_tile(rows, in_features, out_features, output_dtype, tiles):
    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    dtype = torch.bfloat16 if output_dtype == "bfloat16" else torch.float32
    out = torch.empty((rows, out_features), dtype=dtype, device="meta")
    kernel = (
        linear_kernel if output_dtype == "bfloat16" else linear_to_float_kernel
    )
    started = time.perf_counter()
    key = _boot.compile_jit_kernel(
        kernel,
        (x, weight, out),
        tiles,
        provider="pypto.matmul",
    )
    return key, time.perf_counter() - started


def run_case(case, tiles, device: str) -> dict:
    torch.manual_seed(19)
    rows = case["rows"]
    in_features = case["in_features"]
    out_features = case["out_features"]
    output_dtype = case.get("output_dtype", "bfloat16")
    x = torch.randn(
        (rows, in_features), dtype=torch.bfloat16, device=device
    ).div_(8)
    weight = torch.randn(
        (out_features, in_features), dtype=torch.bfloat16, device=device
    ).div_(16)
    out_dtype = (
        torch.bfloat16 if output_dtype == "bfloat16" else torch.float32
    )
    out = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    stream = torch.cuda.current_stream(device)
    calls_per_batch = case.get("calls_per_batch", 100)
    baseline = None  # output of the first successful tile (invariance oracle)
    results = []
    for tile in tiles:
        tiles_arg = (
            list(tile)
            if isinstance(tile, tuple)
            else ([tile] if rows == 1 else [1, tile])
        )
        try:
            key, compile_s = compile_tile(
                rows, in_features, out_features, output_dtype, tiles_arg
            )
        except Exception as error:  # noqa: BLE001 - record and continue
            results.append(
                {"tile": tile, "status": "compile-failed", "error": str(error)[:200]}
            )
            continue
        out.zero_()
        _boot.launch_graph(key, (x, weight, out), stream.cuda_stream)
        torch.cuda.synchronize(device)
        if baseline is None:
            baseline = out.clone()
            bit_exact = True
        else:
            bit_exact = bool(torch.equal(out, baseline))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(20):
            _boot.launch_graph(key, (x, weight, out), stream.cuda_stream)
        torch.cuda.synchronize(device)
        batches = []
        for _ in range(30):
            start.record()
            for _ in range(calls_per_batch):
                _boot.launch_graph(key, (x, weight, out), stream.cuda_stream)
            end.record()
            end.synchronize()
            batches.append(start.elapsed_time(end) / calls_per_batch)
        ordered = sorted(batches)
        p50 = ordered[len(ordered) // 2]
        if rows == 1:
            grid = math.ceil(out_features / tile)
        elif isinstance(tile, tuple):
            grid = math.ceil(rows / tile[0]) * math.ceil(out_features / tile[1])
        else:
            grid = rows * math.ceil(out_features / tile)
        results.append(
            {
                "tile": tile,
                "status": "ok",
                "grid_x": grid,
                "p50_ms": p50,
                "min_ms": ordered[0],
                "max_ms": ordered[-1],
                "bit_exact_vs_first_tile": bit_exact,
                "compile_s": round(compile_s, 3),
            }
        )
    return results


CASES = {
    "gate-up-decode-1x4096x24576": {
        "rows": 1, "in_features": 4096, "out_features": 24576,
        "tiles": [128, 32, 16],
    },
    "down-decode-1x12288x4096": {
        "rows": 1, "in_features": 12288, "out_features": 4096,
        "tiles": [128, 32, 16],
    },
    "fp32-lm-head-1x4096x248320": {
        "rows": 1, "in_features": 4096, "out_features": 248320,
        "output_dtype": "float32", "calls_per_batch": 1,
        "tiles": [128, 32, 16],
    },
    "qkv-q-decode-1x4096x4096": {
        "rows": 1, "in_features": 4096, "out_features": 4096,
        "tiles": [128, 64, 32, 16],
    },
    "kv-decode-1x4096x1024": {
        "rows": 1, "in_features": 4096, "out_features": 1024,
        "tiles": [128, 32, 16, 8],
    },
    "gate-up-prefill-31x4096x24576": {
        "rows": 31, "in_features": 4096, "out_features": 24576,
        "calls_per_batch": 20,
        "tiles": [(1, 32), (2, 32), (4, 32), (8, 32), (16, 32)],
    },
    "down-prefill-31x12288x4096": {
        "rows": 31, "in_features": 12288, "out_features": 4096,
        "calls_per_batch": 20,
        "tiles": [(1, 32), (2, 32), (4, 32), (8, 32), (16, 32)],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = args.device
    free, total = torch.cuda.mem_get_info(device)
    print(f"device={device} free={free/2**30:.1f} GiB / {total/2**30:.1f} GiB")
    if free < 4 << 30:
        raise SystemExit("less than 4 GiB GPU memory free; aborting sweep")
    payload = {"kind": "linear-tile-sweep", "cases": {}}
    for name in args.cases:
        case = CASES[name]
        print(f"== {name} ==")
        results = run_case(case, case["tiles"], device)
        for r in results:
            if r["status"] == "ok":
                print(
                    f"  tile={str(r['tile']):>9} grid={r['grid_x']:>5} "
                    f"p50={r['p50_ms']:9.4f} ms "
                    f"bit_exact_vs_first={r['bit_exact_vs_first_tile']} "
                    f"compile={r['compile_s']}s"
                )
            else:
                print(f"  tile={r['tile']:>4} {r['status']}: {r['error']}")
        payload["cases"][name] = results
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
