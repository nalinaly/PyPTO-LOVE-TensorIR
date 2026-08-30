#!/usr/bin/env python3
"""Measure eager, official Inductor CUDA, and PyPTO on a Qwen SwiGLU shape.

This worker intentionally measures one operator invocation family.  It does
not claim an end-to-end model speedup; the caller must provide the real model
shape and retain the raw JSON together with its bounded GPU run identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def _synchronize(torch) -> None:
    torch.cuda.synchronize()


def _eager_swiglu(gate, up):
    wide = gate.float()
    return (wide * wide.sigmoid() * up.float()).to(gate.dtype)


def _profile(torch, function, gate, up) -> dict[str, object] | None:
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None
    _synchronize(torch)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        result = function(gate, up)
        _synchronize(torch)
    del result
    events = [event for event in trace.events() if event.device_type.name == "CUDA"]
    return {
        "cuda_event_count": len(events),
        "kernel_event_count": len(events),
        "kernel_names": sorted({event.name for event in events}),
    }


def _measure(torch, function, gate, up, warmup_calls: int, timed_calls: int):
    _synchronize(torch)
    started = time.perf_counter_ns()
    result = function(gate, up)
    _synchronize(torch)
    cold_ms = (time.perf_counter_ns() - started) / 1e6
    del result
    for _ in range(warmup_calls):
        result = function(gate, up)
        del result
    _synchronize(torch)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(timed_calls):
        result = function(gate, up)
        del result
    end.record()
    end.synchronize()
    return cold_ms, float(begin.elapsed_time(end)) / timed_calls


def _function_for_mode(torch, mode: str):
    if mode == "eager":
        return _eager_swiglu, "eager"
    if mode == "inductor-nv":
        torch._dynamo.reset()
        return torch.compile(
            _eager_swiglu, backend="inductor", dynamic=False, fullgraph=True
        ), "inductor"
    from pypto_plugins.torch import inductor_swiglu

    return inductor_swiglu.run_fp32_swiglu, "pypto"


def run(
    mode: str,
    rows: int,
    columns: int,
    warmup_calls: int,
    timed_calls: int,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("the ablation requires one visible SM120 CUDA device")
    if rows <= 0 or columns <= 0 or columns % 128:
        raise ValueError("rows must be positive and columns must be divisible by 128")
    if warmup_calls < 0 or timed_calls <= 0:
        raise ValueError("warmup_calls must be non-negative and timed_calls positive")
    torch.manual_seed(20260829)
    packed = torch.randn((rows, 2 * columns), device="cuda", dtype=torch.bfloat16)
    gate = packed[:, :columns]
    up = packed[:, columns:]
    if tuple(gate.stride()) != (2 * columns, 1):
        raise RuntimeError("packed Qwen gate/up view did not preserve row pitch")
    function, backend = _function_for_mode(torch, mode)
    cold_ms, warm_ms = _measure(torch, function, gate, up, warmup_calls, timed_calls)
    profile = _profile(torch, function, gate, up)
    # Keep this worker timing-only. Numerical acceptance belongs to the
    # independent operator/model regression suites.
    result = function(gate, up)
    del result
    _synchronize(torch)
    return {
        "schema": 1,
        "kind": "qwen35-9b-inductor-ablation",
        "mode": mode,
        "backend": backend,
        "shape": [rows, columns],
        "packed_input_shape": [rows, 2 * columns],
        "packed_input_stride": list(packed.stride()),
        "warmup_calls": warmup_calls,
        "timed_calls": timed_calls,
        "cold_first_call_ms": cold_ms,
        "warm_call_ms": warm_ms,
        "profile": profile,
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "cuda": str(torch.version.cuda),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("eager", "inductor-nv", "pypto"), required=True
    )
    parser.add_argument("--rows", type=int, default=19)
    parser.add_argument("--columns", type=int, default=12_288)
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--timed-calls", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.mode, args.rows, args.columns, args.warmup_calls, args.timed_calls
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
