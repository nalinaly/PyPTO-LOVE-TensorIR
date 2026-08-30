#!/usr/bin/env python3
"""Print the checked-in 9B SwiGLU ablation evidence as a compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("state/evidence/qwen35-9b-inductor-ablation-current.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if payload.get("kind") != "qwen35-9b-inductor-ablation":
        raise SystemExit("unexpected ablation evidence kind")
    if payload.get("performance_only") is not True or payload.get("schema") != 2:
        raise SystemExit("evidence is not the current performance-only schema")
    print("operator-level Qwen3.5-9B SwiGLU | E=eager N=NV P=PyPTO")
    for phase in ("prefill", "decode"):
        case = payload["phases"][phase]
        geometry = case["geometry"]
        shape = f"{geometry['rows']}x{geometry['columns'] * 2}"
        print()
        if args.compact:
            eager = case["eager"]
            nv = case["inductor_nv"]
            pypto = case["pypto"]
            derived = case["derived"]
            print(
                f"[{phase} {shape}] warm "
                f"E={eager['warm_call_ms']:.6f} "
                f"N={nv['warm_call_ms']:.6f} "
                f"P={pypto['warm_call_ms']:.6f}"
            )
            print(
                "  cold/launch "
                f"E={eager['cold_first_call_ms']:.1f}/{eager['kernel_event_count']} "
                f"N={nv['cold_first_call_ms']:.1f}/{nv['kernel_event_count']} "
                f"P={pypto['cold_first_call_ms']:.1f}/{pypto['kernel_event_count']}"
            )
            print(
                "  delta "
                f"dN={derived['inductor_nv_speedup_vs_eager_percent']:+.2f}% "
                f"dP={derived['pypto_speedup_vs_eager_percent']:+.2f}% "
                f"dC={derived['pypto_compile_longer_than_inductor_nv_percent']:+.2f}% "
                f"LR={derived['compiled_launch_reduction_vs_eager_percent']:.2f}%"
            )
            continue
        print(f"[{phase}] shape={shape}")
        for mode, key in (
            ("eager", "eager"),
            ("inductor-nv", "inductor_nv"),
            ("pypto", "pypto"),
        ):
            result = case[key]
            print(
                f"  {mode:11} warm={result['warm_call_ms']:.6f} ms  "
                f"cold={result['cold_first_call_ms']:.1f} ms  "
                f"launches={result['kernel_event_count']}"
            )
        derived = case["derived"]
        print(
            f"  speedup vs eager: NV={derived['inductor_nv_speedup_vs_eager_percent']:+.2f}%  "
            f"PyPTO={derived['pypto_speedup_vs_eager_percent']:+.2f}%"
        )
        print(
            f"  PyPTO cold vs NV={derived['pypto_compile_longer_than_inductor_nv_percent']:+.2f}%  "
            f"launch reduction={derived['compiled_launch_reduction_vs_eager_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
