#!/usr/bin/env python3
"""Print the checked-in 9B SwiGLU ablation evidence as a compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
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
    print("scope=operator-level | Qwen3.5-9B SwiGLU | denominator=one invocation")
    for phase in ("prefill", "decode"):
        case = payload["phases"][phase]
        geometry = case["geometry"]
        shape = f"{phase} {geometry['rows']}x{geometry['columns'] * 2}"
        for mode, key in (
            ("eager", "eager"),
            ("inductor-nv", "inductor_nv"),
            ("pypto", "pypto"),
        ):
            result = case[key]
            print(
                f"{shape} | {mode}: warm={result['warm_call_ms']:.6f}ms "
                f"cold={result['cold_first_call_ms']:.1f}ms "
                f"launches={result['kernel_event_count']}"
            )
        derived = case["derived"]
        print(
            f"{shape} | NV-vs-eager={derived['inductor_nv_speedup_vs_eager_percent']:+.2f}% "
            f"PyPTO-vs-eager={derived['pypto_speedup_vs_eager_percent']:+.2f}% "
            f"compile-overhead={derived['pypto_compile_longer_than_inductor_nv_percent']:+.2f}% "
            f"launch-reduction={derived['compiled_launch_reduction_vs_eager_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
