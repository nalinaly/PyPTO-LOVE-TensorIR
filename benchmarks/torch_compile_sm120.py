#!/usr/bin/env python3
"""Execute one native tile graph through ``torch.compile(backend='pypto')``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pypto_plugins.torch import runtime_bridge, scheduling
import pypto_plugins.torch_inductor as torch_inductor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    torch._dynamo.reset()
    scheduling.REGISTRY.clear()
    launches: list[dict[str, object]] = []
    original_launch = runtime_bridge.pypto_launch

    def counted_launch(kernel_name, tensors, stream):
        launches.append(
            {
                "kernel": kernel_name,
                "argument_count": len(tensors),
                "stream": int(stream),
            }
        )
        return original_launch(kernel_name, tensors, stream)

    runtime_bridge.pypto_launch = counted_launch
    try:
        torch.manual_seed(7)
        lhs = torch.randn((256, 1024), device="cuda", dtype=torch.bfloat16)
        rhs = torch.randn_like(lhs)

        def function(x, y):
            return (x + y) * 2.0

        expected = function(lhs, rhs)
        compiled = torch.compile(
            function,
            backend="pypto",
            dynamic=False,
            fullgraph=True,
        )
        output = compiled(lhs, rhs)
        torch.cuda.synchronize()

        artifacts = {
            name: {
                "entry": artifact.entry_name,
                "cubin_sha256": artifact.cubin_sha256,
                "cubin_bytes": artifact.cubin_bytes,
                "grid": list(artifact.grid),
                "argument_count": artifact.argument_count,
                "fallback_used": artifact.fallback_used,
            }
            for name, artifact in scheduling.REGISTRY.snapshot()
        }
        max_abs_diff = float(
            (output.float() - expected.float()).abs().max().item()
        )
        correct = bool(torch.equal(output, expected))
        all_native = bool(artifacts) and all(
            not artifact["fallback_used"] and artifact["cubin_bytes"] > 0
            for artifact in artifacts.values()
        )
        result = {
            "schema": 1,
            "backend": "pypto",
            "shape": list(lhs.shape),
            "dtype": "bfloat16",
            "kernel_count": len(artifacts),
            "launch_count": len(launches),
            "launches": launches,
            "artifacts": artifacts,
            "max_abs_diff": max_abs_diff,
            "all_correct": correct,
            "all_native": all_native,
        }
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(encoded, end="")
        if args.output is not None:
            args.output.write_text(encoded, encoding="utf-8")
        return 0 if correct and all_native and len(artifacts) == 1 and len(launches) == 1 else 1
    finally:
        runtime_bridge.pypto_launch = original_launch
        torch_inductor.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
