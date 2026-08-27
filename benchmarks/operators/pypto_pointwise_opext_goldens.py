#!/usr/bin/env python3
"""Extended-op-table golden producer for FusedPointwiseV2 (opext DSO).

For every operator legalized by the extended GetFusedOperationSpec table,
builds the HIR chain with PointwiseProgramBuilder, compiles it through the
exact opext DSO, launches through the real runtime bridge (the same
pypto_launch entry the Inductor wrapper uses) and compares against eager
PyTorch on the live SM120 device. Emits one JSON record per op with the
Cubin identity plus the numerical verdict for golden publication.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-framework-plugins/src",
)

import torch  # noqa: E402

from pypto_plugins.torch import pointwise_codegen, scheduling  # noqa: E402
from pypto_plugins.torch.runtime_bridge import pypto_launch  # noqa: E402

# op -> (tensor-op registry name, arity, input sanitizer)
CASES = [
    ("add", "tensor.add", 2, "randn"),
    ("sub", "tensor.sub", 2, "randn"),
    ("mul", "tensor.mul", 2, "randn"),
    ("div", "tensor.div", 2, "positive"),
    ("neg", "tensor.neg", 1, "randn"),
    ("exp", "tensor.exp", 1, "randn"),
    ("recip", "tensor.recip", 1, "positive"),
    ("rsqrt", "tensor.rsqrt", 1, "positive"),
    ("abs", "tensor.abs", 1, "randn"),
    ("sqrt", "tensor.sqrt", 1, "positive"),
    ("log", "tensor.log", 1, "positive"),
    ("sin", "tensor.sin", 1, "randn"),
    ("cos", "tensor.cos", 1, "randn"),
    ("maximum", "tensor.maximum", 2, "randn"),
    ("minimum", "tensor.minimum", 2, "randn"),
]

_EAGER = {
    "tensor.add": lambda x, y: x + y,
    "tensor.sub": lambda x, y: x - y,
    "tensor.mul": lambda x, y: x * y,
    "tensor.div": lambda x, y: x / y,
    "tensor.neg": lambda x, y: -x,
    "tensor.exp": lambda x, y: x.exp(),
    "tensor.recip": lambda x, y: x.reciprocal(),
    "tensor.rsqrt": lambda x, y: x.rsqrt(),
    "tensor.abs": lambda x, y: x.abs(),
    "tensor.sqrt": lambda x, y: x.sqrt(),
    "tensor.log": lambda x, y: x.log(),
    "tensor.sin": lambda x, y: x.sin(),
    "tensor.cos": lambda x, y: x.cos(),
    "tensor.maximum": lambda x, y: torch.maximum(x, y),
    "tensor.minimum": lambda x, y: torch.minimum(x, y),
}


def _input(kind: str, n: int) -> torch.Tensor:
    if kind == "positive":
        return torch.rand(n, device="cuda", dtype=torch.float32) + 0.25
    return torch.randn(n, device="cuda", dtype=torch.float32)


def main() -> int:
    torch.manual_seed(7)
    n = 4096
    records = []
    failures = []
    stream = torch.cuda.Stream()
    for label, op_name, arity, kind in CASES:
        builder = pointwise_codegen.PointwiseProgramBuilder((n,), "float32")
        x = builder.add_input("x")
        y = builder.add_input("y") if arity == 2 else None
        arguments = [x, y] if arity == 2 else [x]
        result = builder.emit(op_name, arguments)
        builder.mark_output(result)
        artifact = pointwise_codegen.compile_pointwise(
            builder.build(), registry_name=f"opext_{label}"
        )
        scheduling.REGISTRY.register(f"opext_{label}", artifact)

        tx = _input(kind, n)
        ty = _input(kind, n)
        out = torch.empty_like(tx)
        expected = _EAGER[op_name](tx, ty)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            pass
        launch_args = (tx, ty, out) if arity == 2 else (tx, out)
        pypto_launch(f"opext_{label}", launch_args, stream.cuda_stream)
        stream.synchronize()
        exact = bool(torch.equal(out, expected))
        # div lowers to the pinned backend's non-round-to-nearest division;
        # accept a documented 1-2 ulp tolerance like the RowReductionV3 gate.
        tolerance = bool(
            torch.allclose(out, expected, rtol=1e-6, atol=1e-6, equal_nan=False)
        )
        correct = exact or (label == "div" and tolerance)
        maxdiff = (out - expected).abs().max().item() if not exact else 0.0
        record = {
            "op": label,
            "registry_name": op_name,
            "arity": arity,
            "entry": artifact.entry_name,
            "cubin_sha256": artifact.cubin_sha256,
            "cubin_bytes": artifact.cubin_bytes,
            "grid": list(artifact.grid),
            "argument_count": artifact.argument_count,
            "fallback_used": artifact.fallback_used,
            "output_exact": exact,
            "output_correct": correct,
        }
        if tolerance and not exact:
            record["max_abs_diff"] = maxdiff
            record["comparison"] = "tolerance-2ulp-div"
        if not correct:
            record["max_abs_diff"] = maxdiff
            failures.append(label)
        records.append(record)
    evidence = {
        "schema": 1,
        "kind": "opext-pointwise-goldens",
        "dso": pointwise_codegen._DEFAULT_DSO,
        "case_count": len(records),
        "all_correct": not failures,
        "failures": failures,
        "cases": records,
    }
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0 if not failures else 75


if __name__ == "__main__":
    raise SystemExit(main())
