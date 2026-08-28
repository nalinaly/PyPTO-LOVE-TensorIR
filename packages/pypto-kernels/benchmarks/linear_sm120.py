#!/usr/bin/env python3
"""Numerical gate for real Qwen3.5 dense-linear and LM-head shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import torch

KERNEL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT / "src"))

from _qwen35_models import (  # noqa: E402
    Qwen35Shape,
    load_release_shapes,
    parse_release_rows,
)
from pypto_kernels import linear  # noqa: E402
from pypto_kernels._boot import bootstrap, loaded_dso_path  # noqa: E402


RTOL = 5.0e-2
ATOL = 5.0e-1
SEED = 41


def _reference(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        return torch.nn.functional.linear(input_, weight)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def _run_projection(
    shape: Qwen35Shape,
    rows: int,
    stream: torch.cuda.Stream,
) -> dict[str, object]:
    # Qwen's packed MLP gate/up projection is the widest ordinary dense linear.
    out_features = 2 * shape.intermediate_size
    torch.manual_seed(SEED + rows * 1009 + shape.hidden_size)
    input_ = (
        torch.randn(rows, shape.hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    weight = (
        torch.randn(
            out_features,
            shape.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    reference = _reference(input_, weight)
    actual = linear.linear(input_, weight, stream=stream)
    stream.synchronize()
    difference = (actual.float() - reference.float()).abs()
    correct = bool(
        torch.allclose(actual.float(), reference.float(), rtol=RTOL, atol=ATOL)
    )
    return {
        "case": f"linear_gate_up_{shape.model}_rows{rows}",
        "operator": "linear",
        "model": shape.model,
        "rows": rows,
        "input_shape": list(input_.shape),
        "input_stride": list(input_.stride()),
        "weight_shape": list(weight.shape),
        "weight_stride": list(weight.stride()),
        "output_shape": list(actual.shape),
        "output_stride": list(actual.stride()),
        "input_dtype": str(input_.dtype),
        "output_dtype": str(actual.dtype),
        "launches": 1,
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "correct": correct,
    }


def _run_lm_head(
    shape: Qwen35Shape,
    rows: int,
    stream: torch.cuda.Stream,
) -> dict[str, object]:
    torch.manual_seed(SEED + rows * 1013 + shape.hidden_size)
    input_ = (
        torch.randn(rows, shape.hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    weight = (
        torch.randn(
            shape.vocab_size,
            shape.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    # SGLang's plain BF16 LM head returns BF16-rounded logits; the PyPTO hook
    # preserves those values while widening storage to FP32.
    reference = _reference(input_, weight).float()
    actual = linear.linear_to_float(input_, weight, stream=stream)
    stream.synchronize()
    difference = (actual - reference).abs()
    correct = bool(torch.allclose(actual, reference, rtol=RTOL, atol=ATOL))
    return {
        "case": f"linear_to_float_lm_head_{shape.model}_rows{rows}",
        "operator": "linear_to_float",
        "model": shape.model,
        "rows": rows,
        "input_shape": list(input_.shape),
        "input_stride": list(input_.stride()),
        "weight_shape": list(weight.shape),
        "weight_stride": list(weight.stride()),
        "output_shape": list(actual.shape),
        "output_stride": list(actual.stride()),
        "input_dtype": str(input_.dtype),
        "output_dtype": str(actual.dtype),
        "launches": 1,
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "correct": correct,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--rows", default="1,19")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == KERNEL_ROOT or KERNEL_ROOT in output.parents:
        raise ValueError("linear output must be outside the source package")
    rows = parse_release_rows(args.rows)
    shapes = load_release_shapes(args.model_root)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("this regression requires one visible SM120 CUDA device")
    stream = torch.cuda.Stream()
    cases = []
    for shape in shapes:
        for row_count in rows:
            cases.append(_run_projection(shape, row_count, stream))
            cases.append(_run_lm_head(shape, row_count, stream))
    dso = loaded_dso_path()
    all_correct = all(bool(case["correct"]) for case in cases)
    result = {
        "schema": 1,
        "kind": "pypto-qwen35-linear-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "seed": SEED,
        "thresholds": {"rtol": RTOL, "atol": ATOL},
        "models": [shape.record() for shape in shapes],
        "rows": list(rows),
        "cases": cases,
        "all_correct": all_correct,
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": (
            bootstrap()["compiler"].get_nvidia_backend_build_info().pypto_revision
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if all_correct else 75


if __name__ == "__main__":
    raise SystemExit(main())
