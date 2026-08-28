#!/usr/bin/env python3
"""Focused SM120 numerical gate for real Qwen3.5 Q/K preparation shapes."""

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
from pypto_kernels import qk_rmsnorm_rope  # noqa: E402
from pypto_kernels._boot import bootstrap, loaded_dso_path  # noqa: E402


RTOL = 5.0e-2
ATOL = 5.0e-2
SEED = 3


def _run_case(
    shape: Qwen35Shape,
    tokens: int,
    stream: torch.cuda.Stream,
) -> dict[str, object]:
    q_heads = shape.attention_heads
    kv_heads = shape.kv_heads
    head_dim = shape.attention_head_dim
    rotary_dim = shape.rotary_dim
    max_positions = 262144
    torch.manual_seed(SEED + tokens * 1009 + q_heads * 9176)
    q_gate = torch.randn(
        tokens,
        2 * q_heads * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        tokens,
        kv_heads * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    angles = torch.randn(
        max_positions,
        rotary_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    cos_sin_cache = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).to(
        torch.bfloat16
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 17

    graph_key = qk_rmsnorm_rope.compile_for(
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        int(q_gate.stride(0)),
        int(key.stride(0)),
        False,
        False,
    )
    q_out, k_out, gate_out = qk_rmsnorm_rope.qk_rmsnorm_rope_gate(
        q_gate,
        key,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_heads=q_heads,
        kv_heads=kv_heads,
        stream=stream,
    )
    stream.synchronize()

    q_gate_heads = q_gate.view(tokens, q_heads, 2 * head_dim)
    q_input = q_gate_heads[..., :head_dim]
    gate_reference = q_gate_heads[..., head_dim:]
    k_input = key.view(tokens, kv_heads, head_dim)

    def norm_reference(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        wide = value.float()
        normalized = (
            wide
            * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1.0e-6)
            * (1.0 + weight.float())
        )
        return normalized.to(torch.bfloat16).float()

    def partial_neox_reference(value: torch.Tensor) -> torch.Tensor:
        half = rotary_dim // 2
        low = value[..., :half]
        high = value[..., half:rotary_dim]
        tail = value[..., rotary_dim:]
        selected = cos_sin_cache[positions].float()
        cos = selected[:, :half].unsqueeze(1)
        sin = selected[:, half:].unsqueeze(1)
        return torch.cat(
            (low * cos - high * sin, high * cos + low * sin, tail), dim=-1
        ).to(torch.bfloat16)

    q_reference = partial_neox_reference(norm_reference(q_input, q_weight))
    k_reference = partial_neox_reference(norm_reference(k_input, k_weight))
    q_actual = q_out.view_as(q_reference)
    k_actual = k_out.view_as(k_reference)
    q_abs = (q_actual.float() - q_reference.float()).abs()
    k_abs = (k_actual.float() - k_reference.float()).abs()
    q_close = bool(
        torch.allclose(q_actual.float(), q_reference.float(), rtol=RTOL, atol=ATOL)
    )
    k_close = bool(
        torch.allclose(k_actual.float(), k_reference.float(), rtol=RTOL, atol=ATOL)
    )
    gate_exact = bool(torch.equal(gate_out, gate_reference))
    return {
        "case": f"qk_{shape.model}_rows{tokens}",
        "model": shape.model,
        "rows": tokens,
        "graph_key": graph_key,
        "launches": 1,
        "dtype": "bfloat16",
        "input_shapes": {
            "q_gate": list(q_gate.shape),
            "key": list(key.shape),
            "cos_sin_cache": list(cos_sin_cache.shape),
            "positions": list(positions.shape),
        },
        "input_strides": {
            "q_gate": list(q_gate.stride()),
            "key": list(key.stride()),
            "cos_sin_cache": list(cos_sin_cache.stride()),
            "positions": list(positions.stride()),
        },
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "q_max_abs_diff": float(q_abs.max()),
        "q_max_relative_diff": float(
            (q_abs / q_reference.float().abs().clamp_min(1.0e-6)).max()
        ),
        "k_max_abs_diff": float(k_abs.max()),
        "k_max_relative_diff": float(
            (k_abs / k_reference.float().abs().clamp_min(1.0e-6)).max()
        ),
        "gate_exact": gate_exact,
        "q_close": q_close,
        "k_close": k_close,
        "correct": gate_exact and q_close and k_close,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--rows", default="1,19")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == KERNEL_ROOT or KERNEL_ROOT in output.parents:
        raise ValueError("QK output must be outside the source package")
    rows = parse_release_rows(args.rows)
    shapes = load_release_shapes(args.model_root)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("this regression requires one visible SM120 CUDA device")
    stream = torch.cuda.Stream()
    cases = [
        _run_case(shape, row_count, stream)
        for shape in shapes
        for row_count in rows
    ]
    dso = loaded_dso_path()
    all_correct = all(bool(case["correct"]) for case in cases)
    result = {
        "schema_version": 2,
        "kind": "pypto-qk-rmsnorm-partial-rope-gate-sm120",
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
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all_correct else 75


if __name__ == "__main__":
    raise SystemExit(main())
