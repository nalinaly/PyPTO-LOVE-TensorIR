#!/usr/bin/env python3
"""Focused SM120 numerical gate for fused QK preparation."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pypto_kernels import qk_rmsnorm_rope
from pypto_kernels._boot import DSO_PATH, bootstrap


RTOL = 5.0e-2
ATOL = 5.0e-2


def main() -> int:
    torch.manual_seed(3)
    tokens, q_heads, kv_heads = 2, 8, 2
    head_dim, rotary_dim, max_positions = 256, 64, 262144
    stream = torch.cuda.Stream()

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
    q_weight = (
        torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    k_weight = (
        torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    angles = torch.randn(
        max_positions,
        rotary_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    cos_sin_cache = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).to(
        torch.bfloat16
    )
    positions = torch.tensor([0, 17], device="cuda", dtype=torch.int64)

    graph_key = qk_rmsnorm_rope.compile_for(
        tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions
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
        normalized = (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1.0e-6)
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
    q_max_abs = float(q_abs.max())
    k_max_abs = float(k_abs.max())
    q_max_relative = float(
        (q_abs / q_reference.float().abs().clamp_min(1.0e-6)).max()
    )
    k_max_relative = float(
        (k_abs / k_reference.float().abs().clamp_min(1.0e-6)).max()
    )
    gate_exact = bool(torch.equal(gate_out, gate_reference))
    q_close = bool(
        torch.allclose(q_actual.float(), q_reference.float(), rtol=RTOL, atol=ATOL)
    )
    k_close = bool(
        torch.allclose(k_actual.float(), k_reference.float(), rtol=RTOL, atol=ATOL)
    )
    correct = gate_exact and q_close and k_close

    dso = pathlib.Path(DSO_PATH)
    result = {
        "schema_version": 1,
        "kind": "pypto-qk-rmsnorm-partial-rope-gate-sm120",
        "run_id": os.environ["PYPTO_RUN_ID"],
        "graph_key": graph_key,
        "launches": 1,
        "shape": {
            "tokens": tokens,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "max_positions": max_positions,
        },
        "thresholds": {"rtol": RTOL, "atol": ATOL},
        "q_max_abs_diff": q_max_abs,
        "q_max_relative_diff": q_max_relative,
        "k_max_abs_diff": k_max_abs,
        "k_max_relative_diff": k_max_relative,
        "gate_exact": gate_exact,
        "q_close": q_close,
        "k_close": k_close,
        "correct": correct,
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": (
            bootstrap()["compiler"].get_nvidia_backend_build_info().pypto_revision
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    run_dir = pathlib.Path(__file__).parents[3] / "runs" / result["run_id"]
    (run_dir / "qk-exec-result.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if correct else 75


if __name__ == "__main__":
    raise SystemExit(main())
