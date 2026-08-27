#!/usr/bin/env python3
"""Execution acceptance for the Qwen3.5 PyPTO operators on SM120."""

import json
import hashlib
import math
import os
import pathlib
import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import torch

from pypto_kernels._boot import DSO_PATH, bootstrap
from pypto_kernels import (
    attention,
    fused_add,
    fused_add_rmsnorm,
    gdn,
    gated_rmsnorm,
    linear,
    rmsnorm,
    rope,
    silu_and_mul,
)


def main() -> int:
    torch.manual_seed(3)
    stream = torch.cuda.Stream()
    cases = []
    for m, n in ((256, 1024), (4096, 1024), (1, 3584)):
        g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 2
        u = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 2
        out = silu_and_mul.silu_and_mul(g, u, stream=stream)
        stream.synchronize()
        ref = torch.nn.functional.silu(g.float()) * u.float()
        cases.append(
            {
                "case": f"silu_and_mul {m}x{n}",
                "implementation": "native-tile-dsl",
                "launches": 1,
                "max_abs_diff": float((out.float() - ref).abs().max()),
                "correct": bool(torch.allclose(out.float(), ref, rtol=5e-2, atol=5e-2)),
            }
        )
        a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        out2 = fused_add.fused_add(a, b, stream=stream)
        stream.synchronize()
        ref2 = a.float() + b.float()
        cases.append(
            {
                "case": f"fused_add {m}x{n}",
                "implementation": "native-tile-dsl",
                "launches": 1,
                "max_abs_diff": float((out2.float() - ref2).abs().max()),
                "correct": bool(
                    torch.allclose(out2.float(), ref2, rtol=5e-2, atol=5e-2)
                ),
            }
        )
    # Every broadcast-dependent former B-class operator below is one compile
    # and one launch. Launch arguments follow builder input discovery order.
    rows, cols = 256, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    rms_weight = torch.randn(cols, device="cuda", dtype=torch.bfloat16) * 0.1
    rms_out = rmsnorm.rmsnorm(x, rms_weight, stream=stream)
    stream.synchronize()
    rms_ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * (1.0 + rms_weight.float())
    )
    cases.append(
        {
            "case": "rmsnorm_bf16 256x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((rms_out.float() - rms_ref).abs().max()),
            "correct": bool(
                torch.allclose(rms_out.float(), rms_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rms_residual = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    fused_norm_out, residual_out = fused_add_rmsnorm.fused_add_rmsnorm(
        x, rms_residual, rms_weight, stream=stream
    )
    stream.synchronize()
    residual_ref = x + rms_residual
    fused_norm_ref = (
        residual_ref.float()
        * torch.rsqrt(residual_ref.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * (1.0 + rms_weight.float())
    )
    fused_norm_diff = float((fused_norm_out.float() - fused_norm_ref).abs().max())
    residual_diff = float((residual_out.float() - residual_ref.float()).abs().max())
    cases.append(
        {
            "case": "fused_add_rmsnorm_bf16 256x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": max(fused_norm_diff, residual_diff),
            "normalized_max_abs_diff": fused_norm_diff,
            "residual_max_abs_diff": residual_diff,
            "correct": bool(
                torch.equal(residual_out, residual_ref)
                and torch.allclose(
                    fused_norm_out.float(),
                    fused_norm_ref,
                    rtol=5e-2,
                    atol=5e-2,
                )
            ),
        }
    )

    gated_x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    gated_gate = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    gated_weight = 1.0 + torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
    gated_out = gated_rmsnorm.gated_rmsnorm(
        gated_x, gated_gate, gated_weight, stream=stream
    )
    stream.synchronize()
    gated_ref = (
        gated_x.float()
        * torch.rsqrt(gated_x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * gated_weight.float()
        * torch.nn.functional.silu(gated_gate.float())
    )
    cases.append(
        {
            "case": "gated_rmsnorm_bf16 256x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((gated_out.float() - gated_ref).abs().max()),
            "correct": bool(
                torch.allclose(gated_out.float(), gated_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rows, half = 256, 64
    x_rope = torch.randn(rows, 2 * half, device="cuda", dtype=torch.bfloat16)
    cos_half = torch.rand(rows, half, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.rand(rows, half, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=1).contiguous()
    sin = torch.cat((sin_half, sin_half), dim=1).contiguous()
    rope_out = rope.rope(x_rope, cos, sin, stream=stream)
    stream.synchronize()
    x_low, x_high = x_rope.float().chunk(2, dim=1)
    rope_ref = torch.cat(
        (
            x_low * cos_half.float() - x_high * sin_half.float(),
            x_high * cos_half.float() + x_low * sin_half.float(),
        ),
        dim=1,
    )
    cases.append(
        {
            "case": "rope_bf16 256x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((rope_out.float() - rope_ref).abs().max()),
            "correct": bool(
                torch.allclose(rope_out.float(), rope_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rows, tokens, head_dim, value_dim = 32, 128, 128, 128
    query_attn = torch.randn(rows, head_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    key_attn = torch.randn(tokens, head_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    value_attn = (
        torch.randn(tokens, value_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    )
    attention_out = attention.attention(query_attn, key_attn, value_attn, stream=stream)
    stream.synchronize()
    score_ref = query_attn.float() @ key_attn.float().T / math.sqrt(head_dim)
    attention_ref = torch.softmax(score_ref, dim=-1) @ value_attn.float()
    cases.append(
        {
            "case": "attention_bf16 32x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((attention_out.float() - attention_ref).abs().max()),
            "correct": bool(
                torch.allclose(
                    attention_out.float(), attention_ref, rtol=1e-1, atol=8e-2
                )
            ),
        }
    )

    linear_input = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    linear_weight = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    linear_out = linear.linear(linear_input, linear_weight, stream=stream)
    stream.synchronize()
    linear_ref = linear_input.float() @ linear_weight.float().T
    cases.append(
        {
            "case": "linear_bf16 32x1024x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((linear_out.float() - linear_ref).abs().max()),
            "correct": bool(
                torch.allclose(linear_out.float(), linear_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    heads, dk, dv = 16, 128, 128
    query = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    decay = torch.rand(heads, dk, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    key = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    value = torch.randn(heads, dv, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05

    gdn_out = gdn.gdn_read(query, decay, gate, key, value, state, stream=stream)
    stream.synchronize()

    qd_ref = query.float() * decay.float()
    read_ref = torch.einsum("hd,hdv->hv", qd_ref, state.float())
    compose_ref = (
        query.float() * torch.nn.functional.softplus(gate.float()) * key.float()
    )
    dot_ref = compose_ref.sum(-1, keepdim=True)
    gdn_ref = read_ref + dot_ref * value.float()
    cases.append(
        {
            "case": "gdn_complete_read_bf16 16x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((gdn_out.float() - gdn_ref).abs().max()),
            "correct": bool(
                torch.allclose(gdn_out.float(), gdn_ref, rtol=8e-2, atol=8e-2)
            ),
        }
    )

    # GDN state update is one native tile graph and one launch.
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05
    state_decay = torch.rand(heads, dk, 1, device="cuda", dtype=torch.bfloat16)
    beta_key = torch.randn(heads, dk, 1, device="cuda", dtype=torch.bfloat16) * 0.05
    update_value = torch.randn(heads, 1, dv, device="cuda", dtype=torch.bfloat16) * 0.1
    updated = gdn.gdn_state_update(
        state,
        state_decay,
        beta_key,
        update_value,
        stream=stream,
    )
    stream.synchronize()
    update_ref = (
        state.float() * state_decay.float() + beta_key.float() * update_value.float()
    )
    cases.append(
        {
            "case": "gdn_state_update_bf16 16x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((updated.float() - update_ref).abs().max()),
            "correct": bool(
                torch.allclose(updated.float(), update_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )
    ok = all(c["correct"] for c in cases)
    dso = pathlib.Path(DSO_PATH)
    result = {
        "schema": 2,
        "kind": "pypto-kernels-exec-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": bootstrap()["compiler"]
        .get_nvidia_backend_build_info()
        .pypto_revision,
        "native_tile_ops": [
            "silu_and_mul",
            "fused_add",
            "rmsnorm",
            "fused_add_rmsnorm",
            "gated_rmsnorm",
            "rope",
            "attention",
            "linear",
            "gdn_read",
            "gdn_state_update",
        ],
        "all_correct": ok,
        "cases": cases,
    }
    rendered = json.dumps(result, indent=1)
    pathlib.Path(__file__).with_name("exec_results.json").write_text(
        rendered + "\n", encoding="utf-8"
    )
    print(rendered)
    return 0 if ok else 75


if __name__ == "__main__":
    raise SystemExit(main())
