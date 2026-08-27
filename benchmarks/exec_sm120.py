#!/usr/bin/env python3
"""Execution acceptance for the Qwen3.5 PyPTO operators on SM120."""

import json
import hashlib
import os
import pathlib
import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import torch

from pypto_kernels._boot import DSO_PATH, bootstrap, compile_graph, launch_graph
from pypto_kernels._graph import (
    gdn_compose_graph,
    gdn_delta_combine_graph,
    gdn_delta_graph,
    gdn_q_decay_graph,
    gdn_state_read_graph,
    gdn_state_update_graph,
    row_sum_graph,
    tiles_for,
)
from pypto_kernels import attention, fused_add, rmsnorm, rope, silu_and_mul


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
        # gdn compose: the executable half of the GDN read path.
        # NOTE operand order = builder input order (g, k, q), not call order.
        hh, dk = 16, 128
        q = torch.randn(hh, dk, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(hh, dk, device="cuda", dtype=torch.bfloat16) * 0.5
        k = torch.randn(hh, dk, device="cuda", dtype=torch.bfloat16)
        out3 = torch.empty_like(q)
        key = compile_graph(gdn_compose_graph(hh, dk), [128])
        launch_graph(key, (g, k, q, out3), stream.cuda_stream)
        stream.synchronize()
        ref3 = q.float() * (
            torch.nn.functional.silu(g.float()) * 0  # noqa
            + torch.nn.functional.softplus(g.float()) * k.float()
        )
        cases.append(
            {
                "case": "gdn_compose 16x128",
                "launches": 1,
                "max_abs_diff": float((out3.float() - ref3).abs().max()),
                "correct": bool(
                    torch.allclose(out3.float(), ref3, rtol=5e-2, atol=5e-2)
                ),
            }
        )

    # Every broadcast-dependent former B-class operator below is one compile
    # and one launch. Launch arguments follow builder input discovery order.
    rows, cols = 256, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    rms_out = rmsnorm.rmsnorm(x, stream=stream)
    stream.synchronize()
    rms_ref = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1.0e-6
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

    rows, half = 256, 128
    x1 = torch.randn(rows, half, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn(rows, half, device="cuda", dtype=torch.bfloat16)
    cos = torch.rand(rows, 1, device="cuda", dtype=torch.bfloat16)
    sin = torch.rand(rows, 1, device="cuda", dtype=torch.bfloat16)
    even_out = torch.empty_like(x1)
    even_key = compile_graph(rope.build_even(rows, half), tiles_for(rows, half))
    # build_even input order: x1, cos, x2, sin.
    launch_graph(even_key, (x1, cos, x2, sin, even_out), stream.cuda_stream)
    stream.synchronize()
    even_ref = x1.float() * cos.float() - x2.float() * sin.float()
    cases.append(
        {
            "case": "rope_even_bf16 256x128",
            "launches": 1,
            "max_abs_diff": float((even_out.float() - even_ref).abs().max()),
            "correct": bool(
                torch.allclose(even_out.float(), even_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    odd_out = torch.empty_like(x1)
    odd_key = compile_graph(rope.build_odd(rows, half), tiles_for(rows, half))
    # build_odd input order: x1, sin, x2, cos.
    launch_graph(odd_key, (x1, sin, x2, cos, odd_out), stream.cuda_stream)
    stream.synchronize()
    odd_ref = x1.float() * sin.float() + x2.float() * cos.float()
    cases.append(
        {
            "case": "rope_odd_bf16 256x128",
            "launches": 1,
            "max_abs_diff": float((odd_out.float() - odd_ref).abs().max()),
            "correct": bool(
                torch.allclose(odd_out.float(), odd_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rows, tokens = 256, 128
    exponent = torch.rand(rows, tokens, device="cuda", dtype=torch.bfloat16)
    inverse_sum = torch.rand(rows, 1, device="cuda", dtype=torch.bfloat16)
    scaled = torch.empty_like(exponent)
    scale_key = compile_graph(
        attention.build_softmax_scale(rows, tokens),
        tiles_for(rows, tokens),
    )
    launch_graph(scale_key, (exponent, inverse_sum, scaled), stream.cuda_stream)
    stream.synchronize()
    scale_ref = exponent.float() * inverse_sum.float()
    cases.append(
        {
            "case": "attention_softmax_scale_bf16 256x128",
            "launches": 1,
            "max_abs_diff": float((scaled.float() - scale_ref).abs().max()),
            "correct": bool(
                torch.allclose(scaled.float(), scale_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    heads, dv = 16, 128
    value = torch.randn(heads, dv, device="cuda", dtype=torch.bfloat16)
    dot = torch.randn(heads, 1, device="cuda", dtype=torch.bfloat16)
    delta = torch.empty_like(value)
    delta_key = compile_graph(gdn_delta_graph(heads, dv), tiles_for(heads, dv))
    launch_graph(delta_key, (value, dot, delta), stream.cuda_stream)
    stream.synchronize()
    delta_ref = value.float() * dot.float()
    cases.append(
        {
            "case": "gdn_delta_bf16 16x128",
            "launches": 1,
            "max_abs_diff": float((delta.float() - delta_ref).abs().max()),
            "correct": bool(
                torch.allclose(delta.float(), delta_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    # Complete attention floor: one normalize graph plus one value-mix matmul.
    rows, tokens, value_dim = 256, 128, 128
    exponent = torch.rand(rows, tokens, device="cuda", dtype=torch.bfloat16) + 0.125
    probabilities = torch.empty_like(exponent)
    normalize_key = compile_graph(
        attention.build_softmax_normalize(rows, tokens),
        tiles_for(rows, tokens),
    )
    launch_graph(normalize_key, (exponent, probabilities), stream.cuda_stream)
    stream.synchronize()
    probability_ref = exponent.float() / exponent.float().sum(-1, keepdim=True)
    probability_ok = torch.allclose(
        probabilities.float(), probability_ref, rtol=5e-2, atol=5e-3
    )
    cases.append(
        {
            "case": "attention_softmax_normalize_bf16 256x128",
            "launches": 1,
            "max_abs_diff": float(
                (probabilities.float() - probability_ref).abs().max()
            ),
            "correct": bool(probability_ok),
        }
    )

    value_matrix = (
        torch.randn(tokens, value_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    )
    mixed = torch.empty(rows, value_dim, device="cuda", dtype=torch.bfloat16)
    mix_key = compile_graph(
        attention.build_value_mix(rows, tokens, value_dim),
        tiles_for(rows, value_dim),
    )
    launch_graph(mix_key, (probabilities, value_matrix, mixed), stream.cuda_stream)
    stream.synchronize()
    mixed_ref = probability_ref @ value_matrix.float()
    cases.append(
        {
            "case": "attention_value_mix_bf16 256x128x128",
            "launches": 1,
            "path_launches": 2,
            "max_abs_diff": float((mixed.float() - mixed_ref).abs().max()),
            "correct": bool(
                probability_ok
                and torch.allclose(mixed.float(), mixed_ref, rtol=8e-2, atol=5e-2)
            ),
        }
    )

    # Complete five-graph GDN read path. Each component is independently one
    # graph/one launch; no wrapper presents the five launches as one operator.
    heads, dk, dv = 16, 128, 128
    query = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    decay = torch.rand(heads, dk, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    key = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    value = torch.randn(heads, dv, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05

    query_decay = torch.empty_like(query)
    qd_key = compile_graph(gdn_q_decay_graph(heads, dk), [128])
    launch_graph(qd_key, (query, decay, query_decay), stream.cuda_stream)

    state_read = torch.empty(heads, 1, dv, device="cuda", dtype=torch.bfloat16)
    read_key = compile_graph(gdn_state_read_graph(heads, dk, dv), tiles_for(heads, dv))
    launch_graph(
        read_key,
        (query_decay.view(heads, 1, dk), state, state_read),
        stream.cuda_stream,
    )

    composed = torch.empty_like(query)
    compose_key = compile_graph(gdn_compose_graph(heads, dk), [128])
    # gdn_compose_graph input order: gate, key, query.
    launch_graph(compose_key, (gate, key, query, composed), stream.cuda_stream)

    dot = torch.empty(heads, 1, device="cuda", dtype=torch.bfloat16)
    dot_key = compile_graph(row_sum_graph(heads, dk), tiles_for(heads))
    launch_graph(dot_key, (composed, dot), stream.cuda_stream)

    gdn_out = torch.empty(heads, dv, device="cuda", dtype=torch.bfloat16)
    combine_key = compile_graph(
        gdn_delta_combine_graph(heads, dv), tiles_for(heads, dv)
    )
    # combine input order: value, dot, read.
    launch_graph(
        combine_key,
        (value, dot, state_read.view(heads, dv), gdn_out),
        stream.cuda_stream,
    )
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
            "launches": 5,
            "component_launches": {
                "q_decay": 1,
                "state_read": 1,
                "compose": 1,
                "dot": 1,
                "delta_combine": 1,
            },
            "max_abs_diff": float((gdn_out.float() - gdn_ref).abs().max()),
            "correct": bool(
                torch.allclose(gdn_out.float(), gdn_ref, rtol=8e-2, atol=8e-2)
            ),
        }
    )

    # GDN state update is one rank-3 pointwise graph and one launch.
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05
    state_decay = torch.rand(heads, dk, 1, device="cuda", dtype=torch.bfloat16)
    beta_key = torch.randn(heads, dk, 1, device="cuda", dtype=torch.bfloat16) * 0.05
    update_value = torch.randn(heads, 1, dv, device="cuda", dtype=torch.bfloat16) * 0.1
    updated = torch.empty_like(state)
    update_key = compile_graph(
        gdn_state_update_graph(heads, dk, dv), tiles_for(heads, dk, dv)
    )
    # state-update input order: state, decay, beta_key, value.
    launch_graph(
        update_key,
        (state, state_decay, beta_key, update_value, updated),
        stream.cuda_stream,
    )
    stream.synchronize()
    update_ref = (
        state.float() * state_decay.float() + beta_key.float() * update_value.float()
    )
    cases.append(
        {
            "case": "gdn_state_update_bf16 16x128x128",
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
        "native_tile_ops": ["silu_and_mul", "fused_add", "rmsnorm"],
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
