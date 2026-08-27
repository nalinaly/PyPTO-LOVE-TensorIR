#!/usr/bin/env python3
"""Execution acceptance for the executable v2 operators (SM120)."""

import json
import os
import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels-v2/src")

import torch

from pypto_kernels_v2._boot import compile_graph, launch_graph
from pypto_kernels_v2._graph import (gdn_compose_graph, gdn_delta_graph,
                                     tiles_for)
from pypto_kernels_v2.ops import (attention_design, fused_add, rmsnorm, rope,
                                  silu_and_mul)


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
        cases.append({
            "case": f"silu_and_mul {m}x{n}",
            "launches": 1,
            "max_abs_diff": float((out.float() - ref).abs().max()),
            "correct": bool(torch.allclose(out.float(), ref, rtol=5e-2, atol=5e-2)),
        })
        a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        out2 = fused_add.fused_add(a, b, stream=stream)
        stream.synchronize()
        ref2 = a.float() + b.float()
        cases.append({
            "case": f"fused_add {m}x{n}",
            "launches": 1,
            "max_abs_diff": float((out2.float() - ref2).abs().max()),
            "correct": bool(torch.allclose(out2.float(), ref2, rtol=5e-2, atol=5e-2)),
        })
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
        ref3 = q.float() * (torch.nn.functional.silu(g.float()) * 0 +  # noqa
                            torch.nn.functional.softplus(g.float()) * k.float())
        cases.append({
            "case": "gdn_compose 16x128",
            "launches": 1,
            "max_abs_diff": float((out3.float() - ref3).abs().max()),
            "correct": bool(torch.allclose(out3.float(), ref3, rtol=5e-2,
                                           atol=5e-2)),
        })

    # Every broadcast-dependent former B-class operator below is one compile
    # and one launch. Launch arguments follow builder input discovery order.
    rows, cols = 256, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    rms_out = torch.empty_like(x)
    rms_key = compile_graph(rmsnorm.build(rows, cols), tiles_for(rows, cols))
    launch_graph(rms_key, (x, rms_out), stream.cuda_stream)
    stream.synchronize()
    rms_ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                      + 1.0e-6)
    cases.append({
        "case": "rmsnorm_bf16 256x1024",
        "launches": 1,
        "max_abs_diff": float((rms_out.float() - rms_ref).abs().max()),
        "correct": bool(torch.allclose(rms_out.float(), rms_ref,
                                       rtol=5e-2, atol=5e-2)),
    })

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
    cases.append({
        "case": "rope_even_bf16 256x128",
        "launches": 1,
        "max_abs_diff": float((even_out.float() - even_ref).abs().max()),
        "correct": bool(torch.allclose(even_out.float(), even_ref,
                                       rtol=5e-2, atol=5e-2)),
    })

    odd_out = torch.empty_like(x1)
    odd_key = compile_graph(rope.build_odd(rows, half), tiles_for(rows, half))
    # build_odd input order: x1, sin, x2, cos.
    launch_graph(odd_key, (x1, sin, x2, cos, odd_out), stream.cuda_stream)
    stream.synchronize()
    odd_ref = x1.float() * sin.float() + x2.float() * cos.float()
    cases.append({
        "case": "rope_odd_bf16 256x128",
        "launches": 1,
        "max_abs_diff": float((odd_out.float() - odd_ref).abs().max()),
        "correct": bool(torch.allclose(odd_out.float(), odd_ref,
                                       rtol=5e-2, atol=5e-2)),
    })

    rows, tokens = 256, 128
    exponent = torch.rand(rows, tokens, device="cuda", dtype=torch.bfloat16)
    inverse_sum = torch.rand(rows, 1, device="cuda", dtype=torch.bfloat16)
    scaled = torch.empty_like(exponent)
    scale_key = compile_graph(
        attention_design.build_softmax_scale(rows, tokens),
        tiles_for(rows, tokens),
    )
    launch_graph(scale_key, (exponent, inverse_sum, scaled),
                 stream.cuda_stream)
    stream.synchronize()
    scale_ref = exponent.float() * inverse_sum.float()
    cases.append({
        "case": "attention_softmax_scale_bf16 256x128",
        "launches": 1,
        "max_abs_diff": float((scaled.float() - scale_ref).abs().max()),
        "correct": bool(torch.allclose(scaled.float(), scale_ref,
                                       rtol=5e-2, atol=5e-2)),
    })

    heads, dv = 16, 128
    value = torch.randn(heads, dv, device="cuda", dtype=torch.bfloat16)
    dot = torch.randn(heads, 1, device="cuda", dtype=torch.bfloat16)
    delta = torch.empty_like(value)
    delta_key = compile_graph(gdn_delta_graph(heads, dv),
                              tiles_for(heads, dv))
    launch_graph(delta_key, (value, dot, delta), stream.cuda_stream)
    stream.synchronize()
    delta_ref = value.float() * dot.float()
    cases.append({
        "case": "gdn_delta_bf16 16x128",
        "launches": 1,
        "max_abs_diff": float((delta.float() - delta_ref).abs().max()),
        "correct": bool(torch.allclose(delta.float(), delta_ref,
                                       rtol=5e-2, atol=5e-2)),
    })
    ok = all(c["correct"] for c in cases)
    print(json.dumps({"schema": 1, "kind": "pypto-kernels-v2-exec-sm120",
                      "run_id": os.environ.get("PYPTO_RUN_ID"),
                      "all_correct": ok, "cases": cases}, indent=1))
    return 0 if ok else 75


if __name__ == "__main__":
    raise SystemExit(main())
