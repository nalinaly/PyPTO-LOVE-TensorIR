#!/usr/bin/env python3
"""Execution acceptance for the executable v2 operators (SM120)."""

import json
import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels-v2/src")

import torch

from pypto_kernels_v2._boot import compile_graph, launch_graph
from pypto_kernels_v2._graph import gdn_compose_graph
from pypto_kernels_v2.ops import fused_add, silu_and_mul


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
    ok = all(c["correct"] for c in cases)
    print(json.dumps({"schema": 1, "kind": "pypto-kernels-v2-exec-sm120",
                      "all_correct": ok, "cases": cases}, indent=1))
    return 0 if ok else 75


if __name__ == "__main__":
    raise SystemExit(main())
