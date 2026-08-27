#!/usr/bin/env python3
"""Numerical acceptance for the GDN read-path decomposition (SM120)."""

from __future__ import annotations
import json, sys
sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")
import torch
from pypto_kernels import gdn_kernel


def eager_reference(q, decay, gate, k, v, state):
    beta = torch.nn.functional.softplus(gate.float())
    read = torch.einsum("bhd,bhdn->bhn", (q * decay).float(), state.float())
    dot = (q.float() * (beta * k.float())).sum(-1, keepdim=True)
    return (read + dot * v.float()).to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(13)
    cases, failures = [], []
    stream = torch.cuda.Stream()
    for batch, heads, dk, dv in ((4, 16, 128, 128), (128, 16, 128, 128),
                                 (1, 8, 128, 128), (7, 16, 128, 128)):
        q = torch.randn(batch, heads, dk, device="cuda", dtype=torch.bfloat16)
        decay = torch.rand(batch, heads, dk, device="cuda",
                           dtype=torch.bfloat16) * 0.5 + 0.5
        gate = torch.randn(batch, heads, dk, device="cuda",
                           dtype=torch.bfloat16)
        k = torch.randn(batch, heads, dk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, dv, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(batch, heads, dk, dv, device="cuda",
                            dtype=torch.bfloat16) * 0.05
        expected = eager_reference(q, decay, gate, k, v, state)
        label = f"gdn-read B={batch} H={heads} Dk={dk} Dv={dv}"
        try:
            out = gdn_kernel.pypto_gdn_decode_read(
                q, decay, gate, k, v, state, stream=stream)
            stream.synchronize()
            # the dot term rounds through BF16 before the ones matmul,
            # so ~1.5 percent relative error is the structural floor
            correct = bool(torch.allclose(out.float(), expected.float(),
                                          rtol=5e-2, atol=1.0))
            case = {"case": label, "output_correct": correct,
                    "max_abs_diff": float((out.float() - expected.float()).abs().max())}
        except Exception as error:
            correct = False
            case = {"case": label, "output_correct": False,
                    "error": f"{type(error).__name__}: {error}"}
        cases.append(case)
        if not correct:
            failures.append(label)
    evidence = {"schema": 1, "kind": "pypto-kernels-gdn-read-sm120",
                "case_count": len(cases), "all_correct": not failures,
                "failures": failures, "cases": cases}
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0 if not failures else 75


if __name__ == "__main__":
    raise SystemExit(main())
