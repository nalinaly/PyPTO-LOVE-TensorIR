#!/usr/bin/env python3
"""Numerical acceptance for the pypto-kernels RMSNorm decomposition (SM120).

Compares the five-kernel PyPTO decomposition against eager PyTorch RMSNorm
over the shapes a Qwen-class model uses, in BF16 like the model itself.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src",
)

import torch  # noqa: E402

from pypto_kernels import rmsnorm  # noqa: E402


def eager_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    magnitude = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(magnitude + eps)).to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(5)
    cases = []
    failures = []
    for rows, columns in ((256, 1024), (4096, 1024), (512, 2048), (1, 1024)):
        x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            pass
        expected = eager_rmsnorm(x, 1e-6)
        try:
            output = rmsnorm.pypto_rmsnorm(x, eps=1e-6, stream=stream)
            stream.synchronize()
            difference = (output.float() - expected.float()).abs()
            relative = (difference / expected.float().abs().clamp_min(1e-6))
            correct = bool(
                torch.allclose(
                    output.float(), expected.float(), rtol=2e-2, atol=2e-3
                )
            )
            case = {
                "shape": [rows, columns],
                "output_correct": correct,
                "max_abs_diff": float(difference.max()),
                "max_relative": float(relative.max()),
                "mismatch_ratio": float(
                    (difference > 2e-2 * expected.float().abs().clamp_min(1e-6))
                    .float().mean()
                ),
            }
        except Exception as error:  # noqa: BLE001 - recorded as marker
            case = {
                "shape": [rows, columns],
                "output_correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
            correct = False
        cases.append(case)
        if not correct:
            failures.append([rows, columns])
    evidence = {
        "schema": 1,
        "kind": "pypto-kernels-rmsnorm-decomposed-sm120",
        "dso": rmsnorm._DSO_PATH,
        "case_count": len(cases),
        "all_correct": not failures,
        "failures": failures,
        "cases": cases,
    }
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0 if not failures else 75


if __name__ == "__main__":
    raise SystemExit(main())
