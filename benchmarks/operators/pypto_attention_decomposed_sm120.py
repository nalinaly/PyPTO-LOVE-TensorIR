#!/usr/bin/env python3
"""Numerical acceptance for the decomposed attention kernels (SM120).

Compares the PyPTO decode and prefill decompositions against eager PyTorch
attention (FP32 softmax over BF16 operands, the SDPA math reference).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src",
)

import torch  # noqa: E402

from pypto_kernels import attention  # noqa: E402


def eager_attention(q, k, v, scale, mask=None):
    """q [B,H,M,D], k [B,H,T,D], v [B,H,T,D]; mask [M,T] or None."""

    scores = torch.einsum("bhmd,bhtd->bhmt", q.float(), k.float()) * scale
    if mask is not None:
        scores = scores.masked_fill(mask.view(1, 1, *mask.shape) == 0,
                                    float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmt,bhtd->bhmd", probs, v.float()).to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(9)
    cases = []
    failures = []
    stream = torch.cuda.Stream()

    decode_shapes = (
        (4, 16, 1024, 128),
        (128, 16, 1024, 128),
        (1, 16, 384, 128),
        (7, 8, 256, 128),
    )
    for batch, heads, tokens, dim in decode_shapes:
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, tokens, dim, device="cuda",
                        dtype=torch.bfloat16)
        key_t = k.transpose(-1, -2).contiguous()
        v = torch.randn(batch, heads, tokens, dim, device="cuda",
                        dtype=torch.bfloat16)
        scale = dim ** -0.5
        expected = eager_attention(
            q.view(batch, heads, 1, dim), k, v, scale).view(batch, heads, dim)
        label = f"decode B={batch} H={heads} T={tokens} D={dim}"
        try:
            output = attention.pypto_attention_decode(
                q, key_t, v, scale=scale, stream=stream)
            stream.synchronize()
            correct = bool(torch.allclose(output.float(), expected.float(),
                                          rtol=3e-2, atol=3e-2))
            case = {
                "case": label,
                "output_correct": correct,
                "max_abs_diff": float((output.float() - expected.float()).abs().max()),
            }
        except Exception as error:  # noqa: BLE001 - recorded as marker
            correct = False
            case = {"case": label, "output_correct": False,
                    "error": f"{type(error).__name__}: {error}"}
        cases.append(case)
        if not correct:
            failures.append(label)

    prefill_shapes = (
        (1, 16, 512, 128),
        (2, 16, 384, 128),
        (1, 8, 1024, 128),
    )
    for batch, heads, query_tokens, dim in prefill_shapes:
        tokens = query_tokens  # causal prefill over the prompt itself
        q = torch.randn(batch, heads, query_tokens, dim, device="cuda",
                        dtype=torch.bfloat16)
        k = torch.randn(batch, heads, tokens, dim, device="cuda",
                        dtype=torch.bfloat16)
        key_t = k.transpose(-1, -2).contiguous()
        v = torch.randn(batch, heads, tokens, dim, device="cuda",
                        dtype=torch.bfloat16)
        scale = dim ** -0.5
        mask = torch.tril(torch.ones(query_tokens, tokens, device="cuda",
                                  dtype=torch.bfloat16))
        expected = eager_attention(q, k, v, scale, mask)
        label = f"prefill B={batch} H={heads} M={query_tokens} D={dim}"
        try:
            output = attention.pypto_attention_prefill(
                q, key_t, v, scale=scale, mask=mask, stream=stream)
            stream.synchronize()
            correct = bool(torch.allclose(output.float(), expected.float(),
                                          rtol=3e-2, atol=3e-2))
            case = {
                "case": label,
                "output_correct": correct,
                "max_abs_diff": float((output.float() - expected.float()).abs().max()),
            }
        except Exception as error:  # noqa: BLE001 - recorded as marker
            correct = False
            case = {"case": label, "output_correct": False,
                    "error": f"{type(error).__name__}: {error}"}
        cases.append(case)
        if not correct:
            failures.append(label)

    evidence = {
        "schema": 1,
        "kind": "pypto-kernels-attention-decomposed-sm120",
        "dso": attention.bootstrap is not None,
        "case_count": len(cases),
        "all_correct": not failures,
        "failures": failures,
        "cases": cases,
    }
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0 if not failures else 75


if __name__ == "__main__":
    raise SystemExit(main())
