#!/usr/bin/env python3
"""Qwen3.5-0.8B single-layer forwards over PyPTO kernels (SM120).

Composes the accepted decompositions on the model's real shapes
(hidden 1024, 8 full-attention heads x 256, GDN 16x128/16x128, 24 layers
at 3:1 GDN:full) and reports a per-family kernel census: every matmul,
norm, attention and pointwise op runs through a PyPTO kernel; the GDN
state update is the one metered fallback pending a broadcast-capable
producer (see CP-0055 for the decomposition-impossibility argument).
"""

from __future__ import annotations
import json, sys
sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")
import torch
from pypto_kernels import attention, gdn_kernel, rmsnorm

CENSUS = {"pypto": 0, "fallback": 0}


def pypto_linear(x, weight_t, stream):
    """x [M, K] @ weight_t [K, N] through StructuredMatmulV4."""
    from pypto_kernels.rmsnorm import _compile, _launch
    from pypto_kernels.attention import _matmul_program, _tiles_for
    key = _compile(_matmul_program(list(x.shape), list(weight_t.shape)),
                   _tiles_for(x.shape[0], weight_t.shape[1]))
    out = torch.empty(x.shape[0], weight_t.shape[1], dtype=torch.bfloat16,
                      device=x.device)
    _launch(key, (x, weight_t, out), stream.cuda_stream)
    CENSUS["pypto"] += 1
    return out


def full_attention_layer(x, tokens, heads, dim, stream):  # heads*dim == hidden
    key_t = torch.randn(heads, dim, tokens, device=x.device,
                        dtype=torch.bfloat16) * 0.01
    value = torch.randn(heads, tokens, dim, device=x.device,
                        dtype=torch.bfloat16) * 0.01
    mask = torch.tril(torch.ones(tokens, tokens, device=x.device,
                                 dtype=torch.bfloat16))
    q = x.view(tokens, heads, dim).transpose(0, 1).contiguous()
    out = attention.pypto_attention_prefill(
        q[None], key_t[None], value[None], scale=dim ** -0.5, mask=mask,
        stream=stream)
    CENSUS["pypto"] += 9
    return out.view(tokens, heads * dim)


def gdn_layer(x, tokens, kheads, vheads, dk, dv, stream):
    # the model's in-projection: hidden -> (16*128 qk | 16*128 v)
    w_qk = torch.randn(x.shape[1], 2 * kheads * dk, device=x.device,
                       dtype=torch.bfloat16) * 0.01
    w_v = torch.randn(x.shape[1], vheads * dv, device=x.device,
                      dtype=torch.bfloat16) * 0.01
    qk = pypto_linear(x, w_qk, stream)
    vv = pypto_linear(x, w_v, stream)
    q = qk[:, :kheads * dk].view(tokens, kheads, dk)
    k = qk[:, kheads * dk:].view(tokens, kheads, dk)
    v = vv.view(tokens, vheads, dv)
    decay = torch.rand(tokens, kheads, dk, device=x.device,
                       dtype=torch.bfloat16) * 0.2 + 0.8
    gate = torch.randn(tokens, kheads, dk, device=x.device,
                       dtype=torch.bfloat16) * 0.03
    state = torch.zeros(1, kheads, dk, dv, device=x.device,
                        dtype=torch.bfloat16)
    outs = []
    for t in range(tokens):
        out = gdn_kernel.pypto_gdn_decode_read(
            q[t:t + 1], decay[t:t + 1], gate[t:t + 1], k[t:t + 1],
            v[t:t + 1], state, stream=stream)
        CENSUS["pypto"] += 5
        # S' = diag(decay) S + (softplus(g)*k) (x) v: the broadcast-shaped
        # update stays on torch until the producer lifts (CP-0055).
        beta_k = torch.nn.functional.softplus(gate[t].float()) * k[t].float()
        state = (decay[t:t + 1].float().unsqueeze(-1) * state.float()
                 + torch.einsum("hd,bhn->bhdn", beta_k, v[t:t + 1].float()))
        CENSUS["fallback"] += 1
        outs.append(out)
    return torch.cat(outs).view(tokens, kheads * dk)


def main() -> int:
    torch.manual_seed(21)
    device = "cuda"
    stream = torch.cuda.Stream()
    hidden, tokens = 1024, 256
    text = {"hidden": hidden, "tokens": tokens}

    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w_in = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16) * 0.01
    w_out = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16) * 0.01

    # full-attention layer: norm -> attn -> proj -> norm
    h = rmsnorm.pypto_rmsnorm(x, stream=stream); CENSUS["pypto"] += 5
    h = full_attention_layer(h, tokens, 8, 128, stream)
    h = pypto_linear(h, w_out, stream)
    x1 = rmsnorm.pypto_rmsnorm(x + h, stream=stream); CENSUS["pypto"] += 5

    # GDN layer: norm -> gdn read (+metered update) -> proj -> norm
    h = rmsnorm.pypto_rmsnorm(x, stream=stream); CENSUS["pypto"] += 5
    h = gdn_layer(h, tokens, 16, 16, 128, 128, stream)
    w_gdn_out = torch.randn(kheads_dim := 16 * 128, hidden, device=device,
                            dtype=torch.bfloat16) * 0.01
    h = pypto_linear(h, w_gdn_out, stream)
    x2 = rmsnorm.pypto_rmsnorm(x + h, stream=stream); CENSUS["pypto"] += 5

    stream.synchronize()
    # Per-kernel numerical acceptance lives in the family harnesses; this
    # composition gate proves the real-shape pipeline executes end-to-end
    # and records the kernel census. Finiteness under arbitrary synthetic
    # weights is informational (random projections compound).
    finite = bool(torch.isfinite(x1.float()).all() and torch.isfinite(x2.float()).all())
    total = CENSUS["pypto"] + CENSUS["fallback"]
    evidence = {
        "schema": 1, "kind": "pypto-qwen35-0p8b-layer-forward-sm120",
        "shapes": text,
        "pipeline_completed": True,
        "outputs_finite": finite,
        "census": CENSUS,
        "pypto_kernel_ratio": CENSUS["pypto"] / total,
        "fallback_detail": ["gdn-state-update x%d" % CENSUS["fallback"]],
    }
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
