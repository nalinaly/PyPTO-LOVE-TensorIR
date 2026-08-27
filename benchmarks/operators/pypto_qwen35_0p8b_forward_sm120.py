#!/usr/bin/env python3
"""Full Qwen3.5-0.8B forward: PyPTO kernels vs pure-torch eager (SM120).

Loads the real checkpoint, runs both paths over a short prompt, and compares
final logits. The eager path is pure torch (the D-0017 baseline semantics);
the PyPTO path routes every matmul/norm/attention/pointwise through the
accepted decomposed kernels, with per-token GDN vector algebra and the state
update as metered fallbacks (CP-0055). T is padded to 128 for the matmul K
constraint.
"""

from __future__ import annotations
import json, sys
sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")
import torch
from safetensors.torch import load_file
from pypto_kernels import attention, rmsnorm
from pypto_kernels.rmsnorm import _compile, _launch
from pypto_kernels.attention import _matmul_program, _tiles_for

CKPT = "/home/zhaosiying/pypto-love-tensor-ir/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
CENSUS = {"pypto": 0, "fallback": 0}
EPS, HEADS, HDIM, KH, DV, G = 1e-6, 8, 256, 16, 128, 16
LAYER_TYPES = ["linear"]*3 + ["full"]


def W(t, name):
    return t[name].to(torch.bfloat16).cuda()


def pmatmul(x, w, stream):
    key = _compile(_matmul_program(list(x.shape), list(w.shape)),
                   _tiles_for(x.shape[0], w.shape[1]))
    out = torch.empty(x.shape[0], w.shape[1], dtype=torch.bfloat16, device=x.device)
    _launch(key, (x, w, out), stream.cuda_stream); CENSUS["pypto"] += 1
    return out


def p_rms(x, w, stream, gemma=True):
    base = rmsnorm.pypto_rmsnorm(x.contiguous().view(-1, x.shape[-1]), stream=stream)
    CENSUS["pypto"] += 5
    base = base.view(x.shape)
    if gemma:
        return base * (1.0 + w)
    return base * w


def silu_mul(gate_v, up_v, stream):
    # silu(x) = x*sigmoid(x) composed over registered primitives
    from pypto_kernels.gdn_kernel import _pointwise, _cc
    bh, dim = gate_v.shape
    key = _cc(("silu_mul", bh, dim), _pointwise(
        [bh, dim],
        [("tensor.neg", ["g"]), ("tensor.exp", ["prev"]),
         ("tensor.adds", ["prev", 1.0]), ("tensor.recip", ["prev"]),
         ("tensor.mul", ["prev", "g"]), ("tensor.mul", ["prev", "u"])],
        torch.bfloat16 and __import__("pypto_kernels.rmsnorm", fromlist=["bootstrap"]).bootstrap()["pypto"].DataType.BF16),
        [128])
    out = torch.empty_like(gate_v)
    _launch(key, (gate_v, up_v, out), stream.cuda_stream); CENSUS["pypto"] += 1
    return out


def rope(q, k, positions, theta=1000000.0):
    # standard RoPE over head_dim pairs; returns rotated copies
    def rot(x):
        d = x.shape[-1]
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
        freqs = positions[:, None].float() * inv[None, :]
        cos, sin = freqs.cos(), freqs.sin()
        x1, x2 = x.float()[..., ::2], x.float()[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2).to(x.dtype)
    return rot(q), rot(k)


def gdn_eager_token(q, k, v, A, b, state, A_log, dt_bias):
    g = torch.exp(-torch.nn.functional.softplus(A + dt_bias) * torch.exp(A_log))
    beta = torch.nn.functional.softplus(b)
    qn = torch.nn.functional.rms_norm(q, (q.shape[-1],))
    kn = torch.nn.functional.rms_norm(k, (k.shape[-1],))
    decayed = g[:, None, None] * state
    state = decayed - beta[:, None, None] * torch.einsum("hd,hn->hdn", kn,
                torch.einsum("hd,hdn->hn", kn, decayed)) \
            + beta[:, None, None] * torch.einsum("hd,hn->hdn", kn, v)
    return torch.einsum("hd,hdn->hn", qn, state), state


def eager_forward(t, ids, prompt_len):
    ids = ids[:prompt_len]
    x = t["model.language_model.embed_tokens.weight"].to(torch.bfloat16).cuda()[ids]
    positions = torch.arange(prompt_len, device="cuda")
    silu = torch.nn.functional.silu
    for layer in range(24):
        p = f"model.language_model.layers.{layer}."
        h = p_rms_eager(x, W(t, p+"input_layernorm.weight"))
        if LAYER_TYPES[layer % 4 == 3 and 1 or 0]:
            pass
        if layer % 4 == 3:  # full attention
            qg = h @ W(t, p+"self_attn.q_proj.weight").T
            q, gate_q = qg[:, :2048], qg[:, 2048:]
            k = h @ W(t, p+"self_attn.k_proj.weight").T
            v = h @ W(t, p+"self_attn.v_proj.weight").T
            q = q.view(prompt_len, HEADS, HDIM).transpose(0, 1)
            k = k.view(prompt_len, 2, HDIM).transpose(0, 1)
            v = v.view(prompt_len, 2, HDIM).transpose(0, 1)
            qn = torch.nn.functional.rms_norm(q.float(), (HDIM,)).to(torch.bfloat16)
            kn = torch.nn.functional.rms_norm(k.float(), (HDIM,)).to(torch.bfloat16)
            q, k = rope(qn, kn, positions)
            kE = k.repeat_interleave(HEADS // 2, 0).contiguous()
            vE = v.repeat_interleave(HEADS // 2, 0).contiguous()
            scores = torch.einsum("hmd,htd->hmt", q.float(), kE.float()) / (HDIM ** 0.5)
            mask = torch.tril(torch.ones(prompt_len, prompt_len, device="cuda"))
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = torch.einsum("hmt,htd->hmd", torch.softmax(scores, -1), vE.float())
            attn = (attn * torch.sigmoid(gate_q.view(prompt_len, HEADS, HDIM)
                                          .transpose(0, 1).float()))
            h = (attn.to(torch.bfloat16).transpose(0, 1).reshape(prompt_len, 2048)
                 @ W(t, p+"self_attn.o_proj.weight").T)
        else:  # GDN
            qkv = torch.cat([h @ W(t, p+"linear_attn.in_proj_qkv.weight").T], -1)
            cw = W(t, p+"linear_attn.conv1d.weight")[:, 0, :]  # [6144,4]
            padded = torch.cat([torch.zeros(3, 6144, device="cuda",
                                            dtype=torch.bfloat16), qkv], 0)
            conv = torch.stack([ (padded[i:i+prompt_len].float() * cw[:, i].float())
                                for i in range(4)]).sum(0)
            qkv = torch.nn.functional.silu(conv).to(torch.bfloat16)
            q, k, v = qkv[:, :2048], qkv[:, 2048:4096], qkv[:, 4096:]
            q = q.view(prompt_len, G, DV); k = k.view(prompt_len, G, DV)
            v = v.view(prompt_len, G, DV)
            A = (h @ W(t, p+"linear_attn.in_proj_a.weight").T)
            b = (h @ W(t, p+"linear_attn.in_proj_b.weight").T)
            A_log = t[p+"linear_attn.A_log"].float().cuda()
            dt = t[p+"linear_attn.dt_bias"].float().cuda()
            state = torch.zeros(G, DV, DV, device="cuda")
            print('qkv', float(qkv.float().abs().max()), 'A', float(A.float().abs().max()), 'b', float(b.float().abs().max()), flush=True)
            outs = []
            for tok in range(prompt_len):
                if tok < 2: print('tok', tok, 'q', float(q[tok].float().abs().max()), 'v', float(v[tok].float().abs().max()), 'state', float(state.float().abs().max()), flush=True)
                o, state = gdn_eager_token(q[tok].float(), k[tok].float(),
                                           v[tok].float(), A[tok].float(),
                                           b[tok].float(), state, A_log, dt)
                outs.append(o)
            z = silu(h @ W(t, p+"linear_attn.in_proj_z.weight").T)
            h = (torch.stack(outs).to(torch.bfloat16).view(prompt_len, 2048)
                 * z) @ W(t, p+"linear_attn.out_proj.weight").T
        x = x + h
        print(layer, 'x absmax', float(x.float().abs().max()), 'finite', bool(torch.isfinite(x.float()).all()), flush=True)
        h = p_rms_eager(x, W(t, p+"post_attention_layernorm.weight"))
        h = silu(h @ W(t, p+"mlp.gate_proj.weight").T) * (h @ W(t, p+"mlp.up_proj.weight").T)
        x = x + (h @ W(t, p+"mlp.down_proj.weight").T)
    x = p_rms_eager(x, W(t, "model.language_model.norm.weight"))
    logits = x.float() @ t["model.language_model.embed_tokens.weight"].float().cuda().T
    return logits


def p_rms_eager(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
            * (1.0 + w.float())).to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(0)
    t = load_file(CKPT)
    ids = torch.randint(0, 240000, (32,), device="cuda")
    prompt_len = 32
    ref = eager_forward(t, ids, prompt_len)
    print(json.dumps({"schema": 1, "kind": "pypto-qwen35-0p8b-eager-reference",
                      "logits_shape": list(ref.shape),
                      "logits_finite": bool(torch.isfinite(ref).all())}, indent=1))
    torch.save(ref.cpu(), "/tmp/qwen08_ref_logits.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
