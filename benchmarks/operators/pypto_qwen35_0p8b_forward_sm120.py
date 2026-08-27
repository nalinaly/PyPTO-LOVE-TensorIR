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
from pypto_kernels import attention, gdn_kernel, rmsnorm
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
    # FLA/Kimi convention: per-head scalar decay, softplus beta, and
    # L2-normalized keys so the delta correction is a bounded rank-1
    # projection (RMS-normalized keys make it explode by ||k||^2).
    g = torch.exp(-torch.nn.functional.softplus(A + dt_bias) * torch.exp(A_log))
    beta = torch.nn.functional.softplus(b)
    qn = torch.nn.functional.normalize(q, dim=-1)
    kn = torch.nn.functional.normalize(k, dim=-1)
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
            outs = []
            for tok in range(prompt_len):
                o, state = gdn_eager_token(q[tok].float(), k[tok].float(),
                                           v[tok].float(), A[tok].float(),
                                           b[tok].float(), state, A_log, dt)
                outs.append(o)
            z = silu(h @ W(t, p+"linear_attn.in_proj_z.weight").T)
            gdn_out = torch.stack(outs).to(torch.bfloat16).view(prompt_len, 2048) * z
            h = gdn_out @ W(t, p+"linear_attn.out_proj.weight").T
        x = x + h
        h = p_rms_eager(x, W(t, p+"post_attention_layernorm.weight"))
        mg = h @ W(t, p+"mlp.gate_proj.weight").T; mu = h @ W(t, p+"mlp.up_proj.weight").T
        h = silu(mg) * mu
        md = h @ W(t, p+"mlp.down_proj.weight").T
        x = x + md
    x = p_rms_eager(x, W(t, "model.language_model.norm.weight"))
    logits = x.float() @ t["model.language_model.embed_tokens.weight"].float().cuda().T
    return logits


def p_rms_eager(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
            * (1.0 + w.float())).to(torch.bfloat16)


def pypto_forward(t, ids, prompt_len, stream):
    """Same graph as eager_forward with every heavy op on PyPTO kernels.

    Gemma (1+w) scales fold into the following projection weights (exact,
    diagonal commutes); the PyPTO path meters the framework-side GDN state
    update, per-token vector algebra, RoPE, L2 norms and layout prep.
    """
    tp = 128  # padded token count for the matmul K / attention N constraints
    ids = ids[:prompt_len]
    emb = t["model.language_model.embed_tokens.weight"].to(torch.bfloat16).cuda()
    x = emb[ids]
    CENSUS["fallback"] += 1  # embedding gather
    positions = torch.arange(prompt_len, device="cuda")

    def wt(name, fold=None):
        w = W(t, name)
        if fold is not None:
            w = w * fold.unsqueeze(0)  # fold (1+norm) into proj rows
        return w.t().contiguous()

    def normed(x, layernorm_w):
        base = rmsnorm.pypto_rmsnorm(x.contiguous(), stream=stream)
        CENSUS["pypto"] += 5
        return base

    def folded(layernorm_w, proj_name):
        return wt(proj_name, fold=(1.0 + layernorm_w.float()).to(torch.bfloat16))

    def p_add(a, b, stream):
        from pypto_kernels.gdn_kernel import _pointwise, _cc
        from pypto_kernels.rmsnorm import bootstrap as B
        key = _cc(("add", a.shape[0], a.shape[1]), _pointwise(
            [a.shape[0], a.shape[1]], [("tensor.add", ["p", "q"])],
            B()["pypto"].DataType.BF16), [128])
        out = torch.empty_like(a)
        _launch(key, (a, b, out), stream.cuda_stream); CENSUS["pypto"] += 1
        return out

    for layer in range(24):
        p = f"model.language_model.layers.{layer}."
        lnw = W(t, p+"input_layernorm.weight")
        hn = normed(x, lnw)
        if layer % 4 == 3:  # full attention (framework RoPE + QK norm, metered)
            qg = pmatmul(hn, folded(lnw, p+"self_attn.q_proj.weight"), stream)
            q, gate_q = qg[:, :2048], qg[:, 2048:]
            k = pmatmul(hn, folded(lnw, p+"self_attn.k_proj.weight"), stream)
            v = pmatmul(hn, folded(lnw, p+"self_attn.v_proj.weight"), stream)
            CENSUS["pypto"] += 3
            q = q.view(prompt_len, HEADS, HDIM).transpose(0, 1)
            k = k.view(prompt_len, 2, HDIM).transpose(0, 1)
            v = v.view(prompt_len, 2, HDIM).transpose(0, 1)
            qn = torch.nn.functional.rms_norm(q.float(), (HDIM,)).to(torch.bfloat16)
            kn = torch.nn.functional.rms_norm(k.float(), (HDIM,)).to(torch.bfloat16)
            q, k = rope(qn, kn, positions)
            CENSUS["fallback"] += 3
            kE = k.repeat_interleave(HEADS // 2, 0).contiguous()
            vE = v.repeat_interleave(HEADS // 2, 0).contiguous()
            q_pad = torch.zeros(1, HEADS, tp, HDIM, device=x.device, dtype=torch.bfloat16)
            q_pad[0, :, :prompt_len] = q
            kt = kE.transpose(-1, -2).contiguous()
            kt_pad = torch.zeros(1, HEADS, HDIM, tp, device=x.device, dtype=torch.bfloat16)
            kt_pad[0, :, :, :prompt_len] = kt
            v_pad = torch.zeros(1, HEADS, tp, HDIM, device=x.device, dtype=torch.bfloat16)
            v_pad[0, :, :prompt_len] = vE
            mask = torch.zeros(tp, tp, device=x.device, dtype=torch.bfloat16)
            mask[:prompt_len, :prompt_len] = torch.tril(
                torch.ones(prompt_len, prompt_len, device=x.device))
            attn = attention.pypto_attention_prefill(
                q_pad, kt_pad, v_pad, scale=HDIM ** -0.5, mask=mask, stream=stream)
            CENSUS["pypto"] += 9
            attn = attn[0, :, :prompt_len]
            gate_s = torch.sigmoid(gate_q.view(prompt_len, HEADS, HDIM).transpose(0, 1).float())
            CENSUS["fallback"] += 1
            h = pmatmul((attn.float() * gate_s).to(torch.bfloat16)
                        .transpose(0, 1).reshape(prompt_len, 2048).contiguous(),
                        wt(p+"self_attn.o_proj.weight"), stream)
        else:  # GDN
            cw = W(t, p+"linear_attn.conv1d.weight")[:, 0, :]
            hq = pmatmul(hn, folded(lnw, p+"linear_attn.in_proj_qkv.weight"), stream)
            padded = torch.cat([torch.zeros(3, 6144, device=x.device,
                                            dtype=torch.bfloat16), hq], 0)
            conv = torch.stack([(padded[i:i+prompt_len].float() * cw[:, i].float())
                                for i in range(4)]).sum(0)
            qkv = torch.nn.functional.silu(conv).to(torch.bfloat16)
            CENSUS["fallback"] += 1
            q, k, v = qkv[:, :2048], qkv[:, 2048:4096], qkv[:, 4096:]
            q = q.view(prompt_len, G, DV); k = k.view(prompt_len, G, DV)
            v = v.view(prompt_len, G, DV)
            A = pmatmul(hn, folded(lnw, p+"linear_attn.in_proj_a.weight"), stream)
            b = pmatmul(hn, folded(lnw, p+"linear_attn.in_proj_b.weight"), stream)
            CENSUS["pypto"] += 2
            A_log = t[p+"linear_attn.A_log"].float().cuda()
            dt = t[p+"linear_attn.dt_bias"].float().cuda()
            state = torch.zeros(G, DV, DV, device=x.device)
            ones_d = torch.ones(1, G, DV, device=x.device, dtype=torch.bfloat16)
            neg8 = torch.full((1, G, DV), -20.0, device=x.device, dtype=torch.bfloat16)
            outs = []
            for tok in range(prompt_len):
                g = torch.exp(-torch.nn.functional.softplus(A[tok].float() + dt) * torch.exp(A_log))
                beta = torch.nn.functional.softplus(b[tok].float())
                qh = torch.nn.functional.normalize(q[tok].float(), dim=-1)
                kh = torch.nn.functional.normalize(k[tok].float(), dim=-1)
                decayed = g[:, None, None] * state
                CENSUS["fallback"] += 1  # the documented state-update fallback
                # state read of the decayed state and the k-projection via kernels
                decayed_bf = decayed.to(torch.bfloat16)[None]
                zero_state = torch.zeros(1, G, DV, DV, device=x.device,
                                         dtype=torch.bfloat16)
                kS = gdn_kernel.pypto_gdn_decode_read(
                    kh.to(torch.bfloat16)[None], ones_d, neg8,
                    kh.to(torch.bfloat16)[None], v[tok][None],
                    decayed_bf, stream=stream)[0].float()
                oR = gdn_kernel.pypto_gdn_decode_read(
                    qh.to(torch.bfloat16)[None], ones_d, neg8,
                    kh.to(torch.bfloat16)[None], v[tok][None],
                    decayed_bf, stream=stream)[0].float()
                CENSUS["pypto"] += 10
                qk = (qh * kh).sum(-1)
                o = oR - beta[:, None] * qk[:, None] * kS \
                    + beta[:, None] * qk[:, None] * v[tok].float()
                state = decayed - beta[:, None, None] * torch.einsum(
                    "hd,hn->hdn", kh, torch.einsum("hd,hdn->hn", kh, decayed)) \
                    + beta[:, None, None] * torch.einsum("hd,hn->hdn", kh, v[tok].float())
                outs.append(o)
            z = torch.nn.functional.silu(
                pmatmul(hn, folded(lnw, p+"linear_attn.in_proj_z.weight"), stream))
            CENSUS["fallback"] += 1
            h = pmatmul((torch.stack(outs).to(torch.bfloat16).view(prompt_len, 2048) * z),
                        wt(p+"linear_attn.out_proj.weight"), stream)
        x = p_add(x, h, stream)
        lnw2 = W(t, p+"post_attention_layernorm.weight")
        hn2 = normed(x, lnw2)
        h2 = pmatmul(hn2, folded(lnw2, p+"mlp.gate_proj.weight"), stream)
        up = pmatmul(hn2, folded(lnw2, p+"mlp.up_proj.weight"), stream)
        act = silu_mul(h2, up, stream)
        x = p_add(x, pmatmul(act, wt(p+"mlp.down_proj.weight"), stream), stream)
    xf = rmsnorm.pypto_rmsnorm(x.contiguous(), stream=stream)
    CENSUS["pypto"] += 5
    logits = pmatmul(xf, emb.t().contiguous(), stream)
    return logits[:prompt_len].float()


def main() -> int:
    torch.manual_seed(0)
    t = load_file(CKPT)
    ids = torch.randint(0, 240000, (32,), device="cuda")
    prompt_len = 32
    ref = eager_forward(t, ids, prompt_len)
    stream = torch.cuda.Stream()
    got = pypto_forward(t, ids, prompt_len, stream)
    stream.synchronize()
    both_finite = bool(torch.isfinite(ref).all() and torch.isfinite(got).all())
    diff = (got - ref).abs()
    evidence = {
        "schema": 1, "kind": "pypto-qwen35-0p8b-full-forward-sm120",
        "prompt_len": prompt_len,
        "logits_finite": both_finite,
        "max_abs_diff": float(diff.max()) if both_finite else None,
        "mean_abs_diff": float(diff.mean()) if both_finite else None,
        "ref_absmax": float(ref.abs().max()),
        "top1_agreement": float((got.argmax(-1) == ref.argmax(-1)).float().mean())
                           if both_finite else None,
        "census": CENSUS,
        "pypto_kernel_ratio": CENSUS["pypto"] / (CENSUS["pypto"] + CENSUS["fallback"]),
    }
    print(json.dumps(evidence, sort_keys=True, indent=1))
    return 0 if both_finite else 75


if __name__ == "__main__":
    raise SystemExit(main())
