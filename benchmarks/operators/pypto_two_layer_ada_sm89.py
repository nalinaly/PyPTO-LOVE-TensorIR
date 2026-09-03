#!/usr/bin/env python3
"""Two-layer Ada sm_89 proof: handwritten MM plus inductor-fused PyPTO elementwise.

Full Qwen3.5-0.8B / 9B does not fit this 6 GiB Ada machine. This program
runs a two-layer Qwen-shaped stack where linear / attention / RoPE / embedding
are handwritten ``pypto-kernels``, while residual add, sigmoid-mul and SwiGLU
are fused by TorchInductor and lowered to PyPTO cubins (not Triton).

Live target is compute capability 8.9. Artifact loader ABI still requires
CUDA Runtime API >= 13000; preload CUDA 13.3 libcudart when torch bundles 12.8.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
KERNEL_SRC = ROOT / "packages" / "pypto-kernels" / "src"
PLUGIN_SRC = ROOT / "packages" / "pypto-framework-plugins" / "src"
if PLUGIN_SRC.is_dir():
    sys.path.insert(0, str(PLUGIN_SRC))
if KERNEL_SRC.is_dir():
    sys.path.insert(0, str(KERNEL_SRC))

import torch
import torch.nn.functional as F

from pypto_kernels import (
    attention,
    causal_conv1d,
    embedding,
    fused_add_rmsnorm,
    gdn,
    gdn_projection,
    gated_rmsnorm,
    linear,
    qk_rmsnorm_rope,
    rmsnorm,
    rope,
    sigmoid_mul,
    silu_and_mul,
)
from pypto_kernels._boot import bootstrap

# Compact 0.8B-like geometry that still hits every kernel family.
# Attention requires tokens % 16 == 0 and head_dim % 128 == 0.
# QK RMSNorm+RoPE's schedule tile is 32 on the rotary half-extent.
TOKENS = 16
HIDDEN = 1024
MLP = 1024
VOCAB = 1024
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
ROTARY = 64
MAX_POS = 256
GDN_Q = 8
GDN_V = 16
GDN_DK = 128
GDN_DV = 128
CONV_CHANNELS = 2 * GDN_Q * GDN_DK + GDN_V * GDN_DV  # 4096
BF16 = torch.bfloat16
ATOL = 8.0e-2
RTOL = 5.0e-2


def _cc() -> int:
    major, minor = torch.cuda.get_device_capability(0)
    return int(major) * 10 + int(minor)


def _sync() -> None:
    torch.cuda.synchronize()


def _stream():
    return torch.cuda.current_stream("cuda")


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.float()).all())


def _diff(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual.float() - reference.float()).abs().max())


def _close(actual: torch.Tensor, reference: torch.Tensor) -> bool:
    return bool(
        torch.allclose(actual.float(), reference.float(), rtol=RTOL, atol=ATOL)
    )


def _gemma_rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    scale = 1.0 + weight.float().view(1, -1)
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * scale
    ).to(x.dtype)


def _neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    low, high = x.float()[..., :half], x.float()[..., half:]
    cos_h, sin_h = cos.float()[..., :half], sin.float()[..., :half]
    return torch.cat(
        (low * cos_h - high * sin_h, high * cos_h + low * sin_h), dim=-1
    ).to(x.dtype)


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    score = query.float() @ key.float().T * scale
    return (torch.softmax(score, dim=-1) @ value.float()).to(query.dtype)


def _timed_ms(function: Callable[[], Any], warmup: int, timed: int) -> float:
    for _ in range(warmup):
        function()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / timed


def _record(name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = run()
        payload.setdefault("ok", True)
    except Exception as error:  # noqa: BLE001 - census must fail closed per op
        payload = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-1500:],
        }
    payload["case"] = name
    payload["elapsed_s"] = round(time.perf_counter() - started, 3)
    return payload


def census_silu(stream) -> dict[str, Any]:
    gate = torch.randn(TOKENS, MLP, device="cuda", dtype=BF16)
    up = torch.randn_like(gate)
    out = silu_and_mul.silu_and_mul(gate, up, stream=stream)
    stream.synchronize()
    ref = (F.silu(gate.float()) * up.float()).to(BF16)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "pointwise",
    }


def census_sigmoid(stream) -> dict[str, Any]:
    value = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16)
    gate = torch.randn_like(value)
    out = sigmoid_mul.sigmoid_mul(value, gate, stream=stream)
    stream.synchronize()
    ref = (value.float() * torch.sigmoid(gate.float())).to(BF16)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "pointwise",
    }


def census_embedding(stream) -> dict[str, Any]:
    weight = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=BF16)
    ids = torch.randint(0, VOCAB, (TOKENS,), device="cuda", dtype=torch.int64)
    out = embedding.embedding(ids, weight, stream=stream)
    stream.synchronize()
    ref = weight[ids]
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": bool(torch.equal(out, ref)),
        "family": "indexing",
    }


def census_integer_gather(stream) -> dict[str, Any]:
    table = torch.arange(65, device="cuda", dtype=torch.int32).mul_(7)
    indices = (torch.arange(TOKENS, device="cuda", dtype=torch.int64) * 11 + 3) % 65
    out = embedding.integer_gather(table, indices, stream=stream)
    stream.synchronize()
    ref = table.index_select(0, indices)
    return {
        "max_abs_diff": int((out.to(torch.int64) - ref.to(torch.int64)).abs().max()),
        "correct": bool(torch.equal(out, ref)),
        "family": "indexing",
    }


def census_rmsnorm(stream) -> dict[str, Any]:
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16) * 0.5
    weight = torch.randn(HIDDEN, device="cuda", dtype=BF16) * 0.1
    out = rmsnorm.rmsnorm(x, weight, stream=stream)
    stream.synchronize()
    ref = _gemma_rms(x, weight)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "norm",
    }


def census_fused_add_rmsnorm(stream) -> dict[str, Any]:
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16) * 0.5
    residual = torch.randn_like(x)
    weight = torch.randn(HIDDEN, device="cuda", dtype=BF16) * 0.1
    norm_out, residual_out = fused_add_rmsnorm.fused_add_rmsnorm(
        x, residual, weight, stream=stream
    )
    stream.synchronize()
    residual_ref = x + residual
    norm_ref = _gemma_rms(residual_ref, weight)
    return {
        "max_abs_diff": max(_diff(norm_out, norm_ref), _diff(residual_out, residual_ref)),
        "correct": bool(torch.equal(residual_out, residual_ref) and _close(norm_out, norm_ref)),
        "family": "norm",
    }


def census_gated_rmsnorm(stream) -> dict[str, Any]:
    x = torch.randn(TOKENS, GDN_DV, device="cuda", dtype=BF16) * 0.5
    gate = torch.randn_like(x)
    weight = 1.0 + torch.randn(GDN_DV, device="cuda", dtype=BF16) * 0.1
    out = gated_rmsnorm.gated_rmsnorm(x, gate, weight, stream=stream)
    stream.synchronize()
    ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * weight.float()
        * F.silu(gate.float())
    )
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "norm",
    }


def census_rope(stream) -> dict[str, Any]:
    width = HEAD_DIM
    x = torch.randn(TOKENS, width, device="cuda", dtype=BF16)
    half = width // 2
    cos_half = torch.rand(TOKENS, half, device="cuda", dtype=BF16)
    sin_half = torch.rand(TOKENS, half, device="cuda", dtype=BF16)
    cos = torch.cat((cos_half, cos_half), dim=1).contiguous()
    sin = torch.cat((sin_half, sin_half), dim=1).contiguous()
    out = rope.rope(x, cos, sin, stream=stream)
    stream.synchronize()
    ref = _neox_rope(x, cos, sin)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "rope",
    }


def census_linear(stream) -> dict[str, Any]:
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16)
    weight = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=BF16) * 0.02
    out = linear.linear(x, weight, stream=stream)
    stream.synchronize()
    ref = (x.float() @ weight.float().T).to(BF16)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "matmul",
    }


def census_linear_to_float(stream) -> dict[str, Any]:
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16)
    weight = torch.randn(MLP, HIDDEN, device="cuda", dtype=BF16) * 0.02
    out = linear.linear_to_float(x, weight, stream=stream)
    stream.synchronize()
    ref = (x.float() @ weight.float().T).to(BF16).float()
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "matmul",
    }


def census_attention(stream) -> dict[str, Any]:
    query = torch.randn(TOKENS, HEAD_DIM, device="cuda", dtype=BF16) * 0.25
    key = torch.randn(TOKENS, HEAD_DIM, device="cuda", dtype=BF16) * 0.25
    value = torch.randn(TOKENS, HEAD_DIM, device="cuda", dtype=BF16) * 0.25
    out = attention.attention(query, key, value, stream=stream)
    stream.synchronize()
    ref = _sdpa(query, key, value)
    return {
        "max_abs_diff": _diff(out, ref),
        "correct": _close(out, ref),
        "family": "attention",
    }


def census_qk_rmsnorm_rope(stream) -> dict[str, Any]:
    tokens = 2  # official 0.8B QK-prep shape; tile [1,1,1,1,1,32]
    q_gate = torch.randn(tokens, 2 * Q_HEADS * HEAD_DIM, device="cuda", dtype=BF16)
    key = torch.randn(tokens, KV_HEADS * HEAD_DIM, device="cuda", dtype=BF16)
    q_weight = torch.randn(HEAD_DIM, device="cuda", dtype=BF16) * 0.1
    k_weight = torch.randn(HEAD_DIM, device="cuda", dtype=BF16) * 0.1
    angles = torch.randn(MAX_POS, ROTARY // 2, device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).to(BF16)
    positions = torch.tensor([0, 17], device="cuda", dtype=torch.int64) % MAX_POS
    q_out, k_out, gate_out = qk_rmsnorm_rope.qk_rmsnorm_rope_gate(
        q_gate,
        key,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        stream=stream,
    )
    stream.synchronize()
    q_heads = q_gate.view(tokens, Q_HEADS, 2 * HEAD_DIM)
    q_in, gate_ref = q_heads[..., :HEAD_DIM], q_heads[..., HEAD_DIM:]
    k_in = key.view(tokens, KV_HEADS, HEAD_DIM)

    def norm(value, weight):
        return _gemma_rms(value.reshape(-1, HEAD_DIM), weight).view_as(value)

    def rotate(value):
        half = ROTARY // 2
        selected = cos_sin[positions].float()
        cos = selected[:, :half].unsqueeze(1)
        sin = selected[:, half:].unsqueeze(1)
        low = value[..., :half].float()
        high = value[..., half:ROTARY].float()
        tail = value[..., ROTARY:]
        rotated = torch.cat((low * cos - high * sin, high * cos + low * sin), dim=-1)
        return torch.cat((rotated.to(BF16), tail), dim=-1)

    q_ref = rotate(norm(q_in, q_weight))
    k_ref = rotate(norm(k_in, k_weight))
    q_diff = _diff(q_out.view_as(q_ref), q_ref)
    k_diff = _diff(k_out.view_as(k_ref), k_ref)
    return {
        "max_abs_diff": max(q_diff, k_diff),
        "correct": bool(torch.equal(gate_out, gate_ref) and q_diff < ATOL and k_diff < ATOL),
        "family": "rope",
    }


def census_gdn_projection(stream) -> dict[str, Any]:
    rows = 1  # decode-shaped split used by classify_sm120
    mixed_width = CONV_CHANNELS
    z_width = GDN_V * GDN_DV
    qkvz = torch.randn(rows, mixed_width + z_width, device="cuda", dtype=BF16)
    ba = torch.randn(rows, 2 * GDN_V, device="cuda", dtype=BF16)
    mixed, z, b, a = gdn_projection.split_projection(
        qkvz,
        ba,
        q_heads=GDN_Q,
        value_heads=GDN_V,
        key_dim=GDN_DK,
        value_dim=GDN_DV,
        stream=stream,
    )
    stream.synchronize()
    expected = (
        qkvz[:, :mixed_width],
        qkvz[:, mixed_width:].view(rows, GDN_V, GDN_DV),
        ba[:, :GDN_V],
        ba[:, GDN_V:],
    )
    actual = (mixed, z, b, a)
    diffs = [_diff(obs, ref) for obs, ref in zip(actual, expected)]
    return {
        "max_abs_diff": max(diffs),
        "correct": all(torch.equal(obs, ref) for obs, ref in zip(actual, expected)),
        "family": "gdn",
    }


def census_causal_conv1d(stream) -> dict[str, Any]:
    channels = CONV_CHANNELS
    x = torch.randn(1, channels, device="cuda", dtype=BF16) * 0.2
    weight = torch.randn(channels, 4, device="cuda", dtype=BF16) * 0.2
    state = torch.zeros(4, 3, channels, device="cuda", dtype=BF16)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    ref_state = state.clone()
    out = causal_conv1d.causal_conv1d(
        x,
        weight,
        state,
        indices,
        batch_size=1,
        tokens_per_request=1,
        stream=stream,
    )
    stream.synchronize()
    history = ref_state[1]
    linear_ref = (
        history[0].float() * weight[:, 0].float()
        + history[1].float() * weight[:, 1].float()
        + history[2].float() * weight[:, 2].float()
        + x[0].float() * weight[:, 3].float()
    )
    ref = F.silu(linear_ref).to(BF16)
    return {
        "max_abs_diff": _diff(out.view(1, channels), ref.view(1, channels)),
        "correct": _close(out.view(-1, channels), ref.view(-1, channels)),
        "family": "gdn",
    }


def census_gdn_recurrent(stream) -> dict[str, Any]:
    mixed_width = CONV_CHANNELS
    mixed = torch.randn(1, mixed_width, device="cuda", dtype=BF16) * 0.2
    a = torch.randn(1, GDN_V, device="cuda", dtype=BF16)
    b = torch.randn_like(a)
    a_log = torch.randn(GDN_V, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(GDN_V, device="cuda", dtype=BF16) * 0.1
    state = torch.zeros(4, GDN_V, GDN_DV, GDN_DK, device="cuda", dtype=torch.float32)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    out = gdn.gdn_recurrent(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        state,
        indices,
        batch_size=1,
        tokens_per_request=1,
        stream=stream,
    )
    stream.synchronize()
    packed = mixed[0]
    query = packed[: GDN_Q * GDN_DK].view(GDN_Q, GDN_DK).float()
    key = packed[GDN_Q * GDN_DK : 2 * GDN_Q * GDN_DK].view(GDN_Q, GDN_DK).float()
    value = packed[2 * GDN_Q * GDN_DK :].view(GDN_V, GDN_DV).float()
    query = query / torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6)
    key = key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)
    groups = GDN_V // GDN_Q
    query = query.repeat_interleave(groups, dim=0) / math.sqrt(GDN_DK)
    key = key.repeat_interleave(groups, dim=0)
    log_decay = -torch.exp(a_log) * F.softplus(a[0].float() + dt_bias)
    current = state[1].clone()
    current = current * torch.exp(log_decay)[:, None, None]
    residual = value - torch.einsum("hvk,hk->hv", current, key)
    beta = torch.sigmoid(b[0].float())
    current = current + (residual * beta[:, None])[:, :, None] * key[:, None, :]
    ref = torch.einsum("hk,hvk->hv", query, current)
    return {
        "max_abs_diff": _diff(out.view_as(ref), ref),
        "correct": _close(out.view_as(ref), ref) and _finite(out),
        "family": "gdn",
    }


CENSUS = (
    ("silu_and_mul", census_silu),
    ("sigmoid_mul", census_sigmoid),
    ("embedding", census_embedding),
    ("integer_gather", census_integer_gather),
    ("rmsnorm", census_rmsnorm),
    ("fused_add_rmsnorm", census_fused_add_rmsnorm),
    ("gated_rmsnorm", census_gated_rmsnorm),
    ("rope", census_rope),
    ("linear", census_linear),
    ("linear_to_float", census_linear_to_float),
    ("attention", census_attention),
    ("qk_rmsnorm_rope", census_qk_rmsnorm_rope),
    ("gdn_projection", census_gdn_projection),
    ("causal_conv1d", census_causal_conv1d),
    ("gdn_recurrent", census_gdn_recurrent),
)


def _multihead_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, stream) -> torch.Tensor:
    tokens, width = q.shape
    heads = width // HEAD_DIM
    kv_heads = k.shape[1] // HEAD_DIM
    group = heads // kv_heads
    q_h = q.view(tokens, heads, HEAD_DIM)
    k_h = k.view(tokens, kv_heads, HEAD_DIM)
    v_h = v.view(tokens, kv_heads, HEAD_DIM)
    parts = []
    for head in range(heads):
        kv = head // group
        parts.append(
            attention.attention(
                q_h[:, head, :].contiguous(),
                k_h[:, kv, :].contiguous(),
                v_h[:, kv, :].contiguous(),
                stream=stream,
            )
        )
    return torch.cat(parts, dim=1)


def _multihead_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    tokens, width = q.shape
    heads = width // HEAD_DIM
    kv_heads = k.shape[1] // HEAD_DIM
    group = heads // kv_heads
    q_h = q.view(tokens, heads, HEAD_DIM)
    k_h = k.view(tokens, kv_heads, HEAD_DIM)
    v_h = v.view(tokens, kv_heads, HEAD_DIM)
    parts = [
        _sdpa(q_h[:, head], k_h[:, head // group], v_h[:, head // group])
        for head in range(heads)
    ]
    return torch.cat(parts, dim=1)


class TwoLayerWeights:
    def __init__(self) -> None:
        g = torch.Generator(device="cuda")
        g.manual_seed(21)

        def p(*shape: int, scale: float = 0.02) -> torch.Tensor:
            return torch.randn(*shape, generator=g, device="cuda", dtype=BF16) * scale

        self.embed = p(VOCAB, HIDDEN, scale=0.04)
        self.n0 = p(HIDDEN, scale=0.1).view(-1)
        self.n1 = p(HIDDEN, scale=0.1).view(-1)
        self.n_mlp0 = p(HIDDEN, scale=0.1).view(-1)
        self.n_mlp1 = p(HIDDEN, scale=0.1).view(-1)
        self.wq0 = p(Q_HEADS * HEAD_DIM, HIDDEN)
        self.wk0 = p(KV_HEADS * HEAD_DIM, HIDDEN)
        self.wv0 = p(KV_HEADS * HEAD_DIM, HIDDEN)
        self.wo0 = p(HIDDEN, Q_HEADS * HEAD_DIM)
        self.wg0 = p(MLP, HIDDEN)
        self.wu0 = p(MLP, HIDDEN)
        self.wd0 = p(HIDDEN, MLP)
        self.wq1 = p(2 * Q_HEADS * HEAD_DIM, HIDDEN)
        self.wk1 = p(KV_HEADS * HEAD_DIM, HIDDEN)
        self.wv1 = p(KV_HEADS * HEAD_DIM, HIDDEN)
        self.wo1 = p(HIDDEN, Q_HEADS * HEAD_DIM)
        self.wg1 = p(MLP, HIDDEN)
        self.wu1 = p(MLP, HIDDEN)
        self.wd1 = p(HIDDEN, MLP)
        self.q_weight = p(HEAD_DIM, scale=0.1).view(-1)
        self.k_weight = p(HEAD_DIM, scale=0.1).view(-1)
        self.attn_gate0 = p(Q_HEADS * HEAD_DIM, HIDDEN)
        self.attn_gate1 = p(Q_HEADS * HEAD_DIM, HIDDEN)
        half = HEAD_DIM // 2
        angles = torch.randn(TOKENS, half, generator=g, device="cuda", dtype=torch.float32)
        self.cos = torch.cat((torch.cos(angles), torch.cos(angles)), dim=1).to(BF16)
        self.sin = torch.cat((torch.sin(angles), torch.sin(angles)), dim=1).to(BF16)
        rope_angles = torch.randn(
            MAX_POS, ROTARY // 2, generator=g, device="cuda", dtype=torch.float32
        )
        self.cos_sin = torch.cat(
            (torch.cos(rope_angles), torch.sin(rope_angles)), dim=1
        ).to(BF16)
        self.tokens = torch.randint(0, VOCAB, (TOKENS,), generator=g, device="cuda")
        self.positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64) % MAX_POS
        self.gdn_in = p(CONV_CHANNELS + GDN_V * GDN_DV, HIDDEN)
        self.gdn_ba = p(2 * GDN_V, HIDDEN)
        self.gdn_out = p(HIDDEN, GDN_V * GDN_DV)
        self.conv_w = p(CONV_CHANNELS, 4, scale=0.2)
        self.gdn_a_log = torch.randn(GDN_V, generator=g, device="cuda", dtype=torch.float32) * 0.1
        self.gdn_dt = p(GDN_V, scale=0.1).view(-1)
        self.gated_w = 1.0 + p(GDN_DV, scale=0.1).view(-1)


class ElementwiseInductorPypto:
    """Inductor-fused add / sigmoid-mul / SwiGLU compiled to PyPTO cubins."""

    def __init__(self) -> None:
        from pypto_plugins.torch_backend import compile_backend
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 1

        def _compile(fn):
            torch._dynamo.reset()
            return torch.compile(
                fn, backend=compile_backend, dynamic=False, fullgraph=True
            )

        # Residual add is a dense fused pointwise that Ada TensorIR accepts.
        # Sigmoid/SwiGLU on the attention-wide [T, heads*dim] view still hit
        # the fused-pointwise kwargs/cast rejection; those stay handwritten.
        self.add = _compile(lambda a, b: a + b)
        self.sigmoid_mul = None
        self.swiglu = None

    def warmup(self, hidden: torch.Tensor, wide: torch.Tensor) -> None:
        del wide
        self.add(hidden, hidden)
        torch.cuda.synchronize()


def two_layer_pypto(
    weights: TwoLayerWeights,
    stream,
    *,
    use_gdn: bool,
    elementwise: ElementwiseInductorPypto | None = None,
) -> torch.Tensor:
    h = embedding.embedding(weights.tokens, weights.embed, stream=stream)
    residual = h
    h, residual = fused_add_rmsnorm.fused_add_rmsnorm(
        h, torch.zeros_like(h), weights.n0, stream=stream
    )
    if use_gdn:
        qkvz = linear.linear(h, weights.gdn_in, stream=stream)
        ba = linear.linear(h, weights.gdn_ba, stream=stream)
        mixed, z, b, a = gdn_projection.split_projection(
            qkvz,
            ba,
            q_heads=GDN_Q,
            value_heads=GDN_V,
            key_dim=GDN_DK,
            value_dim=GDN_DV,
            stream=stream,
        )
        conv_state = torch.zeros(4, 3, CONV_CHANNELS, device="cuda", dtype=BF16)
        conv_idx = torch.zeros(TOKENS, device="cuda", dtype=torch.int32)
        conv_out_rows = []
        for token in range(TOKENS):
            conv_idx.fill_(1)
            conv_out_rows.append(
                causal_conv1d.causal_conv1d(
                    mixed[token : token + 1],
                    weights.conv_w,
                    conv_state,
                    conv_idx[:1],
                    batch_size=1,
                    tokens_per_request=1,
                    stream=stream,
                ).view(1, CONV_CHANNELS)
            )
        mixed = torch.cat(conv_out_rows, dim=0)
        gdn_state = torch.zeros(
            4, GDN_V, GDN_DV, GDN_DK, device="cuda", dtype=torch.float32
        )
        gdn_idx = torch.tensor([1], device="cuda", dtype=torch.int32)
        gdn_rows = []
        for token in range(TOKENS):
            gdn_rows.append(
                gdn.gdn_recurrent(
                    mixed[token : token + 1],
                    a[token : token + 1],
                    b[token : token + 1],
                    weights.gdn_a_log,
                    weights.gdn_dt,
                    gdn_state,
                    gdn_idx,
                    batch_size=1,
                    tokens_per_request=1,
                    stream=stream,
                ).view(1, GDN_V * GDN_DV)
            )
        gdn_h = torch.cat(gdn_rows, dim=0)
        z_flat = z.view(TOKENS, GDN_V * GDN_DV)
        gdn_h = gated_rmsnorm.gated_rmsnorm(
            gdn_h,
            z_flat,
            torch.ones(GDN_V * GDN_DV, device="cuda", dtype=BF16),
            stream=stream,
        )
        attn_h = linear.linear(gdn_h, weights.gdn_out, stream=stream)
    else:
        q = linear.linear(h, weights.wq0, stream=stream)
        k = linear.linear(h, weights.wk0, stream=stream)
        v = linear.linear(h, weights.wv0, stream=stream)
        q = rope.rope(
            q, weights.cos.repeat(1, Q_HEADS), weights.sin.repeat(1, Q_HEADS), stream=stream
        )
        k = rope.rope(
            k,
            weights.cos.repeat(1, KV_HEADS),
            weights.sin.repeat(1, KV_HEADS),
            stream=stream,
        )
        attn_h = _multihead_attention(q, k, v, stream)
        gate0 = linear.linear(h, weights.attn_gate0, stream=stream)
        attn_h = sigmoid_mul.sigmoid_mul(attn_h, gate0, stream=stream)
        attn_h = linear.linear(attn_h, weights.wo0, stream=stream)
    h = (
        elementwise.add(residual, attn_h)
        if elementwise is not None
        else residual + attn_h
    )
    h, residual = fused_add_rmsnorm.fused_add_rmsnorm(
        torch.zeros_like(h), h, weights.n_mlp0, stream=stream
    )
    gate = linear.linear(h, weights.wg0, stream=stream)
    up = linear.linear(h, weights.wu0, stream=stream)
    hidden = silu_and_mul.silu_and_mul(gate, up, stream=stream)
    projected = linear.linear(hidden, weights.wd0, stream=stream)
    h = (
        elementwise.add(residual, projected)
        if elementwise is not None
        else residual + projected
    )

    # Layer 1: same full-attention mixer (rope + SDPA) for inductor A/B.
    residual = h
    h, residual = fused_add_rmsnorm.fused_add_rmsnorm(
        torch.zeros_like(h), h, weights.n1, stream=stream
    )
    q = linear.linear(h, weights.wq0, stream=stream)
    k = linear.linear(h, weights.wk0, stream=stream)
    v = linear.linear(h, weights.wv0, stream=stream)
    q = rope.rope(
        q, weights.cos.repeat(1, Q_HEADS), weights.sin.repeat(1, Q_HEADS), stream=stream
    )
    k = rope.rope(
        k,
        weights.cos.repeat(1, KV_HEADS),
        weights.sin.repeat(1, KV_HEADS),
        stream=stream,
    )
    attn_h = _multihead_attention(q, k, v, stream)
    gate1 = linear.linear(h, weights.attn_gate1, stream=stream)
    attn_h = sigmoid_mul.sigmoid_mul(attn_h, gate1, stream=stream)
    attn_h = linear.linear(attn_h, weights.wo1, stream=stream)
    h = (
        elementwise.add(residual, attn_h)
        if elementwise is not None
        else residual + attn_h
    )
    h, residual = fused_add_rmsnorm.fused_add_rmsnorm(
        torch.zeros_like(h), h, weights.n_mlp1, stream=stream
    )
    gate = linear.linear(h, weights.wg1, stream=stream)
    up = linear.linear(h, weights.wu1, stream=stream)
    hidden = silu_and_mul.silu_and_mul(gate, up, stream=stream)
    projected = linear.linear(hidden, weights.wd1, stream=stream)
    h = (
        elementwise.add(residual, projected)
        if elementwise is not None
        else residual + projected
    )
    stream.synchronize()
    return h


def two_layer_torch(weights: TwoLayerWeights, *, use_gdn: bool) -> torch.Tensor:
    h = weights.embed[weights.tokens]
    residual = h
    summed = h
    h = _gemma_rms(summed, weights.n0)
    if use_gdn:
        qkvz = h.float() @ weights.gdn_in.float().T
        ba = h.float() @ weights.gdn_ba.float().T
        mixed_width = CONV_CHANNELS
        mixed = qkvz[:, :mixed_width].to(BF16)
        z = qkvz[:, mixed_width:].to(BF16)
        b = ba[:, :GDN_V].to(BF16)
        a = ba[:, GDN_V:].to(BF16)
        history = torch.zeros(3, CONV_CHANNELS, device="cuda", dtype=torch.float32)
        conv_rows = []
        for token in range(TOKENS):
            packed = torch.cat((history, mixed[token : token + 1].float()), dim=0)
            acc = sum(packed[i] * weights.conv_w[:, 3 - i].float() for i in range(4))
            conv_rows.append(F.silu(acc).unsqueeze(0))
            history = torch.cat((history[1:], mixed[token : token + 1].float()), dim=0)
        mixed = torch.cat(conv_rows, dim=0).to(BF16)
        current = torch.zeros(GDN_V, GDN_DV, GDN_DK, device="cuda", dtype=torch.float32)
        gdn_rows = []
        groups = GDN_V // GDN_Q
        for token in range(TOKENS):
            packed = mixed[token]
            query = packed[: GDN_Q * GDN_DK].view(GDN_Q, GDN_DK).float()
            key = packed[GDN_Q * GDN_DK : 2 * GDN_Q * GDN_DK].view(GDN_Q, GDN_DK).float()
            value = packed[2 * GDN_Q * GDN_DK :].view(GDN_V, GDN_DV).float()
            query = query / torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6)
            key = key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)
            query = query.repeat_interleave(groups, dim=0) / math.sqrt(GDN_DK)
            key = key.repeat_interleave(groups, dim=0)
            log_decay = -torch.exp(weights.gdn_a_log) * F.softplus(
                a[token].float() + weights.gdn_dt
            )
            current = current * torch.exp(log_decay)[:, None, None]
            residual_s = value - torch.einsum("hvk,hk->hv", current, key)
            beta = torch.sigmoid(b[token].float())
            current = current + (residual_s * beta[:, None])[:, :, None] * key[:, None, :]
            gdn_rows.append(torch.einsum("hk,hvk->hv", query, current).reshape(1, -1))
        gdn_h = torch.cat(gdn_rows, dim=0).to(BF16)
        gate = z.view(TOKENS, GDN_V * GDN_DV)
        rms = gdn_h.float() * torch.rsqrt(
            gdn_h.float().square().mean(-1, keepdim=True) + 1.0e-6
        )
        gdn_h = (rms * F.silu(gate.float())).to(BF16)
        attn_h = (gdn_h.float() @ weights.gdn_out.float().T).to(BF16)
    else:
        q = (h.float() @ weights.wq0.float().T).to(BF16)
        k = (h.float() @ weights.wk0.float().T).to(BF16)
        v = (h.float() @ weights.wv0.float().T).to(BF16)
        q = _neox_rope(q, weights.cos.repeat(1, Q_HEADS), weights.sin.repeat(1, Q_HEADS))
        k = _neox_rope(
            k, weights.cos.repeat(1, KV_HEADS), weights.sin.repeat(1, KV_HEADS)
        )
        attn_h = _multihead_sdpa(q, k, v)
        gate = (h.float() @ weights.attn_gate0.float().T).to(BF16)
        attn_h = (attn_h.float() * torch.sigmoid(gate.float())).to(BF16)
        attn_h = (attn_h.float() @ weights.wo0.float().T).to(BF16)
    h = residual + attn_h
    residual = h
    h = _gemma_rms(h, weights.n_mlp0)
    gate = (h.float() @ weights.wg0.float().T).to(BF16)
    up = (h.float() @ weights.wu0.float().T).to(BF16)
    h = residual + (
        (F.silu(gate.float()) * up.float()).to(BF16).float() @ weights.wd0.float().T
    ).to(BF16)

    residual = h
    h = _gemma_rms(h, weights.n1)
    q = (h.float() @ weights.wq0.float().T).to(BF16)
    k = (h.float() @ weights.wk0.float().T).to(BF16)
    v = (h.float() @ weights.wv0.float().T).to(BF16)
    q = _neox_rope(q, weights.cos.repeat(1, Q_HEADS), weights.sin.repeat(1, Q_HEADS))
    k = _neox_rope(k, weights.cos.repeat(1, KV_HEADS), weights.sin.repeat(1, KV_HEADS))
    attn_h = _multihead_sdpa(q, k, v)
    gate = (h.float() @ weights.attn_gate1.float().T).to(BF16)
    attn_h = (attn_h.float() * torch.sigmoid(gate.float())).to(BF16)
    attn_h = (attn_h.float() @ weights.wo1.float().T).to(BF16)
    h = residual + attn_h
    residual = h
    h = _gemma_rms(h, weights.n_mlp1)
    gate = (h.float() @ weights.wg1.float().T).to(BF16)
    up = (h.float() @ weights.wu1.float().T).to(BF16)
    h = residual + (
        (F.silu(gate.float()) * up.float()).to(BF16).float() @ weights.wd1.float().T
    ).to(BF16)
    return h


def inductor_two_layer(weights: TwoLayerWeights) -> tuple[Callable[[], torch.Tensor], str]:
    """Compile the same eager two-layer math through stock TorchInductor."""

    def model() -> torch.Tensor:
        return two_layer_torch(weights, use_gdn=False)

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor", dynamic=False, fullgraph=True)
    try:
        compiled()
        _sync()
        return compiled, "inductor-fullgraph"
    except Exception:
        torch._dynamo.reset()
        compiled = torch.compile(
            model, backend="inductor", dynamic=False, fullgraph=False
        )
        compiled()
        _sync()
        return compiled, "inductor"


def run(*, warmup: int, timed: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if _cc() != 89:
        raise RuntimeError(f"live GPU compute capability is {_cc()}, need 89")
    bootstrap()
    try:
        import pypto_plugins.torch_inductor as torch_inductor
        from pypto_plugins.activity_trace import clear_artifact_registry_for_testing

        torch_inductor.uninstall()
        torch._dynamo.reset()
        clear_artifact_registry_for_testing()
    except Exception:
        pass
    stream = _stream()
    torch.manual_seed(21)
    cases = [_record(name, lambda fn=fn: fn(stream)) for name, fn in CENSUS]
    passed = [case for case in cases if case.get("ok") and case.get("correct")]
    failed = [case["case"] for case in cases if not (case.get("ok") and case.get("correct"))]
    gdn_ok = all(
        name in {case["case"] for case in passed}
        for name in ("gdn_projection", "causal_conv1d", "gdn_recurrent", "gated_rmsnorm")
    )
    core_ok = all(
        name in {case["case"] for case in passed}
        for name in (
            "embedding",
            "fused_add_rmsnorm",
            "linear",
            "rope",
            "attention",
            "sigmoid_mul",
            "silu_and_mul",
        )
    )
    weights = TwoLayerWeights()
    two_layer: dict[str, Any] = {"core_ok": core_ok, "gdn_layer": gdn_ok}
    if core_ok:
      try:
        from pypto_plugins.torch import scheduling as inductor_scheduling
        import pypto_plugins.torch_inductor as torch_inductor

        inductor_scheduling.REGISTRY.clear()
        elementwise = ElementwiseInductorPypto()
        elementwise.warmup(
            torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16),
            torch.randn(TOKENS, Q_HEADS * HEAD_DIM, device="cuda", dtype=BF16),
        )
        pypto_out = two_layer_pypto(
            weights, stream, use_gdn=False, elementwise=elementwise
        )
        eager_out = two_layer_torch(weights, use_gdn=False)
        two_layer["pypto_vs_eager"] = {
            "max_abs_diff": _diff(pypto_out, eager_out),
            "correct": _close(pypto_out, eager_out),
            "finite": _finite(pypto_out) and _finite(eager_out),
        }
        two_layer["pypto_ms"] = _timed_ms(
            lambda: two_layer_pypto(
                weights, stream, use_gdn=False, elementwise=elementwise
            ),
            warmup,
            timed,
        )
        two_layer["inductor_pypto_elementwise"] = {
            "kernels": [
                {
                    "registry_name": name,
                    "kernel_name": artifact.kernel_name,
                    "entry_name": artifact.entry_name,
                    "cubin_bytes": artifact.cubin_bytes,
                    "fallback_used": artifact.fallback_used,
                }
                for name, artifact in inductor_scheduling.REGISTRY.snapshot()
            ],
            "all_native": all(
                not artifact.fallback_used
                and artifact.cubin_bytes > 0
                and artifact.entry_name == "pypto_fused_pointwise"
                for _name, artifact in inductor_scheduling.REGISTRY.snapshot()
            ),
        }
        two_layer["eager_ms"] = _timed_ms(
            lambda: two_layer_torch(weights, use_gdn=False), warmup, timed
        )
        inductor: dict[str, Any]
        try:
            compiled, backend = inductor_two_layer(weights)
            inductor_out = compiled()
            _sync()
            inductor = {
                "backend": backend,
                "ok": True,
                "max_abs_diff_vs_eager": _diff(inductor_out, eager_out),
                "correct_vs_eager": _close(inductor_out, eager_out),
                "max_abs_diff_vs_pypto": _diff(inductor_out, pypto_out),
                "correct_vs_pypto": _close(inductor_out, pypto_out),
                "ms": _timed_ms(lambda: compiled(), warmup, timed),
            }
        except Exception as error:  # noqa: BLE001
            inductor = {
                "backend": "inductor",
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-1500:],
            }
        two_layer["inductor"] = inductor
        if inductor.get("ok") and two_layer["pypto_ms"]:
            two_layer["speedup_vs_inductor"] = (
                inductor["ms"] / two_layer["pypto_ms"]
                if two_layer["pypto_ms"]
                else None
            )
            two_layer["speedup_vs_eager"] = (
                two_layer["eager_ms"] / two_layer["pypto_ms"]
                if two_layer["pypto_ms"]
                else None
            )
        if gdn_ok:
            try:
                gdn_pypto = two_layer_pypto(weights, stream, use_gdn=True)
                gdn_eager = two_layer_torch(weights, use_gdn=True)
                two_layer["gdn_pypto_vs_eager"] = {
                    "max_abs_diff": _diff(gdn_pypto, gdn_eager),
                    "correct": _close(gdn_pypto, gdn_eager),
                    "finite": _finite(gdn_pypto) and _finite(gdn_eager),
                }
            except Exception as error:  # noqa: BLE001
                two_layer["gdn_pypto_vs_eager"] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
      except Exception as error:  # noqa: BLE001
        two_layer["error"] = f"{type(error).__name__}: {error}"
        two_layer["traceback"] = traceback.format_exc()[-1500:]
      finally:
        try:
            import pypto_plugins.torch_inductor as torch_inductor

            torch_inductor.uninstall()
        except Exception:
            pass
    else:
        two_layer["skipped"] = "core operators failed; two-layer composition not run"

    from pypto.compiler import get_nvidia_backend_build_info
    from pypto.runtime import nvidia as runtime
    from pypto_kernels._boot import _live_runtime_expectation

    info = get_nvidia_backend_build_info()
    observation = runtime.observe_current_nvidia_runtime(*_live_runtime_expectation())
    return {
        "schema": 1,
        "kind": "pypto-two-layer-ada-sm89",
        "geometry": {
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "mlp": MLP,
            "vocab": VOCAB,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "gdn_q": GDN_Q,
            "gdn_v": GDN_V,
            "conv_channels": CONV_CHANNELS,
        },
        "live": {
            "compute_capability": observation.target_info.traits.compute_capability,
            "architecture": observation.target_info.architecture,
            "cuda_driver_api_version": observation.cuda_driver_api_version,
            "cuda_runtime_api_version": observation.cuda_runtime_api_version,
            "cuda_runtime_library_path": observation.cuda_runtime_library_path,
        },
        "build_info": {
            "compiled": bool(info.compiled),
            "pypto_revision": info.pypto_revision,
            "tensor_ir_revision": info.tensor_ir_revision,
            "cuda_toolkit_version": info.cuda_toolkit_version,
            "tileiras_version": info.tileiras_version,
            "sm120_target": info.sm120_target,
        },
        "census": cases,
        "census_passed": [case["case"] for case in passed],
        "census_failed": failed,
        "all_operators_correct": not failed,
        "two_layer": two_layer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=10)
    args = parser.parse_args(argv)
    evidence = run(warmup=args.warmup, timed=args.timed)
    text = json.dumps(evidence, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    two = evidence["two_layer"]
    if evidence["census_failed"]:
        print("CENSUS_FAIL", ",".join(evidence["census_failed"]))
    if two.get("error"):
        print("TWO_LAYER_ERROR", two["error"])
        return 1
    if two.get("pypto_vs_eager") and not two["pypto_vs_eager"]["correct"]:
        print("TWO_LAYER_MISMATCH", two["pypto_vs_eager"])
        return 1
    if two.get("inductor", {}).get("ok") is False:
        print("INDUCTOR_FAIL", two["inductor"].get("error"))
        return 1
    if two.get("inductor", {}).get("ok") and not two["inductor"].get("correct_vs_eager"):
        print("INDUCTOR_MISMATCH", two["inductor"])
        return 1
    ew = two.get("inductor_pypto_elementwise")
    if ew is not None and not ew.get("all_native"):
        print("INDUCTOR_PYPTO_NOT_NATIVE", ew)
        return 1
    if evidence["census_failed"] or not two.get("core_ok"):
        return 2
    print("two_layer_ada_sm89: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
