"""Q/K RMSNorm, partial NeoX RoPE and gate split in one PyPTO graph."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_graph, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_EPSILON = 1.0e-6
_lock = threading.RLock()
_cache: dict[tuple[int, ...], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def qk_rmsnorm_rope_gate_kernel(
    q_gate: pl.Tensor,
    key: pl.Tensor,
    q_weight: pl.Tensor,
    k_weight: pl.Tensor,
    cos_sin_cache: pl.Tensor,
    repeated_positions: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Normalize Q/K heads, rotate their prefix, and copy Q gates."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for token in pl.range(q_gate.shape[0]):
            for q_head in pl.range(
                q_gate.shape[1] // (2 * q_weight.shape[1])
            ):
                q = pl.load(
                    q_gate,
                    [token, q_head * 2 * q_weight.shape[1]],
                    [1, q_weight.shape[1]],
                )
                gate = pl.load(
                    q_gate,
                    [token, (q_head * 2 + 1) * q_weight.shape[1]],
                    [1, q_weight.shape[1]],
                )
                weight = pl.load(
                    q_weight, [0, 0], [1, q_weight.shape[1]]
                )
                q_wide = pl.cast(q, target_type=pl.FP32)
                q_square = pl.mul(q_wide, q_wide)
                q_scratch = pl.create_tile(
                    [1, q_weight.shape[1]],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                q_sum = pl.row_sum(q_square, q_scratch)
                q_mean = pl.mul(q_sum, 1.0 / q_weight.shape[1])
                q_shifted = pl.add(q_mean, 1.0e-6)
                q_inv = pl.rsqrt(q_shifted)
                q_norm = pl.row_expand_mul(q_wide, q_inv)
                q_weight_wide = pl.cast(weight, target_type=pl.FP32)
                q_scale = pl.add(q_weight_wide, 1.0)
                q_scaled = pl.mul(q_norm, q_scale)
                q_rounded = pl.cast(q_scaled, target_type=pl.BF16)
                q_ready = pl.cast(q_rounded, target_type=pl.FP32)

                position = pl.read(repeated_positions, [token, 0])
                cache = pl.load(
                    cos_sin_cache,
                    [position, 0],
                    [1, cos_sin_cache.shape[1]],
                )
                cos = pl.tile.slice(
                    cache, [1, cos_sin_cache.shape[1] // 2], [0, 0]
                )
                sin = pl.tile.slice(
                    cache,
                    [1, cos_sin_cache.shape[1] // 2],
                    [0, cos_sin_cache.shape[1] // 2],
                )
                q_low = pl.tile.slice(
                    q_ready, [1, cos_sin_cache.shape[1] // 2], [0, 0]
                )
                q_high = pl.tile.slice(
                    q_ready,
                    [1, cos_sin_cache.shape[1] // 2],
                    [0, cos_sin_cache.shape[1] // 2],
                )
                q_tail = pl.tile.slice(
                    q_ready,
                    [1, q_weight.shape[1] - cos_sin_cache.shape[1]],
                    [0, cos_sin_cache.shape[1]],
                )
                q_rot_low = pl.sub(pl.mul(q_low, cos), pl.mul(q_high, sin))
                q_rot_high = pl.add(pl.mul(q_high, cos), pl.mul(q_low, sin))
                q_rotary = pl.tile.concat(q_rot_low, q_rot_high)
                q_complete = pl.tile.concat(q_rotary, q_tail)
                q_result = pl.cast(q_complete, target_type=pl.BF16)
                pl.store(
                    q_result,
                    [token, q_head, 0],
                    out,
                )
                pl.store(
                    gate,
                    [
                        token,
                        q_gate.shape[1] // (2 * q_weight.shape[1])
                        + key.shape[1] // k_weight.shape[1]
                        + q_head,
                        0,
                    ],
                    out,
                )

            for k_head in pl.range(key.shape[1] // k_weight.shape[1]):
                k = pl.load(
                    key,
                    [token, k_head * k_weight.shape[1]],
                    [1, k_weight.shape[1]],
                )
                weight = pl.load(
                    k_weight, [0, 0], [1, k_weight.shape[1]]
                )
                k_wide = pl.cast(k, target_type=pl.FP32)
                k_square = pl.mul(k_wide, k_wide)
                k_scratch = pl.create_tile(
                    [1, k_weight.shape[1]],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                k_sum = pl.row_sum(k_square, k_scratch)
                k_mean = pl.mul(k_sum, 1.0 / k_weight.shape[1])
                k_shifted = pl.add(k_mean, 1.0e-6)
                k_inv = pl.rsqrt(k_shifted)
                k_norm = pl.row_expand_mul(k_wide, k_inv)
                k_weight_wide = pl.cast(weight, target_type=pl.FP32)
                k_scale = pl.add(k_weight_wide, 1.0)
                k_scaled = pl.mul(k_norm, k_scale)
                k_rounded = pl.cast(k_scaled, target_type=pl.BF16)
                k_ready = pl.cast(k_rounded, target_type=pl.FP32)

                position = pl.read(repeated_positions, [token, 0])
                cache = pl.load(
                    cos_sin_cache,
                    [position, 0],
                    [1, cos_sin_cache.shape[1]],
                )
                cos = pl.tile.slice(
                    cache, [1, cos_sin_cache.shape[1] // 2], [0, 0]
                )
                sin = pl.tile.slice(
                    cache,
                    [1, cos_sin_cache.shape[1] // 2],
                    [0, cos_sin_cache.shape[1] // 2],
                )
                k_low = pl.tile.slice(
                    k_ready, [1, cos_sin_cache.shape[1] // 2], [0, 0]
                )
                k_high = pl.tile.slice(
                    k_ready,
                    [1, cos_sin_cache.shape[1] // 2],
                    [0, cos_sin_cache.shape[1] // 2],
                )
                k_tail = pl.tile.slice(
                    k_ready,
                    [1, k_weight.shape[1] - cos_sin_cache.shape[1]],
                    [0, cos_sin_cache.shape[1]],
                )
                k_rot_low = pl.sub(pl.mul(k_low, cos), pl.mul(k_high, sin))
                k_rot_high = pl.add(pl.mul(k_high, cos), pl.mul(k_low, sin))
                k_rotary = pl.tile.concat(k_rot_low, k_rot_high)
                k_complete = pl.tile.concat(k_rotary, k_tail)
                k_result = pl.cast(k_complete, target_type=pl.BF16)
                pl.store(
                    k_result,
                    [
                        token,
                        q_gate.shape[1] // (2 * q_weight.shape[1]) + k_head,
                        0,
                    ],
                    out,
                )
    return out


def _validate_shape(
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
) -> None:
    if min(tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions) <= 0:
        raise ValueError("QK preparation needs positive static dimensions")
    if head_dim % 128 or rotary_dim % 2 or rotary_dim >= head_dim:
        raise ValueError(
            "QK preparation needs head_dim divisible by 128 and an even "
            "partial rotary_dim smaller than head_dim"
        )


def build(
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
) -> Any:
    _validate_shape(tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions)
    import torch

    q_gate = torch.empty(
        (tokens, 2 * q_heads * head_dim), dtype=torch.bfloat16, device="meta"
    )
    key = torch.empty(
        (tokens, kv_heads * head_dim), dtype=torch.bfloat16, device="meta"
    )
    weight = torch.empty((1, head_dim), dtype=torch.bfloat16, device="meta")
    cache = torch.empty(
        (max_positions, rotary_dim), dtype=torch.bfloat16, device="meta"
    )
    positions = torch.empty(
        (tokens, rotary_dim), dtype=torch.int64, device="meta"
    )
    out = torch.empty(
        (tokens, 2 * q_heads + kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="meta",
    )
    return qk_rmsnorm_rope_gate_kernel.specialize(
        q_gate, key, weight, weight, cache, positions, out
    )


def compile_for(
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
) -> str:
    _validate_shape(tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions)
    cache_key = (tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    program = build(*cache_key)
    graph_key = compile_graph(program, [1, 1, 1, 1, 1, 32])
    with _lock:
        _cache[cache_key] = graph_key
    return graph_key


def qk_rmsnorm_rope_gate(
    q_gate: Any,
    key: Any,
    q_weight: Any,
    k_weight: Any,
    cos_sin_cache: Any,
    positions: Any,
    *,
    q_heads: int,
    kv_heads: int,
    stream: Any = None,
) -> tuple[Any, Any, Any]:
    """Return Q, K and gate views backed by one packed graph output."""

    import torch

    tensors = (q_gate, key, q_weight, k_weight, cos_sin_cache)
    if any(
        tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("QK preparation needs contiguous BF16 tensor inputs")
    if positions.ndim != 1 or positions.dtype is not torch.int64:
        raise ValueError("QK preparation positions must be rank-1 INT64")
    tokens = int(q_gate.shape[0])
    head_dim = int(q_weight.numel())
    rotary_dim = int(cos_sin_cache.shape[1])
    max_positions = int(cos_sin_cache.shape[0])
    _validate_shape(tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions)
    if tuple(q_gate.shape) != (tokens, 2 * q_heads * head_dim):
        raise ValueError("q_gate shape is incompatible with q_heads/head_dim")
    if tuple(key.shape) != (tokens, kv_heads * head_dim):
        raise ValueError("key shape is incompatible with kv_heads/head_dim")
    if tuple(q_weight.shape) != (head_dim,) or tuple(k_weight.shape) != (head_dim,):
        raise ValueError("Q/K weights must be flat head_dim vectors")
    if stream is None:
        stream = torch.cuda.current_stream(q_gate.device)
    graph_key = compile_for(
        tokens, q_heads, kv_heads, head_dim, rotary_dim, max_positions
    )
    packed = torch.empty(
        (tokens, 2 * q_heads + kv_heads, head_dim),
        dtype=q_gate.dtype,
        device=q_gate.device,
    )
    repeated_positions = positions.as_strided((tokens, rotary_dim), (1, 0))
    launch_graph(
        graph_key,
        (
            q_gate,
            key,
            q_weight.view(1, head_dim),
            k_weight.view(1, head_dim),
            cos_sin_cache,
            repeated_positions,
            packed,
        ),
        stream.cuda_stream,
    )
    q_size = q_heads * head_dim
    k_size = kv_heads * head_dim
    q = packed[:, :q_heads, :].view(tokens, q_size)
    k = packed[:, q_heads : q_heads + kv_heads, :].view(tokens, k_size)
    gate = packed[:, q_heads + kv_heads :, :]
    return q, k, gate
