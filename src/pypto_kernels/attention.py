"""Dense scaled dot-product attention as one native PyPTO tile graph."""

from __future__ import annotations

import math
import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_ROW_TILE = 1
_VALUE_TILE = 64
_lock = threading.RLock()
_cache: dict[tuple[int, int, int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def attention_kernel(
    query: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """QK matmul, stable softmax and probability/value matmul in one graph."""

    with pl.at(level=pl.Level.CORE_GROUP):
        key_tile = pl.load(
            key,
            [0, 0],
            [key.shape[0], key.shape[1]],
            target_memory=pl.MemorySpace.Mat,
        )
        value_tile = pl.load(
            value,
            [0, 0],
            [value.shape[0], value.shape[1]],
            target_memory=pl.MemorySpace.Mat,
        )
        key_transposed = pl.tile.transpose_view(key_tile)
        for row in pl.range(query.shape[0]):
            query_tile = pl.load(
                query,
                [row, 0],
                [1, query.shape[1]],
                target_memory=pl.MemorySpace.Mat,
            )
            score = pl.matmul(
                query_tile,
                key_transposed,
                out_dtype=pl.FP32,
            )
            scaled = pl.mul(score, scale)
            max_scratch = pl.create_tile(
                [1, key.shape[0]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_max = pl.row_max(scaled, max_scratch)
            centered = pl.row_expand_sub(scaled, row_max)
            exponent = pl.exp(centered)
            sum_scratch = pl.create_tile(
                [1, key.shape[0]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_sum = pl.row_sum(exponent, sum_scratch)
            probability = pl.row_expand_div(exponent, row_sum)
            probability_bf16 = pl.cast(probability, target_type=pl.BF16)
            mixed = pl.matmul(probability_bf16, value_tile, out_dtype=pl.FP32)
            result = pl.cast(mixed, target_type=pl.BF16)
            pl.store(result, [row, 0], out)
    return out


def _validate_shape(rows: int, tokens: int, head_dim: int, value_dim: int) -> None:
    if (
        rows <= 0
        or tokens <= 0
        or head_dim <= 0
        or value_dim <= 0
        or head_dim % 128
        or tokens % 16
        or value_dim % 16
    ):
        raise ValueError(
            "attention needs positive dimensions, head_dim divisible by 128, "
            "and token/value dimensions divisible by 16"
        )


def build(rows: int, tokens: int, head_dim: int, value_dim: int) -> Any:
    _validate_shape(rows, tokens, head_dim, value_dim)
    import torch

    query = torch.empty((rows, head_dim), dtype=torch.bfloat16, device="meta")
    key = torch.empty((tokens, head_dim), dtype=torch.bfloat16, device="meta")
    value = torch.empty((tokens, value_dim), dtype=torch.bfloat16, device="meta")
    out = torch.empty((rows, value_dim), dtype=torch.bfloat16, device="meta")
    return attention_kernel.specialize(
        query, key, value, 1.0 / math.sqrt(head_dim), out
    )


def compile_for(rows: int, tokens: int, head_dim: int, value_dim: int) -> str:
    _validate_shape(rows, tokens, head_dim, value_dim)
    shape_key = (rows, tokens, head_dim, value_dim)
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    query = torch.empty((rows, head_dim), dtype=torch.bfloat16, device="meta")
    key = torch.empty((tokens, head_dim), dtype=torch.bfloat16, device="meta")
    value = torch.empty((tokens, value_dim), dtype=torch.bfloat16, device="meta")
    out = torch.empty((rows, value_dim), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        attention_kernel,
        (query, key, value, 1.0 / math.sqrt(head_dim), out),
        [_ROW_TILE, _VALUE_TILE],
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def attention(query: Any, key: Any, value: Any, stream: Any = None) -> Any:
    """Return dense attention from one native tile graph launch."""

    import torch

    tensors = (query, key, value)
    if any(
        tensor.ndim != 2
        or tensor.dtype is not torch.bfloat16
        or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("attention needs contiguous rank-2 BF16 Q/K/V")
    rows, head_dim = (int(query.shape[0]), int(query.shape[1]))
    tokens = int(key.shape[0])
    value_dim = int(value.shape[1])
    if tuple(key.shape) != (tokens, head_dim) or int(value.shape[0]) != tokens:
        raise ValueError("attention Q/K/V dimensions are incompatible")
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    graph_key = compile_for(rows, tokens, head_dim, value_dim)
    out = torch.empty((rows, value_dim), dtype=torch.bfloat16, device=query.device)
    launch_graph(graph_key, (query, key, value, out), stream.cuda_stream)
    return out


def status(
    rows: int = 32,
    tokens: int = 128,
    head_dim: int = 128,
    value_dim: int = 128,
) -> dict[str, str]:
    try:
        return {
            "status": "compiled",
            "key": compile_for(rows, tokens, head_dim, value_dim),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
