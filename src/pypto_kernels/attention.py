"""Dense scaled dot-product attention as one native PyPTO tile graph."""

from __future__ import annotations

import math
import threading
from typing import Any

from ._boot import bootstrap, compile_graph, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_ROW_TILE = 1
_VALUE_TILE = 64
_lock = threading.RLock()
_cache: dict[tuple[int, int, int, int], str] = {}
_paged_decode_cache: dict[tuple[int, int, int, int, int], str] = {}

STATUS = "native-tile executable"
PAGED_DECODE_STATUS = "native-tile source candidate"
GRAPHS = 2


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


@pl.jit
def paged_attention_decode_kernel(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    physical_indices: pl.Tensor,
    valid_tokens: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Gather one request's paged KV rows and mask its static KV bucket."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for q_head in pl.range(query.shape[0]):
            kv_heads = key_cache.shape[1] // query.shape[1]
            queries_per_kv = query.shape[0] // kv_heads
            kv_head = q_head // queries_per_kv

            keys = pl.tile.create(
                [query.shape[1], physical_indices.shape[0]],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
                transpose=True,
            )
            values = pl.tile.create(
                [physical_indices.shape[0], query.shape[1]],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
            )
            for slot in pl.range(physical_indices.shape[0]):
                physical = pl.read(physical_indices, [slot, 0])
                cache_column = kv_head * query.shape[1]
                keys = pl.tile.gather_row(
                    keys,
                    key_cache,
                    [0, slot],
                    [physical, cache_column],
                    [1, query.shape[1]],
                    transpose=True,
                )
                values = pl.tile.gather_row(
                    values,
                    value_cache,
                    [slot, 0],
                    [physical, cache_column],
                    [1, query.shape[1]],
                )

            query_tile = pl.load(
                query,
                [q_head, 0],
                [1, query.shape[1]],
                target_memory=pl.MemorySpace.Mat,
            )
            score = pl.matmul(query_tile, keys, out_dtype=pl.FP32)
            scaled = pl.mul(score, scale)
            positions = pl.tile.ci(
                0,
                [1, physical_indices.shape[0]],
                dtype=pl.INT32,
            )
            valid_token_count = pl.read(valid_tokens, [0, 0])
            valid_mask = pl.cmps(positions, valid_token_count, cmp_type=2)
            mask_scratch = pl.tile.create(
                [1, 32], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec
            )
            masked_scaled = pl.sels(
                valid_mask,
                scaled,
                mask_scratch,
                -3.4028234663852886e38,
            )
            max_scratch = pl.create_tile(
                [1, physical_indices.shape[0]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_max = pl.row_max(masked_scaled, max_scratch)
            centered = pl.row_expand_sub(masked_scaled, row_max)
            exponent = pl.exp(centered)
            sum_scratch = pl.create_tile(
                [1, physical_indices.shape[0]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_sum = pl.row_sum(exponent, sum_scratch)
            probability = pl.row_expand_div(exponent, row_sum)
            probability_bf16 = pl.cast(probability, target_type=pl.BF16)
            mixed = pl.matmul(probability_bf16, values, out_dtype=pl.FP32)
            result = pl.cast(mixed, target_type=pl.BF16)
            pl.store(result, [q_head, 0], out)
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


def _validate_paged_decode_shape(
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
) -> None:
    if (
        q_heads <= 0
        or kv_heads <= 0
        or tokens <= 0
        or head_dim <= 0
        or cache_rows <= 0
        or q_heads % kv_heads
        or head_dim % 128
        or tokens % 16
    ):
        raise ValueError(
            "paged decode needs positive dimensions, q_heads divisible by "
            "kv_heads, head_dim divisible by 128 and a 16-token KV bucket"
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


def build_paged_decode(
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
) -> Any:
    _validate_paged_decode_shape(q_heads, kv_heads, tokens, head_dim, cache_rows)
    import torch

    query = torch.empty((q_heads, head_dim), dtype=torch.bfloat16, device="meta")
    key_cache = torch.empty(
        (cache_rows, kv_heads * head_dim), dtype=torch.bfloat16, device="meta"
    )
    value_cache = torch.empty_like(key_cache)
    physical_indices = torch.empty(
        (tokens, head_dim), dtype=torch.int32, device="meta"
    )
    valid_tokens = torch.empty((1, tokens), dtype=torch.int32, device="meta")
    out = torch.empty_like(query)
    return paged_attention_decode_kernel.specialize(
        query,
        key_cache,
        value_cache,
        physical_indices,
        valid_tokens,
        1.0 / math.sqrt(head_dim),
        out,
    )


def compile_paged_decode_for(
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
) -> str:
    _validate_paged_decode_shape(q_heads, kv_heads, tokens, head_dim, cache_rows)
    shape_key = (q_heads, kv_heads, tokens, head_dim, cache_rows)
    cached = _paged_decode_cache.get(shape_key)
    if cached is not None:
        return cached
    program = build_paged_decode(*shape_key)
    graph_key = compile_graph(program, [_ROW_TILE, _VALUE_TILE])
    with _lock:
        _paged_decode_cache[shape_key] = graph_key
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


def paged_attention_decode(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    physical_indices: Any,
    valid_tokens: Any,
    *,
    kv_heads: int,
    stream: Any = None,
) -> Any:
    """Run one request's GQA decode over physical SGLang KV-pool rows."""

    import torch

    if (
        query.ndim != 2
        or key_cache.ndim != 2
        or value_cache.ndim != 2
        or any(
            tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous()
            for tensor in (query, key_cache, value_cache)
        )
    ):
        raise ValueError("paged decode needs contiguous rank-2 BF16 Q/K/V storage")
    if (
        physical_indices.ndim != 1
        or physical_indices.dtype is not torch.int32
        or not physical_indices.is_contiguous()
    ):
        raise ValueError("paged decode physical indices must be contiguous rank-1 INT32")
    if (
        valid_tokens.ndim != 1
        or valid_tokens.numel() != 1
        or valid_tokens.dtype is not torch.int32
        or not valid_tokens.is_contiguous()
    ):
        raise ValueError(
            "paged decode valid token count must be one contiguous INT32 element"
        )
    if valid_tokens.device != query.device or physical_indices.device != query.device:
        raise ValueError("paged decode metadata must be on the query device")
    q_heads, head_dim = map(int, query.shape)
    cache_rows = int(key_cache.shape[0])
    tokens = int(physical_indices.shape[0])
    _validate_paged_decode_shape(q_heads, kv_heads, tokens, head_dim, cache_rows)
    expected_cache_shape = (cache_rows, kv_heads * head_dim)
    if tuple(key_cache.shape) != expected_cache_shape or tuple(
        value_cache.shape
    ) != expected_cache_shape:
        raise ValueError("paged decode cache shape is incompatible with KV heads")
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    graph_key = compile_paged_decode_for(
        q_heads, kv_heads, tokens, head_dim, cache_rows
    )
    out = torch.empty_like(query)
    launch_graph(
        graph_key,
        (
            query,
            key_cache,
            value_cache,
            physical_indices.as_strided((tokens, head_dim), (1, 0)),
            valid_tokens.as_strided((1, tokens), (0, 0)),
            out,
        ),
        stream.cuda_stream,
    )
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
