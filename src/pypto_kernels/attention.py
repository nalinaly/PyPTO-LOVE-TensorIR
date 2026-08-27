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
_paged_decode_cache: dict[tuple[int, int, int, int, int, int, int], str] = {}
_paged_cache_write_cache: dict[tuple[int, int], str] = {}

STATUS = "native-tile executable"
PAGED_DECODE_STATUS = "native-tile source candidate"
GRAPHS = 3


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
    req_to_token: pl.Tensor,
    request_index: pl.Tensor,
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
            request_id = pl.read(request_index, [0, 0])
            valid_token_count_i64 = pl.read(valid_tokens, [0, 0])
            valid_token_count = pl.cast(valid_token_count_i64, pl.INT32)

            keys = pl.tile.create(
                [query.shape[1], request_index.shape[1]],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
                transpose=True,
            )
            values = pl.tile.create(
                [request_index.shape[1], query.shape[1]],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
            )
            for slot in pl.range(request_index.shape[1]):
                physical = pl.read(req_to_token, [request_id, slot])
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
                [1, request_index.shape[1]],
                dtype=pl.INT32,
            )
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
                [1, request_index.shape[1]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_max = pl.row_max(masked_scaled, max_scratch)
            centered = pl.row_expand_sub(masked_scaled, row_max)
            exponent = pl.exp(centered)
            sum_scratch = pl.create_tile(
                [1, request_index.shape[1]],
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


@pl.jit
def paged_cache_write_kernel(
    key_cache: pl.InOut[pl.Tensor],
    value_cache: pl.InOut[pl.Tensor],
    physical_row: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Write one flattened GQA K/V row to the selected physical cache slot."""

    with pl.at(level=pl.Level.CORE_GROUP):
        physical_row_i64 = pl.read(physical_row, [0, 0])
        physical_row_i32 = pl.cast(physical_row_i64, pl.INT32)
        key_tile = pl.load(
            key, [0, 0], [1, key.shape[1]], target_memory=pl.MemorySpace.Vec
        )
        value_tile = pl.load(
            value, [0, 0], [1, value.shape[1]], target_memory=pl.MemorySpace.Vec
        )
        pl.store(key_tile, [physical_row_i32, 0], key_cache)
        pl.store(value_tile, [physical_row_i32, 0], value_cache)
        anchor = pl.add(key_tile, value_tile)
        pl.store(anchor, [0, 0], out)
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
    request_rows: int,
    max_context_len: int,
) -> None:
    if (
        q_heads <= 0
        or kv_heads <= 0
        or tokens <= 0
        or head_dim <= 0
        or cache_rows <= 0
        or request_rows <= 0
        or max_context_len < tokens
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
    request_rows: int,
    max_context_len: int,
) -> Any:
    _validate_paged_decode_shape(
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
    )
    import torch

    query = torch.empty((q_heads, head_dim), dtype=torch.bfloat16, device="meta")
    key_cache = torch.empty(
        (cache_rows, kv_heads * head_dim), dtype=torch.bfloat16, device="meta"
    )
    value_cache = torch.empty_like(key_cache)
    req_to_token = torch.empty(
        (request_rows, max_context_len), dtype=torch.int32, device="meta"
    )
    request_index = torch.empty((1, tokens), dtype=torch.int64, device="meta")
    valid_tokens = torch.empty((1, tokens), dtype=torch.int64, device="meta")
    out = torch.empty_like(query)
    return paged_attention_decode_kernel.specialize(
        query,
        key_cache,
        value_cache,
        req_to_token,
        request_index,
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
    request_rows: int,
    max_context_len: int,
) -> str:
    _validate_paged_decode_shape(
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
    )
    shape_key = (
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
    )
    cached = _paged_decode_cache.get(shape_key)
    if cached is not None:
        return cached
    program = build_paged_decode(*shape_key)
    graph_key = compile_graph(program, [_ROW_TILE, _VALUE_TILE])
    with _lock:
        _paged_decode_cache[shape_key] = graph_key
    return graph_key


def _validate_paged_cache_write_shape(cache_rows: int, row_width: int) -> None:
    if cache_rows <= 0 or row_width <= 0 or row_width % 128:
        raise ValueError(
            "paged cache write needs positive dimensions and row width "
            "divisible by 128"
        )


def build_paged_cache_write(cache_rows: int, row_width: int) -> Any:
    _validate_paged_cache_write_shape(cache_rows, row_width)
    import torch

    key_cache = torch.empty(
        (cache_rows, row_width), dtype=torch.bfloat16, device="meta"
    )
    value_cache = torch.empty_like(key_cache)
    physical_row = torch.empty((1, row_width), dtype=torch.int64, device="meta")
    key = torch.empty((1, row_width), dtype=torch.bfloat16, device="meta")
    value = torch.empty_like(key)
    out = torch.empty_like(key)
    return paged_cache_write_kernel.specialize(
        key_cache, value_cache, physical_row, key, value, out
    )


def compile_paged_cache_write_for(cache_rows: int, row_width: int) -> str:
    _validate_paged_cache_write_shape(cache_rows, row_width)
    shape_key = (cache_rows, row_width)
    cached = _paged_cache_write_cache.get(shape_key)
    if cached is not None:
        return cached
    graph_key = compile_graph(build_paged_cache_write(*shape_key), [128])
    with _lock:
        _paged_cache_write_cache[shape_key] = graph_key
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
    req_to_token: Any,
    request_index: Any,
    valid_tokens: Any,
    *,
    kv_heads: int,
    bucket_tokens: int,
    stream: Any = None,
) -> Any:
    """Run one request's GQA decode over a padded physical KV bucket.

    ``request_index`` and ``valid_tokens`` are one-element device views of the
    live request row and sequence length. The valid length must be in
    ``[1, bucket_tokens]``; keeping both values on-device avoids decode-time
    host synchronization and an intermediate ``kv_indices`` kernel.
    """

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
        req_to_token.ndim != 2
        or req_to_token.dtype is not torch.int32
        or not req_to_token.is_contiguous()
    ):
        raise ValueError("paged decode request table must be contiguous rank-2 INT32")
    if (
        request_index.ndim != 1
        or request_index.numel() != 1
        or request_index.dtype is not torch.int64
        or not request_index.is_contiguous()
    ):
        raise ValueError(
            "paged decode request index must be one contiguous INT64 element"
        )
    if (
        valid_tokens.ndim != 1
        or valid_tokens.numel() != 1
        or valid_tokens.dtype is not torch.int64
        or not valid_tokens.is_contiguous()
    ):
        raise ValueError(
            "paged decode valid token count must be one contiguous INT64 element"
        )
    if (
        req_to_token.device != query.device
        or request_index.device != query.device
        or valid_tokens.device != query.device
    ):
        raise ValueError("paged decode metadata must be on the query device")
    q_heads, head_dim = map(int, query.shape)
    cache_rows = int(key_cache.shape[0])
    tokens = int(bucket_tokens)
    request_rows, max_context_len = map(int, req_to_token.shape)
    _validate_paged_decode_shape(
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
    )
    expected_cache_shape = (cache_rows, kv_heads * head_dim)
    if tuple(key_cache.shape) != expected_cache_shape or tuple(
        value_cache.shape
    ) != expected_cache_shape:
        raise ValueError("paged decode cache shape is incompatible with KV heads")
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    graph_key = compile_paged_decode_for(
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
    )
    out = torch.empty_like(query)
    launch_graph(
        graph_key,
        (
            query,
            key_cache,
            value_cache,
            req_to_token,
            request_index.as_strided((1, tokens), (0, 0)),
            valid_tokens.as_strided((1, tokens), (0, 0)),
            out,
        ),
        stream.cuda_stream,
    )
    return out


def paged_cache_write(
    key_cache: Any,
    value_cache: Any,
    physical_row: Any,
    key: Any,
    value: Any,
    *,
    stream: Any = None,
) -> Any:
    """Write one K/V row through a mutation-declared PyPTO graph."""

    import torch

    if (
        key_cache.ndim != 2
        or value_cache.ndim != 2
        or key_cache.dtype is not torch.bfloat16
        or value_cache.dtype is not torch.bfloat16
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
        or tuple(key_cache.shape) != tuple(value_cache.shape)
    ):
        raise ValueError("paged cache write needs matching contiguous rank-2 BF16 caches")
    if (
        key.ndim != 2
        or value.ndim != 2
        or key.dtype is not torch.bfloat16
        or value.dtype is not torch.bfloat16
        or not key.is_contiguous()
        or not value.is_contiguous()
        or tuple(key.shape) != tuple(value.shape)
        or int(key.shape[0]) != 1
    ):
        raise ValueError("paged cache write needs matching one-row contiguous BF16 updates")
    if (
        physical_row.ndim != 1
        or physical_row.numel() != 1
        or physical_row.dtype is not torch.int64
        or not physical_row.is_contiguous()
    ):
        raise ValueError("paged cache write physical row must be one INT64 element")
    cache_rows, row_width = map(int, key_cache.shape)
    if tuple(key.shape) != (1, row_width):
        raise ValueError("paged cache write update width must match the cache row")
    if any(
        tensor.device != key_cache.device
        for tensor in (value_cache, physical_row, key, value)
    ):
        raise ValueError("paged cache write tensors must share one device")
    if stream is None:
        stream = torch.cuda.current_stream(key_cache.device)
    graph_key = compile_paged_cache_write_for(cache_rows, row_width)
    out = torch.empty_like(key)
    launch_graph(
        graph_key,
        (
            key_cache,
            value_cache,
            physical_row.as_strided((1, row_width), (0, 0)),
            key,
            value,
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
