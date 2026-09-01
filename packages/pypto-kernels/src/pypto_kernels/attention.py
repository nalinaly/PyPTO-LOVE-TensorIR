"""Dense scaled dot-product attention as one native PyPTO tile graph."""

from __future__ import annotations

import math
import threading
from typing import Any

from ._boot import bootstrap, compile_graph, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_ROW_TILE = 1
# 32-wide value tiles match the measured SM120 scheduling cliff
# (see linear.py): 64-column tiles under-fill the grid and lose the
# bandwidth-bound decode path; numerics are tile-invariant because the
# KV reduction order is unchanged.
_VALUE_TILE = 32
_MAX_FUSED_PREFILL_KV_HEADS = 2
_lock = threading.RLock()
_cache: dict[tuple[int, int, int, int], str] = {}
_masked_cache: dict[tuple[int, ...], str] = {}
_paged_gather_cache: dict[tuple[int, ...], str] = {}
_paged_decode_cache: dict[tuple[int, ...], str] = {}
_paged_cache_write_cache: dict[tuple[int, ...], str] = {}
_paged_prefill_cache: dict[tuple[int, ...], str] = {}
_row_pitch_copy_cache: dict[tuple[int, int, int], str] = {}

STATUS = "native-tile executable"
PAGED_DECODE_STATUS = "native-tile executable"
GRAPHS = 7


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
def masked_attention_kernel(
    query: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    additive_mask: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Dense attention with one explicit FP32 additive score mask."""

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
            score = pl.matmul(query_tile, key_transposed, out_dtype=pl.FP32)
            scaled = pl.mul(score, scale)
            mask = pl.load(
                additive_mask,
                [row, 0],
                [1, key.shape[0]],
                target_memory=pl.MemorySpace.Vec,
            )
            masked = pl.add(scaled, mask)
            max_scratch = pl.create_tile(
                [1, key.shape[0]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            row_max = pl.row_max(masked, max_scratch)
            centered = pl.row_expand_sub(masked, row_max)
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
def paged_gather_kernel(
    cache: pl.Tensor,
    req_to_token: pl.Tensor,
    request_index: pl.Tensor,
    virtual_to_physical: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Materialize one request's mapped cache rows into contiguous storage."""

    with pl.at(level=pl.Level.CORE_GROUP):
        request_id = pl.read(request_index, [0, 0])
        for slot in pl.range(out.shape[0]):
            virtual = pl.read(req_to_token, [request_id, slot])
            physical_i64 = pl.read(virtual_to_physical, [virtual, 0])
            physical = pl.cast(physical_i64, pl.INT32)
            value = pl.load(
                cache,
                [physical, 0],
                [1, cache.shape[1]],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(value, [slot, 0], out)
    return out


@pl.jit
def paged_attention_decode_kernel(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    req_to_token: pl.Tensor,
    request_index: pl.Tensor,
    valid_tokens: pl.Tensor,
    virtual_to_physical: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Gather each request's paged KV rows and mask its static KV bucket."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for batch_row in pl.range(query.shape[0]):
            for q_head in pl.range(query.shape[1]):
                kv_heads = key_cache.shape[1] // query.shape[2]
                queries_per_kv = query.shape[1] // kv_heads
                kv_head = q_head // queries_per_kv
                request_id = pl.read(request_index, [batch_row, 0])
                valid_token_count_i64 = pl.read(valid_tokens, [batch_row, 0])
                valid_token_count = pl.cast(valid_token_count_i64, pl.INT32)

                keys = pl.tile.create(
                    [query.shape[2], request_index.shape[1]],
                    dtype=pl.BF16,
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                values = pl.tile.create(
                    [request_index.shape[1], query.shape[2]],
                    dtype=pl.BF16,
                    target_memory=pl.MemorySpace.Mat,
                )
                for slot in pl.range(request_index.shape[1]):
                    virtual = pl.read(req_to_token, [request_id, slot])
                    physical_i64 = pl.read(virtual_to_physical, [virtual, 0])
                    physical = pl.cast(physical_i64, pl.INT32)
                    cache_column = kv_head * query.shape[2]
                    keys = pl.tile.gather_row(
                        keys,
                        key_cache,
                        [0, slot],
                        [physical, cache_column],
                        [1, query.shape[2]],
                        transpose=True,
                    )
                    values = pl.tile.gather_row(
                        values,
                        value_cache,
                        [slot, 0],
                        [physical, cache_column],
                        [1, query.shape[2]],
                    )

                query_box = pl.load(
                    query,
                    [batch_row, q_head, 0],
                    [1, 1, query.shape[2]],
                    target_memory=pl.MemorySpace.Mat,
                )
                query_tile = pl.reshape(query_box, [1, query.shape[2]])
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
                result_box = pl.reshape(result, [1, 1, query.shape[2]])
                pl.store(result_box, [batch_row, q_head, 0], out)
    return out


@pl.jit
def paged_cache_write_kernel(
    key_cache: pl.InOut[pl.Tensor],
    value_cache: pl.InOut[pl.Tensor],
    virtual_row: pl.Tensor,
    virtual_to_physical: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Write flattened GQA K/V rows to selected physical cache slots."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(key.shape[0]):
            virtual_row_i64 = pl.read(virtual_row, [row, 0])
            physical_row_i64 = pl.read(
                virtual_to_physical, [virtual_row_i64, 0]
            )
            physical_row_i32 = pl.cast(physical_row_i64, pl.INT32)
            key_tile = pl.load(
                key,
                [row, 0],
                [1, key.shape[1]],
                target_memory=pl.MemorySpace.Vec,
            )
            value_tile = pl.load(
                value,
                [row, 0],
                [1, value.shape[1]],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(key_tile, [physical_row_i32, 0], key_cache)
            pl.store(value_tile, [physical_row_i32, 0], value_cache)
            anchor = pl.add(key_tile, value_tile)
            pl.store(anchor, [row, 0], out)
    return out


@pl.jit
def row_pitch_copy_kernel(
    source: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Copy row-pitched BF16 input into dense row-major storage."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(source.shape[0]):
            for column_block in pl.range(source.shape[1] // 128):
                value = pl.load(
                    source, [row, column_block * 128], [1, 128]
                )
                copied = pl.add(value, 0.0)
                pl.store(copied, [row, column_block * 128], out)
    return out


@pl.jit
def paged_attention_prefill_kernel(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    req_to_token: pl.Tensor,
    request_index: pl.Tensor,
    prefix_tokens: pl.Tensor,
    virtual_to_physical: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Run causal GQA prefill with head-major Q/output TensorView operands."""

    with pl.at(level=pl.Level.CORE_GROUP):
        request_id = pl.read(request_index, [0, 0])
        prefix_token_count = pl.read(prefix_tokens, [0, 0])
        positions = pl.tile.ci(
            0,
            [1, request_index.shape[1]],
            dtype=pl.INT32,
        )
        q_heads = query.shape[0]
        head_dim = query.shape[2]
        kv_heads = value_cache.shape[1] // head_dim
        queries_per_kv = q_heads // kv_heads
        for kv_head in pl.range(kv_heads):
            keys = pl.tile.create(
                [head_dim, request_index.shape[1]],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
                transpose=True,
            )
            values = pl.tile.create(
                [request_index.shape[1], head_dim],
                dtype=pl.BF16,
                target_memory=pl.MemorySpace.Mat,
            )
            for slot in pl.range(request_index.shape[1]):
                virtual = pl.read(req_to_token, [request_id, slot])
                physical_i64 = pl.read(virtual_to_physical, [virtual, 0])
                physical = pl.cast(physical_i64, pl.INT32)
                cache_column = kv_head * head_dim
                keys = pl.tile.gather_row(
                    keys,
                    key_cache,
                    [0, slot],
                    [physical, cache_column],
                    [1, head_dim],
                    transpose=True,
                )
                values = pl.tile.gather_row(
                    values,
                    value_cache,
                    [slot, 0],
                    [physical, cache_column],
                    [1, head_dim],
                )
            for q_group in pl.range(queries_per_kv):
                q_head = kv_head * queries_per_kv + q_group
                for query_row in pl.range(query.shape[1]):
                    query_box = pl.load(
                        query,
                        [q_head, query_row, 0],
                        [1, 1, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    query_tile = pl.reshape(query_box, [1, head_dim])
                    score = pl.matmul(query_tile, keys, out_dtype=pl.FP32)
                    scaled = pl.mul(score, scale)
                    query_row_i32 = pl.cast(query_row, pl.INT32)
                    valid_token_count = prefix_token_count + query_row_i32
                    valid_token_count = valid_token_count + pl.cast(1, pl.INT32)
                    valid_mask = pl.cmps(
                        positions, valid_token_count, cmp_type=2
                    )
                    mask_scratch = pl.tile.create(
                        [1, 32],
                        dtype=pl.UINT8,
                        target_memory=pl.MemorySpace.Vec,
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
                    probability_bf16 = pl.cast(
                        probability, target_type=pl.BF16
                    )
                    mixed = pl.matmul(
                        probability_bf16, values, out_dtype=pl.FP32
                    )
                    result = pl.cast(mixed, target_type=pl.BF16)
                    result_box = pl.reshape(result, [1, 1, head_dim])
                    pl.store(result_box, [q_head, query_row, 0], out)
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
    batch_size: int,
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int,
    mapping_rows: int,
    query_row_stride: int,
    result_row_stride: int,
) -> None:
    if (
        batch_size <= 0
        or q_heads <= 0
        or kv_heads <= 0
        or tokens <= 0
        or head_dim <= 0
        or cache_rows <= 0
        or request_rows <= 0
        or max_context_len < tokens
        or cache_row_stride < kv_heads * head_dim
        or mapping_rows <= 0
        or query_row_stride < q_heads * head_dim
        or result_row_stride < q_heads * head_dim
        or q_heads % kv_heads
        or head_dim % 128
        or tokens % 16
    ):
        raise ValueError(
            "paged decode needs positive dimensions, q_heads divisible by "
            "kv_heads, head_dim divisible by 128 and a 16-token KV bucket"
        )


def _paged_cache_row_stride(
    key_cache: Any, value_cache: Any, *, operation: str
) -> int:
    """Validate a dense-row or row-pitched BF16 cache pair."""

    import torch

    if (
        key_cache.ndim != 2
        or value_cache.ndim != 2
        or key_cache.dtype is not torch.bfloat16
        or value_cache.dtype is not torch.bfloat16
        or tuple(key_cache.shape) != tuple(value_cache.shape)
        or tuple(key_cache.stride()) != tuple(value_cache.stride())
        or int(key_cache.stride(1)) != 1
        or int(key_cache.stride(0)) < int(key_cache.shape[1])
    ):
        raise ValueError(
            f"{operation} needs matching rank-2 BF16 caches with contiguous "
            "rows and a non-overlapping static row pitch"
        )
    return int(key_cache.stride(0))


def _tensor_row_stride(tensor: Any, *, operation: str) -> int:
    """Validate one rank-2 BF16 tensor with dense or row-pitched rows."""

    import torch

    if (
        tensor.ndim != 2
        or tensor.dtype is not torch.bfloat16
        or int(tensor.stride(1)) != 1
        or int(tensor.stride(0)) < int(tensor.shape[1])
    ):
        raise ValueError(
            f"{operation} needs rank-2 BF16 rows with unit inner stride and "
            "a non-overlapping static row pitch"
        )
    return int(tensor.stride(0))


def _mapping_rows(virtual_to_physical: Any, *, device: Any, operation: str) -> int:
    """Validate the page-size-one virtual-to-physical table."""

    import torch

    if (
        virtual_to_physical.ndim != 1
        or virtual_to_physical.dtype is not torch.int64
        or not virtual_to_physical.is_contiguous()
        or virtual_to_physical.device != device
        or virtual_to_physical.numel() <= 0
    ):
        raise ValueError(
            f"{operation} needs one contiguous device INT64 virtual-to-physical table"
        )
    return int(virtual_to_physical.numel())


def _dense_tiles(rows: int) -> list[int]:
    return (
        [_VALUE_TILE]
        if rows == 1
        else [min(_ROW_TILE, rows), _VALUE_TILE]
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
        _dense_tiles(rows),
        provider="pypto.attention",
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def compile_masked_for(
    rows: int,
    tokens: int,
    head_dim: int,
    value_dim: int,
    query_row_stride: int,
    key_row_stride: int,
    value_row_stride: int,
    mask_row_stride: int,
    result_row_stride: int | None = None,
) -> str:
    """Compile dense attention with an explicit additive score mask."""

    _validate_shape(rows, tokens, head_dim, value_dim)
    if result_row_stride is None:
        result_row_stride = value_dim
    if (
        query_row_stride < head_dim
        or key_row_stride < head_dim
        or value_row_stride < value_dim
        or mask_row_stride < tokens
        or result_row_stride < value_dim
    ):
        raise ValueError("masked attention row strides overlap logical rows")
    shape_key = (
        rows,
        tokens,
        head_dim,
        value_dim,
        query_row_stride,
        key_row_stride,
        value_row_stride,
        mask_row_stride,
        result_row_stride,
    )
    cached = _masked_cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    query = (
        torch.empty((rows, head_dim), dtype=torch.bfloat16, device="meta")
        if query_row_stride == head_dim
        else torch.empty_strided(
            (rows, head_dim),
            (query_row_stride, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    )
    key = (
        torch.empty((tokens, head_dim), dtype=torch.bfloat16, device="meta")
        if key_row_stride == head_dim
        else torch.empty_strided(
            (tokens, head_dim),
            (key_row_stride, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    )
    value = (
        torch.empty((tokens, value_dim), dtype=torch.bfloat16, device="meta")
        if value_row_stride == value_dim
        else torch.empty_strided(
            (tokens, value_dim),
            (value_row_stride, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    )
    additive_mask = (
        torch.empty((rows, tokens), dtype=torch.float32, device="meta")
        if mask_row_stride == tokens
        else torch.empty_strided(
            (rows, tokens),
            (mask_row_stride, 1),
            dtype=torch.float32,
            device="meta",
        )
    )
    out = (
        torch.empty((rows, value_dim), dtype=torch.bfloat16, device="meta")
        if result_row_stride == value_dim
        else torch.empty_strided(
            (rows, value_dim),
            (result_row_stride, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    )
    graph_key = compile_jit_kernel(
        masked_attention_kernel,
        (
            query,
            key,
            value,
            additive_mask,
            1.0 / math.sqrt(head_dim),
            out,
        ),
        _dense_tiles(rows),
        provider="pypto.attention",
        source_node="pypto_kernels.attention:masked_attention",
    )
    with _lock:
        _masked_cache[shape_key] = graph_key
    return graph_key


def build_paged_decode(
    batch_size: int,
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    query_row_stride: int | None = None,
    result_row_stride: int | None = None,
) -> Any:
    cache_width = kv_heads * head_dim
    if cache_row_stride is None:
        cache_row_stride = cache_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if query_row_stride is None:
        query_row_stride = q_heads * head_dim
    if result_row_stride is None:
        result_row_stride = q_heads * head_dim
    _validate_paged_decode_shape(
        batch_size,
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    import torch

    query = torch.empty_strided(
        (batch_size, q_heads, head_dim),
        (query_row_stride, head_dim, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    key_cache = torch.empty_strided(
        (cache_rows, cache_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    value_cache = torch.empty_strided(
        (cache_rows, cache_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    req_to_token = torch.empty(
        (request_rows, max_context_len), dtype=torch.int32, device="meta"
    )
    request_index = torch.empty_strided(
        (batch_size, tokens), (1, 0), dtype=torch.int64, device="meta"
    )
    valid_tokens = torch.empty_strided(
        (batch_size, tokens), (1, 0), dtype=torch.int64, device="meta"
    )
    virtual_to_physical = torch.empty(
        (mapping_rows, 1), dtype=torch.int64, device="meta"
    )
    out = torch.empty_strided(
        (batch_size, q_heads, head_dim),
        (result_row_stride, head_dim, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    return paged_attention_decode_kernel.specialize(
        query,
        key_cache,
        value_cache,
        req_to_token,
        request_index,
        valid_tokens,
        virtual_to_physical,
        1.0 / math.sqrt(head_dim),
        out,
    )


def compile_paged_decode_for(
    batch_size: int,
    q_heads: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    query_row_stride: int | None = None,
    result_row_stride: int | None = None,
) -> str:
    cache_width = kv_heads * head_dim
    if cache_row_stride is None:
        cache_row_stride = cache_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if query_row_stride is None:
        query_row_stride = q_heads * head_dim
    if result_row_stride is None:
        result_row_stride = q_heads * head_dim
    _validate_paged_decode_shape(
        batch_size,
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    shape_key = (
        batch_size,
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    cached = _paged_decode_cache.get(shape_key)
    if cached is not None:
        return cached
    program = build_paged_decode(*shape_key)
    tile_shape = _paged_decode_tiles(batch_size, q_heads)
    graph_key = compile_graph(
        program,
        tile_shape,
        provider="pypto.attention",
        source_node="pypto_kernels.attention:paged_decode",
    )
    with _lock:
        _paged_decode_cache[shape_key] = graph_key
    return graph_key


def _paged_decode_tiles(batch_size: int, q_heads: int) -> list[int]:
    """Match canonical removal of unit batch/head iteration dimensions."""

    if batch_size <= 0 or q_heads <= 0:
        raise ValueError("paged decode tiles need positive batch and Q heads")
    iteration_rank = int(batch_size != 1) + int(q_heads != 1)
    return [*([1] * iteration_rank), _VALUE_TILE]


def _paged_decode_partition_count(q_heads: int) -> int:
    if q_heads <= 0:
        raise ValueError("paged decode partitioning needs positive Q heads")
    return q_heads


def _validate_paged_cache_write_shape(
    cache_rows: int,
    update_rows: int,
    row_width: int,
    cache_row_stride: int,
    mapping_rows: int,
    key_row_stride: int,
    value_row_stride: int,
) -> None:
    if (
        cache_rows <= 0
        or update_rows <= 0
        or row_width <= 0
        or row_width % 128
        or cache_row_stride < row_width
        or mapping_rows <= 0
        or key_row_stride < row_width
        or value_row_stride < row_width
    ):
        raise ValueError(
            "paged cache write needs positive dimensions and row width "
            "divisible by 128"
        )


def build_paged_cache_write(
    cache_rows: int,
    update_rows: int,
    row_width: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    key_row_stride: int | None = None,
    value_row_stride: int | None = None,
) -> Any:
    if cache_row_stride is None:
        cache_row_stride = row_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if key_row_stride is None:
        key_row_stride = row_width
    if value_row_stride is None:
        value_row_stride = row_width
    _validate_paged_cache_write_shape(
        cache_rows,
        update_rows,
        row_width,
        cache_row_stride,
        mapping_rows,
        key_row_stride,
        value_row_stride,
    )
    import torch

    key_cache = torch.empty_strided(
        (cache_rows, row_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    value_cache = torch.empty_strided(
        (cache_rows, row_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    physical_row = torch.empty_strided(
        (update_rows, row_width), (1, 0), dtype=torch.int64, device="meta"
    )
    virtual_to_physical = torch.empty(
        (mapping_rows, 1), dtype=torch.int64, device="meta"
    )
    key = torch.empty_strided(
        (update_rows, row_width),
        (key_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    value = torch.empty_strided(
        (update_rows, row_width),
        (value_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    out = torch.empty(
        (update_rows, row_width), dtype=torch.bfloat16, device="meta"
    )
    return paged_cache_write_kernel.specialize(
        key_cache,
        value_cache,
        physical_row,
        virtual_to_physical,
        key,
        value,
        out,
    )


def compile_paged_cache_write_for(
    cache_rows: int,
    update_rows: int,
    row_width: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    key_row_stride: int | None = None,
    value_row_stride: int | None = None,
) -> str:
    if cache_row_stride is None:
        cache_row_stride = row_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if key_row_stride is None:
        key_row_stride = row_width
    if value_row_stride is None:
        value_row_stride = row_width
    _validate_paged_cache_write_shape(
        cache_rows,
        update_rows,
        row_width,
        cache_row_stride,
        mapping_rows,
        key_row_stride,
        value_row_stride,
    )
    shape_key = (
        cache_rows,
        update_rows,
        row_width,
        cache_row_stride,
        mapping_rows,
        key_row_stride,
        value_row_stride,
    )
    cached = _paged_cache_write_cache.get(shape_key)
    if cached is not None:
        return cached
    tile_shape = [128] if update_rows == 1 else [1, 128]
    graph_key = compile_graph(
        build_paged_cache_write(*shape_key),
        tile_shape,
        provider="pypto.attention",
        source_node="pypto_kernels.attention:paged_cache_write",
    )
    with _lock:
        _paged_cache_write_cache[shape_key] = graph_key
    return graph_key


def _validate_paged_prefill_shape(
    query_rows: int,
    q_heads: int,
    kv_heads: int,
    bucket_tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int,
    mapping_rows: int,
    query_row_stride: int,
    result_row_stride: int,
) -> None:
    if (
        query_rows <= 0
        or q_heads <= 0
        or kv_heads <= 0
        or q_heads % kv_heads
        or bucket_tokens <= 0
        or bucket_tokens % 16
        or head_dim <= 0
        or head_dim % 128
        or cache_rows <= 0
        or request_rows <= 0
        or max_context_len < bucket_tokens
        or cache_row_stride < kv_heads * head_dim
        or mapping_rows <= 0
        or query_row_stride < q_heads * head_dim
        or result_row_stride < q_heads * head_dim
    ):
        raise ValueError(
            "paged prefill needs positive GQA geometry, a 16-token bucket, "
            "head_dim divisible by 128 and a covering request table"
        )


def _validate_paged_gather_shape(
    tokens: int,
    row_width: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int,
    mapping_rows: int,
) -> None:
    if (
        tokens <= 0
        or tokens % 16
        or row_width <= 0
        or row_width % 128
        or cache_rows <= 0
        or request_rows <= 0
        or max_context_len < tokens
        or cache_row_stride < row_width
        or mapping_rows <= 0
    ):
        raise ValueError(
            "paged gather needs positive dimensions, a 16-token bucket, "
            "and a 128-aligned cache row"
        )


def build_paged_gather(
    tokens: int,
    row_width: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
) -> Any:
    if cache_row_stride is None:
        cache_row_stride = row_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    _validate_paged_gather_shape(
        tokens,
        row_width,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
    )
    import torch

    cache = torch.empty_strided(
        (cache_rows, row_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    req_to_token = torch.empty(
        (request_rows, max_context_len), dtype=torch.int32, device="meta"
    )
    request_index = torch.empty_strided(
        (1, tokens), (0, 0), dtype=torch.int64, device="meta"
    )
    virtual_to_physical = torch.empty(
        (mapping_rows, 1), dtype=torch.int64, device="meta"
    )
    out = torch.empty((tokens, row_width), dtype=torch.bfloat16, device="meta")
    return paged_gather_kernel.specialize(
        cache,
        req_to_token,
        request_index,
        virtual_to_physical,
        out,
    )


def compile_paged_gather_for(
    tokens: int,
    row_width: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
) -> str:
    if cache_row_stride is None:
        cache_row_stride = row_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    _validate_paged_gather_shape(
        tokens,
        row_width,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
    )
    shape_key = (
        tokens,
        row_width,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
    )
    cached = _paged_gather_cache.get(shape_key)
    if cached is not None:
        return cached
    graph_key = compile_graph(
        build_paged_gather(*shape_key),
        [1, 128],
        provider="pypto.attention",
        source_node="pypto_kernels.attention:paged_gather",
    )
    with _lock:
        _paged_gather_cache[shape_key] = graph_key
    return graph_key


def paged_gather(
    cache: Any,
    req_to_token: Any,
    request_index: Any,
    virtual_to_physical: Any,
    *,
    bucket_tokens: int,
    stream: Any = None,
) -> Any:
    """Materialize one mapped request's cache rows with one native graph."""

    import torch

    cache_row_stride = _paged_cache_row_stride(
        cache, cache, operation="paged gather"
    )
    mapping_rows = _mapping_rows(
        virtual_to_physical, device=cache.device, operation="paged gather"
    )
    if (
        req_to_token.ndim != 2
        or req_to_token.dtype is not torch.int32
        or not req_to_token.is_contiguous()
        or request_index.ndim != 1
        or request_index.numel() != 1
        or request_index.dtype is not torch.int64
        or not request_index.is_contiguous()
        or any(
            tensor.device != cache.device
            for tensor in (req_to_token, request_index, virtual_to_physical)
        )
    ):
        raise ValueError(
            "paged gather needs contiguous device request metadata with one INT64 index"
        )
    cache_rows, row_width = map(int, cache.shape)
    request_rows, max_context_len = map(int, req_to_token.shape)
    _validate_paged_gather_shape(
        int(bucket_tokens),
        row_width,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
    )
    if stream is None:
        stream = torch.cuda.current_stream(cache.device)
    graph_key = compile_paged_gather_for(
        int(bucket_tokens),
        row_width,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
    )
    out = torch.empty(
        (int(bucket_tokens), row_width), dtype=torch.bfloat16, device=cache.device
    )
    request_index_view = request_index.as_strided((1, int(bucket_tokens)), (0, 0))
    mapping_view = virtual_to_physical.view(mapping_rows, 1)
    launch_graph(
        graph_key,
        (cache, req_to_token, request_index_view, mapping_view, out),
        stream.cuda_stream,
    )
    return out


def _paged_prefill_partition_count(kv_heads: int) -> int:
    if kv_heads <= 0:
        raise ValueError("paged prefill partitioning needs positive KV heads")
    return 1 if kv_heads <= _MAX_FUSED_PREFILL_KV_HEADS else kv_heads


def _paged_prefill_tiles(
    query_rows: int, q_heads: int, kv_heads: int
) -> list[int]:
    """Match unit-axis removal plus the multi-KV concatenate space."""

    if query_rows <= 0 or q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("paged prefill tiles need positive divisible GQA geometry")
    iteration_rank = (
        int(query_rows != 1) + int(q_heads != 1) + int(kv_heads > 1)
    )
    return [*([1] * iteration_rank), 32]


def _launch_paged_prefill_graph(
    graph_key: str, operands: tuple[Any, ...], stream: Any
) -> None:
    launch_graph(graph_key, operands, stream.cuda_stream)


def _compile_row_pitch_copy(rows: int, columns: int, row_stride: int) -> str:
    if (
        rows <= 0
        or columns <= 0
        or columns % 128
        or row_stride < columns
    ):
        raise ValueError(
            "row-pitch copy needs positive BF16 rows, 128-aligned columns and "
            "a covering row stride"
        )
    shape_key = (rows, columns, row_stride)
    cached = _row_pitch_copy_cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    source = torch.empty_strided(
        (rows, columns),
        (row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    out = torch.empty((rows, columns), dtype=torch.bfloat16, device="meta")
    graph_key = compile_graph(
        row_pitch_copy_kernel.specialize(source, out),
        [1, 128],
        provider="pypto.attention",
        source_node="pypto_kernels.attention:row_pitch_copy",
    )
    with _lock:
        _row_pitch_copy_cache[shape_key] = graph_key
    return graph_key


def _contiguous_prefill_query(query: Any, stream: Any) -> Any:
    """Stage a pitched query through one native graph when rank-3 loads cannot."""

    import torch

    rows, columns = map(int, query.shape)
    if int(query.stride(0)) == columns:
        return query
    out = torch.empty(
        (rows, columns), dtype=torch.bfloat16, device=query.device
    )
    graph_key = _compile_row_pitch_copy(rows, columns, int(query.stride(0)))
    launch_graph(graph_key, (query, out), stream.cuda_stream)
    return out


def build_paged_prefill(
    query_rows: int,
    q_heads: int,
    kv_heads: int,
    bucket_tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    query_row_stride: int | None = None,
    result_row_stride: int | None = None,
) -> Any:
    cache_width = kv_heads * head_dim
    if cache_row_stride is None:
        cache_row_stride = cache_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if query_row_stride is None:
        query_row_stride = q_heads * head_dim
    if result_row_stride is None:
        result_row_stride = q_heads * head_dim
    _validate_paged_prefill_shape(
        query_rows,
        q_heads,
        kv_heads,
        bucket_tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    import torch

    query = torch.empty_strided(
        (q_heads, query_rows, head_dim),
        (head_dim, query_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    key_cache = torch.empty_strided(
        (cache_rows, cache_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    value_cache = torch.empty_strided(
        (cache_rows, cache_width),
        (cache_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    req_to_token = torch.empty(
        (request_rows, max_context_len), dtype=torch.int32, device="meta"
    )
    request_index = torch.empty_strided(
        (1, bucket_tokens), (0, 0), dtype=torch.int64, device="meta"
    )
    prefix_tokens = torch.empty_strided(
        (1, bucket_tokens), (0, 0), dtype=torch.int32, device="meta"
    )
    virtual_to_physical = torch.empty(
        (mapping_rows, 1), dtype=torch.int64, device="meta"
    )
    out = torch.empty_strided(
        (q_heads, query_rows, head_dim),
        (head_dim, result_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    return paged_attention_prefill_kernel.specialize(
        query,
        key_cache,
        value_cache,
        req_to_token,
        request_index,
        prefix_tokens,
        virtual_to_physical,
        1.0 / math.sqrt(head_dim),
        out,
    )


def compile_paged_prefill_for(
    query_rows: int,
    q_heads: int,
    kv_heads: int,
    bucket_tokens: int,
    head_dim: int,
    cache_rows: int,
    request_rows: int,
    max_context_len: int,
    cache_row_stride: int | None = None,
    mapping_rows: int | None = None,
    query_row_stride: int | None = None,
    result_row_stride: int | None = None,
) -> str:
    cache_width = kv_heads * head_dim
    if cache_row_stride is None:
        cache_row_stride = cache_width
    if mapping_rows is None:
        mapping_rows = cache_rows
    if query_row_stride is None:
        query_row_stride = q_heads * head_dim
    if result_row_stride is None:
        result_row_stride = q_heads * head_dim
    shape_key = (
        query_rows,
        q_heads,
        kv_heads,
        bucket_tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    _validate_paged_prefill_shape(*shape_key)
    cached = _paged_prefill_cache.get(shape_key)
    if cached is not None:
        return cached
    graph_key = compile_graph(
        build_paged_prefill(*shape_key),
        _paged_prefill_tiles(query_rows, q_heads, kv_heads),
        provider="pypto.attention",
        source_node="pypto_kernels.attention:paged_prefill",
    )
    with _lock:
        _paged_prefill_cache[shape_key] = graph_key
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


def masked_attention(
    query: Any,
    key: Any,
    value: Any,
    additive_mask: Any,
    stream: Any = None,
    out: Any = None,
) -> Any:
    """Run dense attention with a caller-provided FP32 additive score mask."""

    import torch

    query_row_stride = _tensor_row_stride(
        query, operation="masked attention query"
    )
    key_row_stride = _tensor_row_stride(key, operation="masked attention key")
    value_row_stride = _tensor_row_stride(
        value, operation="masked attention value"
    )
    rows, head_dim = map(int, query.shape)
    tokens = int(key.shape[0])
    value_dim = int(value.shape[1])
    if tuple(key.shape) != (tokens, head_dim) or int(value.shape[0]) != tokens:
        raise ValueError("masked attention Q/K/V dimensions are incompatible")
    if (
        additive_mask.ndim != 2
        or tuple(additive_mask.shape) != (rows, tokens)
        or additive_mask.dtype is not torch.float32
        or int(additive_mask.stride(1)) != 1
        or int(additive_mask.stride(0)) < tokens
        or additive_mask.device != query.device
        or any(tensor.device != query.device for tensor in (key, value))
    ):
        raise ValueError(
            "masked attention needs a row-pitched FP32 [rows,tokens] mask "
            "on the Q/K/V device"
        )
    if out is None:
        out = torch.empty(
            (rows, value_dim), dtype=torch.bfloat16, device=query.device
        )
    elif (
        out.ndim != 2
        or out.dtype is not torch.bfloat16
        or tuple(out.shape) != (rows, value_dim)
        or int(out.stride(1)) != 1
        or int(out.stride(0)) < value_dim
        or out.device != query.device
    ):
        raise ValueError(
            "masked attention output must be a row-pitched BF16 [rows,value_dim] view"
        )
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    graph_key = compile_masked_for(
        rows,
        tokens,
        head_dim,
        value_dim,
        query_row_stride,
        key_row_stride,
        value_row_stride,
        int(additive_mask.stride(0)),
        int(out.stride(0)),
    )
    launch_graph(
        graph_key,
        (query, key, value, additive_mask, out),
        stream.cuda_stream,
    )
    return out


def causal_mask(
    valid_lengths: list[int],
    bucket_tokens: int,
    *,
    device: Any,
    stream: Any = None,
) -> Any:
    """Build an additive causal mask on the caller's CUDA stream."""

    import torch

    if not valid_lengths or bucket_tokens <= 0 or bucket_tokens % 16:
        raise ValueError("causal mask needs non-empty 16-aligned dimensions")
    if any(length <= 0 or length > bucket_tokens for length in valid_lengths):
        raise ValueError("causal mask valid lengths must fit the token bucket")
    if stream is None:
        stream = torch.cuda.current_stream(device)
    with torch.cuda.stream(stream):
        mask = torch.full(
            (len(valid_lengths), bucket_tokens),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=device,
        )
        for row, length in enumerate(valid_lengths):
            mask[row, :length] = 0.0
    return mask


def paged_attention_decode(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    req_to_token: Any,
    request_index: Any,
    valid_tokens: Any,
    virtual_to_physical: Any,
    *,
    kv_heads: int,
    bucket_tokens: int,
    stream: Any = None,
) -> Any:
    """Run batched GQA decode over padded physical KV buckets.

    ``request_index`` and ``valid_tokens`` contain one device value per batch
    row. Each valid length must be in ``[1, bucket_tokens]``; keeping both
    tensors on-device avoids decode-time host synchronization and an
    intermediate ``kv_indices`` kernel. Decode is partitioned into one Q head
    per launch. Every Q head reuses one compiled artifact, while zero-copy
    row-pitched views select its Q column, mapped KV head and output column.
    This keeps the complete static token bucket in one exact softmax while
    avoiding the tile assembler's multi-head fusion explosion.
    """

    import torch

    query_row_stride = _tensor_row_stride(query, operation="paged decode query")
    cache_row_stride = _paged_cache_row_stride(
        key_cache, value_cache, operation="paged decode"
    )
    mapping_rows = _mapping_rows(
        virtual_to_physical, device=query.device, operation="paged decode"
    )
    if (
        req_to_token.ndim != 2
        or req_to_token.dtype is not torch.int32
        or not req_to_token.is_contiguous()
    ):
        raise ValueError("paged decode request table must be contiguous rank-2 INT32")
    if (
        request_index.ndim != 1
        or request_index.dtype is not torch.int64
        or not request_index.is_contiguous()
    ):
        raise ValueError(
            "paged decode request indices must be contiguous INT64"
        )
    if (
        valid_tokens.ndim != 1
        or valid_tokens.dtype is not torch.int64
        or not valid_tokens.is_contiguous()
    ):
        raise ValueError(
            "paged decode valid token counts must be contiguous INT64"
        )
    if (
        key_cache.device != query.device
        or value_cache.device != query.device
        or req_to_token.device != query.device
        or request_index.device != query.device
        or valid_tokens.device != query.device
    ):
        raise ValueError("paged decode metadata must be on the query device")
    cache_rows = int(key_cache.shape[0])
    tokens = int(bucket_tokens)
    batch_size, query_width = map(int, query.shape)
    request_rows, max_context_len = map(int, req_to_token.shape)
    if request_index.numel() != batch_size or valid_tokens.numel() != batch_size:
        raise ValueError("paged decode needs one request index and length per batch row")
    cache_width = int(key_cache.shape[1])
    if kv_heads <= 0 or cache_width <= 0 or cache_width % kv_heads:
        raise ValueError("paged decode cache width must divide into positive KV heads")
    head_dim = cache_width // kv_heads
    if query_width % head_dim:
        raise ValueError("paged decode query width must divide into Q heads")
    q_heads = query_width // head_dim
    result_row_stride = query_width
    _validate_paged_decode_shape(
        batch_size,
        q_heads,
        kv_heads,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    expected_cache_shape = (cache_rows, kv_heads * head_dim)
    if tuple(key_cache.shape) != expected_cache_shape or tuple(
        value_cache.shape
    ) != expected_cache_shape:
        raise ValueError("paged decode cache shape is incompatible with KV heads")
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    queries_per_kv = q_heads // kv_heads
    out = torch.empty(
        (batch_size, query_width),
        dtype=torch.bfloat16,
        device=query.device,
    )
    request_index_view = request_index.as_strided(
        (batch_size, tokens), (1, 0)
    )
    valid_tokens_view = valid_tokens.as_strided(
        (batch_size, tokens), (1, 0)
    )
    mapping_view = virtual_to_physical.view(mapping_rows, 1)
    if batch_size == 1:
        # Single-launch path: one graph covers every Q head, so the grid
        # spans all heads (16x the CTAs of a per-head launch on the
        # latency-bound decode bucket) and one launch replaces sixteen.
        # Per-head arithmetic and reduction order are unchanged.
        merged_key = compile_paged_decode_for(
            1,
            q_heads,
            kv_heads,
            tokens,
            head_dim,
            cache_rows,
            request_rows,
            max_context_len,
            cache_row_stride,
            mapping_rows,
            query_row_stride,
            result_row_stride,
        )
        merged_query = query.as_strided(
            (1, q_heads, head_dim),
            (query_row_stride, head_dim, 1),
        )
        merged_result = out.as_strided(
            (1, q_heads, head_dim),
            (result_row_stride, head_dim, 1),
        )
        launch_graph(
            merged_key,
            (
                merged_query,
                key_cache,
                value_cache,
                req_to_token,
                request_index_view,
                valid_tokens_view,
                mapping_view,
                merged_result,
            ),
            stream.cuda_stream,
        )
        return out
    graph_key = compile_paged_decode_for(
        batch_size,
        1,
        1,
        tokens,
        head_dim,
        cache_rows,
        request_rows,
        max_context_len,
        cache_row_stride,
        mapping_rows,
        query_row_stride,
        result_row_stride,
    )
    query_storage_offset = int(query.storage_offset())
    key_storage_offset = int(key_cache.storage_offset())
    value_storage_offset = int(value_cache.storage_offset())
    result_storage_offset = int(out.storage_offset())
    # as_strided only creates tensor metadata: no gather/copy/concatenation
    # kernel is inserted between these same-stream PyPTO launches.
    for q_head in range(_paged_decode_partition_count(q_heads)):
        kv_head = q_head // queries_per_kv
        query_column = q_head * head_dim
        cache_column = kv_head * head_dim
        query_group = query.as_strided(
            (batch_size, 1, head_dim),
            (query_row_stride, head_dim, 1),
            storage_offset=query_storage_offset + query_column,
        )
        key_group = key_cache.as_strided(
            (cache_rows, head_dim),
            (cache_row_stride, 1),
            storage_offset=key_storage_offset + cache_column,
        )
        value_group = value_cache.as_strided(
            (cache_rows, head_dim),
            (cache_row_stride, 1),
            storage_offset=value_storage_offset + cache_column,
        )
        result_group = out.as_strided(
            (batch_size, 1, head_dim),
            (result_row_stride, head_dim, 1),
            storage_offset=result_storage_offset + query_column,
        )
        launch_graph(
            graph_key,
            (
                query_group,
                key_group,
                value_group,
                req_to_token,
                request_index_view,
                valid_tokens_view,
                mapping_view,
                result_group,
            ),
            stream.cuda_stream,
        )
    return out


def paged_cache_write(
    key_cache: Any,
    value_cache: Any,
    physical_row: Any,
    virtual_to_physical: Any,
    key: Any,
    value: Any,
    *,
    stream: Any = None,
) -> Any:
    """Write K/V rows through one mutation-declared PyPTO graph."""

    import torch

    cache_row_stride = _paged_cache_row_stride(
        key_cache, value_cache, operation="paged cache write"
    )
    mapping_rows = _mapping_rows(
        virtual_to_physical,
        device=key_cache.device,
        operation="paged cache write",
    )
    key_row_stride = _tensor_row_stride(key, operation="paged cache write key")
    value_row_stride = _tensor_row_stride(
        value, operation="paged cache write value"
    )
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("paged cache write needs matching BF16 update shapes")
    if (
        physical_row.ndim != 1
        or physical_row.dtype is not torch.int64
        or not physical_row.is_contiguous()
    ):
        raise ValueError("paged cache write physical rows must be contiguous INT64")
    cache_rows, row_width = map(int, key_cache.shape)
    update_rows = int(key.shape[0])
    if tuple(key.shape) != (update_rows, row_width):
        raise ValueError("paged cache write update width must match the cache row")
    if physical_row.numel() != update_rows:
        raise ValueError("paged cache write needs one physical row per update row")
    if any(
        tensor.device != key_cache.device
        for tensor in (
            value_cache,
            physical_row,
            virtual_to_physical,
            key,
            value,
        )
    ):
        raise ValueError("paged cache write tensors must share one device")
    if stream is None:
        stream = torch.cuda.current_stream(key_cache.device)
    graph_key = compile_paged_cache_write_for(
        cache_rows,
        update_rows,
        row_width,
        cache_row_stride,
        mapping_rows,
        key_row_stride,
        value_row_stride,
    )
    out = torch.empty(
        (update_rows, row_width), dtype=torch.bfloat16, device=key.device
    )
    launch_graph(
        graph_key,
        (
            key_cache,
            value_cache,
            physical_row.as_strided((update_rows, row_width), (1, 0)),
            virtual_to_physical.view(mapping_rows, 1),
            key,
            value,
            out,
        ),
        stream.cuda_stream,
    )
    return out


def paged_attention_prefill(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    req_to_token: Any,
    request_index: Any,
    prefix_tokens: Any,
    virtual_to_physical: Any,
    *,
    kv_heads: int,
    bucket_tokens: int,
    stream: Any = None,
) -> Any:
    """Run causal GQA prefill after the same-stream cache-write graph."""

    import torch

    query_row_stride = _tensor_row_stride(query, operation="paged prefill query")
    cache_row_stride = _paged_cache_row_stride(
        key_cache, value_cache, operation="paged prefill"
    )
    mapping_rows = _mapping_rows(
        virtual_to_physical, device=query.device, operation="paged prefill"
    )
    if (
        req_to_token.ndim != 2
        or req_to_token.dtype is not torch.int32
        or not req_to_token.is_contiguous()
    ):
        raise ValueError("paged prefill request table must be contiguous rank-2 INT32")
    if (
        request_index.ndim != 1
        or request_index.numel() != 1
        or request_index.dtype is not torch.int64
        or not request_index.is_contiguous()
        or prefix_tokens.ndim != 1
        or prefix_tokens.numel() != 1
        or prefix_tokens.dtype is not torch.int32
        or not prefix_tokens.is_contiguous()
    ):
        raise ValueError(
            "paged prefill needs one INT64 request index and one INT32 prefix length"
        )
    if any(
        tensor.device != query.device
        for tensor in (
            key_cache,
            value_cache,
            req_to_token,
            request_index,
            prefix_tokens,
            virtual_to_physical,
        )
    ):
        raise ValueError("paged prefill tensors must share one device")
    query_rows, query_width = map(int, query.shape)
    cache_rows, cache_width = map(int, key_cache.shape)
    if kv_heads <= 0 or cache_width <= 0 or cache_width % kv_heads:
        raise ValueError("paged prefill cache width must divide into positive KV heads")
    head_dim = cache_width // kv_heads
    if query_width % head_dim:
        raise ValueError("paged prefill query width must divide into Q heads")
    q_heads = query_width // head_dim
    request_rows, max_context_len = map(int, req_to_token.shape)
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    query = _contiguous_prefill_query(query, stream)
    query_row_stride = int(query.stride(0))
    query_view = query.as_strided(
        (q_heads, query_rows, head_dim),
        (head_dim, query_row_stride, 1),
        storage_offset=int(query.storage_offset()),
    )
    out = torch.empty(
        (query_rows, q_heads, head_dim),
        dtype=torch.bfloat16,
        device=query.device,
    )
    out_view = out.as_strided(
        (q_heads, query_rows, head_dim),
        (head_dim, query_width, 1),
        storage_offset=int(out.storage_offset()),
    )
    request_index_view = request_index.as_strided((1, bucket_tokens), (0, 0))
    prefix_tokens_view = prefix_tokens.as_strided((1, bucket_tokens), (0, 0))
    mapping_view = virtual_to_physical.view(mapping_rows, 1)
    partitions = _paged_prefill_partition_count(kv_heads)
    if partitions == 1:
        graph_key = compile_paged_prefill_for(
            query_rows,
            q_heads,
            kv_heads,
            int(bucket_tokens),
            head_dim,
            cache_rows,
            request_rows,
            max_context_len,
            cache_row_stride,
            mapping_rows,
            query_row_stride,
            query_width,
        )
        _launch_paged_prefill_graph(
            graph_key,
            (
                query_view,
                key_cache,
                value_cache,
                req_to_token,
                request_index_view,
                prefix_tokens_view,
                mapping_view,
                out_view,
            ),
            stream,
        )
    else:
        queries_per_kv = q_heads // kv_heads
        graph_key = compile_paged_prefill_for(
            query_rows,
            queries_per_kv,
            1,
            int(bucket_tokens),
            head_dim,
            cache_rows,
            request_rows,
            max_context_len,
            cache_row_stride,
            mapping_rows,
            query_row_stride,
            query_width,
        )
        for kv_head in range(kv_heads):
            query_start = kv_head * queries_per_kv
            query_limit = query_start + queries_per_kv
            cache_start = kv_head * head_dim
            cache_limit = cache_start + head_dim
            _launch_paged_prefill_graph(
                graph_key,
                (
                    query_view[query_start:query_limit, :, :],
                    key_cache[:, cache_start:cache_limit],
                    value_cache[:, cache_start:cache_limit],
                    req_to_token,
                    request_index_view,
                    prefix_tokens_view,
                    mapping_view,
                    out_view[query_start:query_limit, :, :],
                ),
                stream,
            )
    return out.view(query_rows, query_width)


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
