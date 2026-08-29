"""Single-graph Gated DeltaNet recurrence for Qwen3.5."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_graph, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_lock = threading.RLock()
_recurrent_cache: dict[tuple[int, int, int, int, int, int, int, int, str], str] = {}
_MAX_FUSED_TOKENS = 1

STATUS = "native-tile recurrent executable"
GRAPHS = 1
UPDATE_GRAPHS = 0


@pl.jit
def gdn_recurrent_kernel(
    mixed_qkv: pl.Tensor,
    a: pl.Tensor,
    b: pl.Tensor,
    A_log: pl.Tensor,
    dt_bias: pl.Tensor,
    state: pl.InOut[pl.Tensor],
    state_indices: pl.Tensor,
    key_dim: pl.INT64,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Run ordered Gated DeltaNet recurrence and update the FP32 state pool."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for batch_row in pl.range(mixed_qkv.shape[0]):
            state_index_i64 = pl.read(state_indices, [batch_row, 0])
            state_index = pl.cast(state_index_i64, pl.INT32)
            q_heads = (mixed_qkv.shape[2] - out.shape[2] * out.shape[3]) // (
                2 * key_dim
            )
            value_heads_per_q_head = out.shape[2] // q_heads
            for value_head in pl.range(out.shape[2]):
                query_head = value_head // value_heads_per_q_head
                state_offset = value_head * key_dim * out.shape[3]
                state_box = pl.load(
                    state,
                    [state_index, state_offset],
                    [1, key_dim * out.shape[3]],
                )
                current_state = pl.reshape(state_box, [out.shape[3], key_dim])
                for token in pl.range(mixed_qkv.shape[1]):
                    query_offset = query_head * key_dim
                    key_offset = q_heads * key_dim + query_offset
                    value_offset = 2 * q_heads * key_dim + (value_head * out.shape[3])
                    query = pl.load(
                        mixed_qkv,
                        [batch_row, token, query_offset],
                        [1, 1, key_dim],
                    )
                    key = pl.load(
                        mixed_qkv,
                        [batch_row, token, key_offset],
                        [1, 1, key_dim],
                    )
                    value = pl.load(
                        mixed_qkv,
                        [batch_row, token, value_offset],
                        [1, 1, out.shape[3]],
                    )
                    query = pl.reshape(query, [1, key_dim])
                    key = pl.reshape(key, [1, key_dim])
                    value = pl.reshape(value, [out.shape[3], 1])

                    query_wide = pl.cast(query, target_type=pl.FP32)
                    query_square = pl.mul(query_wide, query_wide)
                    query_scratch = pl.create_tile(
                        [1, key_dim],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    query_norm2 = pl.row_sum(query_square, query_scratch)
                    query_norm2 = pl.add(query_norm2, 1.0e-6)
                    query_inv_norm = pl.rsqrt(query_norm2)
                    query_normalized = pl.row_expand_mul(query_wide, query_inv_norm)
                    query_scaled = pl.mul(query_normalized, scale)

                    key_wide = pl.cast(key, target_type=pl.FP32)
                    key_square = pl.mul(key_wide, key_wide)
                    key_scratch = pl.create_tile(
                        [1, key_dim],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    key_norm2 = pl.row_sum(key_square, key_scratch)
                    key_norm2 = pl.add(key_norm2, 1.0e-6)
                    key_inv_norm = pl.rsqrt(key_norm2)
                    key_normalized = pl.row_expand_mul(key_wide, key_inv_norm)

                    a_box = pl.load(
                        a,
                        [batch_row, token, value_head, 0],
                        [1, 1, 1, out.shape[3]],
                    )
                    b_box = pl.load(
                        b,
                        [batch_row, token, value_head, 0],
                        [1, 1, 1, out.shape[3]],
                    )
                    A_box = pl.load(
                        A_log,
                        [value_head, 0],
                        [1, out.shape[3]],
                    )
                    dt_box = pl.load(
                        dt_bias,
                        [value_head, 0],
                        [1, out.shape[3]],
                    )
                    a_wide = pl.cast(
                        pl.reshape(a_box, [out.shape[3], 1]),
                        target_type=pl.FP32,
                    )
                    b_wide = pl.cast(
                        pl.reshape(b_box, [out.shape[3], 1]),
                        target_type=pl.FP32,
                    )
                    A_wide = pl.reshape(A_box, [out.shape[3], 1])
                    dt_wide = pl.cast(
                        pl.reshape(dt_box, [out.shape[3], 1]),
                        target_type=pl.FP32,
                    )

                    gate_x = pl.add(a_wide, dt_wide)
                    gate_abs = pl.abs(gate_x)
                    gate_tail = pl.exp(pl.neg(gate_abs))
                    gate_tail = pl.log(pl.add(gate_tail, 1.0))
                    softplus = pl.add(pl.maximums(gate_x, 0.0), gate_tail)
                    log_decay = pl.neg(pl.mul(pl.exp(A_wide), softplus))
                    decay = pl.exp(log_decay)
                    decayed_state = pl.row_expand_mul(current_state, decay)

                    beta_x = pl.neg(b_wide)
                    beta_abs = pl.abs(beta_x)
                    beta_tail = pl.exp(pl.neg(beta_abs))
                    beta_tail = pl.log(pl.add(beta_tail, 1.0))
                    beta_softplus = pl.add(pl.maximums(beta_x, 0.0), beta_tail)
                    beta = pl.exp(pl.neg(beta_softplus))
                    beta_storage = pl.cast(beta, target_type=pl.BF16)
                    beta = pl.cast(beta_storage, target_type=pl.FP32)

                    key_column = pl.tile.transpose_view(key_normalized)
                    state_key = pl.matmul(decayed_state, key_column, out_dtype=pl.FP32)
                    value_wide = pl.cast(value, target_type=pl.FP32)
                    residual_value = pl.sub(value_wide, state_key)
                    delta_value = pl.mul(residual_value, beta)
                    delta_full = pl.row_expand(current_state, delta_value)
                    outer = pl.col_expand_mul(delta_full, key_normalized)
                    current_state = pl.add(decayed_state, outer)

                    state_transposed = pl.tile.transpose_view(current_state)
                    output_wide = pl.matmul(
                        query_scaled, state_transposed, out_dtype=pl.FP32
                    )
                    output = pl.cast(output_wide, target_type=pl.BF16)
                    output_box = pl.reshape(output, [1, 1, 1, out.shape[3]])
                    pl.store(
                        output_box,
                        [batch_row, token, value_head, 0],
                        out,
                    )

                state_result = pl.reshape(
                    current_state,
                    [1, key_dim * out.shape[3]],
                )
                pl.store(state_result, [state_index, state_offset], state)
    return out


def build_recurrent(
    batch_size: int,
    tokens_per_request: int,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    state_slots: int,
    state_slot_stride: int | None = None,
    index_dtype: str = "int32",
) -> Any:
    """Build one ordered recurrent graph for decode or uniform prefill."""

    if (
        batch_size <= 0
        or tokens_per_request <= 0
        or q_heads <= 0
        or value_heads <= 0
        or value_heads % q_heads
        or key_dim <= 0
        or key_dim % 128
        or value_dim <= 0
        or value_dim % 16
        or state_slots <= 0
    ):
        raise ValueError("GDN recurrent graph received invalid static geometry")
    state_width = value_heads * value_dim * key_dim
    if state_slot_stride is None:
        state_slot_stride = state_width
    if state_slot_stride < state_width:
        raise ValueError("GDN recurrent state slot stride overlaps adjacent slots")
    import torch

    if index_dtype == "int32":
        torch_index_dtype = torch.int32
    elif index_dtype == "int64":
        torch_index_dtype = torch.int64
    else:
        raise ValueError("GDN recurrent state indices must use int32 or int64")
    mixed_width = 2 * q_heads * key_dim + value_heads * value_dim
    mixed_qkv = torch.empty(
        (batch_size, tokens_per_request, mixed_width),
        dtype=torch.bfloat16,
        device="meta",
    )
    gate_strides = (
        tokens_per_request * value_heads,
        value_heads,
        1,
        0,
    )
    a = torch.empty_strided(
        (batch_size, tokens_per_request, value_heads, value_dim),
        gate_strides,
        dtype=torch.bfloat16,
        device="meta",
    )
    b = torch.empty_strided(
        (batch_size, tokens_per_request, value_heads, value_dim),
        gate_strides,
        dtype=torch.bfloat16,
        device="meta",
    )
    parameter_strides = (1, 0)
    A_log = torch.empty_strided(
        (value_heads, value_dim),
        parameter_strides,
        dtype=torch.float32,
        device="meta",
    )
    dt_bias = torch.empty_strided(
        (value_heads, value_dim),
        parameter_strides,
        dtype=torch.bfloat16,
        device="meta",
    )
    state = torch.empty_strided(
        (state_slots, state_width),
        (state_slot_stride, 1),
        dtype=torch.float32,
        device="meta",
    )
    state_indices = torch.empty((batch_size, 1), dtype=torch_index_dtype, device="meta")
    out = torch.empty(
        (batch_size, tokens_per_request, value_heads, value_dim),
        dtype=torch.bfloat16,
        device="meta",
    )
    return gdn_recurrent_kernel.specialize(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        key_dim,
        key_dim**-0.5,
        out,
    )


def _recurrent_tiles(
    batch_size: int,
    tokens_per_request: int,
    q_heads: int,
    value_heads: int,
    value_dim: int,
) -> list[int]:
    """Tile only non-unit dimensions of the typed TensorIR iteration space."""

    head_groups = value_heads // q_heads
    outer_extents = (batch_size, tokens_per_request, q_heads, head_groups)
    tiles = [1 for extent in outer_extents if extent > 1]
    tiles.append(1 << min(7, value_dim.bit_length() - 1))
    return tiles


def compile_recurrent(
    batch_size: int,
    tokens_per_request: int,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    state_slots: int,
    state_slot_stride: int,
    index_dtype: str,
) -> str:
    if tokens_per_request > _MAX_FUSED_TOKENS:
        raise ValueError(
            "GDN recurrent fused primitive is bounded to one ordered token"
        )
    shape_key = (
        batch_size,
        tokens_per_request,
        q_heads,
        value_heads,
        key_dim,
        value_dim,
        state_slots,
        state_slot_stride,
        index_dtype,
    )
    cached = _recurrent_cache.get(shape_key)
    if cached is not None:
        return cached
    program = build_recurrent(*shape_key)
    tiles = _recurrent_tiles(
        batch_size, tokens_per_request, q_heads, value_heads, value_dim
    )
    graph_key = compile_graph(
        program,
        tiles,
        provider="pypto.gdn",
        source_node="pypto_kernels.gdn:recurrent",
    )
    with _lock:
        _recurrent_cache[shape_key] = graph_key
    return graph_key


def gdn_recurrent(
    mixed_qkv: Any,
    a: Any,
    b: Any,
    A_log: Any,
    dt_bias: Any,
    state: Any,
    state_indices: Any,
    *,
    batch_size: int,
    tokens_per_request: int,
    stream: Any = None,
) -> Any:
    """Execute ordered PyPTO recurrence and update ``state`` in place.

    Long prefills use ordered, CUDA-graph-capturable single-token launches so
    every recurrent state boundary has the numerically accepted flat layout.
    """

    import torch

    if (
        mixed_qkv.ndim != 2
        or mixed_qkv.dtype is not torch.bfloat16
        or not mixed_qkv.is_contiguous()
        or a.ndim != 2
        or b.ndim != 2
        or a.dtype is not torch.bfloat16
        or b.dtype is not torch.bfloat16
        or not a.is_contiguous()
        or not b.is_contiguous()
        or tuple(a.shape) != tuple(b.shape)
    ):
        raise ValueError("GDN recurrent QKV/a/b inputs must be contiguous rank-2 BF16")
    if (
        A_log.ndim != 1
        or dt_bias.ndim != 1
        or A_log.dtype is not torch.float32
        or dt_bias.dtype is not torch.bfloat16
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
        or tuple(A_log.shape) != tuple(dt_bias.shape)
    ):
        raise ValueError(
            "GDN recurrent A_log/dt_bias must be matching FP32/BF16 vectors"
        )
    if state.ndim != 4 or state.dtype is not torch.float32:
        raise ValueError("GDN recurrent state must be rank-4 FP32")
    state_slots, value_heads, value_dim, key_dim = map(int, state.shape)
    if (
        state.stride(3) != 1
        or state.stride(2) != key_dim
        or state.stride(1) != value_dim * key_dim
        or state.stride(0) < value_heads * value_dim * key_dim
    ):
        raise ValueError(
            "GDN recurrent state payload must be contiguous within each slot"
        )
    if batch_size <= 0 or tokens_per_request <= 0:
        raise ValueError("GDN recurrent batch and token extents must be positive")
    rows = batch_size * tokens_per_request
    if mixed_qkv.shape[0] != rows or a.shape != (rows, value_heads):
        raise ValueError("GDN recurrent rows or value-head gates are incompatible")
    if A_log.numel() != value_heads:
        raise ValueError("GDN recurrent gate parameter count must equal value heads")
    qk_width = int(mixed_qkv.shape[1]) - value_heads * value_dim
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError("GDN recurrent packed QKV width is incompatible with state")
    q_heads = qk_width // (2 * key_dim)
    if q_heads <= 0 or value_heads % q_heads:
        raise ValueError("GDN recurrent Q/value head grouping is invalid")
    if (
        state_indices.ndim != 1
        or state_indices.numel() != batch_size
        or state_indices.dtype not in (torch.int32, torch.int64)
        or not state_indices.is_contiguous()
    ):
        raise ValueError(
            "GDN recurrent needs one contiguous INT32/INT64 state index per request"
        )
    if any(
        tensor.device != mixed_qkv.device
        for tensor in (a, b, A_log, dt_bias, state, state_indices)
    ):
        raise ValueError("GDN recurrent tensors must share one device")
    index_dtype = "int32" if state_indices.dtype is torch.int32 else "int64"
    state_slot_stride = int(state.stride(0))
    if stream is None:
        stream = torch.cuda.current_stream(mixed_qkv.device)
    A_view = A_log.as_strided((value_heads, value_dim), (1, 0))
    dt_view = dt_bias.as_strided((value_heads, value_dim), (1, 0))
    state_view = state.view(state_slots, -1)
    out = torch.empty(
        (batch_size, tokens_per_request, value_heads, value_dim),
        dtype=torch.bfloat16,
        device=mixed_qkv.device,
    )

    def launch_views(graph_key: str, views: tuple[Any, ...]) -> None:
        launch_graph(graph_key, views, stream.cuda_stream)

    def launch_chunk(*, request_row: int, token_start: int, token_count: int) -> None:
        flat_start = request_row * tokens_per_request + token_start
        graph_key = compile_recurrent(
            1,
            token_count,
            q_heads,
            value_heads,
            key_dim,
            value_dim,
            state_slots,
            state_slot_stride,
            index_dtype,
        )
        mixed_view = mixed_qkv.narrow(0, flat_start, token_count).view(
            1, token_count, -1
        )
        gate_strides = (token_count * value_heads, value_heads, 1, 0)
        a_view = (
            a.narrow(0, flat_start, token_count)
            .view(1, token_count, value_heads)
            .as_strided((1, token_count, value_heads, value_dim), gate_strides)
        )
        b_view = (
            b.narrow(0, flat_start, token_count)
            .view(1, token_count, value_heads)
            .as_strided((1, token_count, value_heads, value_dim), gate_strides)
        )
        out_view = (
            out.view(rows, value_heads, value_dim)
            .narrow(0, flat_start, token_count)
            .view(1, token_count, value_heads, value_dim)
        )
        launch_views(
            graph_key,
            (
                mixed_view,
                a_view,
                b_view,
                A_view,
                dt_view,
                state_view,
                state_indices.narrow(0, request_row, 1).view(1, 1),
                out_view,
            ),
        )

    if tokens_per_request <= _MAX_FUSED_TOKENS:
        graph_key = compile_recurrent(
            batch_size,
            tokens_per_request,
            q_heads,
            value_heads,
            key_dim,
            value_dim,
            state_slots,
            state_slot_stride,
            index_dtype,
        )
        mixed_view = mixed_qkv.view(batch_size, tokens_per_request, -1)
        gate_strides = (
            tokens_per_request * value_heads,
            value_heads,
            1,
            0,
        )
        a_view = a.view(batch_size, tokens_per_request, value_heads).as_strided(
            (batch_size, tokens_per_request, value_heads, value_dim),
            gate_strides,
        )
        b_view = b.view(batch_size, tokens_per_request, value_heads).as_strided(
            (batch_size, tokens_per_request, value_heads, value_dim),
            gate_strides,
        )
        launch_views(
            graph_key,
            (
                mixed_view,
                a_view,
                b_view,
                A_view,
                dt_view,
                state_view,
                state_indices.view(batch_size, 1),
                out,
            ),
        )
    else:
        for request_row in range(batch_size):
            for token_start in range(0, tokens_per_request, _MAX_FUSED_TOKENS):
                launch_chunk(
                    request_row=request_row,
                    token_start=token_start,
                    token_count=min(
                        _MAX_FUSED_TOKENS,
                        tokens_per_request - token_start,
                    ),
                )
    return out


def recurrent_status(
    batch_size: int = 2,
    tokens_per_request: int = 1,
    q_heads: int = 8,
    value_heads: int = 16,
    key_dim: int = 128,
    value_dim: int = 128,
    state_slots: int = 65,
) -> dict[str, str]:
    state_stride = value_heads * value_dim * key_dim
    try:
        return {
            "status": "compiled",
            "key": compile_recurrent(
                batch_size,
                tokens_per_request,
                q_heads,
                value_heads,
                key_dim,
                value_dim,
                state_slots,
                state_stride,
                "int32",
            ),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
