"""Qwen GDN width-4 causal depthwise convolution as one native tile graph."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_graph, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_CHANNEL_TILE = 128
_KERNEL_WIDTH = 4
_lock = threading.RLock()
_cache: dict[tuple[int, int, int, int, int, str], str] = {}
_MAX_FUSED_TOKENS = 1

STATUS = "native-tile stateful executable"
GRAPHS = 1


@pl.jit
def causal_conv1d_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    state: pl.InOut[pl.Tensor],
    state_indices: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Apply width-four causal convolution and update each slot's history."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for batch_row in pl.range(x.shape[0]):
            state_index_i64 = pl.read(state_indices, [batch_row, 0])
            state_index = pl.cast(state_index_i64, pl.INT32)
            for channel_block in pl.range(x.shape[2] // 128):
                channel_offset = channel_block * 128
                history0_box = pl.load(
                    state,
                    [state_index, channel_offset],
                    [1, 128],
                )
                history1_box = pl.load(
                    state,
                    [state_index, x.shape[2] + channel_offset],
                    [1, 128],
                )
                history2_box = pl.load(
                    state,
                    [state_index, 2 * x.shape[2] + channel_offset],
                    [1, 128],
                )
                history0 = pl.cast(
                    pl.reshape(history0_box, [128, 1]), target_type=pl.FP32
                )
                history1 = pl.cast(
                    pl.reshape(history1_box, [128, 1]), target_type=pl.FP32
                )
                history2 = pl.cast(
                    pl.reshape(history2_box, [128, 1]), target_type=pl.FP32
                )

                weight_tile = pl.load(weight, [channel_block * 128, 0], [128, 4])
                weight_wide = pl.cast(weight_tile, target_type=pl.FP32)
                weight0 = pl.tile.slice(weight_wide, [128, 1], [0, 0])
                weight1 = pl.tile.slice(weight_wide, [128, 1], [0, 1])
                weight2 = pl.tile.slice(weight_wide, [128, 1], [0, 2])
                weight3 = pl.tile.slice(weight_wide, [128, 1], [0, 3])

                for token in pl.range(x.shape[1]):
                    current_box = pl.load(
                        x,
                        [batch_row, token, channel_block * 128],
                        [1, 1, 128],
                    )
                    current = pl.reshape(current_box, [128, 1])
                    current_wide = pl.cast(current, target_type=pl.FP32)
                    term0 = pl.mul(history0, weight0)
                    term1 = pl.mul(history1, weight1)
                    term2 = pl.mul(history2, weight2)
                    term3 = pl.mul(current_wide, weight3)
                    convolution = pl.add(pl.add(term0, term1), pl.add(term2, term3))
                    negative = pl.neg(convolution)
                    exponent = pl.exp(negative)
                    denominator = pl.add(exponent, 1.0)
                    sigmoid = pl.recip(denominator)
                    activated = pl.mul(convolution, sigmoid)
                    result = pl.cast(activated, target_type=pl.BF16)
                    result_box = pl.reshape(result, [1, 1, 128])
                    pl.store(
                        result_box,
                        [batch_row, token, channel_block * 128],
                        out,
                    )
                    history0 = history1
                    history1 = history2
                    history2 = current_wide

                final_history0 = pl.reshape(
                    pl.cast(history0, target_type=pl.BF16), [1, 128]
                )
                final_history1 = pl.reshape(
                    pl.cast(history1, target_type=pl.BF16), [1, 128]
                )
                final_history2 = pl.reshape(
                    pl.cast(history2, target_type=pl.BF16), [1, 128]
                )
                pl.store(final_history0, [state_index, channel_offset], state)
                pl.store(
                    final_history1,
                    [state_index, x.shape[2] + channel_offset],
                    state,
                )
                pl.store(
                    final_history2,
                    [state_index, 2 * x.shape[2] + channel_offset],
                    state,
                )
    return out


def _validate_shape(
    batch_size: int,
    tokens_per_request: int,
    channels: int,
    state_slots: int,
    state_slot_stride: int,
) -> None:
    if channels <= 0 or channels % _CHANNEL_TILE:
        raise ValueError("causal_conv1d channels must be positive and divisible by 128")
    if batch_size <= 0 or tokens_per_request <= 0 or state_slots <= 0:
        raise ValueError("causal_conv1d batch, tokens and state slots must be positive")
    if state_slot_stride < channels * (_KERNEL_WIDTH - 1):
        raise ValueError("causal_conv1d state slots overlap")


def build(
    batch_size: int,
    tokens_per_request: int,
    channels: int,
    state_slots: int = 65,
    state_slot_stride: int | None = None,
    index_dtype: str = "int32",
) -> Any:
    state_width = channels * (_KERNEL_WIDTH - 1)
    if state_slot_stride is None:
        state_slot_stride = state_width
    _validate_shape(
        batch_size,
        tokens_per_request,
        channels,
        state_slots,
        state_slot_stride,
    )
    import torch

    if index_dtype == "int32":
        torch_index_dtype = torch.int32
    elif index_dtype == "int64":
        torch_index_dtype = torch.int64
    else:
        raise ValueError("causal_conv1d state indices must use int32 or int64")
    x = torch.empty(
        (batch_size, tokens_per_request, channels),
        dtype=torch.bfloat16,
        device="meta",
    )
    weight = torch.empty((channels, _KERNEL_WIDTH), dtype=torch.bfloat16, device="meta")
    state = torch.empty_strided(
        (state_slots, state_width),
        (state_slot_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    state_indices = torch.empty((batch_size, 1), dtype=torch_index_dtype, device="meta")
    return causal_conv1d_kernel.specialize(x, weight, state, state_indices, x)


def compile_for(
    batch_size: int,
    tokens_per_request: int,
    channels: int,
    state_slots: int,
    state_slot_stride: int,
    index_dtype: str,
) -> str:
    if tokens_per_request > _MAX_FUSED_TOKENS:
        raise ValueError(
            "causal_conv1d fused primitive is bounded to one ordered token"
        )
    _validate_shape(
        batch_size,
        tokens_per_request,
        channels,
        state_slots,
        state_slot_stride,
    )
    shape_key = (
        batch_size,
        tokens_per_request,
        channels,
        state_slots,
        state_slot_stride,
        index_dtype,
    )
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    graph_key = compile_graph(build(*shape_key), [1, 1, _CHANNEL_TILE])
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def causal_conv1d(
    x: Any,
    weight: Any,
    state: Any,
    state_indices: Any,
    *,
    batch_size: int,
    tokens_per_request: int,
    stream: Any = None,
) -> Any:
    """Run ordered stateful width-four convolution and SiLU PyPTO launches."""

    import torch

    if (
        x.ndim != 2
        or x.dtype is not torch.bfloat16
        or not x.is_contiguous()
        or weight.ndim != 2
        or weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("causal_conv1d needs contiguous rank-2 BF16 tensors")
    rows, channels = map(int, x.shape)
    if tuple(weight.shape) != (channels, _KERNEL_WIDTH):
        raise ValueError("causal_conv1d weight must have shape [channels, 4]")
    if rows != batch_size * tokens_per_request:
        raise ValueError("causal_conv1d rows disagree with batch/token geometry")
    if state.ndim != 3 or state.dtype is not torch.bfloat16:
        raise ValueError("causal_conv1d state must be rank-3 BF16")
    state_slots = int(state.shape[0])
    if tuple(state.shape[1:]) != (_KERNEL_WIDTH - 1, channels):
        raise ValueError("causal_conv1d state payload must have shape [3,channels]")
    if (
        state.stride(2) != 1
        or state.stride(1) != channels
        or state.stride(0) < channels * (_KERNEL_WIDTH - 1)
    ):
        raise ValueError(
            "causal_conv1d state payload must be contiguous within each slot"
        )
    if (
        state_indices.ndim != 1
        or state_indices.numel() != batch_size
        or state_indices.dtype not in (torch.int32, torch.int64)
        or not state_indices.is_contiguous()
    ):
        raise ValueError("causal_conv1d needs one INT32/INT64 state index per request")
    if any(tensor.device != x.device for tensor in (weight, state, state_indices)):
        raise ValueError("causal_conv1d tensors must share one device")
    index_dtype = "int32" if state_indices.dtype is torch.int32 else "int64"
    state_slot_stride = int(state.stride(0))
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    state_view = state.view(state_slots, -1)
    out = torch.empty(
        (batch_size, tokens_per_request, channels),
        dtype=x.dtype,
        device=x.device,
    )

    def launch_views(graph_key: str, views: tuple[Any, ...]) -> None:
        launch_graph(graph_key, views, stream.cuda_stream)

    if tokens_per_request == 1:
        graph_key = compile_for(
            batch_size,
            1,
            channels,
            state_slots,
            state_slot_stride,
            index_dtype,
        )
        launch_views(
            graph_key,
            (
                x.view(batch_size, 1, channels),
                weight,
                state_view,
                state_indices.view(batch_size, 1),
                out,
            ),
        )
    else:
        graph_key = compile_for(
            1,
            1,
            channels,
            state_slots,
            state_slot_stride,
            index_dtype,
        )
        out_rows = out.view(rows, channels)
        for request_row in range(batch_size):
            for token in range(tokens_per_request):
                flat_row = request_row * tokens_per_request + token
                launch_views(
                    graph_key,
                    (
                        x.narrow(0, flat_row, 1).view(1, 1, channels),
                        weight,
                        state_view,
                        state_indices.narrow(0, request_row, 1).view(1, 1),
                        out_rows.narrow(0, flat_row, 1).view(1, 1, channels),
                    ),
                )
    return out.view(rows, channels)


def status(
    batch_size: int = 1,
    tokens_per_request: int = 64,
    channels: int = 2048,
    state_slots: int = 65,
) -> dict[str, str]:
    state_stride = channels * (_KERNEL_WIDTH - 1)
    try:
        return {
            "status": "compiled",
            "key": compile_for(
                batch_size,
                tokens_per_request,
                channels,
                state_slots,
                state_stride,
                "int32",
            ),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
