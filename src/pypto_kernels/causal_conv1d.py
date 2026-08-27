"""Qwen GDN width-4 causal depthwise convolution as one native tile graph."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_CHANNEL_TILE = 128
_KERNEL_WIDTH = 4
_lock = threading.RLock()
_cache: dict[tuple[int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def causal_conv1d_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Apply four causal depthwise taps and SiLU per channel tile."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for channel_block in pl.range(x.shape[0] // 128):
            x_tile = pl.load(x, [channel_block * 128, 0], [128, x.shape[1]])
            weight_tile = pl.load(weight, [channel_block * 128, 0], [128, 4])
            x_wide = pl.cast(x_tile, target_type=pl.FP32)
            weight_wide = pl.cast(weight_tile, target_type=pl.FP32)

            zero3 = pl.tile.full([128, 3], dtype=pl.FP32, value=0.0)
            body0 = pl.tile.slice(x_wide, [128, x.shape[1] - 3], [0, 0])
            shifted0 = pl.tile.concat(zero3, body0)
            weight0 = pl.tile.slice(weight_wide, [128, 1], [0, 0])
            term0 = pl.row_expand_mul(shifted0, weight0)

            zero2 = pl.tile.full([128, 2], dtype=pl.FP32, value=0.0)
            body1 = pl.tile.slice(x_wide, [128, x.shape[1] - 2], [0, 0])
            shifted1 = pl.tile.concat(zero2, body1)
            weight1 = pl.tile.slice(weight_wide, [128, 1], [0, 1])
            term1 = pl.row_expand_mul(shifted1, weight1)

            zero1 = pl.tile.full([128, 1], dtype=pl.FP32, value=0.0)
            body2 = pl.tile.slice(x_wide, [128, x.shape[1] - 1], [0, 0])
            shifted2 = pl.tile.concat(zero1, body2)
            weight2 = pl.tile.slice(weight_wide, [128, 1], [0, 2])
            term2 = pl.row_expand_mul(shifted2, weight2)

            weight3 = pl.tile.slice(weight_wide, [128, 1], [0, 3])
            term3 = pl.row_expand_mul(x_wide, weight3)

            sum01 = pl.add(term0, term1)
            sum23 = pl.add(term2, term3)
            convolution = pl.add(sum01, sum23)
            negative = pl.mul(convolution, -1.0)
            exponent = pl.exp(negative)
            denominator = pl.add(exponent, 1.0)
            sigmoid = pl.recip(denominator)
            activated = pl.mul(convolution, sigmoid)
            result = pl.cast(activated, target_type=pl.BF16)
            pl.store(result, [channel_block * 128, 0], out)
    return out


def _validate_shape(channels: int, tokens: int) -> None:
    if channels <= 0 or channels % _CHANNEL_TILE:
        raise ValueError("causal_conv1d channels must be positive and divisible by 128")
    if tokens < _KERNEL_WIDTH:
        raise ValueError("causal_conv1d prefill needs at least four tokens")


def build(channels: int, tokens: int) -> Any:
    _validate_shape(channels, tokens)
    import torch

    x = torch.empty((channels, tokens), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((channels, _KERNEL_WIDTH), dtype=torch.bfloat16, device="meta")
    return causal_conv1d_kernel.specialize(x, weight, x)


def compile_for(channels: int, tokens: int) -> str:
    _validate_shape(channels, tokens)
    shape_key = (channels, tokens)
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    x = torch.empty((channels, tokens), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((channels, _KERNEL_WIDTH), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        causal_conv1d_kernel,
        (x, weight, x),
        [_CHANNEL_TILE, 1],
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def causal_conv1d(x: Any, weight: Any, stream: Any = None) -> Any:
    """Return width-4 causal depthwise convolution plus SiLU in one launch."""

    import torch

    if (
        x.ndim != 2
        or x.dtype is not torch.bfloat16
        or not x.is_contiguous()
        or weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("causal_conv1d needs contiguous rank-2 BF16 tensors")
    channels, tokens = map(int, x.shape)
    if tuple(weight.shape) != (channels, _KERNEL_WIDTH):
        raise ValueError("causal_conv1d weight must have shape [channels, 4]")
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    graph_key = compile_for(channels, tokens)
    out = torch.empty_like(x)
    launch_graph(graph_key, (x, weight, out), stream.cuda_stream)
    return out


def status(channels: int = 2048, tokens: int = 64) -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for(channels, tokens)}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
