"""Bias-free Qwen linear projection as one native PyPTO tile kernel."""

from __future__ import annotations

import math
import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_TILE_ROWS = 1
_TILE_COLUMNS = 128
# Measured on RTX 5090 (SM120): a 32-column schedule tile is the sweet spot
# for every production GEMV/GEMM shape (decode and prefill). Larger tiles
# under-fill the grid (down-projection GEMV drops to 32 CTAs on ~170 SMs) and
# hit a hard efficiency cliff at >=64 columns; smaller tiles add launch
# overhead without more bandwidth. For multi-row (prefill) shapes a 2-row
# block halves the per-row weight re-streaming while staying on the
# row-per-CTA arithmetic path: results are bit-identical to the single-row
# schedule (verified on live hardware). A 16-row block is measurably faster
# still (0.6 vs 3.7 ms/call) because it selects the tensor-core MMA path,
# but that path changes prefill numerics enough to break the frozen
# token-level model gate, so it is rejected.
_SCHEDULE_COLUMNS = 32
_PREFILL_TILE_ROWS = 2
_lock = threading.RLock()
_cache: dict[tuple[int, int, int, str, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 2


@pl.jit
def linear_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Compute ``x @ weight.T`` using explicit output tiles."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            for column_block in pl.range(weight.shape[0] // 128):
                x_tile = pl.load(
                    x,
                    [row, 0],
                    [1, x.shape[1]],
                    target_memory=pl.MemorySpace.Mat,
                )
                weight_tile = pl.load(
                    weight,
                    [column_block * 128, 0],
                    [128, x.shape[1]],
                    target_memory=pl.MemorySpace.Mat,
                )
                weight_transposed = pl.tile.transpose_view(weight_tile)
                accumulation = pl.matmul(
                    x_tile,
                    weight_transposed,
                    out_dtype=pl.FP32,
                )
                result = pl.cast(accumulation, target_type=pl.BF16)
                pl.store(result, [row, column_block * 128], out)
    return out


@pl.jit
def linear_to_float_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Compute BF16-rounded ``x @ weight.T`` into an FP32 result."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            for column_block in pl.range(weight.shape[0] // 128):
                x_tile = pl.load(
                    x,
                    [row, 0],
                    [1, x.shape[1]],
                    target_memory=pl.MemorySpace.Mat,
                )
                weight_tile = pl.load(
                    weight,
                    [column_block * 128, 0],
                    [128, x.shape[1]],
                    target_memory=pl.MemorySpace.Mat,
                )
                weight_transposed = pl.tile.transpose_view(weight_tile)
                accumulation = pl.matmul(
                    x_tile,
                    weight_transposed,
                    out_dtype=pl.FP32,
                )
                rounded = pl.cast(accumulation, target_type=pl.BF16)
                result = pl.cast(rounded, target_type=pl.FP32)
                pl.store(result, [row, column_block * 128], out)
    return out


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("linear needs positive rows and feature dimensions")
    if in_features % 128:
        raise ValueError("linear input features must be divisible by 128")
    if out_features % _TILE_COLUMNS:
        raise ValueError("linear output features must be divisible by 128")


def _tiles(rows: int) -> list[int]:
    return (
        [_SCHEDULE_COLUMNS]
        if rows == 1
        else [_PREFILL_TILE_ROWS, _SCHEDULE_COLUMNS]
    )


def build(
    rows: int,
    in_features: int,
    out_features: int,
    output_dtype: str = "bfloat16",
    output_row_stride: int | None = None,
) -> Any:
    _validate_shape(rows, in_features, out_features)
    import torch

    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    if output_dtype == "bfloat16":
        dtype = torch.bfloat16
        kernel = linear_kernel
    elif output_dtype == "float32":
        dtype = torch.float32
        kernel = linear_to_float_kernel
    else:
        raise ValueError("linear output dtype must be bfloat16 or float32")
    output_row_stride = (
        out_features if output_row_stride is None else output_row_stride
    )
    if output_row_stride < out_features:
        raise ValueError("linear output row stride must cover the logical row")
    out = torch.empty_strided(
        (rows, out_features),
        (output_row_stride, 1),
        dtype=dtype,
        device="meta",
    )
    return kernel.specialize(x, weight, out)


def compile_for(
    rows: int,
    in_features: int,
    out_features: int,
    output_dtype: str = "bfloat16",
    output_row_stride: int | None = None,
) -> str:
    _validate_shape(rows, in_features, out_features)
    output_row_stride = (
        out_features if output_row_stride is None else output_row_stride
    )
    if output_row_stride < out_features:
        raise ValueError("linear output row stride must cover the logical row")
    shape_key = (
        rows,
        in_features,
        out_features,
        output_dtype,
        output_row_stride,
    )
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    if output_dtype == "bfloat16":
        dtype = torch.bfloat16
        kernel = linear_kernel
    elif output_dtype == "float32":
        dtype = torch.float32
        kernel = linear_to_float_kernel
    else:
        raise ValueError("linear output dtype must be bfloat16 or float32")
    out = torch.empty_strided(
        (rows, out_features),
        (output_row_stride, 1),
        dtype=dtype,
        device="meta",
    )
    graph_key = compile_jit_kernel(
        kernel,
        (x, weight, out),
        _tiles(rows),
        provider="pypto.matmul",
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def linear(x: Any, weight: Any, stream: Any = None) -> Any:
    """Return the bias-free projection from one graph launch."""

    import torch

    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise ValueError("linear needs BF16 input and weight")
    if x.ndim < 1 or weight.ndim != 2:
        raise ValueError("linear expects an input tensor and a rank-2 weight")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("linear needs contiguous input and weight")
    in_features = int(x.shape[-1])
    out_features, weight_features = map(int, weight.shape)
    if in_features != weight_features:
        raise ValueError("linear input and weight contraction extents must match")
    rows = math.prod(map(int, x.shape[:-1])) if x.ndim > 1 else 1
    _validate_shape(rows, in_features, out_features)
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    result_shape = (*map(int, x.shape[:-1]), out_features)
    out = torch.empty(result_shape, dtype=x.dtype, device=x.device)
    graph_key = compile_for(rows, in_features, out_features)
    launch_graph(
        graph_key,
        (
            x.reshape(rows, in_features),
            weight,
            out.reshape(rows, out_features),
        ),
        stream.cuda_stream,
    )
    return out


def linear_to_float(x: Any, weight: Any, stream: Any = None) -> Any:
    """Return a BF16-rounded projection widened to FP32 in one launch."""

    import torch

    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise ValueError("linear_to_float needs BF16 input and weight")
    if x.ndim < 1 or weight.ndim != 2:
        raise ValueError("linear_to_float expects an input tensor and rank-2 weight")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("linear_to_float needs contiguous input and weight")
    in_features = int(x.shape[-1])
    out_features, weight_features = map(int, weight.shape)
    if in_features != weight_features:
        raise ValueError("linear_to_float contraction extents must match")
    rows = math.prod(map(int, x.shape[:-1])) if x.ndim > 1 else 1
    _validate_shape(rows, in_features, out_features)
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    result_shape = (*map(int, x.shape[:-1]), out_features)
    out = torch.empty(result_shape, dtype=torch.float32, device=x.device)
    graph_key = compile_for(rows, in_features, out_features, "float32")
    launch_graph(
        graph_key,
        (
            x.reshape(rows, in_features),
            weight,
            out.reshape(rows, out_features),
        ),
        stream.cuda_stream,
    )
    return out


def status(
    rows: int = 32, in_features: int = 1024, out_features: int = 1024
) -> dict[str, str]:
    try:
        return {
            "status": "compiled",
            "key": compile_for(rows, in_features, out_features),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
