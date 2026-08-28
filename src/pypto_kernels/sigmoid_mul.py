"""Attention output gating as one native PyPTO tile kernel and one launch."""

from __future__ import annotations

import math
import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_TILE_WIDTH = 128
_lock = threading.RLock()
_cache: dict[tuple, str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def sigmoid_mul_kernel(
    value: pl.Tensor,
    gate: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Compute ``value * sigmoid(gate)`` over explicit contiguous tiles."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(value.shape[0]):
            for block in pl.range(value.shape[1] // 128):
                value_tile = pl.load(value, [row, block * 128], [1, 128])
                gate_tile = pl.load(gate, [row, block * 128], [1, 128])
                gate_neg = pl.neg(gate_tile)
                gate_exp = pl.exp(gate_neg)
                denominator = pl.add(gate_exp, 1.0)
                sigmoid = pl.recip(denominator)
                result = pl.mul(value_tile, sigmoid)
                pl.store(result, [row, block * 128], out)
    return out


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError("sigmoid_mul needs a non-empty positive shape")
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    if columns % _TILE_WIDTH:
        raise ValueError(
            f"sigmoid_mul trailing extent must be divisible by {_TILE_WIDTH}; "
            f"got {columns}"
        )
    return rows, columns


def compile_for(
    shape: tuple[int, ...],
    dtype_name: str = "bfloat16",
    value_row_stride: int | None = None,
    gate_row_stride: int | None = None,
) -> str:
    matrix_shape = _matrix_shape(shape)
    rows, columns = matrix_shape
    if value_row_stride is None:
        value_row_stride = columns
    if gate_row_stride is None:
        gate_row_stride = columns
    if value_row_stride < columns or gate_row_stride < columns:
        raise ValueError("sigmoid_mul row strides must cover each logical row")
    cache_key = (
        shape,
        dtype_name,
        value_row_stride,
        gate_row_stride,
        _TILE_WIDTH,
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    import torch

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    value = torch.empty_strided(
        matrix_shape, (value_row_stride, 1), dtype=dtype, device="meta"
    )
    gate = torch.empty_strided(
        matrix_shape, (gate_row_stride, 1), dtype=dtype, device="meta"
    )
    out = torch.empty((rows, columns), dtype=dtype, device="meta")
    graph_key = compile_jit_kernel(
        sigmoid_mul_kernel,
        (value, gate, out),
        [_TILE_WIDTH],
    )
    with _lock:
        _cache[cache_key] = graph_key
    return graph_key


def sigmoid_mul(
    value: Any,
    gate: Any,
    stream: Any = None,
    *,
    inplace: bool = False,
) -> Any:
    """Apply the full-attention output gate with one graph launch."""

    import torch

    if (
        value.shape != gate.shape
        or value.dtype is not gate.dtype
        or value.dtype not in (torch.bfloat16, torch.float32)
        or value.ndim != 2
        or value.stride(1) != 1
        or gate.stride(1) != 1
        or value.stride(0) < value.shape[1]
        or gate.stride(0) < gate.shape[1]
    ):
        raise ValueError("sigmoid_mul needs equal dense-row or row-pitched BF16/FP32 tensors")
    if stream is None:
        stream = torch.cuda.current_stream(value.device)
    shape = tuple(int(extent) for extent in value.shape)
    dtype_name = "bfloat16" if value.dtype is torch.bfloat16 else "float32"
    graph_key = compile_for(
        shape,
        dtype_name,
        int(value.stride(0)),
        int(gate.stride(0)),
    )
    if inplace and not value.is_contiguous():
        raise ValueError("in-place sigmoid_mul requires a dense output tensor")
    out = (
        value
        if inplace
        else torch.empty(
            tuple(value.shape), dtype=value.dtype, device=value.device
        )
    )
    launch_graph(graph_key, (value, gate, out), stream.cuda_stream)
    return out


def status() -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for((256, 1024))}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
