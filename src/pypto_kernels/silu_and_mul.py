"""SiLU-and-mul as one native PyPTO tile kernel and one launch."""

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
def silu_and_mul_kernel(
    gate: pl.Tensor,
    up: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Tile source: ``silu(gate) * up`` over contiguous matrix rows."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(gate.shape[0]):
            for block in pl.range(gate.shape[1] // 128):
                tile_gate = pl.load(gate, [row, block * 128], [1, 128])
                tile_up = pl.load(up, [row, block * 128], [1, 128])
                gate_neg = pl.mul(tile_gate, -1.0)
                exp_neg = pl.exp(gate_neg)
                denominator = pl.add(exp_neg, 1.0)
                sigmoid = pl.recip(denominator)
                silu_gate = pl.mul(tile_gate, sigmoid)
                result = pl.mul(silu_gate, tile_up)
                pl.store(result, [row, block * 128], out)
    return out


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError("silu_and_mul needs a non-empty positive shape")
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    if columns % _TILE_WIDTH:
        raise ValueError(
            "silu_and_mul trailing extent must be divisible by "
            f"{_TILE_WIDTH}; got {columns}"
        )
    return rows, columns


def _tiles(rows: int) -> list[int]:
    return [_TILE_WIDTH] if rows == 1 else [1, _TILE_WIDTH]


def compile_for(
    shape: tuple[int, ...],
    dtype_name: str = "bfloat16",
    *,
    gate_row_stride: int | None = None,
    up_row_stride: int | None = None,
    out_row_stride: int | None = None,
) -> str:
    matrix_shape = _matrix_shape(shape)
    rows, columns = matrix_shape
    gate_row_stride = columns if gate_row_stride is None else gate_row_stride
    up_row_stride = columns if up_row_stride is None else up_row_stride
    out_row_stride = columns if out_row_stride is None else out_row_stride
    if min(gate_row_stride, up_row_stride, out_row_stride) < columns:
        raise ValueError("silu_and_mul row strides must cover the logical row")
    cache_key = (
        shape,
        dtype_name,
        gate_row_stride,
        up_row_stride,
        out_row_stride,
        _TILE_WIDTH,
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    import torch

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    gate = torch.empty_strided(
        matrix_shape, (gate_row_stride, 1), dtype=dtype, device="meta"
    )
    up = torch.empty_strided(
        matrix_shape, (up_row_stride, 1), dtype=dtype, device="meta"
    )
    out = torch.empty_strided(
        matrix_shape, (out_row_stride, 1), dtype=dtype, device="meta"
    )
    graph_key = compile_jit_kernel(
        silu_and_mul_kernel,
        (gate, up, out),
        _tiles(rows),
    )
    with _lock:
        _cache[cache_key] = graph_key
    return graph_key


def silu_and_mul(
    gate: Any,
    up: Any,
    stream: Any = None,
    *,
    out: Any = None,
) -> Any:
    """Return ``silu(gate) * up`` from one native tile-DSL graph launch."""

    import torch

    if stream is None:
        stream = torch.cuda.current_stream(gate.device)
    if gate.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
        raise ValueError("silu_and_mul needs BF16 operands")
    shape = tuple(gate.shape)
    if tuple(up.shape) != shape:
        raise ValueError("silu_and_mul needs matching operands")
    matrix_shape = _matrix_shape(shape)
    gate_matrix = gate.view(matrix_shape)
    up_matrix = up.view(matrix_shape)
    if gate_matrix.stride(1) != 1 or up_matrix.stride(1) != 1:
        raise ValueError("silu_and_mul needs unit inner strides")
    if out is None:
        out = torch.empty(shape, dtype=gate.dtype, device=gate.device)
    if tuple(out.shape) != shape or out.dtype is not gate.dtype:
        raise ValueError("silu_and_mul output shape and dtype must match")
    out_matrix = out.view(matrix_shape)
    if out_matrix.stride(1) != 1:
        raise ValueError("silu_and_mul output needs unit inner stride")
    graph_key = compile_for(
        shape,
        "bfloat16",
        gate_row_stride=int(gate_matrix.stride(0)),
        up_row_stride=int(up_matrix.stride(0)),
        out_row_stride=int(out_matrix.stride(0)),
    )
    launch_graph(
        graph_key,
        (gate_matrix, up_matrix, out_matrix),
        stream.cuda_stream,
    )
    return out
