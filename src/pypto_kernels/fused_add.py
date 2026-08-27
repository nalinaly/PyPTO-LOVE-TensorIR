"""Fused residual add as one native PyPTO tile kernel and one launch."""

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
def fused_add_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Tile source: out = a + b over contiguous ``[rows, channels]``."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(a.shape[0]):
            for block in pl.range(a.shape[1] // 128):
                tile_a = pl.load(a, [row, block * 128], [1, 128])
                tile_b = pl.load(b, [row, block * 128], [1, 128])
                tile_out = pl.add(tile_a, tile_b)
                pl.store(tile_out, [row, block * 128], out)
    return out


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError("fused_add needs a non-empty positive shape")
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    if columns % _TILE_WIDTH:
        raise ValueError(
            f"fused_add trailing extent must be divisible by {_TILE_WIDTH}; "
            f"got {columns}"
        )
    return rows, columns


def compile_for(shape: tuple[int, ...], dtype_name: str = "bfloat16") -> str:
    matrix_shape = _matrix_shape(shape)
    cache_key = (shape, dtype_name, _TILE_WIDTH)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    import torch

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    sample = torch.empty(matrix_shape, dtype=dtype, device="meta")
    graph_key = compile_jit_kernel(
        fused_add_kernel,
        (sample, sample, sample),
        [_TILE_WIDTH],
    )
    with _lock:
        _cache[cache_key] = graph_key
    return graph_key


def fused_add(a: Any, b: Any, stream: Any = None) -> Any:
    """Return ``a + b`` from one native tile-DSL graph launch."""

    import torch

    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    if a.dtype is not torch.bfloat16 or tuple(a.shape) != tuple(b.shape):
        raise ValueError("fused_add needs matching BF16 operands")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("fused_add needs contiguous operands")
    shape = tuple(a.shape)
    matrix_shape = _matrix_shape(shape)
    graph_key = compile_for(shape, "bfloat16")
    out = torch.empty_like(a)
    launch_graph(
        graph_key,
        (a.view(matrix_shape), b.view(matrix_shape), out.view(matrix_shape)),
        stream.cuda_stream,
    )
    return out
