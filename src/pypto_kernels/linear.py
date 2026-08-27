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
_lock = threading.RLock()
_cache: dict[tuple[int, int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


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


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("linear needs positive rows and feature dimensions")
    if in_features % 128:
        raise ValueError("linear input features must be divisible by 128")
    if out_features % _TILE_COLUMNS:
        raise ValueError("linear output features must be divisible by 128")


def build(rows: int, in_features: int, out_features: int) -> Any:
    _validate_shape(rows, in_features, out_features)
    import torch

    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    out = torch.empty((rows, out_features), dtype=torch.bfloat16, device="meta")
    return linear_kernel.specialize(x, weight, out)


def compile_for(rows: int, in_features: int, out_features: int) -> str:
    _validate_shape(rows, in_features, out_features)
    shape_key = (rows, in_features, out_features)
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    x = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty(
        (out_features, in_features), dtype=torch.bfloat16, device="meta"
    )
    out = torch.empty((rows, out_features), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        linear_kernel,
        (x, weight, out),
        [_TILE_ROWS, _TILE_COLUMNS],
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
    graph_key = compile_for(rows, in_features, out_features)
    result_shape = (*map(int, x.shape[:-1]), out_features)
    out = torch.empty(result_shape, dtype=x.dtype, device=x.device)
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
