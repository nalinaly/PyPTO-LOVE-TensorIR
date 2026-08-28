"""Qwen fused residual add and weighted RMSNorm as one native tile graph."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_CHANNEL_TILE = 128
_EPSILON = 1.0e-6
_lock = threading.RLock()
_cache: dict[tuple[int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def fused_add_rmsnorm_kernel(
    x: pl.Tensor,
    residual: pl.Tensor,
    weight: pl.Tensor,
    normalized_out: pl.Out[pl.Tensor],
    residual_out: pl.Out[pl.Tensor],
):
    """Add BF16 residuals, normalize in FP32, and return both outputs."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            x_part = pl.load(x, [row, 0], [1, x.shape[1]])
            residual_part = pl.load(residual, [row, 0], [1, x.shape[1]])
            weight_part = pl.load(weight, [0, 0], [1, x.shape[1]])
            residual_sum = pl.add(x_part, residual_part)
            residual_wide = pl.cast(residual_sum, target_type=pl.FP32)
            square = pl.mul(residual_wide, residual_wide)
            scratch = pl.create_tile(
                [1, x.shape[1]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            square_sum = pl.row_sum(square, scratch)
            mean_square = pl.mul(square_sum, 1.0 / x.shape[1])
            shifted = pl.add(mean_square, 1.0e-6)
            inv_rms = pl.rsqrt(shifted)
            normalized = pl.row_expand_mul(residual_wide, inv_rms)
            weight_wide = pl.cast(weight_part, target_type=pl.FP32)
            scale = pl.add(weight_wide, 1.0)
            weighted = pl.mul(normalized, scale)
            result = pl.cast(weighted, target_type=pl.BF16)
            pl.store(result, [row, 0], normalized_out)
            pl.store(residual_sum, [row, 0], residual_out)
    return normalized_out, residual_out


def _validate_shape(rows: int, columns: int) -> None:
    if rows <= 0 or columns <= 0 or columns % _CHANNEL_TILE:
        raise ValueError(
            "fused_add_rmsnorm needs positive [rows, columns] with columns "
            f"divisible by {_CHANNEL_TILE}"
        )


def _tiles(rows: int) -> list[int]:
    return [_CHANNEL_TILE] if rows == 1 else [1, _CHANNEL_TILE]


def build(rows: int, columns: int, eps: float = _EPSILON) -> Any:
    if eps != _EPSILON:
        raise ValueError(f"fused_add_rmsnorm specializes epsilon {_EPSILON}")
    _validate_shape(rows, columns)
    import torch

    sample = torch.empty((rows, columns), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, columns), dtype=torch.bfloat16, device="meta")
    return fused_add_rmsnorm_kernel.specialize(sample, sample, weight, sample, sample)


def compile_for(rows: int, columns: int, eps: float = _EPSILON) -> str:
    if eps != _EPSILON:
        raise ValueError(f"fused_add_rmsnorm specializes epsilon {_EPSILON}")
    _validate_shape(rows, columns)
    shape_key = (rows, columns)
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    sample = torch.empty((rows, columns), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, columns), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        fused_add_rmsnorm_kernel,
        (sample, sample, weight, sample, sample),
        _tiles(rows),
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def fused_add_rmsnorm(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float = _EPSILON,
    stream: Any = None,
) -> tuple[Any, Any]:
    """Return ``(normalized, residual_sum)`` from one graph launch."""

    import torch

    if (
        x.ndim != 2
        or x.dtype is not torch.bfloat16
        or residual.dtype is not torch.bfloat16
        or tuple(residual.shape) != tuple(x.shape)
        or not x.is_contiguous()
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "fused_add_rmsnorm needs matching contiguous rank-2 BF16 inputs"
        )
    rows, columns = map(int, x.shape)
    if (
        weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) not in ((columns,), (1, columns))
    ):
        raise ValueError("fused_add_rmsnorm weight must be contiguous BF16 [columns]")
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    graph_key = compile_for(rows, columns, eps)
    normalized = torch.empty_like(x)
    residual_sum = torch.empty_like(x)
    launch_graph(
        graph_key,
        (x, residual, weight.view(1, columns), normalized, residual_sum),
        stream.cuda_stream,
    )
    return normalized, residual_sum


def status(rows: int = 256, columns: int = 1024) -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for(rows, columns)}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
