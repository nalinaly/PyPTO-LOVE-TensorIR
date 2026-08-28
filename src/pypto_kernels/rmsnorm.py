"""Qwen weighted RMSNorm as one native PyPTO tile kernel."""

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
def rmsnorm_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Normalize one row tile and apply the ``1 + weight`` scale."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            x_part = pl.load(x, [row, 0], [1, x.shape[1]])
            weight_part = pl.load(weight, [0, 0], [1, x.shape[1]])
            x_wide = pl.cast(x_part, target_type=pl.FP32)
            square = pl.mul(x_wide, x_wide)
            scratch = pl.create_tile(
                [1, x.shape[1]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            square_sum = pl.row_sum(square, scratch)
            mean_square = pl.mul(square_sum, 1.0 / x.shape[1])
            shifted = pl.add(mean_square, 1.0e-6)
            inv_rms = pl.rsqrt(shifted)
            normalized = pl.row_expand_mul(x_wide, inv_rms)
            weight_wide = pl.cast(weight_part, target_type=pl.FP32)
            scale = pl.add(weight_wide, 1.0)
            weighted = pl.mul(normalized, scale)
            result = pl.cast(weighted, target_type=pl.BF16)
            pl.store(result, [row, 0], out)
    return out


def _validate_shape(rows: int, columns: int) -> None:
    if rows <= 0 or columns <= 0 or columns % _CHANNEL_TILE:
        raise ValueError(
            "rmsnorm needs positive [rows, columns] with columns divisible "
            f"by {_CHANNEL_TILE}; got [{rows}, {columns}]"
        )


def _tiles(rows: int) -> list[int]:
    return [_CHANNEL_TILE] if rows == 1 else [1, _CHANNEL_TILE]


def build(rows: int, cols: int, eps: float = _EPSILON) -> Any:
    """Specialize and return the visible native tile IR."""

    if eps != _EPSILON:
        raise ValueError(f"rmsnorm currently specializes epsilon {_EPSILON}")
    _validate_shape(rows, cols)
    import torch

    sample = torch.empty((rows, cols), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, cols), dtype=torch.bfloat16, device="meta")
    return rmsnorm_kernel.specialize(sample, weight, sample)


def compile_for(rows: int, cols: int, eps: float = _EPSILON) -> str:
    if eps != _EPSILON:
        raise ValueError(f"rmsnorm currently specializes epsilon {_EPSILON}")
    _validate_shape(rows, cols)
    cached = _cache.get((rows, cols))
    if cached is not None:
        return cached

    import torch

    sample = torch.empty((rows, cols), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, cols), dtype=torch.bfloat16, device="meta")
    key = compile_jit_kernel(
        rmsnorm_kernel,
        (sample, weight, sample),
        _tiles(rows),
    )
    with _lock:
        _cache[(rows, cols)] = key
    return key


def rmsnorm(
    x: Any,
    weight: Any,
    eps: float = _EPSILON,
    stream: Any = None,
) -> Any:
    """Return Qwen's weighted RMS-normalized BF16 ``x`` in one launch."""

    import torch

    if x.ndim != 2 or x.dtype is not torch.bfloat16 or not x.is_contiguous():
        raise ValueError("rmsnorm needs a contiguous rank-2 BF16 tensor")
    rows, cols = (int(x.shape[0]), int(x.shape[1]))
    if (
        weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) not in ((cols,), (1, cols))
    ):
        raise ValueError("rmsnorm weight must be contiguous BF16 with shape [columns]")
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    key = compile_for(rows, cols, eps)
    out = torch.empty_like(x)
    launch_graph(key, (x, weight.view(1, cols), out), stream.cuda_stream)
    return out


def status(rows: int = 256, cols: int = 1024) -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for(rows, cols)}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
