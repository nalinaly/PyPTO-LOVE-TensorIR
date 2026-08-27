"""GDN output RMSNorm and SiLU gate as one native PyPTO tile graph."""

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
def gated_rmsnorm_kernel(
    x: pl.Tensor,
    gate: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Compute ``RMSNorm(x, weight) * SiLU(gate)`` per row tile."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            x_part = pl.load(x, [row, 0], [1, x.shape[1]])
            gate_part = pl.load(gate, [row, 0], [1, x.shape[1]])
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
            weighted = pl.mul(normalized, weight_wide)
            gate_wide = pl.cast(gate_part, target_type=pl.FP32)
            gate_neg = pl.mul(gate_wide, -1.0)
            gate_exp = pl.exp(gate_neg)
            gate_denominator = pl.add(gate_exp, 1.0)
            gate_sigmoid = pl.recip(gate_denominator)
            gate_silu = pl.mul(gate_wide, gate_sigmoid)
            gated = pl.mul(weighted, gate_silu)
            result = pl.cast(gated, target_type=pl.BF16)
            pl.store(result, [row, 0], out)
    return out


def _validate_shape(rows: int, columns: int) -> None:
    if rows <= 0 or columns <= 0 or columns % _CHANNEL_TILE:
        raise ValueError(
            "gated_rmsnorm needs positive [rows, columns] with columns "
            f"divisible by {_CHANNEL_TILE}"
        )


def build(rows: int, columns: int, eps: float = _EPSILON) -> Any:
    if eps != _EPSILON:
        raise ValueError(f"gated_rmsnorm specializes epsilon {_EPSILON}")
    _validate_shape(rows, columns)
    import torch

    sample = torch.empty((rows, columns), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, columns), dtype=torch.bfloat16, device="meta")
    return gated_rmsnorm_kernel.specialize(sample, sample, weight, sample)


def compile_for(rows: int, columns: int, eps: float = _EPSILON) -> str:
    if eps != _EPSILON:
        raise ValueError(f"gated_rmsnorm specializes epsilon {_EPSILON}")
    _validate_shape(rows, columns)
    shape_key = (rows, columns)
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    sample = torch.empty((rows, columns), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((1, columns), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        gated_rmsnorm_kernel,
        (sample, sample, weight, sample),
        [1, _CHANNEL_TILE],
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def gated_rmsnorm(
    x: Any,
    gate: Any,
    weight: Any,
    eps: float = _EPSILON,
    stream: Any = None,
) -> Any:
    """Return gated weighted RMSNorm from one graph launch."""

    import torch

    if (
        x.ndim != 2
        or x.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or tuple(gate.shape) != tuple(x.shape)
        or not x.is_contiguous()
        or not gate.is_contiguous()
    ):
        raise ValueError("gated_rmsnorm needs matching contiguous rank-2 BF16 inputs")
    rows, columns = map(int, x.shape)
    if (
        weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) not in ((columns,), (1, columns))
    ):
        raise ValueError("gated_rmsnorm weight must be contiguous BF16 [columns]")
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    graph_key = compile_for(rows, columns, eps)
    out = torch.empty_like(x)
    launch_graph(
        graph_key,
        (x, gate, weight.view(1, columns), out),
        stream.cuda_stream,
    )
    return out


def status(rows: int = 256, columns: int = 128) -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for(rows, columns)}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
