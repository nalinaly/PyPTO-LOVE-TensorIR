"""NeoX rotary position embedding as one native PyPTO tile kernel."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import compile_jit_kernel, launch_graph, bootstrap

bootstrap()
import pypto.language as pl  # noqa: E402

_SCHEDULE_WIDTH = 64
_lock = threading.RLock()
_cache: dict[tuple[int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def rope_kernel(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Rotate the low/high halves of every row and store one full output."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(x.shape[0]):
            x_low = pl.load(x, [row, 0], [1, x.shape[1] // 2])
            x_high = pl.load(
                x,
                [row, x.shape[1] // 2],
                [1, x.shape[1] // 2],
            )
            cos_part = pl.load(cos, [row, 0], [1, x.shape[1] // 2])
            sin_part = pl.load(sin, [row, 0], [1, x.shape[1] // 2])

            low_cos = pl.mul(x_low, cos_part)
            high_sin = pl.mul(x_high, sin_part)
            low = pl.sub(low_cos, high_sin)
            high_cos = pl.mul(x_high, cos_part)
            low_sin = pl.mul(x_low, sin_part)
            high = pl.add(high_cos, low_sin)
            pl.store(low, [row, 0], out)
            pl.store(high, [row, x.shape[1] // 2], out)
    return out


def _validate_shape(rows: int, half: int) -> None:
    if rows <= 0 or half <= 0 or (2 * half) % _SCHEDULE_WIDTH:
        raise ValueError(
            "RoPE needs positive rows/half with full head width divisible by "
            f"{_SCHEDULE_WIDTH}; got rows={rows}, half={half}"
        )


def _tiles(rows: int) -> list[int]:
    return [1, _SCHEDULE_WIDTH] if rows == 1 else [1, 1, _SCHEDULE_WIDTH]


def build(rows: int, half: int) -> Any:
    _validate_shape(rows, half)
    import torch

    x = torch.empty((rows, 2 * half), dtype=torch.bfloat16, device="meta")
    frequency = torch.empty((rows, 2 * half), dtype=torch.bfloat16, device="meta")
    return rope_kernel.specialize(x, frequency, frequency, x)


def compile_for(rows: int, half: int) -> str:
    _validate_shape(rows, half)
    cached = _cache.get((rows, half))
    if cached is not None:
        return cached

    import torch

    x = torch.empty((rows, 2 * half), dtype=torch.bfloat16, device="meta")
    frequency = torch.empty((rows, 2 * half), dtype=torch.bfloat16, device="meta")
    key = compile_jit_kernel(
        rope_kernel,
        (x, frequency, frequency, x),
        _tiles(rows),
    )
    with _lock:
        _cache[(rows, half)] = key
    return key


def rope(x: Any, cos: Any, sin: Any, stream: Any = None) -> Any:
    """Return the full rotated tensor from one native tile graph launch."""

    import torch

    if x.ndim != 2 or x.dtype is not torch.bfloat16 or not x.is_contiguous():
        raise ValueError("RoPE needs a contiguous rank-2 BF16 input")
    rows, width = (int(x.shape[0]), int(x.shape[1]))
    if width % 2:
        raise ValueError("RoPE input width must be even")
    half = width // 2
    if (
        tuple(cos.shape) != (rows, width)
        or tuple(sin.shape) != (rows, width)
        or cos.dtype is not torch.bfloat16
        or sin.dtype is not torch.bfloat16
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError("RoPE cos/sin must be contiguous BF16 and match x")
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    key = compile_for(rows, half)
    out = torch.empty_like(x)
    launch_graph(key, (x, cos, sin, out), stream.cuda_stream)
    return out


def status(rows: int = 256, half: int = 64) -> dict[str, str]:
    try:
        return {"status": "compiled", "key": compile_for(rows, half)}
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
