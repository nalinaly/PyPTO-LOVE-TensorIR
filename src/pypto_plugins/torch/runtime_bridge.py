"""Wrapper-side launch bridge for Inductor-generated PyPTO kernels.

The generated Inductor Python wrapper imports ``pypto_launch`` from this
module and calls it with the kernel name, the launch arguments and the
raw current stream. Executing the compiled artifact through the PyPTO
``NvidiaExecutable`` lifecycle is the next layer; until it lands the
bridge fails closed so no silent fallback executes the model.
"""

from __future__ import annotations

from typing import Any

from .errors import StrictCoverageError


def pypto_launch(kernel_name: str, args: tuple[Any, ...], stream: int) -> None:
    """Fail-closed placeholder for the PyPTO kernel launch bridge."""

    raise StrictCoverageError(
        f"the PyPTO runtime launch bridge for Inductor kernel "
        f"{kernel_name!r} is not implemented yet; refusing fallback"
    )


__all__ = ("pypto_launch",)
