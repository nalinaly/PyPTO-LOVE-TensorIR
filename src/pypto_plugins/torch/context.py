"""Per-compile PyPTO mode, independent of Torch's process-global registry."""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class PyPTOInductorMode:
    """Immutable policy active for one logical Inductor compilation."""

    strict: bool

    def __post_init__(self) -> None:
        if type(self.strict) is not bool:
            raise TypeError(f"strict must be a bool, got {type(self.strict).__name__}")


_CURRENT_MODE: contextvars.ContextVar[PyPTOInductorMode | None] = contextvars.ContextVar(
    "pypto_inductor_mode",
    default=None,
)


def current_mode() -> PyPTOInductorMode | None:
    """Return the mode for this task/context, or ``None`` outside PyPTO."""

    return _CURRENT_MODE.get()


@contextlib.contextmanager
def activate_mode(*, strict: bool = True) -> Iterator[PyPTOInductorMode]:
    """Activate PyPTO for the current logical compilation.

    Nested contexts may strengthen a development context to strict mode, but
    cannot weaken an already-strict compilation. ``ContextVar`` propagates to
    asyncio tasks and copied contexts without changing unrelated threads.
    """

    requested = PyPTOInductorMode(strict=strict)
    outer = current_mode()
    if outer is not None and outer.strict and not requested.strict:
        raise ValueError("a nested PyPTO mode cannot weaken strict coverage")
    effective = PyPTOInductorMode(
        strict=requested.strict or (outer.strict if outer is not None else False)
    )
    token = _CURRENT_MODE.set(effective)
    try:
        yield effective
    finally:
        _CURRENT_MODE.reset(token)


def bind_current_context(function: Callable[P, R]) -> Callable[P, R]:
    """Bind the current context for an explicitly spawned worker callback.

    Thread pools do not inherit ``ContextVar`` state. The caller must opt into
    propagation at submission time; each invocation receives a fresh copy so
    concurrent calls do not attempt to enter the same ``Context`` object.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    captured = contextvars.copy_context()

    @functools.wraps(function)
    def bound(*args: P.args, **kwargs: P.kwargs) -> R:
        return captured.copy().run(function, *args, **kwargs)

    return bound


__all__ = (
    "PyPTOInductorMode",
    "activate_mode",
    "bind_current_context",
    "current_mode",
)
