"""Fail-closed constructor dispatch for TorchInductor's global CUDA slot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import BackendNotReadyError, StrictCoverageError
from .context import current_mode


Constructor = Callable[..., Any]
WrapperConstructor = Any


def _require_constructor(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not callable(value):
        suffix = " or None" if optional else ""
        raise TypeError(f"{name} must be callable{suffix}")


def _require_wrapper(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    create = getattr(value, "create", None)
    supports_caching = getattr(value, "supports_caching", None)
    if not callable(create) or type(supports_caching) is not bool:
        raise TypeError(
            f"{name} must expose callable create(...) and bool supports_caching"
        )


@dataclass(frozen=True, slots=True)
class DeviceCodegenSnapshot:
    """The exact original CUDA constructor tuple captured after Torch init."""

    scheduling: Constructor
    wrapper_codegen: WrapperConstructor
    cpp_wrapper_codegen: WrapperConstructor | None = None
    fx_wrapper_codegen: WrapperConstructor | None = None

    def __post_init__(self) -> None:
        _require_constructor(self.scheduling, "scheduling")
        _require_wrapper(self.wrapper_codegen, "wrapper_codegen")
        _require_wrapper(self.cpp_wrapper_codegen, "cpp_wrapper_codegen", optional=True)
        _require_wrapper(self.fx_wrapper_codegen, "fx_wrapper_codegen", optional=True)

    @classmethod
    def from_device_codegen(cls, value: object) -> "DeviceCodegenSnapshot":
        """Copy the pinned DeviceCodegen fields without retaining a mutable object."""

        missing = [
            name
            for name in (
                "scheduling",
                "wrapper_codegen",
                "cpp_wrapper_codegen",
                "fx_wrapper_codegen",
            )
            if not hasattr(value, name)
        ]
        if missing:
            raise TypeError(f"DeviceCodegen object lacks pinned fields: {missing}")
        return cls(
            scheduling=getattr(value, "scheduling"),
            wrapper_codegen=getattr(value, "wrapper_codegen"),
            cpp_wrapper_codegen=getattr(value, "cpp_wrapper_codegen"),
            fx_wrapper_codegen=getattr(value, "fx_wrapper_codegen"),
        )


@dataclass(frozen=True, slots=True)
class ConstructorDispatcher:
    """Choose an original or PyPTO constructor from the active context."""

    path: str
    original: Constructor | None
    pypto: Constructor | None
    reject_in_pypto: bool = False

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ValueError("dispatcher path must be a non-empty string")
        _require_constructor(self.original, "original", optional=True)
        _require_constructor(self.pypto, "pypto", optional=True)
        if type(self.reject_in_pypto) is not bool:
            raise TypeError("reject_in_pypto must be a bool")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        mode = current_mode()
        if mode is None:
            if self.original is None:
                raise BackendNotReadyError(
                    f"the original CUDA {self.path} constructor is unavailable"
                )
            return self.original(*args, **kwargs)
        if self.reject_in_pypto:
            qualifier = "strict " if mode.strict else ""
            raise StrictCoverageError(
                f"{qualifier}PyPTO does not support Inductor {self.path} code generation"
            )
        if self.pypto is None:
            raise BackendNotReadyError(
                f"PyPTO {self.path} constructor is not implemented; refusing CUDA fallback"
            )
        return self.pypto(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class WrapperCodegenDispatcher:
    """Proxy the pinned wrapper ``supports_caching``/``create`` interface."""

    path: str
    original: WrapperConstructor | None
    pypto: WrapperConstructor | None
    reject_in_pypto: bool = False

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ValueError("wrapper dispatcher path must be a non-empty string")
        _require_wrapper(self.original, "original wrapper", optional=True)
        _require_wrapper(self.pypto, "PyPTO wrapper", optional=True)
        if type(self.reject_in_pypto) is not bool:
            raise TypeError("reject_in_pypto must be a bool")

    def _selected(self) -> WrapperConstructor:
        mode = current_mode()
        if mode is None:
            if self.original is None:
                raise BackendNotReadyError(
                    f"the original CUDA {self.path} constructor is unavailable"
                )
            return self.original
        if self.reject_in_pypto:
            qualifier = "strict " if mode.strict else ""
            raise StrictCoverageError(
                f"{qualifier}PyPTO does not support Inductor {self.path} code generation"
            )
        if self.pypto is None:
            raise BackendNotReadyError(
                f"PyPTO {self.path} constructor is not implemented; refusing CUDA fallback"
            )
        return self.pypto

    @property
    def supports_caching(self) -> bool:
        """Mirror the selected wrapper's cacheability in the current context."""

        value = self._selected().supports_caching
        if type(value) is not bool:
            raise TypeError(f"selected {self.path} supports_caching must be a bool")
        return value

    def create(
        self,
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: Any,
        partition_signatures: Any = None,
    ) -> Any:
        """Delegate the pinned four-argument wrapper construction contract."""

        return self._selected().create(
            is_subgraph,
            subgraph_name,
            parent_wrapper,
            partition_signatures,
        )


@dataclass(frozen=True, slots=True)
class DeviceCodegenDispatch:
    """Constructor tuple suitable for the pinned DeviceCodegen registration API."""

    scheduling: ConstructorDispatcher
    wrapper_codegen: WrapperCodegenDispatcher
    cpp_wrapper_codegen: WrapperCodegenDispatcher
    fx_wrapper_codegen: WrapperCodegenDispatcher


def make_device_codegen_dispatch(
    original: DeviceCodegenSnapshot,
    *,
    pypto_scheduling: Constructor,
    pypto_wrapper_codegen: WrapperConstructor,
) -> DeviceCodegenDispatch:
    """Build dispatchers without mutating Torch's registry.

    The later installation transaction supplies real PyPTO constructors, then
    registers these four callables atomically. C++ and FX wrappers delegate only
    outside PyPTO mode and fail explicitly inside it.
    """

    _require_constructor(pypto_scheduling, "pypto_scheduling")
    _require_wrapper(pypto_wrapper_codegen, "pypto_wrapper_codegen")
    return DeviceCodegenDispatch(
        scheduling=ConstructorDispatcher(
            "scheduling",
            original.scheduling,
            pypto_scheduling,
        ),
        wrapper_codegen=WrapperCodegenDispatcher(
            "Python wrapper",
            original.wrapper_codegen,
            pypto_wrapper_codegen,
        ),
        cpp_wrapper_codegen=WrapperCodegenDispatcher(
            "C++ wrapper",
            original.cpp_wrapper_codegen,
            None,
            reject_in_pypto=True,
        ),
        fx_wrapper_codegen=WrapperCodegenDispatcher(
            "FX wrapper",
            original.fx_wrapper_codegen,
            None,
            reject_in_pypto=True,
        ),
    )


__all__ = (
    "ConstructorDispatcher",
    "DeviceCodegenDispatch",
    "DeviceCodegenSnapshot",
    "WrapperCodegenDispatcher",
    "make_device_codegen_dispatch",
)
