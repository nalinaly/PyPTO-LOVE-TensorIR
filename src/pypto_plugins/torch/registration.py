"""Reversible CUDA backend registration transaction for TorchInductor.

``install()`` captures the original CUDA DeviceCodegen after
``init_backend_registration()``, rebuilds it with the PyPTO dispatcher
pair (PyPTO scheduling constructor + PyPTO Python wrapper constructor),
and installs the tuple into ``device_codegens``/``custom_backend_passes``
in one assignment each. ``uninstall()`` restores the captured snapshot.
Outside PyPTO mode every dispatch delegates to the captured originals,
so an installed plugin is inert until a PyPTO context is active.
"""

from __future__ import annotations

import threading
from typing import Any

from ..errors import BackendNotReadyError
from . import scheduling as pypto_scheduling
from .dispatch import (
    DeviceCodegenSnapshot,
    WrapperCodegenDispatcher,
    ConstructorDispatcher,
    make_device_codegen_dispatch,
)

_lock = threading.Lock()
_snapshot: DeviceCodegenSnapshot | None = None
_installed = False
_original_triton_hash_with_backend: Any = None

PYPTO_BACKEND_HASH = "pypto-sm120-strided-pointwise-fp32-dso-pid-20260828"


def _pypto_triton_hash_with_backend() -> str:
    """Stable backend hash inside PyPTO mode; original outside it."""

    from .context import current_mode

    if current_mode() is None:
        assert _original_triton_hash_with_backend is not None
        return _original_triton_hash_with_backend()
    return PYPTO_BACKEND_HASH


def _install_triton_hash_guard() -> None:
    """Stop inductor's autotune-cache backend hash from touching Triton."""

    global _original_triton_hash_with_backend
    import torch.utils._triton as torch_triton

    if getattr(torch_triton, "_pypto_hash_guard", False):
        return
    _original_triton_hash_with_backend = torch_triton.triton_hash_with_backend
    torch_triton.triton_hash_with_backend = _pypto_triton_hash_with_backend
    torch_triton._pypto_hash_guard = True


def _uninstall_triton_hash_guard() -> None:
    global _original_triton_hash_with_backend
    import torch.utils._triton as torch_triton

    if not getattr(torch_triton, "_pypto_hash_guard", False):
        return
    assert _original_triton_hash_with_backend is not None
    torch_triton.triton_hash_with_backend = _original_triton_hash_with_backend
    delattr(torch_triton, "_pypto_hash_guard")
    _original_triton_hash_with_backend = None


def _common_module() -> Any:
    from torch._inductor.codegen import common

    return common


def _ensure_original_registration(common: Any) -> Any:
    if "cuda" in common.device_codegens:
        return common.device_codegens["cuda"]
    common.init_backend_registration()
    if "cuda" not in common.device_codegens:
        raise BackendNotReadyError("TorchInductor did not register a CUDA backend")
    return common.device_codegens["cuda"]


def _resolve_scheduling_class(constructor: Any) -> Any:
    """Unwrap the pinned ``lambda scheduling: backends[...](scheduling)``.

    The pinned registration stores a closure over the configured CUDA
    backend class; probing it once with a placeholder scheduler returns a
    real instance whose type is the class the PyPTO scheduling must
    subclass so delegation outside PyPTO mode is exact.
    """

    if isinstance(constructor, type):
        return constructor
    instance = constructor(None)
    return type(instance)


def _pypto_wrapper_constructor(original_wrapper: Any) -> Any:
    class PyptoPythonWrapperCodegen(original_wrapper):  # type: ignore[misc,valid-type]
        # Generated wrappers depend on process-owned artifact/executable objects.
        # The plugin provides its own revision-bound cache and deliberately
        # disables Inductor's disk FX cache, which cannot restore those objects.
        supports_caching = False

        def codegen_kernel_call(
            self,
            name: str,
            kernel_args: Any,
            *,
            triton: bool = True,
            **kwargs: Any,
        ) -> str:
            if triton is False:
                return super().codegen_kernel_call(
                    name, kernel_args, triton=False, **kwargs
                )
            return super().codegen_kernel_call(
                name, kernel_args, triton=triton, **kwargs
            )

    return PyptoPythonWrapperCodegen


def install() -> DeviceCodegenSnapshot:
    """Install the PyPTO dispatch tuple over the CUDA slot, reversibly."""

    global _snapshot, _installed
    with _lock:
        if _installed:
            assert _snapshot is not None
            return _snapshot
        common = _common_module()
        original = _ensure_original_registration(common)
        snapshot = DeviceCodegenSnapshot.from_device_codegen(original)
        base_scheduling_class = _resolve_scheduling_class(snapshot.scheduling)
        pypto_scheduling_ctor = pypto_scheduling.make_pypto_cuda_scheduling(
            base_scheduling_class
        )
        pypto_wrapper_ctor = _pypto_wrapper_constructor(snapshot.wrapper_codegen)
        dispatch = make_device_codegen_dispatch(
            snapshot,
            pypto_scheduling=pypto_scheduling_ctor,
            pypto_wrapper_codegen=pypto_wrapper_ctor,
        )
        common.device_codegens["cuda"] = common.DeviceCodegen(
            dispatch.scheduling,
            dispatch.wrapper_codegen,
            dispatch.cpp_wrapper_codegen,
            dispatch.fx_wrapper_codegen,
        )
        _install_triton_hash_guard()
        _snapshot = snapshot
        _installed = True
        return snapshot


def uninstall() -> None:
    """Restore the captured original CUDA registration."""

    global _snapshot, _installed
    with _lock:
        if not _installed or _snapshot is None:
            return
        common = _common_module()
        common.device_codegens["cuda"] = common.DeviceCodegen(
            _snapshot.scheduling,
            _snapshot.wrapper_codegen,
            _snapshot.cpp_wrapper_codegen,
            _snapshot.fx_wrapper_codegen,
        )
        _uninstall_triton_hash_guard()
        _snapshot = None
        _installed = False


def _is_dispatcher(value: Any) -> bool:
    return isinstance(value, ConstructorDispatcher)


def _is_wrapper_dispatcher(value: Any) -> bool:
    return isinstance(value, WrapperCodegenDispatcher)


def installed() -> bool:
    return _installed


__all__ = ("install", "uninstall", "installed")
