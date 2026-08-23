"""SGLang general-plugin entry point.

Registration is fail-closed and happens before ServerArgs construction in the
pinned SGLang release. Actual Attention/GDN classes are added after the
standalone operator ABI exists.
"""

from __future__ import annotations

from .errors import BackendNotReadyError
from .torch_inductor import install, prepare_process_strict
from .versions import assert_sglang_compatible, assert_torch_compatible


LINEAR_BACKEND_RESOLVER_TARGET = (
    "sglang.srt.layers.attention.linear.utils.resolve_linear_attn_backends"
)


def _attention_factory(_runner):
    raise BackendNotReadyError(
        "PyPTOAttentionBackend is not implemented yet; refusing SGLang compute fallback."
    )


def _resolve_linear_backends_around(original_fn, prefill_default=None):
    """Force every selected GDN phase, including verify, to PyPTO."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selections = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
        mamba.linear_attn_verify_backend,
    }
    uses_pypto = "pypto" in selections
    decode_backend = mamba.linear_attn_decode_backend or mamba.linear_attn_backend
    prefill_backend = mamba.linear_attn_prefill_backend or mamba.linear_attn_backend
    verify_backend = mamba.linear_attn_verify_backend
    if uses_pypto and (decode_backend != "pypto" or prefill_backend != "pypto"):
        raise ValueError(
            "PyPTO linear attention requires both decode and prefill backends "
            f"to be 'pypto', got decode={decode_backend!r}, "
            f"prefill={prefill_backend!r}"
        )
    if uses_pypto and verify_backend not in (None, "pypto"):
        raise ValueError(
            "PyPTO linear attention requires verify backend 'pypto', "
            f"got {verify_backend!r}"
        )
    backends = original_fn(
        prefill_default=None if uses_pypto else prefill_default
    )
    if not uses_pypto:
        return backends
    return type(backends)(
        decode=type(backends.decode).CUSTOM,
        prefill=type(backends.prefill).CUSTOM,
        verify=type(backends.verify).CUSTOM,
    )


def _register_impl() -> None:
    prepare_process_strict()
    assert_torch_compatible()
    assert_sglang_compatible()
    install()

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from sglang.srt.server_args import (
        add_attention_backend_choices,
        add_linear_attn_kernel_backend_choices,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    add_attention_backend_choices(["pypto"])
    add_linear_attn_kernel_backend_choices(["pypto"])
    register_attention_backend("pypto")(_attention_factory)
    HookRegistry.register(
        LINEAR_BACKEND_RESOLVER_TARGET,
        _resolve_linear_backends_around,
        HookType.AROUND,
    )


def register() -> None:
    """Register the pinned SGLang adapter or terminate fail-closed.

    SGLang's general-plugin loader deliberately catches ordinary
    :class:`Exception` instances so one optional plugin cannot take down the
    server. That policy is unsafe for a user-selected strict compute backend:
    logging a compatibility error and continuing could silently select a
    default provider. ``SystemExit`` derives from ``BaseException`` rather than
    ``Exception``, so it crosses that loader boundary and stops the worker.
    """
    try:
        _register_impl()
    except Exception as exc:
        raise SystemExit(f"PyPTO SGLang plugin registration failed: {exc}") from exc
