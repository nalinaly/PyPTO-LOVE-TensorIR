"""SGLang general-plugin entry point.

Registration is fail-closed and happens before ServerArgs construction in the
pinned SGLang release. Actual Attention/GDN classes are added after the
standalone operator ABI exists.
"""

from __future__ import annotations

from .errors import BackendNotReadyError
from .versions import assert_sglang_compatible, assert_torch_compatible


def _attention_factory(_runner):
    raise BackendNotReadyError(
        "PyPTOAttentionBackend is not implemented yet; refusing SGLang compute fallback."
    )


def _register_impl() -> None:
    assert_torch_compatible()
    assert_sglang_compatible()

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from sglang.srt.server_args import (
        add_attention_backend_choices,
        add_linear_attn_kernel_backend_choices,
    )

    add_attention_backend_choices(["pypto"])
    add_linear_attn_kernel_backend_choices(["pypto"])
    register_attention_backend("pypto")(_attention_factory)


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
