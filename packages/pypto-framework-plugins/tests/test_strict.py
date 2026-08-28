from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from pypto_plugins.sglang_plugin import _resolve_linear_backends_around
from pypto_plugins.torch_backend import _compile_config_patches
from pypto_plugins.torch_inductor import prepare_process_strict


def test_torch_backend_options_cannot_override_cuda_backend() -> None:
    with pytest.raises(ValueError, match="cuda_backend"):
        _compile_config_patches({"cuda_backend": "triton"})
    patches = _compile_config_patches(None)
    assert patches["cuda_backend"] == "pypto"
    assert patches["implicit_fallbacks"] is False


def test_fx_subprocess_mode_is_rejected_before_torch_import(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_FX_COMPILE_MODE", "async+SUBPROCESS")
    with pytest.raises(SystemExit, match="in-process FX compilation"):
        prepare_process_strict()


def test_linear_attention_forces_verify_to_pypto() -> None:
    class Backend(Enum):
        TRITON = "triton"
        CUSTOM = "custom"

    class Backends:
        def __init__(self, *, decode, prefill, verify):
            self.decode = decode
            self.prefill = prefill
            self.verify = verify

    mamba = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend=None,
        linear_attn_prefill_backend=None,
        linear_attn_verify_backend=None,
    )
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: SimpleNamespace(mamba=mamba)

    def original(prefill_default=None):
        assert prefill_default is None
        return Backends(
            decode=Backend.CUSTOM,
            prefill=Backend.CUSTOM,
            verify=Backend.TRITON,
        )

    previous = sys.modules.get("sglang.srt.runtime_context")
    sys.modules["sglang.srt.runtime_context"] = runtime_context
    try:
        result = _resolve_linear_backends_around(
            original, prefill_default="triton"
        )
    finally:
        if previous is None:
            sys.modules.pop("sglang.srt.runtime_context", None)
        else:
            sys.modules["sglang.srt.runtime_context"] = previous
    assert result.decode is Backend.CUSTOM
    assert result.prefill is Backend.CUSTOM
    assert result.verify is Backend.CUSTOM


def test_linear_attention_rejects_mixed_verify_provider() -> None:
    mamba = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend=None,
        linear_attn_prefill_backend=None,
        linear_attn_verify_backend="triton",
    )
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: SimpleNamespace(mamba=mamba)
    previous = sys.modules.get("sglang.srt.runtime_context")
    sys.modules["sglang.srt.runtime_context"] = runtime_context
    try:
        with pytest.raises(ValueError, match="verify backend"):
            _resolve_linear_backends_around(lambda **_kwargs: None)
    finally:
        if previous is None:
            sys.modules.pop("sglang.srt.runtime_context", None)
        else:
            sys.modules["sglang.srt.runtime_context"] = previous


@pytest.mark.parametrize(
    ("base", "decode", "prefill"),
    (
        ("pypto", "triton", None),
        ("pypto", None, "triton"),
        ("triton", "pypto", None),
    ),
)
def test_linear_attention_rejects_mixed_compute_providers(
    base, decode, prefill
) -> None:
    mamba = SimpleNamespace(
        linear_attn_backend=base,
        linear_attn_decode_backend=decode,
        linear_attn_prefill_backend=prefill,
        linear_attn_verify_backend=None,
    )
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: SimpleNamespace(mamba=mamba)
    previous = sys.modules.get("sglang.srt.runtime_context")
    sys.modules["sglang.srt.runtime_context"] = runtime_context
    try:
        with pytest.raises(ValueError, match="both decode and prefill"):
            _resolve_linear_backends_around(lambda **_kwargs: None)
    finally:
        if previous is None:
            sys.modules.pop("sglang.srt.runtime_context", None)
        else:
            sys.modules["sglang.srt.runtime_context"] = previous
