from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypto_plugins.errors import BackendNotReadyError, StrictCoverageError
from pypto_plugins.torch.context import activate_mode
from pypto_plugins.torch.dispatch import (
    ConstructorDispatcher,
    DeviceCodegenSnapshot,
    make_device_codegen_dispatch,
)


def constructor(name):
    return lambda *args, **kwargs: (name, args, kwargs)


def wrapper(name, *, supports_caching=True):
    class Wrapper:
        pass

    Wrapper.supports_caching = supports_caching
    Wrapper.create = staticmethod(
        lambda is_subgraph, subgraph_name, parent_wrapper, partition_signatures: (
            name,
            is_subgraph,
            subgraph_name,
            parent_wrapper,
            partition_signatures,
        )
    )
    return Wrapper


def snapshot() -> DeviceCodegenSnapshot:
    return DeviceCodegenSnapshot(
        scheduling=constructor("original-scheduling"),
        wrapper_codegen=wrapper("original-wrapper"),
        cpp_wrapper_codegen=wrapper("original-cpp"),
        fx_wrapper_codegen=wrapper("original-fx", supports_caching=False),
    )


def test_snapshot_copies_exact_pinned_fields() -> None:
    value = SimpleNamespace(
        scheduling=constructor("s"),
        wrapper_codegen=wrapper("w"),
        cpp_wrapper_codegen=None,
        fx_wrapper_codegen=None,
    )
    copied = DeviceCodegenSnapshot.from_device_codegen(value)
    assert copied.scheduling is value.scheduling
    with pytest.raises(TypeError, match="lacks pinned fields"):
        DeviceCodegenSnapshot.from_device_codegen(SimpleNamespace())


def test_dispatch_preserves_original_outside_pypto_mode() -> None:
    dispatch = make_device_codegen_dispatch(
        snapshot(),
        pypto_scheduling=constructor("pypto-scheduling"),
        pypto_wrapper_codegen=wrapper("pypto-wrapper"),
    )
    assert dispatch.scheduling(1)[0] == "original-scheduling"
    assert dispatch.wrapper_codegen.supports_caching is True
    assert dispatch.wrapper_codegen.create(False, None, None, None)[0] == "original-wrapper"
    assert dispatch.cpp_wrapper_codegen.create(False, None, None, None)[0] == "original-cpp"
    assert dispatch.fx_wrapper_codegen.supports_caching is False
    assert dispatch.fx_wrapper_codegen.create(False, None, None, None)[0] == "original-fx"


def test_dispatch_selects_only_pypto_constructors_in_mode() -> None:
    dispatch = make_device_codegen_dispatch(
        snapshot(),
        pypto_scheduling=constructor("pypto-scheduling"),
        pypto_wrapper_codegen=wrapper("pypto-wrapper"),
    )
    with activate_mode(strict=True):
        assert dispatch.scheduling(1)[0] == "pypto-scheduling"
        assert dispatch.wrapper_codegen.supports_caching is True
        assert dispatch.wrapper_codegen.create(True, "sub", object(), None)[0] == "pypto-wrapper"
        with pytest.raises(StrictCoverageError, match=r"C\+\+ wrapper"):
            dispatch.cpp_wrapper_codegen.create(False, None, None, None)
        with pytest.raises(StrictCoverageError, match="FX wrapper"):
            _ = dispatch.fx_wrapper_codegen.supports_caching


def test_missing_pypto_constructor_never_delegates() -> None:
    dispatcher = ConstructorDispatcher(
        "scheduling",
        original=constructor("original"),
        pypto=None,
    )
    with activate_mode(strict=False):
        with pytest.raises(BackendNotReadyError, match="refusing CUDA fallback"):
            dispatcher()
