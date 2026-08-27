from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import pypto_kernels
from pypto_plugins.errors import FrameworkCompatibilityError
from pypto_plugins.operator_library import (
    EXPECTED_OPERATOR_LIBRARY_VERSION,
    EXPECTED_OPERATOR_MODULES,
    _validate_native_tile_source,
    assert_operator_library_compatible,
    inspect_operator_library,
)


def test_canonical_operator_package_is_native_tile_only() -> None:
    snapshot = assert_operator_library_compatible()
    assert snapshot.version == EXPECTED_OPERATOR_LIBRARY_VERSION == "0.1.0"
    assert snapshot.modules == EXPECTED_OPERATOR_MODULES
    assert Path(snapshot.package_root) == Path(pypto_kernels.__file__).resolve().parent
    assert dict(snapshot.graph_counts) == {
        "attention": 1,
        "causal_conv1d": 1,
        "fused_add": 1,
        "fused_add_rmsnorm": 1,
        "gated_rmsnorm": 1,
        "gdn": 2,
        "linear": 1,
        "rmsnorm": 1,
        "rope": 1,
        "silu_and_mul": 1,
    }
    with pytest.raises(FrozenInstanceError):
        snapshot.version = "changed"


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("__version__", "9.9.9", "version mismatch"),
        ("__all__", ("attention",), "exported operator set"),
    ],
)
def test_package_identity_mismatches_fail_closed(attribute, value, message) -> None:
    fake = SimpleNamespace(
        __name__="pypto_kernels",
        __file__=pypto_kernels.__file__,
        __version__=pypto_kernels.__version__,
        __all__=pypto_kernels.__all__,
    )
    setattr(fake, attribute, value)
    with pytest.raises(FrameworkCompatibilityError, match=message):
        inspect_operator_library(fake)


def test_native_tile_source_requires_explicit_schedule_and_single_launch(tmp_path) -> None:
    module = SimpleNamespace(__name__="pypto_kernels.example")
    incomplete = tmp_path / "example.py"
    incomplete.write_text("@pl.jit\npl.load(x)\npl.store(y)\n", encoding="utf-8")
    with pytest.raises(FrameworkCompatibilityError, match="native tile graph"):
        _validate_native_tile_source(module, incomplete, 1)

    duplicate_launch = tmp_path / "duplicate.py"
    duplicate_launch.write_text(
        "@pl.jit\nwith pl.at(x):\n pl.range(1)\n pl.load(x)\n pl.store(y)\n"
        "launch_graph(a)\nlaunch_graph(b)\n",
        encoding="utf-8",
    )
    with pytest.raises(FrameworkCompatibilityError, match="launch once"):
        _validate_native_tile_source(module, duplicate_launch, 1)


def test_native_tile_source_rejects_retired_labels_and_whole_tensor_ir(tmp_path) -> None:
    module = SimpleNamespace(__name__="pypto_kernels.example")
    for payload, message in (
        (
            "@pl.jit\nwith pl.at(x):\n pl.range(1)\n pl.load(x)\n pl.store(y)\n"
            "launch_graph(a)\n# v2\n",
            "version label",
        ),
        (
            "@pl.jit\nwith pl.at(x):\n pl.range(1)\n pl.load(x)\n pl.store(y)\n"
            "launch_graph(a)\nfrom pypto import tensor\ntensor.add(a, b)\n",
            "whole-tensor",
        ),
    ):
        source = tmp_path / f"case-{message}.py"
        source.write_text(payload, encoding="utf-8")
        with pytest.raises(FrameworkCompatibilityError, match=message):
            _validate_native_tile_source(module, source, 1)


def test_torch_install_checks_operators_before_framework_actions(monkeypatch) -> None:
    import pypto_plugins.torch_inductor as torch_inductor

    calls: list[str] = []
    monkeypatch.setattr(torch_inductor, "_INSTALLED", False)

    def reject_operator_library():
        calls.append("operators")
        raise FrameworkCompatibilityError("operator mismatch")

    monkeypatch.setattr(
        torch_inductor, "assert_operator_library_compatible", reject_operator_library
    )
    monkeypatch.setattr(
        torch_inductor, "assert_backend_executable_ready", lambda: calls.append("ready")
    )
    monkeypatch.setattr(
        torch_inductor, "assert_torch_compatible", lambda: calls.append("torch")
    )
    monkeypatch.setattr(
        torch_inductor, "prepare_process_strict", lambda: calls.append("strict")
    )
    with pytest.raises(FrameworkCompatibilityError, match="operator mismatch"):
        torch_inductor.install()
    assert calls == ["operators"]
    assert torch_inductor._INSTALLED is False
