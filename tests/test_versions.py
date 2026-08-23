from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypto_plugins.errors import FrameworkCompatibilityError
from pypto_plugins import sglang_plugin
from pypto_plugins.versions import (
    EXPECTED_TORCH_COMMIT,
    assert_sglang_compatible,
    assert_torch_compatible,
)


def fake_torch(version: str, commit: str):
    return SimpleNamespace(__version__=version, version=SimpleNamespace(git_version=commit))


def test_torch_exact_commit_is_accepted() -> None:
    assert_torch_compatible(fake_torch("2.13.0+cu130", EXPECTED_TORCH_COMMIT))


def test_torch_version_mismatch_fails_closed() -> None:
    with pytest.raises(FrameworkCompatibilityError):
        assert_torch_compatible(fake_torch("2.13.1", EXPECTED_TORCH_COMMIT))


def test_sglang_requires_source_identity() -> None:
    with pytest.raises(FrameworkCompatibilityError, match="PYPTO_SGLANG_SOURCE_ROOT"):
        assert_sglang_compatible(installed_version="0.5.18", source_root=None)


def test_sglang_registration_error_exits_fail_closed(monkeypatch) -> None:
    def fail_registration() -> None:
        raise FrameworkCompatibilityError("wrong SGLang source")

    monkeypatch.setattr(sglang_plugin, "_register_impl", fail_registration)
    with pytest.raises(SystemExit, match="wrong SGLang source"):
        sglang_plugin.register()
