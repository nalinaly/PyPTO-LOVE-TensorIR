from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypto_plugins import bootstrap
from pypto_plugins.errors import FrameworkCompatibilityError


def entry_point(group: str, *, value: str | None = None):
    target = value or bootstrap.EXPECTED_ENTRY_POINTS[group]
    return SimpleNamespace(
        name="pypto",
        value=target,
        dist=SimpleNamespace(name="pypto-framework-plugins"),
        load=lambda: lambda: None,
    )


def test_installed_entry_points_are_unique_and_exact(monkeypatch) -> None:
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "entry_points",
        lambda *, group: [entry_point(group)],
    )
    bootstrap.verify_installed_entry_points()


def test_missing_entry_point_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "entry_points",
        lambda *, group: [],
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one"):
        bootstrap.verify_entry_point(
            "sglang.srt.plugins",
            bootstrap.EXPECTED_ENTRY_POINTS["sglang.srt.plugins"],
        )


def test_wrong_entry_point_target_fails_closed(monkeypatch) -> None:
    group = "torch_dynamo_backends"
    monkeypatch.setattr(
        bootstrap.importlib.metadata,
        "entry_points",
        lambda *, group: [entry_point(group, value="other:backend")],
    )
    with pytest.raises(FrameworkCompatibilityError, match="resolves to"):
        bootstrap.verify_entry_point(group, bootstrap.EXPECTED_ENTRY_POINTS[group])
