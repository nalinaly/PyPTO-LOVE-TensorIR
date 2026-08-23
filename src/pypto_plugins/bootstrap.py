"""Launcher-side verification that plugin entry points exist and are unique."""

from __future__ import annotations

import importlib.metadata

from .errors import FrameworkCompatibilityError


EXPECTED_DISTRIBUTION = "pypto-framework-plugins"
EXPECTED_ENTRY_POINTS = {
    "torch_dynamo_backends": "pypto_plugins.torch_backend:compile_backend",
    "sglang.srt.plugins": "pypto_plugins.sglang_plugin:register",
}


def _normalized_distribution(name: str) -> str:
    return name.replace("_", "-").lower()


def verify_entry_point(group: str, value: str) -> None:
    """Require one callable PyPTO entry point from this distribution."""

    matches = [
        entry_point
        for entry_point in importlib.metadata.entry_points(group=group)
        if entry_point.name == "pypto"
    ]
    if len(matches) != 1:
        raise FrameworkCompatibilityError(
            f"expected exactly one {group!r} entry point named 'pypto', found {len(matches)}"
        )
    entry_point = matches[0]
    distribution = entry_point.dist.name if entry_point.dist is not None else ""
    if _normalized_distribution(distribution) != EXPECTED_DISTRIBUTION:
        raise FrameworkCompatibilityError(
            f"entry point {group!r} came from {distribution!r}, expected {EXPECTED_DISTRIBUTION!r}"
        )
    if entry_point.value != value:
        raise FrameworkCompatibilityError(
            f"entry point {group!r} resolves to {entry_point.value!r}, expected {value!r}"
        )
    if not callable(entry_point.load()):
        raise FrameworkCompatibilityError(f"entry point {group!r} is not callable")


def verify_installed_entry_points() -> None:
    """Verify both framework discovery surfaces before launching a server."""

    for group, value in EXPECTED_ENTRY_POINTS.items():
        verify_entry_point(group, value)


if __name__ == "__main__":
    verify_installed_entry_points()
