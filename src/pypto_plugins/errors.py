"""Fail-closed plugin errors."""


class FrameworkCompatibilityError(RuntimeError):
    """The installed framework is not the pinned, validated build."""


class BackendNotReadyError(RuntimeError):
    """The adapter exists but its compiler/operator implementation is not ready."""

