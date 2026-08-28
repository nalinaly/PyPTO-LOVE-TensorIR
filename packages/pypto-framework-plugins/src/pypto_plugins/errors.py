"""Fail-closed plugin errors."""

from __future__ import annotations

from pathlib import Path


class FrameworkCompatibilityError(RuntimeError):
    """The installed framework is not the pinned, validated build."""


class BackendNotReadyError(RuntimeError):
    """The adapter exists but its compiler/operator implementation is not ready."""


class StrictCoverageError(RuntimeError):
    """A requested path would violate PyPTO model-forward coverage."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        report_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.report_path = report_path
