from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_release  # noqa: E402


def test_build_surface_freezes_parallelism_and_stages() -> None:
    args = build_release.parser().parse_args([])
    assert args.stage == "all"
    assert args.jobs == 24
    assert build_release.PYPTO_BUILD.is_relative_to(ROOT / "builds")
    assert build_release.NATIVE_BUILD.is_relative_to(ROOT / "builds")
    assert build_release.WHEEL_DIR.is_relative_to(ROOT / "builds")


def test_worker_rejects_non_24_parallelism() -> None:
    with pytest.raises(build_release.ReleaseContractError, match="exactly 24"):
        build_release._worker("wheels", 23)


def test_pypto_build_uses_backend_and_test_defines() -> None:
    source = (ROOT / "tools/build_release.py").read_text(encoding="utf-8")
    assert "PYPTO_ENABLE_NVIDIA_BACKEND=ON" in source
    assert "BUILD_TESTING=ON" in source
    assert "CMAKE_BUILD_PARALLEL_LEVEL" in source
    assert '"--cmake_dir"' in source
    assert '"-j24"' in source
    assert "Total Tests: 13" in source
    assert '"24"' not in source  # all values come from the frozen CPU_JOBS constant


def test_build_tool_has_no_workstation_source_paths() -> None:
    source = (ROOT / "tools/build_release.py").read_text(encoding="utf-8")
    assert "/home/" not in source
    assert "projects/" not in source
    assert "worktrees/" not in source
