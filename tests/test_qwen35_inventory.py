from __future__ import annotations

from pathlib import Path

import pytest

from pypto_plugins.sglang.inventory import (
    CoverageProvider,
    QWEN35_COMPUTE_SITES,
    SGLANG_SOURCE_SPECS,
    audit_qwen35_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SGLANG_ROOT = WORKSPACE_ROOT / "upstream" / "sglang"


def test_pinned_qwen35_compute_inventory_is_complete() -> None:
    if not SGLANG_ROOT.is_dir():
        pytest.skip("workspace SGLang checkout is not present")
    audit = audit_qwen35_inventory(SGLANG_ROOT)
    assert len(audit.sources) == 10
    assert len(audit.sites) == len(QWEN35_COMPUTE_SITES)
    assert all(lines for _source, _symbol, _provider, lines in audit.sites)


def test_inventory_has_all_strict_provider_families() -> None:
    providers = {site.provider for site in QWEN35_COMPUTE_SITES}
    assert providers == {
        CoverageProvider.GENERIC,
        CoverageProvider.MATMUL,
        CoverageProvider.ATTENTION,
        CoverageProvider.GDN,
        CoverageProvider.HOST_ONLY,
    }


def test_model_specific_names_stay_out_of_the_compiler_core() -> None:
    production_roots = (
        WORKSPACE_ROOT / "projects" / "pypto" / "src",
        WORKSPACE_ROOT / "projects" / "pypto" / "include",
        WORKSPACE_ROOT / "projects" / "pypto" / "python",
    )
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".py",
        ".td",
    }
    for root in production_roots:
        offenders = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in source_suffixes
            and "qwen" in path.read_text(errors="ignore").lower()
        ]
        assert offenders == []


def test_fingerprint_mismatch_fails_closed(tmp_path) -> None:
    if not SGLANG_ROOT.is_dir():
        pytest.skip("workspace SGLang checkout is not present")
    first_spec = SGLANG_SOURCE_SPECS[0]
    target = tmp_path / first_spec.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("changed")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        audit_qwen35_inventory(tmp_path)
