from __future__ import annotations

from pathlib import Path

import pytest

from pypto_plugins.sglang.state_lifecycle_inventory import (
    LifecycleAction,
    SGLANG_STATE_SOURCE_SPECS,
    STATE_LIFECYCLE_SCOPE_V1,
    STATE_LIFECYCLE_SITES,
    audit_state_lifecycle_inventory,
    validate_state_lifecycle_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SGLANG_ROOT = WORKSPACE_ROOT / "upstream" / "sglang"


def test_pinned_state_lifecycle_inventory_is_complete() -> None:
    if not SGLANG_ROOT.is_dir():
        pytest.skip("workspace SGLang checkout is not present")
    audit = audit_state_lifecycle_inventory(SGLANG_ROOT)
    assert len(audit.sources) == 11
    assert len(audit.sites) == len(STATE_LIFECYCLE_SITES)
    assert all(lines for _source, _symbol, _action, lines in audit.sites)
    assert {site.action for site in STATE_LIFECYCLE_SITES} == set(LifecycleAction)
    assert len(audit.scope_digest) == 64


def test_state_lifecycle_v1_excludes_unsupported_modes() -> None:
    assert STATE_LIFECYCLE_SCOPE_V1 == {
        "precision": "full-precision",
        "mamba_radix_strategies": (
            "no_buffer",
            "extra_buffer",
            "extra_buffer_lazy",
        ),
        "unified_radix_selected": True,
        "legacy_mamba_radix_selected": False,
        "unified_slot_translation": True,
        "page_major_envelope": True,
        "overlap_schedule": True,
        "speculative": False,
        "mtp": False,
        "replayssm": False,
        "int8_checkpoint": False,
        "hicache": False,
        "disaggregation": False,
        "capture_supported": False,
        "registration_ready": False,
        "requires_single_dso": True,
        "requires_target_info": True,
        "requires_compile_request": True,
        "requires_current_stream_runtime": True,
        "requires_statebundle_runtime": True,
        "implementation_order": (
            "single-dso",
            "target-info",
            "compile-request-current-stream",
            "statebundle-runtime",
            "sglang-plugin-adapter",
        ),
    }
    assert len(validate_state_lifecycle_scope(STATE_LIFECYCLE_SCOPE_V1)) == 64
    with pytest.raises(TypeError):
        STATE_LIFECYCLE_SCOPE_V1["speculative"] = True
    weakened = dict(STATE_LIFECYCLE_SCOPE_V1)
    weakened["registration_ready"] = True
    with pytest.raises(RuntimeError, match="scope mismatch"):
        validate_state_lifecycle_scope(weakened)


def test_state_source_fingerprint_mismatch_fails_closed(tmp_path) -> None:
    if not SGLANG_ROOT.is_dir():
        pytest.skip("workspace SGLang checkout is not present")
    first = SGLANG_STATE_SOURCE_SPECS[0]
    target = tmp_path / first.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("changed")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        audit_state_lifecycle_inventory(tmp_path)


def test_inventory_module_imports_no_framework() -> None:
    path = (
        PROJECT_ROOT
        / "src"
        / "pypto_plugins"
        / "sglang"
        / "state_lifecycle_inventory.py"
    )
    text = path.read_text()
    for forbidden in ("import torch", "import sglang", "import pypto_kernels"):
        assert forbidden not in text
