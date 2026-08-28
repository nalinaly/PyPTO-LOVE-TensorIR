"""Portable source-distribution contracts for the release operator package."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from _qwen35_models import (  # noqa: E402
    _LOCKED,
    load_release_shapes,
    parse_release_rows,
)
BENCHMARKS = (
    "classify_sm120.py",
    "exec_sm120.py",
    "linear_sm120.py",
    "qk_sm120.py",
    "paged_attention_sm120.py",
    "stateful_sm120.py",
    "cuda_graph_stateful_sm120.py",
)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_release_benchmark_has_explicit_external_output(name: str) -> None:
    source = (ROOT / "benchmarks" / name).read_text(encoding="utf-8")
    assert '"--output"' in source
    assert "required=True" in source
    assert "/home/" not in source
    assert "with_name(\"exec_results.json\")" not in source
    assert "with_name(\"classify_results.json\")" not in source


def test_bootstrap_has_no_workstation_runtime_defaults() -> None:
    source = (ROOT / "src/pypto_kernels/_boot.py").read_text(encoding="utf-8")
    assert "/home/" not in source
    assert "610.74" not in source
    assert "PYPTO_KERNEL_CUDA_DRIVER_LABEL" in source
    assert "PYPTO_KERNEL_CUDART" in source
    assert "loaded_dso_path" in source


def test_release_model_geometry_loader_is_exact_and_fail_closed(tmp_path: Path) -> None:
    for model, fields in _LOCKED.items():
        directory = tmp_path / model
        directory.mkdir()
        payload = {
            "text_config": {
                **fields,
                "rope_parameters": {"partial_rotary_factor": 0.25},
            }
        }
        (directory / "config.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    shapes = load_release_shapes(tmp_path)
    assert [(shape.linear_q_heads, shape.linear_value_heads) for shape in shapes] == [
        (16, 16),
        (16, 32),
    ]
    assert [shape.conv_channels for shape in shapes] == [6144, 8192]
    assert parse_release_rows("1,19") == (1, 19)
    with pytest.raises(ValueError, match="frozen value"):
        parse_release_rows("1,2")
    bad = tmp_path / "Qwen3.5-9B/config.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["text_config"]["linear_num_value_heads"] = 16
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="release geometry changed"):
        load_release_shapes(tmp_path)
