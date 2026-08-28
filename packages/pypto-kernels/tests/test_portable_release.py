"""Portable source-distribution contracts for the release operator package."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "classify_sm120.py",
    "exec_sm120.py",
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
