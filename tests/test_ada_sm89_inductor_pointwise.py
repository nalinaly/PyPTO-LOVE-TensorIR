"""Ada sm_89: inductor-fused elementwise compiles to a PyPTO cubin.

Requires CUDA Runtime API >= 13000. Preload CUDA 13.3 libcudart when torch
bundles 12.8:

    LD_PRELOAD=/usr/local/cuda-13.3/lib64/libcudart.so.13
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "operators" / "pypto_inductor_pointwise_ada_sm89.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pypto_inductor_pointwise_ada_sm89", BENCH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inductor_fused_pointwise_emits_pypto_cubin_on_live_sm89() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: cannot claim a GPU pass")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip(
            f"live GPU compute capability is {torch.cuda.get_device_capability()}, need (8, 9)"
        )

    module = _load()
    try:
        evidence = module.run()
    finally:
        import torch as torch_mod
        import pypto_plugins.torch_inductor as torch_inductor

        torch_inductor.uninstall()
        torch_mod._dynamo.reset()
        from pypto_plugins.activity_trace import clear_artifact_registry_for_testing

        clear_artifact_registry_for_testing()

    assert evidence["all_correct"], evidence["cases"]
    assert evidence["all_native_pypto_cubin"], evidence["kernels"]
    assert evidence["kernel_count"] >= 1
    assert all(
        item["entry_name"] == "pypto_fused_pointwise" and not item["fallback_used"]
        for item in evidence["kernels"]
    )
