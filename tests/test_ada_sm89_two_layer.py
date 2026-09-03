"""Ada sm_89: complete PyPTO operator census plus two-layer vs inductor.

Requires CUDA Runtime API >= 13000. Torch 2.11+cu128 reports 12080 unless
CUDA 13.3 libcudart is preloaded, e.g.

    LD_PRELOAD=/usr/local/cuda-13.3/lib64/libcudart.so.13
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "operators" / "pypto_two_layer_ada_sm89.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("pypto_two_layer_ada_sm89", BENCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_layer_census_and_inductor_on_live_sm89() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: cannot claim a GPU pass")
    capability = torch.cuda.get_device_capability()
    live_cc = int(capability[0]) * 10 + int(capability[1])
    if live_cc != 89:
        pytest.skip(f"live GPU compute capability is {live_cc}, need 89")

    module = _load_bench()
    evidence = module.run(warmup=1, timed=2)
    failed = evidence["census_failed"]
    assert not failed, f"operator census failed: {failed}"
    assert evidence["all_operators_correct"]
    two = evidence["two_layer"]
    assert two["core_ok"]
    match = two["pypto_vs_eager"]
    assert match["finite"]
    assert match["correct"], match
    inductor = two["inductor"]
    assert inductor["ok"], inductor.get("error")
    assert inductor["correct_vs_eager"], inductor
    assert two["pypto_ms"] > 0
    assert two["eager_ms"] > 0
    assert inductor["ms"] > 0
