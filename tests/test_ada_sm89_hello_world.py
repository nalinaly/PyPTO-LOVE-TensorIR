"""Live Ada sm_89 compile+launch of the shipped hello_world graph.

Artifact loader ABI requires CUDA Runtime API >= 13000. Torch 2.11+cu128
exposes 12080 unless CUDA 13.3 libcudart is preloaded, e.g.

    LD_PRELOAD=/usr/local/cuda-13.3/lib64/libcudart.so.13
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "raw_hello_world.py"


def _load_hello_world():
    spec = importlib.util.spec_from_file_location("raw_hello_world", DEMO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DEMO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["raw_hello_world"] = module
    spec.loader.exec_module(module)
    return module


def test_shipped_hello_world_compiles_and_launches_on_live_sm89() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: cannot claim a GPU pass")

    capability = torch.cuda.get_device_capability()
    live_cc = int(capability[0]) * 10 + int(capability[1])
    if live_cc != 89:
        pytest.skip(f"live GPU compute capability is {live_cc}, need 89")

    module = _load_hello_world()
    from pypto_kernels import _boot

    torch.manual_seed(7)
    x = torch.randn(module.ROWS, module.COLS, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    assert x.is_cuda and y.is_cuda

    program = module.hello_world.specialize(x, y)
    graph_key = _boot.compile_graph(
        program,
        [module.COL_TILE],
        provider="pypto.tensorir",
        source_node="tests/test_ada_sm89_hello_world.py",
    )
    artifact = _boot.compiled_artifact(graph_key)
    actual = artifact.actual_target
    assert actual.compute_capability == 89
    assert actual.name == "sm_89a"

    stream = torch.cuda.current_stream("cuda")
    _boot.launch_graph(graph_key, (x, y), stream.cuda_stream)
    stream.synchronize()
    first = y.clone()
    reference = x + 2.0
    assert torch.equal(first, reference) or float((first - reference).abs().max()) < 1e-6

    y.zero_()
    _boot.launch_graph(graph_key, (x, y), stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(y, reference) or float((y - reference).abs().max()) < 1e-6
    assert torch.equal(y, first)
