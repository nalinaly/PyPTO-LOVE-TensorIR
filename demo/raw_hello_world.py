"""Run one raw PyPTO DSL kernel on NVIDIA, end to end, no harness.

This is the NVIDIA-native twin of the tutorial article's hello_world: the
same ``@pl.jit`` tile graph, executed directly by CPython. Running

    envs/pypto-release/bin/python demo/raw_hello_world.py

compiles the kernel through PyPTO HIR -> typed TensorIR -> CUDA Tile ->
tileiras, prints the compiled artifact identity, launches it on the current
CUDA stream and checks the result against PyTorch. Everything printed comes
from this one process; nothing is filtered or summarized.
"""

import sys

from pypto_kernels import _boot

_boot.bootstrap()
import pypto.language as pl  # noqa: E402

ROWS = 64
COLS = 4096
COL_TILE = 128


@pl.jit
def hello_world(
    x: pl.Tensor,
    y: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for r in pl.range(x.shape[0]):
            for block in pl.range(x.shape[1] // COL_TILE):
                tile_x = pl.load(x, [r, block * COL_TILE], [1, COL_TILE])
                added = pl.add(tile_x, 2.0)
                pl.store(added, [r, block * COL_TILE], y)
    return y


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1
    torch.manual_seed(7)
    device = "cuda"
    x = torch.randn(ROWS, COLS, dtype=torch.float32, device=device)
    y = torch.zeros_like(x)

    print("== pypto raw hello_world ==")
    print(f"input  : shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    print("-- specializing kernel (freezes shape/stride into PyPTO HIR)")
    program = hello_world.specialize(x, y)
    print(program)
    print("-- compiling: PyPTO HIR -> typed TensorIR -> CUDA Tile -> tileiras -> cubin")
    graph_key = _boot.compile_graph(
        program,
        [COL_TILE],
        provider="pypto.tensorir",
        source_node="demo/raw_hello_world.py",
    )
    print(f"compiled graph key: {graph_key}")
    artifact = _boot.compiled_artifact(graph_key)
    actual = artifact.actual_target
    print(
        "artifact target : "
        f"name={actual.name} cc={actual.compute_capability} "
        f"portability={actual.portability}"
    )
    from pypto.runtime import nvidia as runtime

    observation = runtime.observe_current_nvidia_runtime(*_boot._live_runtime_expectation())
    live = observation.target_info
    print(
        "live GPU target : "
        f"arch={live.architecture} codegen={live.arch_conditional_architecture} "
        f"cc={live.traits.compute_capability}"
    )
    executable = _boot._ready_executable(graph_key)
    print(f"executable state  : {executable.state}")

    print("-- launching on the caller's current CUDA stream")
    stream = torch.cuda.current_stream(device)
    _boot.launch_graph(graph_key, (x, y), stream.cuda_stream)
    stream.synchronize()
    print("-- kernel finished; comparing against PyTorch")

    reference = x + 2.0
    max_abs_diff = float((y - reference).abs().max())
    exact = bool(torch.equal(y, reference))
    print(f"max_abs_diff={max_abs_diff} exact={exact}")
    print("first 8 elements of the kernel output:")
    print(y.flatten()[:8].tolist())
    print("hello_world: OK" if exact or max_abs_diff < 1e-6 else "hello_world: MISMATCH")
    return 0 if (exact or max_abs_diff < 1e-6) else 1


if __name__ == "__main__":
    raise SystemExit(main())
