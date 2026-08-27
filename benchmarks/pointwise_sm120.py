#!/usr/bin/env python3
"""SM120 acceptance for an Inductor-style generated native tile graph."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-framework-plugins/src",
)

import torch

from pypto_plugins.torch import pointwise_codegen
from pypto_plugins.torch.runtime_bridge import pypto_launch
from pypto_plugins.torch.scheduling import REGISTRY


def main() -> int:
    torch.manual_seed(17)
    shape = (256, 1024)
    builder = pointwise_codegen.PointwiseProgramBuilder(shape, "bfloat16")
    x_value = builder.add_input("x")
    y_value = builder.add_input("y")
    summed = builder.emit("tensor.add", [x_value, y_value])
    scaled = builder.emit("tensor.muls", [summed, builder.scalar(2.0)])
    result_value = builder.emit("tensor.neg", [scaled])
    builder.mark_output(result_value)
    program = builder.build()
    native_source = program.native_source(128)
    rendered = str(program.specialize(128))

    kernel_name = "inductor_native_pointwise_acceptance"
    artifact = pointwise_codegen.compile_pointwise(
        program, tile=128, registry_name=kernel_name
    )
    REGISTRY.register(kernel_name, artifact)

    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    pypto_launch(kernel_name, (x, y, out), stream.cuda_stream)
    stream.synchronize()
    reference = -((x + y) * 2.0)
    max_abs_diff = float((out.float() - reference.float()).abs().max())
    correct = bool(torch.equal(out, reference))

    dso = pathlib.Path(pointwise_codegen._DEFAULT_DSO)
    info = pointwise_codegen.bootstrap_pypto()[
        "compiler"
    ].get_nvidia_backend_build_info()
    evidence = {
        "schema": 1,
        "kind": "inductor-native-pointwise-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "pypto_commit": info.pypto_revision,
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "shape": list(shape),
        "dtype": "bfloat16",
        "tile": [1, 128],
        "launches": 1,
        "native_source": {
            "jit": "@pl.jit" in native_source,
            "range_count": native_source.count("pl.range"),
            "load_count": native_source.count("pl.load"),
            "store_count": native_source.count("pl.store"),
            "whole_tensor_ops": "tensor." in native_source,
            "specialized_tile_load_count": rendered.count("pl.tile.load"),
            "specialized_tile_store_count": rendered.count("pl.tile.store"),
        },
        "max_abs_diff": max_abs_diff,
        "all_correct": correct,
        "fallback_used": artifact.fallback_used,
        "cubin_sha256": artifact.cubin_sha256,
    }
    rendered_evidence = json.dumps(evidence, indent=1)
    pathlib.Path(__file__).with_name("pointwise_results.json").write_text(
        rendered_evidence + "\n", encoding="utf-8"
    )
    print(rendered_evidence)
    return 0 if correct and not artifact.fallback_used else 75


if __name__ == "__main__":
    raise SystemExit(main())
