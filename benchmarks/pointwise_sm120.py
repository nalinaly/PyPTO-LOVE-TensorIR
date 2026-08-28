#!/usr/bin/env python3
"""SM120 acceptance for an Inductor-style generated native tile graph."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from pypto_plugins.torch import pointwise_codegen  # noqa: E402
from pypto_plugins.torch.runtime_bridge import pypto_launch  # noqa: E402
from pypto_plugins.torch.scheduling import REGISTRY  # noqa: E402


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

    reduction_program = pointwise_codegen.NativeReductionProgram(
        (256, 128), "bfloat16", "sum"
    )
    reduction_source = reduction_program.native_source()
    reduction_name = "inductor_native_reduction_acceptance"
    reduction_artifact = pointwise_codegen.compile_pointwise(
        reduction_program,
        tile=reduction_program.row_tile,
        registry_name=reduction_name,
    )
    REGISTRY.register(reduction_name, reduction_artifact)
    reduction_input = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    reduction_out = torch.empty((256, 1), device="cuda", dtype=torch.bfloat16)
    pypto_launch(
        reduction_name,
        (reduction_input, reduction_out),
        stream.cuda_stream,
    )
    stream.synchronize()
    reduction_reference = (
        reduction_input.float().sum(-1, keepdim=True).to(torch.bfloat16)
    )
    reduction_diff = float(
        (reduction_out.float() - reduction_reference.float()).abs().max()
    )
    reduction_correct = bool(torch.equal(reduction_out, reduction_reference))

    dso = pointwise_codegen.pypto_dso_path()
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
        "reduction": {
            "shape": [256, 128],
            "output_shape": [256, 1],
            "dtype": "bfloat16",
            "row_tile": reduction_program.row_tile,
            "launches": 1,
            "jit": "@pl.jit" in reduction_source,
            "row_sum": "pl.row_sum" in reduction_source,
            "whole_tensor_ops": "tensor." in reduction_source,
            "max_abs_diff": reduction_diff,
            "correct": reduction_correct,
            "fallback_used": reduction_artifact.fallback_used,
        },
        "all_correct": correct and reduction_correct,
        "fallback_used": artifact.fallback_used,
        "cubin_sha256": artifact.cubin_sha256,
    }
    rendered_evidence = json.dumps(evidence, indent=1)
    pathlib.Path(__file__).with_name("pointwise_results.json").write_text(
        rendered_evidence + "\n", encoding="utf-8"
    )
    print(rendered_evidence)
    return (
        0
        if correct
        and reduction_correct
        and not artifact.fallback_used
        and not reduction_artifact.fallback_used
        else 75
    )


if __name__ == "__main__":
    raise SystemExit(main())
