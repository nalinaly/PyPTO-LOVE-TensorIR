#!/usr/bin/env python3
"""Ada sm_89: Inductor-fused pointwise lowered to a PyPTO cubin.

``torch.compile(backend=compile_backend)`` still lets Inductor fuse the
elementwise chain, then the PyPTO CUDA scheduling compiles that fused body
through TensorIR / tileiras instead of Triton. Matmul / attention stay out
of this path (strict mode fail-closed).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "pypto-framework-plugins" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pypto-kernels" / "src"))

import torch

from pypto_plugins.torch import inductor_swiglu, scheduling
from pypto_plugins.torch_backend import compile_backend
import pypto_plugins.torch_inductor as torch_inductor


def _compile(fn):
    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    torch._dynamo.reset()
    return torch.compile(fn, backend=compile_backend, dynamic=False, fullgraph=True)


def _kernel_records() -> list[dict[str, object]]:
    records = []
    for name, artifact in scheduling.REGISTRY.snapshot():
        records.append(
            {
                "registry_name": name,
                "kernel_name": artifact.kernel_name,
                "entry_name": artifact.entry_name,
                "cubin_bytes": artifact.cubin_bytes,
                "fallback_used": artifact.fallback_used,
            }
        )
    return records


def run() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (8, 9):
        raise RuntimeError(f"live GPU compute capability is {(major, minor)}, need (8, 9)")

    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    torch.manual_seed(7)
    scheduling.REGISTRY.clear()
    from pypto_plugins.activity_trace import clear_artifact_registry_for_testing

    clear_artifact_registry_for_testing()
    cases: list[dict[str, object]] = []

    x = torch.randn((16, 1024), device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    addmul = _compile(lambda a, b: (a + b) * 2)
    addmul_out = addmul(x, y)
    torch.cuda.synchronize()
    addmul_ref = (x + y) * 2
    cases.append(
        {
            "case": "inductor_fused_add_mul",
            "max_abs_diff": float((addmul_out.float() - addmul_ref.float()).abs().max()),
            "correct": bool(torch.equal(addmul_out, addmul_ref)),
        }
    )

    gate = torch.randn_like(x)
    value = torch.randn_like(x)
    sigmoid_mul = _compile(
        lambda v, g: (v.float() * torch.sigmoid(g.float())).to(v.dtype)
    )
    sm_out = sigmoid_mul(value, gate)
    torch.cuda.synchronize()
    sm_ref = (value.float() * torch.sigmoid(gate.float())).to(value.dtype)
    cases.append(
        {
            "case": "inductor_fused_sigmoid_mul",
            "max_abs_diff": float((sm_out.float() - sm_ref.float()).abs().max()),
            "correct": bool(torch.equal(sm_out, sm_ref)),
        }
    )

    up = torch.randn_like(x)
    swiglu = _compile(inductor_swiglu.fp32_swiglu_subgraph)
    sw_out = swiglu(gate, up)
    torch.cuda.synchronize()
    sw_ref = inductor_swiglu.fp32_swiglu_subgraph(gate, up)
    cases.append(
        {
            "case": "inductor_fused_fp32_swiglu",
            "max_abs_diff": float((sw_out.float() - sw_ref.float()).abs().max()),
            "correct": bool(torch.equal(sw_out, sw_ref)),
        }
    )

    kernels = _kernel_records()
    all_native = bool(kernels) and all(
        not item["fallback_used"] and int(item["cubin_bytes"]) > 0
        and str(item["entry_name"]) == "pypto_fused_pointwise"
        for item in kernels
    )
    return {
        "schema": 1,
        "kind": "pypto-inductor-pointwise-ada-sm89",
        "backend": "pypto",
        "live_cc": [major, minor],
        "cases": cases,
        "kernels": kernels,
        "all_correct": all(bool(case["correct"]) for case in cases),
        "all_native_pypto_cubin": all_native,
        "kernel_count": len({item["kernel_name"] for item in kernels}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    try:
        evidence = run()
    finally:
        torch_inductor.uninstall()
    text = json.dumps(evidence, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not evidence["all_correct"] or not evidence["all_native_pypto_cubin"]:
        print("inductor_pointwise_ada_sm89: FAIL")
        return 1
    print("inductor_pointwise_ada_sm89: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
