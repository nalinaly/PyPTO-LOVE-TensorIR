#!/usr/bin/env python3
"""RMSNorm-style reduction+broadcast chain through Inductor/PyPTO (SM120).

The norm chain is the GENERIC-site shape Qwen-class models need everywhere:
row reduction (PyPTO RowReductionV3), [M,1] epilogue pointwise, and the
[M,D] row-broadcast multiply (FusedPointwiseV2 row-expand operands).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-framework-plugins/src",
)

import torch  # noqa: E402
from torch._inductor.codegen import common  # noqa: E402

from pypto_plugins.torch import registration, scheduling  # noqa: E402
from pypto_plugins.torch.context import activate_mode  # noqa: E402


def main() -> int:
    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    common.init_backend_registration()
    registration.install()
    results = []
    failures = []
    try:
        torch.manual_seed(11)
        rows, cols = 256, 1024
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        def rmsnorm(t):
            mean_square = t.pow(2).mean(-1, keepdim=True)
            return t * torch.rsqrt(mean_square + 1e-6)

        def softmax_ref(t):
            shifted = t - t.amax(-1, keepdim=True)
            exponent = torch.exp(shifted)
            return exponent / exponent.sum(-1, keepdim=True)

        cases = {
            "rmsnorm": rmsnorm,
            "gated_rmsnorm": lambda t: rmsnorm(t) * torch.sigmoid(t),
        }
        for label, fn in cases.items():
            expected = fn(x)
            compiled = torch.compile(fn, backend="inductor", dynamic=False)
            error = None
            correct = False
            kernels_before = len(scheduling.REGISTRY._kernels)
            try:
                with activate_mode(strict=True):
                    output = compiled(x)
                    torch.cuda.synchronize()
                    correct = bool(
                        torch.allclose(
                            output, expected, rtol=1e-5, atol=1e-6, equal_nan=True
                        )
                    )
            except Exception as caught:  # noqa: BLE001 - recorded as marker
                error = f"{type(caught).__name__}: {caught}"
            kernels = {
                name: {
                    "cubin_sha256": artifact.cubin_sha256,
                    "fallback_used": artifact.fallback_used,
                }
                for name, artifact in list(scheduling.REGISTRY._kernels.items())[
                    kernels_before:
                ]
            }
            results.append(
                {
                    "case": label,
                    "output_correct": correct,
                    "error": error,
                    "kernels": kernels,
                }
            )
            if error is not None or not correct:
                failures.append(label)
        evidence = {
            "schema": 1,
            "kind": "inductor-pypto-rmsnorm-sm120-smoke",
            "shape": [rows, cols],
            "case_count": len(results),
            "all_correct": not failures,
            "failures": failures,
            "cases": results,
        }
        print(json.dumps(evidence, sort_keys=True, indent=1))
        return 0 if not failures else 75
    finally:
        registration.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
