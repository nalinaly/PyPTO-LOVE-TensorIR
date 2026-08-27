#!/usr/bin/env python3
"""Activation-function Inductor/PyPTO end-to-end coverage smoke (SM120).

Compiles the activation shapes a Qwen-class model needs through
TorchInductor inside strict PyPTO mode and checks output correctness plus
which PyPTO kernels were produced. Ops the table cannot yet express fail
closed and are recorded as coverage gaps, not errors.
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
        torch.manual_seed(3)
        x = torch.randn(2048, device="cuda", dtype=torch.float32)

        cases = {
            "sigmoid": lambda t: torch.sigmoid(t),
            "silu": lambda t: torch.nn.functional.silu(t),
            "relu": lambda t: torch.nn.functional.relu(t),
            "tanh": lambda t: torch.tanh(t),
            "swish_exp": lambda t: t / (1.0 + torch.exp(-t)),
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
                    "activation": label,
                    "output_correct": correct,
                    "error": error,
                    "kernels": kernels,
                }
            )
            if error is not None or not correct:
                failures.append(label)
        evidence = {
            "schema": 1,
            "kind": "inductor-pypto-actfn-sm120-smoke",
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
