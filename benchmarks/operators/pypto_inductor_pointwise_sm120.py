#!/usr/bin/env python3
"""First real-SM120 Inductor-scheduled PyPTO pointwise smoke (CUDA device).

Installs the plugin's reversible CUDA registration, compiles a CUDA
pointwise function through TorchInductor inside the strict PyPTO mode,
and proves the Inductor scheduling path genuinely routed through
``PyptoCudaScheduling`` and produced a non-fallback SM120 Cubin artifact
in the plugin registry. The wrapper launch fails closed through the
pending runtime bridge by design; that marker is recorded.
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
    common.init_backend_registration()
    registration.install()
    try:
        torch.manual_seed(0)
        lhs = torch.randn(1024, device="cuda", dtype=torch.float32)
        rhs = torch.randn(1024, device="cuda", dtype=torch.float32)

        def fn(x, y):
            return x + y

        expected = fn(lhs, rhs)
        compiled = torch.compile(fn, backend="inductor", dynamic=False)
        wrapper_error = None
        with activate_mode(strict=True):
            try:
                output = compiled(lhs, rhs)
                torch.cuda.synchronize()
                correct = bool(torch.equal(output, expected))
            except Exception as error:  # noqa: BLE001 - recorded as marker
                wrapper_error = f"{type(error).__name__}: {error}"[:200]
                correct = False

        kernels = {
            name: {
                "entry": artifact.entry_name,
                "cubin_sha256": artifact.cubin_sha256,
                "cubin_bytes": artifact.cubin_bytes,
                "argument_count": artifact.argument_count,
                "grid": list(artifact.grid),
                "workspace_bytes": artifact.workspace_bytes,
                "fallback_used": artifact.fallback_used,
            }
            for name, artifact in scheduling.REGISTRY._kernels.items()
        }
        evidence = {
            "schema": 1,
            "kind": "inductor-pypto-pointwise-sm120-smoke",
            "scheduling_routed": bool(kernels),
            "kernel_count": len(kernels),
            "kernels": kernels,
            "wrapper_error": wrapper_error,
            "output_correct": correct,
        }
        print(json.dumps(evidence, sort_keys=True, indent=1))
        if not kernels:
            print("FAIL: Inductor never routed through PyPTO scheduling")
            return 75
        artifact = next(iter(scheduling.REGISTRY._kernels.values()))
        if artifact.fallback_used or artifact.cubin_bytes == 0:
            print("FAIL: fallback or empty Cubin")
            return 75
        return 0
    finally:
        registration.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
