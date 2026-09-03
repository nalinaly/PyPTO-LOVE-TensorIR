#!/usr/bin/env python3
"""Capture one Chrome/Kineto PyTorch profiler JSON with stacks and GPU info.

Warmup (inductor-PyPTO residual-add compile) stays outside the profiler so the
trace is a live Ada two-layer forward: handwritten MM kernels plus fused
pointwise cubins. Requires CUDA Runtime API >= 13000; preload CUDA 13.3
libcudart when torch bundles 12.8:

    LD_PRELOAD=/usr/local/cuda-13.3/lib64/libcudart.so.13
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
KERNEL_SRC = ROOT / "packages" / "pypto-kernels" / "src"
PLUGIN_SRC = ROOT / "packages" / "pypto-framework-plugins" / "src"
sys.path.insert(0, str(HERE))
if PLUGIN_SRC.is_dir():
    sys.path.insert(0, str(PLUGIN_SRC))
if KERNEL_SRC.is_dir():
    sys.path.insert(0, str(KERNEL_SRC))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from pypto_kernels._boot import bootstrap
from pypto_two_layer_ada_sm89 import (
    BF16,
    HEAD_DIM,
    HIDDEN,
    Q_HEADS,
    TOKENS,
    ElementwiseInductorPypto,
    TwoLayerWeights,
    _cc,
    _stream,
    two_layer_pypto,
)


def _gpu_info() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    major, minor = torch.cuda.get_device_capability(0)
    try:
        driver = int(torch.backends.cuda.driver_version() or 0)
    except Exception:
        driver = 0
    runtime = int(getattr(torch.version, "cuda", "0").replace(".", "") or 0)
    try:
        import ctypes

        cudart = ctypes.CDLL("libcudart.so")
        ver = ctypes.c_int(0)
        if cudart.cudaRuntimeGetVersion(ctypes.byref(ver)) == 0:
            runtime = int(ver.value)
    except Exception:
        pass
    return {
        "index": 0,
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "compute_major": int(major),
        "compute_minor": int(minor),
        "total_memory_bytes": int(props.total_memory),
        "free_memory_bytes": int(free_b),
        "visible_total_memory_bytes": int(total_b),
        "multi_processor_count": int(props.multi_processor_count),
        "max_threads_per_multi_processor": int(
            getattr(props, "max_threads_per_multi_processor", 0)
        ),
        "warp_size": int(getattr(props, "warp_size", 32)),
        "regs_per_multiprocessor": int(getattr(props, "regs_per_multiprocessor", 0)),
        "shared_memory_per_multiprocessor_bytes": int(
            getattr(props, "shared_memory_per_multiprocessor", 0)
        ),
        "l2_cache_size_bytes": int(getattr(props, "L2_cache_size", 0)),
        "is_integrated": bool(getattr(props, "is_integrated", False)),
        "uuid": str(getattr(props, "uuid", "")),
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cuda_driver_version_reported": driver,
        "cuda_runtime_api_version": runtime,
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if _cc() != 89:
        raise RuntimeError(f"live GPU compute capability is {_cc()}, need 89")

    out_dir = ROOT / "reports" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pytorch-profiler-ada-sm89-two-layer-with-stack.json"

    bootstrap()
    try:
        import pypto_plugins.torch_inductor as torch_inductor
        from pypto_plugins.activity_trace import clear_artifact_registry_for_testing
        from pypto_plugins.torch import scheduling as inductor_scheduling

        torch_inductor.uninstall()
        torch._dynamo.reset()
        clear_artifact_registry_for_testing()
        inductor_scheduling.REGISTRY.clear()
    except Exception:
        pass

    stream = _stream()
    torch.manual_seed(21)
    weights = TwoLayerWeights()
    elementwise = ElementwiseInductorPypto()
    elementwise.warmup(
        torch.randn(TOKENS, HIDDEN, device="cuda", dtype=BF16),
        torch.randn(TOKENS, Q_HEADS * HEAD_DIM, device="cuda", dtype=BF16),
    )
    # Compile / first-launch stay outside the captured window.
    two_layer_pypto(weights, stream, use_gdn=False, elementwise=elementwise)
    torch.cuda.synchronize()

    gpu = _gpu_info()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    started = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
    ) as trace:
        for step in range(2):
            with record_function(f"two_layer_pypto_step_{step}"):
                two_layer_pypto(
                    weights, stream, use_gdn=False, elementwise=elementwise
                )
        torch.cuda.synchronize()
    elapsed_s = round(time.perf_counter() - started, 3)

    trace.export_chrome_trace(str(out_path))
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents") or []
    cats = {}
    python_stack_events = 0
    cuda_kernel_events = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        cat = str(event.get("cat") or "")
        cats[cat] = cats.get(cat, 0) + 1
        name = str(event.get("name") or "")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if cat in {"python_function", "python"} or "stack" in args or name.startswith("two_layer"):
            python_stack_events += 1
        if cat in {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"}:
            cuda_kernel_events += 1
        if isinstance(args.get("Call stack"), str) or isinstance(args.get("Python stack"), list):
            python_stack_events += 1

    payload["pyptoProfileMeta"] = {
        "kind": "pypto-ada-sm89-two-layer-pytorch-profiler",
        "schema": 1,
        "workload": "two_layer_pypto hybrid (handwritten MM + inductor-PyPTO add)",
        "steps_profiled": 2,
        "elapsed_s": elapsed_s,
        "gpu": gpu,
        "event_count": len(events),
        "event_categories": cats,
        "python_stack_events": python_stack_events,
        "cuda_or_gpu_events": cuda_kernel_events,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": True,
        "output": str(out_path),
    }
    if not payload.get("deviceProperties"):
        payload["deviceProperties"] = [
            {
                "id": 0,
                "name": gpu["name"],
                "totalGlobalMem": gpu["total_memory_bytes"],
                "computeMajor": gpu["compute_major"],
                "computeMinor": gpu["compute_minor"],
                "multiProcessorCount": gpu["multi_processor_count"],
                "regsPerMultiprocessor": gpu["regs_per_multiprocessor"],
                "warpSize": gpu["warp_size"],
                "sharedMemPerMultiprocessor": gpu["shared_memory_per_multiprocessor_bytes"],
                "l2CacheSize": gpu["l2_cache_size_bytes"],
                "uuid": gpu["uuid"],
            }
        ]
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = payload["pyptoProfileMeta"]
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)")
    if python_stack_events < 1:
        raise SystemExit("profiler JSON is missing Python stack events")
    if cuda_kernel_events < 1 and not payload.get("deviceProperties"):
        raise SystemExit("profiler JSON is missing GPU information")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
