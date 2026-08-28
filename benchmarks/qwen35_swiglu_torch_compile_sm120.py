#!/usr/bin/env python3
"""GPU regression for real 0.8B/9B packed, row-pitched SwiGLU shapes."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("PYPTO_WORKSPACE_ROOT", str(WORKSPACE_ROOT))
os.environ.setdefault("PYPTO_ENV_PREFIX", str(WORKSPACE_ROOT / "envs/pypto-nvidia"))


def _bind_installed_pypto_for_operator_bootstrap() -> None:
    distribution = importlib.metadata.distribution("pypto")
    core_candidates = [
        distribution.locate_file(file)
        for file in distribution.files or ()
        if Path(file).name.startswith("pypto_core") and Path(file).suffix == ".so"
    ]
    package_spec = importlib.util.find_spec("pypto")
    if len(core_candidates) != 1 or package_spec is None or package_spec.origin is None:
        raise RuntimeError("installed pypto package/extension identity is ambiguous")
    os.environ.setdefault(
        "PYPTO_KERNEL_DSO_PATH",
        str(Path(core_candidates[0]).resolve(strict=True)),
    )
    os.environ.setdefault(
        "PYPTO_KERNEL_PACKAGE_PATH",
        str(Path(package_spec.origin).resolve(strict=True).parent),
    )


_bind_installed_pypto_for_operator_bootstrap()

import torch  # noqa: E402

from pypto_kernels import silu_and_mul  # noqa: E402
from pypto_plugins.activity_trace import artifact_registry_snapshot  # noqa: E402
from pypto_plugins.torch import inductor_swiglu, pointwise_codegen  # noqa: E402
from pypto_plugins.torch.scheduling import REGISTRY  # noqa: E402


MODEL_DIRECTORIES = ("Qwen3.5-0.8B", "Qwen3.5-9B")


def _intermediate_size(model_root: Path, name: str) -> int:
    payload = json.loads((model_root / name / "config.json").read_text())
    text_config = payload.get("text_config", payload)
    value = text_config.get("intermediate_size")
    if type(value) is not int or value <= 0 or value % 128:
        raise RuntimeError(f"{name} has an incompatible intermediate_size={value!r}")
    return value


def _run_case(model: str, rows: int, columns: int) -> dict[str, object]:
    torch.manual_seed(rows * 1009 + columns)
    packed = torch.randn(
        (rows, 2 * columns),
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = packed[:, :columns]
    up = packed[:, columns:]
    expected_stride = (2 * columns, 1)
    if tuple(gate.stride()) != expected_stride or tuple(up.stride()) != expected_stride:
        raise RuntimeError("packed Qwen views lost their row pitch")

    generated = inductor_swiglu.run_fp32_swiglu(gate, up)
    generated_again = inductor_swiglu.run_fp32_swiglu(gate, up)
    handwritten = silu_and_mul.silu_and_mul(gate, up)
    reference = (
        gate.float() * torch.sigmoid(gate.float()) * up.float()
    ).to(torch.bfloat16)
    torch.cuda.synchronize()

    generated_error = (generated.float() - reference.float()).abs()
    handwritten_error = (generated.float() - handwritten.float()).abs()
    repeated_equal = bool(torch.equal(generated, generated_again))
    reference_close = bool(torch.allclose(generated, reference, rtol=1e-2, atol=3.125e-2))
    handwritten_equal = bool(torch.equal(generated, handwritten))
    if not repeated_equal or not reference_close or not handwritten_equal:
        raise RuntimeError(
            f"{model} rows={rows} SwiGLU mismatch: "
            f"repeat={repeated_equal}, reference={reference_close}, "
            f"handwritten={handwritten_equal}"
        )
    return {
        "model": model,
        "rows": rows,
        "columns": columns,
        "gate_stride": list(gate.stride()),
        "up_stride": list(up.stride()),
        "output_stride": list(generated.stride()),
        "max_abs_vs_fp32_formula": float(generated_error.max()),
        "max_abs_vs_handwritten": float(handwritten_error.max()),
        "repeated_equal": repeated_equal,
        "reference_close": reference_close,
        "handwritten_equal": handwritten_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=WORKSPACE_ROOT / "models",
    )
    parser.add_argument("--rows", default="1,19")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if not rows or any(value <= 0 for value in rows):
        raise ValueError("--rows must contain positive comma-separated integers")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("this regression requires one visible SM120 CUDA device")

    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    torch._dynamo.reset()
    REGISTRY.clear()
    pointwise_codegen.clear_caches_for_testing()
    inductor_swiglu.clear_callable_cache_for_testing()

    cases = [
        _run_case(model, row_count, _intermediate_size(args.model_root, model))
        for model in MODEL_DIRECTORIES
        for row_count in rows
    ]
    cache = inductor_swiglu.callable_cache_snapshot()
    artifacts = [
        record
        for record in artifact_registry_snapshot()
        if record.source_node.startswith("torch-inductor:")
    ]
    registry_artifacts = REGISTRY.unique_artifacts()
    expected_case_count = len(MODEL_DIRECTORIES) * len(rows)
    source_nodes = {entry[2] for entry in cache}
    all_native = (
        expected_case_count > 0
        and len(cases) == expected_case_count
        and len(cache) == expected_case_count
        and len(source_nodes) == expected_case_count
        and len(artifacts) == expected_case_count
        and len({record.artifact_id for record in artifacts}) == expected_case_count
        and {record.source_node for record in artifacts} == source_nodes
        and {record.provider for record in artifacts} == {"pypto.generic"}
        and len(registry_artifacts) == expected_case_count
        and {artifact.source_node for artifact in registry_artifacts} == source_nodes
        and all(not artifact.fallback_used for artifact in registry_artifacts)
    )
    payload = {
        "schema": 1,
        "kind": "qwen35-row-pitched-fp32-swiglu-torch-compile-sm120",
        "torch_compile": {
            "backend": "pypto",
            "dynamic": False,
            "fullgraph": True,
        },
        "cases": cases,
        "callable_cache_entries": len(cache),
        "expected_case_count": expected_case_count,
        "artifact_count": len(artifacts),
        "registry_artifact_count": len(registry_artifacts),
        "source_nodes": sorted(source_nodes),
        "providers": sorted({record.provider for record in artifacts}),
        "all_native": all_native,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if all_native else 1


if __name__ == "__main__":
    raise SystemExit(main())
