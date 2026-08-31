#!/usr/bin/env python3
"""Run computational article examples through the NVIDIA compatibility path.

The upstream article files are imported and hashed but never edited.  The
compatibility implementations live in this file: ``hello_world`` uses the
strict PyPTO/TensorIR/CUDA-Tile bridge; the remaining small teaching examples
use independent CUDA Torch references until a corresponding strict adapter is
available.  Hardware-facing and unmapped entries are rejected by policy rather
than silently falling back.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo" / "pypto-lib"
MANIFEST = DEMO_ROOT / "SOURCE_MANIFEST.json"
POLICY_PATH = (
    ROOT / "state" / "evidence" / "article-demo-compatibility-policy-current.json"
)
ARTICLE_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def corpus_sha256(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        manifest.get("files", []), key=lambda item: str(item.get("path", ""))
    ):
        relative = str(record["path"])
        path = (DEMO_ROOT / relative).resolve()
        if DEMO_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"imported corpus file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    temporary.replace(path)


def load_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("kind") != "article-demo-compatibility-policy":
        raise ValueError("unexpected article demo compatibility policy")
    if policy.get("upstream_commit") != ARTICLE_COMMIT:
        raise ValueError("compatibility policy is not article-time locked")
    manifest = load_manifest()
    if (
        policy.get("manifest_sha256") != sha256_file(MANIFEST)
        or policy.get("corpus_sha256") != corpus_sha256(manifest)
    ):
        raise ValueError("compatibility policy manifest hash is stale")
    return policy


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("kind") != "article-demo-provenance":
        raise ValueError("unexpected article demo manifest")
    if manifest.get("upstream", {}).get("commit") != ARTICLE_COMMIT:
        raise ValueError("article demo manifest is not article-time locked")
    return manifest


def source_record(relative: str, manifest: dict[str, Any]) -> dict[str, object]:
    path = (DEMO_ROOT / relative).resolve()
    if DEMO_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"demo path is outside the imported tree: {relative}")
    record = next((item for item in manifest["files"] if item.get("path") == relative), None)
    if not isinstance(record, dict):
        raise ValueError(f"demo is absent from SOURCE_MANIFEST.json: {relative}")
    observed = sha256_file(path)
    if observed != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        raise ValueError(f"imported demo source changed: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def policy_entry(relative: str, policy: dict[str, Any]) -> dict[str, Any]:
    entry = next((item for item in policy["entries"] if item.get("path") == relative), None)
    if not isinstance(entry, dict):
        raise ValueError(f"demo has no compatibility policy: {relative}")
    return entry


def import_demo(relative: str) -> Any:
    path = (DEMO_ROOT / relative).resolve()
    module_name = "_article_demo_" + hashlib.sha256(relative.encode()).hexdigest()[:16]
    if str(DEMO_ROOT) not in sys.path:
        # The copied examples import their shared ``golden`` package by its
        # original top-level name.  Keeping this path for the process lifetime
        # is the compatibility equivalent of running from the article tree.
        sys.path.insert(0, str(DEMO_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _builder(module: Any) -> Callable[[], list[Any]]:
    for name in ("build_specs", "build_tensor_specs"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise ValueError("article demo does not expose build_specs/build_tensor_specs")


def _golden(module: Any) -> Callable[[dict[str, Any]], object]:
    candidates = [
        value
        for name, value in vars(module).items()
        if name.startswith("golden_") and callable(value)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one golden function, found {len(candidates)}")
    return candidates[0]


def _spec_values(module: Any, seed: int) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    specs = list(_builder(module)())
    if not specs:
        raise ValueError("article demo returned no tensor/scalar specs")
    values = {
        spec.name: spec.create_tensor()
        if hasattr(spec, "create_tensor")
        else spec.value.clone()
        if isinstance(spec.value, torch.Tensor)
        else spec.value
        for spec in specs
    }
    expected = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }
    _golden(module)(expected)
    return specs, values, expected


def _reference_adapter(relative: str, values: dict[str, Any]) -> dict[str, Any]:
    """Compute the same mathematical result independently of the imported golden."""

    import torch

    if relative.endswith("gemm_eltwise.py"):
        values["resid"] = values["attn_out"].float() @ values["wo"].float()
        values["resid"] = values["resid"] + values["hidden_states"].float()
    elif relative.endswith("multi_proj.py"):
        x = values["x"].float()
        values["q_out"] = x @ values["wq"].float()
        values["k_out"] = x @ values["wk"].float()
        values["v_out"] = x @ values["wv"].float()
    elif relative.endswith("topk.py"):
        # The imported example's tie policy is checked by the paired values;
        # deterministic random inputs make this path exact in normal runs.
        vals, indices = torch.topk(values["scores"].float(), 16, dim=-1, sorted=True)
        values["topk_vals"] = vals
        values["topk_idx"] = indices.to(torch.int32)
    elif relative.endswith("matmul.py") or relative.endswith("gemm.py"):
        output = values["a"].float() @ values["b"].float()
        values["c"] = output
    elif relative.endswith("layer_norm.py"):
        x = values["x"].float()
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        values["y"] = (x - mean) / torch.sqrt(variance + 1e-5)
        values["y"] = values["y"] * values["gamma"].float() + values["beta"].float()
    elif relative.endswith("rms_norm.py"):
        x = values["x"].float()
        values["y"] = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        values["y"] = values["y"] * values["gamma"].float()
    elif relative.endswith("rope.py"):
        x = values["x"].float()
        half = x.shape[-1] // 2
        values["y"] = torch.cat(
            [
                x[..., :half] * values["cos"][..., :half].float()
                - x[..., half:] * values["sin"][..., :half].float(),
                x[..., half:] * values["cos"][..., half:].float()
                + x[..., :half] * values["sin"][..., half:].float(),
            ],
            dim=-1,
        )
    elif relative.endswith("softmax.py"):
        values["y"] = torch.softmax(values["x"].float(), dim=-1)
    else:
        raise ValueError(f"no computational reference adapter for {relative}")
    return values


def _run_strict_hello(values: dict[str, Any], device: torch.device, source_node: str) -> dict[str, Any]:
    import torch

    from pypto_plugins.torch import pointwise_codegen as pc
    from pypto_plugins.torch.runtime_bridge import pypto_launch
    from pypto_plugins.torch.scheduling import REGISTRY

    x = values["x"].to(device)
    output = torch.empty_like(x)
    scalar = float(values["a"].item())
    flat_shape = (x.numel(),)
    builder = pc.PointwiseProgramBuilder(flat_shape, "float32")
    input_value = builder.add_input("x")
    result_value = builder.emit("tensor.adds", [input_value, builder.scalar(scalar)])
    builder.mark_output(result_value)
    program = builder.build()
    registry_name = "article_demo_hello_" + source_node.replace("/", "_").replace(".", "_")
    artifact = pc.compile_pointwise(program, tile=128, registry_name=registry_name)
    REGISTRY.register(registry_name, artifact)
    stream = torch.cuda.current_stream(device)
    pypto_launch(registry_name, (x.reshape(-1), output.reshape(-1)), stream.cuda_stream)
    stream.synchronize()
    return {
        "outputs": {"y": output},
        "adapter": "strict-pypto-nvidia",
        "artifact": {
            "kernel_name": artifact.kernel_name,
            "entry_name": artifact.entry_name,
            "source_sha256": artifact.pypto_source_sha256,
            "artifact_sha256": artifact.artifact_sha256,
            "cubin_sha256": artifact.cubin_sha256,
            "cache_key_sha256": artifact.artifact_cache_key_sha256,
            "source_node": artifact.source_node,
            "fallback_used": artifact.fallback_used,
            "tile": [128],
            "flattened_shape": list(flat_shape),
            "specialized_scalar_a": scalar,
        },
    }


def _compare(
    outputs: dict[str, Any], expected: dict[str, Any], specs: list[Any]
) -> tuple[bool, list[dict[str, Any]]]:
    import torch

    metrics: list[dict[str, Any]] = []
    all_ok = True
    for spec in specs:
        if not getattr(spec, "is_output", False):
            continue
        actual = outputs[spec.name].detach().cpu()
        reference = expected[spec.name].detach().cpu()
        if actual.dtype.is_floating_point:
            tolerance = 1e-2 if actual.dtype is torch.bfloat16 else 1e-4
            ok = bool(torch.allclose(actual.float(), reference.float(), rtol=tolerance, atol=tolerance))
            max_abs = float((actual.float() - reference.float()).abs().max().item())
        else:
            ok = bool(torch.equal(actual, reference))
            max_abs = (
                0.0
                if ok
                else float(
                    (actual.to(torch.int64) - reference.to(torch.int64))
                    .abs()
                    .max()
                    .item()
                )
            )
        all_ok = all_ok and ok
        metrics.append({
            "name": spec.name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "ok": ok,
            "max_abs_diff": max_abs,
        })
    return all_ok, metrics


def run(
    relative: str, *, device_index: int, seed: int, run_id: str
) -> dict[str, object]:
    import torch

    manifest = load_manifest()
    policy = load_policy()
    entry = policy_entry(relative, policy)
    source = source_record(relative, manifest)
    mode = str(entry["compatibility_mode"])
    started = time.monotonic()
    payload: dict[str, object] = {
        "schema": 2,
        "kind": "article-demo-nvidia-run",
        "article_url": manifest["article"]["url"],
        "upstream_commit": manifest["upstream"]["commit"],
        "run_id": run_id,
        "source": source,
        "compatibility": {
            "mode": mode,
            "adapter": entry.get("adapter"),
            "reason": entry["reason"],
            "policy_sha256": sha256_file(POLICY_PATH),
        },
        "seed": seed,
        "device_index": device_index,
        "adapter_source": {
            "path": "tools/run_article_demo_nvidia.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "status": "not_started",
    }
    if mode not in {"strict-pypto-nvidia", "computational-cuda-reference"}:
        payload.update(
            {
                "status": "skipped",
                "skip_reason": entry["reason"],
                "elapsed_seconds": 0.0,
                "stages": {
                    "source_audit": "pass",
                    "compatibility_compile": "skipped",
                    "device_execution": "skipped",
                    "precision_compare": "skipped",
                },
            }
        )
        return payload
    if not torch.cuda.is_available():
        payload.update(
            {
                "status": "blocked",
                "blocker": "CUDA runtime/device unavailable",
                "stages": {
                    "source_audit": "pass",
                    "compatibility_compile": "not_started",
                    "device_execution": "blocked",
                    "precision_compare": "not_started",
                },
            }
        )
        return payload
    device = torch.device("cuda", device_index)
    module = import_demo(relative)
    specs, cpu_values, expected = _spec_values(module, seed)
    payload["inputs"] = [
        {
            "name": spec.name,
            "shape": list(getattr(spec, "shape", [])),
            "dtype": str(getattr(spec, "dtype", "")),
            "is_output": bool(getattr(spec, "is_output", False)),
        }
        for spec in specs
    ]
    payload["golden"] = {
        "function": next(
            name
            for name, value in vars(module).items()
            if name.startswith("golden_") and callable(value)
        ),
        "outputs": [spec.name for spec in specs if getattr(spec, "is_output", False)],
    }
    device_values = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in cpu_values.items()
    }
    if mode == "strict-pypto-nvidia":
        result = _run_strict_hello(device_values, device, relative)
    else:
        result_values = _reference_adapter(relative, device_values)
        result = {
            "outputs": {
                spec.name: result_values[spec.name]
                for spec in specs
                if getattr(spec, "is_output", False)
            },
            "adapter": "computational-cuda-reference",
        }
    passed, metrics = _compare(result["outputs"], expected, specs)
    payload.update(
        {
            "adapter": result["adapter"],
            "outputs": metrics,
            "golden_pass": passed,
            "strict_compiler_evidence": mode == "strict-pypto-nvidia",
            "elapsed_seconds": time.monotonic() - started,
            "status": "pass" if passed else "fail",
            "stages": {
                "source_audit": "pass",
                "input_generation": "pass",
                "golden_reference": "pass",
                "compatibility_compile": (
                    "strict-pypto-nvidia"
                    if mode == "strict-pypto-nvidia"
                    else "independent-cuda-reference"
                ),
                "device_execution": "pass",
                "precision_compare": "pass" if passed else "fail",
            },
        }
    )
    if "artifact" in result:
        payload["artifact"] = result["artifact"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, help="path relative to demo/pypto-lib")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--run-id",
        default=None,
        help="stable evidence identifier; defaults to article-demo-nvidia-<stem>",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    relative = Path(args.demo).as_posix()
    if relative.startswith("/") or ".." in Path(relative).parts:
        parser.error("--demo must stay below demo/pypto-lib")
    try:
        run_id = args.run_id or f"article-demo-nvidia-{Path(relative).stem}"
        allowed_run_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
        if not run_id or any(ch not in allowed_run_id for ch in run_id):
            parser.error("--run-id contains unsupported characters")
        payload = run(relative, device_index=args.device, seed=args.seed, run_id=run_id)
    except Exception as error:  # noqa: BLE001 - preserve exact blocker in evidence
        source: dict[str, object] = {"path": relative}
        try:
            source = source_record(relative, load_manifest())
        except Exception:  # noqa: BLE001 - retain the original failure detail
            pass
        payload = {
            "schema": 2,
            "kind": "article-demo-nvidia-run",
            "source": source,
            "run_id": args.run_id,
            "adapter_source": {
                "path": "tools/run_article_demo_nvidia.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_json(args.output.resolve(), payload)
    print(f"[ARTICLE-NVIDIA] source={relative}")
    print(f"[ARTICLE-NVIDIA] mode={payload.get('compatibility', {}).get('mode', 'unknown')}")
    print(
        "[ARTICLE-NVIDIA] adapter="
        f"{payload.get('adapter', payload.get('compatibility', {}).get('adapter'))}"
    )
    print(f"[ARTICLE-NVIDIA] status={payload.get('status')} golden_pass={payload.get('golden_pass')}")
    if payload.get("artifact"):
        artifact = payload["artifact"]
        print(
            f"[ARTICLE-NVIDIA] artifact={artifact.get('kernel_name')} "
            f"fallback_used={artifact.get('fallback_used')}"
        )
    print(f"article demo NVIDIA report: {args.output.resolve()}")
    return 0 if payload.get("status") in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
