#!/usr/bin/env python3
"""Run one Qwen3.5-0.8B SGLang ModelRunner forward in one process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import pathlib
import traceback

from run_qwen35_0p8b_pypto_smoke import (
    PROMPT,
    PROMPT_TOKEN_IDS,
    ROOT,
    configure_environment,
)


def output_path() -> pathlib.Path:
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    path = ROOT / "runs" / run_id / "qwen35-0p8b-model-runner-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _model_revision(model_path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        if path.suffix in {".json", ".model"}:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _device_fingerprint(torch_module) -> str:
    properties = torch_module.cuda.get_device_properties(0)
    payload = {
        "capability": list(torch_module.cuda.get_device_capability(0)),
        "name": properties.name,
        "total_memory": properties.total_memory,
        "torch_cuda": torch_module.version.cuda,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _dispatch_inventory_mode(torch_module, monitor):
    from torch.utils._python_dispatch import TorchDispatchMode

    def describe(value):
        if isinstance(value, torch_module.Tensor):
            return {
                "device": str(value.device),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
            }
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): describe(item) for key, item in value.items()}
        if type(value) in (bool, int, float, str) or value is None:
            return value
        return type(value).__name__

    class DispatchInventoryMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            payload = json.dumps(
                {
                    "args": describe(args),
                    "kind": "torch-dispatch-op.v1",
                    "kwargs": describe(kwargs or {}),
                    "op": str(func),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            external_id = monitor.push_user_annotation(payload)
            if type(external_id) is not int:
                raise RuntimeError("CUPTI rejected a Torch dispatch annotation")
            try:
                return func(*args, **(kwargs or {}))
            finally:
                popped_id = monitor.pop_user_annotation()
                if popped_id != external_id:
                    raise RuntimeError("Torch dispatch CUPTI annotation stack mismatch")

    return DispatchInventoryMode()


def _dispatch_kernel_inventory(window: dict[str, object]) -> list[dict[str, object]]:
    annotations: dict[int, dict[str, object]] = {}
    for raw_id, raw_value in window["user_annotations"].items():
        try:
            payload = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("kind") == "torch-dispatch-op.v1":
            annotations[int(raw_id)] = payload
    correlations = {
        int(event["correlation_id"]): int(event["external_id"])
        for event in window["events"]
        if event.get("kind") == "external_correlation"
    }
    groups: dict[str, dict[str, object]] = {}
    for event in window["events"]:
        if event.get("kind") != "kernel":
            continue
        annotation = annotations.get(correlations.get(int(event["correlation_id"])))
        if annotation is None:
            continue
        identity = json.dumps(
            {
                "args": annotation["args"],
                "kernel_name": event["name"],
                "kwargs": annotation["kwargs"],
                "op": annotation["op"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        group = groups.setdefault(
            identity,
            {
                "args": annotation["args"],
                "call_count": 0,
                "gpu_time_ns": 0,
                "kernel_name": event["name"],
                "kwargs": annotation["kwargs"],
                "op": annotation["op"],
            },
        )
        group["call_count"] += 1
        group["gpu_time_ns"] += int(event["end_ns"]) - int(event["start_ns"])
    return sorted(groups.values(), key=lambda item: (-item["gpu_time_ns"], item["op"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument(
        "--coverage-mode",
        choices=("off", "development", "strict"),
        default="off",
    )
    parser.add_argument("--dispatch-inventory", action="store_true")
    args = parser.parse_args()
    configure_environment()
    report: dict[str, object] = {
        "schema": 1,
        "kind": "qwen35-0p8b-pypto-model-runner-smoke",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "model_path": str(ROOT / "models/Qwen3.5-0.8B"),
        "prompt": PROMPT,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "torch_compile": not args.disable_torch_compile,
        "coverage_mode": args.coverage_mode,
        "dispatch_inventory": args.dispatch_inventory,
        "status": "starting",
    }
    model_runner = None
    batch = None
    monitor = None
    monitor_api = None
    trace_window = None
    try:
        import torch
        if args.coverage_mode != "off":
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "strict CUPTI collection must start before CUDA initialization"
                )
            from torch.profiler import _cupti_monitor as monitor_api

            monitor = monitor_api.start_collection(
                output_path().parent / "cupti-monitor"
            )
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import (
            initialize_fp4_gemm_config,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            initialize_fp8_gemm_config,
        )
        from sglang.srt.plugins import load_plugins

        load_plugins()
        from sglang.benchmark import one_batch
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(
            model_path=str(ROOT / "models/Qwen3.5-0.8B"),
            tokenizer_path=str(ROOT / "models/Qwen3.5-0.8B"),
            skip_tokenizer_init=True,
            enable_multimodal=False,
            load_format="safetensors",
            model_loader_extra_config='{"enable_multithread_load": false}',
            weight_loader_drop_cache_after_load=True,
            dtype="bfloat16",
            attention_backend="pypto",
            decode_attention_backend="pypto",
            prefill_attention_backend="pypto",
            linear_attn_backend="pypto",
            linear_attn_decode_backend="pypto",
            linear_attn_prefill_backend="pypto",
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_custom_all_reduce=True,
            page_size=1,
            context_length=64,
            max_total_tokens=64,
            max_prefill_tokens=64,
            chunked_prefill_size=64,
            prefill_max_requests=1,
            max_running_requests=1,
            mem_fraction_static=0.55,
            enable_torch_compile=not args.disable_torch_compile,
            torch_compile_max_bs=1,
            sampling_backend="pytorch",
            grammar_backend="none",
            random_seed=19,
            log_level="error",
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
        )
        _set_envs_and_config(server_args)
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)
        port_args = PortArgs.init_new(server_args)
        one_batch.get_tokenizer = lambda *_args, **_kwargs: None
        model_runner, _tokenizer = one_batch.load_model(
            server_args, port_args, gpu_id=0, tp_rank=0
        )
        report["status"] = "model-ready"
        if monitor is not None:
            warmup_reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
                1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
            )
            _warmup_tokens, _warmup_logits, warmup_batch = model_runner.extend(
                warmup_reqs
            )
            model_runner.synchronize()
            model_runner.cleanup(warmup_batch)
            model_runner.clear()
            torch.cuda.synchronize()
            report["coverage_warmup"] = "complete"
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
        )
        if monitor is None:
            next_token_ids, logits, batch = model_runner.extend(reqs)
        else:
            torch_runner = model_runner.torch_runner
            original_forward = torch_runner.forward

            def traced_forward(*forward_args, **forward_kwargs):
                nonlocal trace_window
                monitor.begin_trace_window()
                try:
                    if args.dispatch_inventory:
                        with _dispatch_inventory_mode(torch, monitor):
                            return original_forward(*forward_args, **forward_kwargs)
                    return original_forward(*forward_args, **forward_kwargs)
                finally:
                    torch.cuda.synchronize()
                    trace_window = monitor.end_trace_window()

            torch_runner.forward = traced_forward
            try:
                next_token_ids, logits, batch = model_runner.extend(reqs)
            finally:
                torch_runner.forward = original_forward
        model_runner.synchronize()
        if monitor is not None:
            monitor_stats = monitor_api.stop_collection()
            monitor = None
            if trace_window is None:
                raise RuntimeError("model-forward CUPTI trace window was not captured")
            trace_path = output_path().parent / "qwen35-0p8b-cupti-window.json"
            _write_json(
                trace_path,
                {"stats": monitor_stats, "trace_window": trace_window},
            )
            if args.dispatch_inventory:
                dispatch_path = (
                    output_path().parent / "qwen35-0p8b-dispatch-inventory.json"
                )
                _write_json(
                    dispatch_path,
                    _dispatch_kernel_inventory(trace_window),
                )
                report["dispatch_inventory_path"] = str(dispatch_path)
            from pypto_plugins.activity_trace import normalize_cupti_window
            from pypto_plugins.coverage import (
                FRAMEWORK_PROFILE,
                TRACE_COLLECTOR,
                TRACE_COLLECTOR_REVISION,
                CoverageAuditor,
                CoverageMode,
                TraceManifest,
                compute_artifact_registry_digest,
                compute_trace_digest,
            )

            normalized = normalize_cupti_window(
                trace_window,
                dropped_records=int(monitor_stats["dropped_records"]),
            )
            artifacts = list(normalized.artifacts)
            events = list(normalized.events)
            manifest = TraceManifest(
                run_id=str(report["run_id"]),
                model_id="Qwen/Qwen3.5-0.8B",
                model_revision=_model_revision(ROOT / "models/Qwen3.5-0.8B"),
                device_fingerprint=_device_fingerprint(torch),
                collector=TRACE_COLLECTOR,
                collector_revision=TRACE_COLLECTOR_REVISION,
                framework_profile=FRAMEWORK_PROFILE,
                artifact_registry_digest=compute_artifact_registry_digest(
                    artifacts
                ),
                trace_digest=compute_trace_digest(events),
                activity_count=len(events),
                closed_world=normalized.closed_world,
            )
            coverage_report = output_path().parent / "qwen35-0p8b-coverage.json"
            mode = (
                CoverageMode.STRICT
                if args.coverage_mode == "strict"
                else CoverageMode.DEVELOPMENT
            )
            with CoverageAuditor(
                mode=mode,
                report_path=coverage_report,
                manifest=manifest,
                artifacts=artifacts,
            ) as auditor:
                for event in events:
                    auditor.record(event)
                coverage_summary = auditor.finalize(
                    event_stream_complete=normalized.closed_world
                )
            report["coverage"] = {
                "report_path": str(coverage_report),
                "trace_path": str(trace_path),
                "summary": asdict(coverage_summary),
            }
        logits_cpu = logits.detach().float().cpu().contiguous()
        top_values, top_indices = torch.topk(logits_cpu[0], 5)
        report.update(
            {
                "status": "forward-complete",
                "next_token_ids": next_token_ids.cpu().tolist(),
                "logits_shape": list(logits_cpu.shape),
                "logits_finite": bool(torch.isfinite(logits_cpu).all()),
                "logits_min": float(logits_cpu.min()),
                "logits_max": float(logits_cpu.max()),
                "logits_sha256": hashlib.sha256(
                    logits_cpu.numpy().tobytes()
                ).hexdigest(),
                "top5_token_ids": top_indices.tolist(),
                "top5_logits": top_values.tolist(),
            }
        )
        _write_json(output_path(), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(output_path(), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    finally:
        if monitor is not None and monitor_api is not None:
            try:
                monitor_api.stop_collection()
            except BaseException:
                pass
        if model_runner is not None and batch is not None:
            model_runner.cleanup(batch)
        try:
            from sglang.srt.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
