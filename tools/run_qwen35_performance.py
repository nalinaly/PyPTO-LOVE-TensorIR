#!/usr/bin/env python3
"""Measure warm exact-prompt Qwen forwards without profiler overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from run_qwen35_0p8b_model_runner_smoke import _model_revision
from run_qwen35_0p8b_pypto_smoke import (
    PROMPT,
    PROMPT_TOKEN_IDS,
    ROOT,
    configure_environment,
)
from run_qwen35_0p8b_stability import _parity


REQUEST_COUNT = 10


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("0.8B", "9B"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    args = parser.parse_args()

    model_slug = "0p8b" if args.model_size == "0.8B" else "9b"
    model_id = f"Qwen/Qwen3.5-{args.model_size}"
    model_path = ROOT / f"models/Qwen3.5-{args.model_size}"
    mem_fraction_static = 0.55 if args.model_size == "0.8B" else 0.78
    cpu_offload_gb = 0 if args.model_size == "0.8B" else 2
    configure_environment()
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / f"qwen35-{model_slug}-performance.json"
    policy_bytes = args.policy.read_bytes()
    policy = json.loads(policy_bytes)
    report: dict[str, object] = {
        "kind": f"qwen35-{model_slug}-warm-performance",
        "model_id": model_id,
        "model_revision": _model_revision(model_path),
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "prompt": PROMPT,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "request_count": REQUEST_COUNT,
        "cpu_offload_gb": cpu_offload_gb,
        "mem_fraction_static": mem_fraction_static,
        "profiler_enabled": False,
        "run_id": run_id,
        "schema": 1,
        "status": "starting",
    }
    model_runner = None
    try:
        if (
            policy.get("status")
            != "frozen-before-final-candidate-measurement"
            or policy.get("model", {}).get("id") != model_id
            or policy.get("prompt") != PROMPT
            or policy.get("prompt_token_ids") != PROMPT_TOKEN_IDS
        ):
            raise ValueError("performance policy does not match model/prompt")
        if (
            hashlib.sha256(args.reference_logits.read_bytes()).hexdigest()
            != policy["reference"]["logits_tensor_file_sha256"]
        ):
            raise ValueError("reference logits file differs from frozen policy")

        import torch
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import (
            initialize_fp4_gemm_config,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            initialize_fp8_gemm_config,
        )
        from sglang.srt.plugins import load_plugins
        from sglang.benchmark import one_batch
        from sglang.srt.server_args import PortArgs, ServerArgs
        from transformers import AutoTokenizer

        load_plugins()
        server_args = ServerArgs(
            model_path=str(model_path),
            tokenizer_path=str(model_path),
            skip_tokenizer_init=True,
            enable_multimodal=False,
            language_model_only=True,
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
            mem_fraction_static=mem_fraction_static,
            cpu_offload_gb=cpu_offload_gb,
            enable_torch_compile=True,
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

        load_start = time.perf_counter_ns()
        model_runner, _tokenizer = one_batch.load_model(
            server_args, port_args, gpu_id=0, tp_rank=0
        )
        report["cold_model_load_ms"] = (time.perf_counter_ns() - load_start) / 1e6
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        expected_ids = policy["reference"]["next_token_ids"]
        expected_text = tokenizer.decode(expected_ids, skip_special_tokens=False)
        reference = torch.load(
            args.reference_logits, map_location="cpu", weights_only=True
        ).float().contiguous()
        if (
            hashlib.sha256(reference.numpy().tobytes()).hexdigest()
            != policy["reference"]["logits_raw_sha256"]
        ):
            raise ValueError("reference logits payload differs from frozen policy")

        warmup_reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
        )
        torch.cuda.synchronize()
        warmup_start = time.perf_counter_ns()
        _warm_ids, _warm_logits, warmup_batch = model_runner.extend(warmup_reqs)
        model_runner.synchronize()
        torch.cuda.synchronize()
        report["cold_compile_warmup_ms"] = (
            time.perf_counter_ns() - warmup_start
        ) / 1e6
        model_runner.cleanup(warmup_batch)
        model_runner.clear()

        requests = []
        latencies_ms: list[float] = []
        for request_index in range(REQUEST_COUNT):
            reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
                1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
            )
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            next_ids, logits, batch = model_runner.extend(reqs)
            model_runner.synchronize()
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter_ns() - start) / 1e6
            latencies_ms.append(latency_ms)
            logits_cpu = logits.detach().float().cpu().contiguous()
            ids = next_ids.cpu().tolist()
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            parity = _parity(policy, reference, logits_cpu)
            requests.append(
                {
                    "request_index": request_index,
                    "latency_ms": latency_ms,
                    "next_token_ids": ids,
                    "decoded_text": decoded,
                    "logits_sha256": hashlib.sha256(
                        logits_cpu.numpy().tobytes()
                    ).hexdigest(),
                    "parity": parity,
                    "passed": bool(
                        ids == expected_ids
                        and decoded == expected_text
                        and parity["passed"]
                    ),
                }
            )
            model_runner.cleanup(batch)
            model_runner.clear()

        report["requests"] = requests
        report["all_passed"] = all(item["passed"] for item in requests)
        report["latency_ms"] = {
            "min": min(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p90_nearest_rank": _percentile(latencies_ms, 0.9),
            "max": max(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
        }
        report["throughput"] = {
            "prompt_tokens_per_second_at_median": (
                len(PROMPT_TOKEN_IDS) * 1000.0 / statistics.median(latencies_ms)
            ),
            "requests_per_second_at_median": (
                1000.0 / statistics.median(latencies_ms)
            ),
        }
        report["unique_logits_sha256"] = sorted(
            {item["logits_sha256"] for item in requests}
        )
        report["status"] = "complete" if report["all_passed"] else "failed"
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if report["all_passed"] else 1
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    finally:
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
