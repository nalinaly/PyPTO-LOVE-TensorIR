#!/usr/bin/env python3
"""Run the exact prompt through stock SGLang providers without PyPTO."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/Qwen3.5-0.8B"
SGLANG_SOURCE = ROOT / "upstream/sglang/python"
SGLANG_OVERLAY = ROOT / "envs/sglang-runtime-py314"
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
PROMPT_TOKEN_IDS = [
    144277,
    103426,
    108169,
    95967,
    236,
    124094,
    26076,
    96212,
    103182,
    108076,
    96799,
    24273,
    95761,
    104224,
    109276,
    95726,
    111104,
    115110,
    10992,
]


def main() -> int:
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "qwen35-0p8b-sglang-reference.json"
    logits_path = run_dir / "qwen35-0p8b-sglang-reference-logits.pt"
    report: dict[str, object] = {
        "kind": "qwen35-0p8b-stock-sglang-reference",
        "model_path": str(MODEL_PATH),
        "prompt": PROMPT,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "run_id": run_id,
        "schema": 1,
        "status": "starting",
    }
    model_runner = None
    batch = None
    try:
        sys.path[:0] = [str(SGLANG_SOURCE), str(SGLANG_OVERLAY)]
        os.environ["PYTHONPATH"] = os.pathsep.join(
            (str(SGLANG_SOURCE), str(SGLANG_OVERLAY))
        )
        os.environ.pop("SGLANG_PLUGINS", None)
        os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"
        os.environ["CPATH"] = "/usr/local/cuda-13.3/targets/x86_64-linux/include"
        os.environ["SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS"] = "1"
        os.environ["SGLANG_IO_WORKERS"] = "1"
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from sglang.benchmark import one_batch
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import (
            initialize_fp4_gemm_config,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            initialize_fp8_gemm_config,
        )
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(
            model_path=str(MODEL_PATH),
            tokenizer_path=str(MODEL_PATH),
            skip_tokenizer_init=True,
            enable_multimodal=False,
            language_model_only=True,
            load_format="safetensors",
            model_loader_extra_config='{"enable_multithread_load": false}',
            weight_loader_drop_cache_after_load=True,
            dtype="bfloat16",
            attention_backend="flashinfer",
            decode_attention_backend="flashinfer",
            prefill_attention_backend="flashinfer",
            linear_attn_backend="flashinfer",
            linear_attn_decode_backend="flashinfer",
            linear_attn_prefill_backend="flashinfer",
            mamba_ssm_dtype="bfloat16",
            bf16_gemm_backend="torch",
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
            enable_torch_compile=False,
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
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
        )
        next_token_ids, logits, batch = model_runner.extend(reqs)
        model_runner.synchronize()
        logits_cpu = logits.detach().float().cpu().contiguous()
        torch.save(logits_cpu, logits_path)
        top_values, top_indices = torch.topk(logits_cpu[0], 5)
        report.update(
            {
                "status": "complete",
                "logits_path": str(logits_path),
                "logits_shape": list(logits_cpu.shape),
                "logits_finite": bool(torch.isfinite(logits_cpu).all()),
                "logits_min": float(logits_cpu.min()),
                "logits_max": float(logits_cpu.max()),
                "logits_sha256": hashlib.sha256(
                    logits_cpu.numpy().tobytes()
                ).hexdigest(),
                "top5_token_ids": top_indices.tolist(),
                "top5_logits": top_values.tolist(),
                "next_token_ids": next_token_ids.cpu().tolist(),
            }
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    finally:
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
