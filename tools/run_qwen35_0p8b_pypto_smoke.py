#!/usr/bin/env python3
"""Run one fail-closed Qwen3.5-0.8B PyPTO Engine request."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SGLANG_SOURCE = ROOT / "upstream/sglang/python"
SGLANG_OVERLAY = ROOT / "envs/sglang-runtime-py314"
KERNEL_SOURCE = ROOT / "worktrees/pypto-kernels-stateful-gate/src"
PLUGIN_SOURCE = ROOT / "projects/pypto-framework-plugins/src"
PYPTO_PACKAGE = ROOT / "worktrees/pypto-paged-decode/python/pypto"
PYPTO_DSO = (
    ROOT / "builds/pypto-paged-f34c3f5/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
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


def configure_environment() -> None:
    python_paths = [SGLANG_SOURCE, SGLANG_OVERLAY, KERNEL_SOURCE, PLUGIN_SOURCE]
    rendered = os.pathsep.join(str(path) for path in python_paths)
    sys.path[:0] = [str(path) for path in python_paths]
    os.environ["PYTHONPATH"] = rendered
    os.environ["SGLANG_PLUGINS"] = "pypto"
    os.environ["SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS"] = "1"
    os.environ["SGLANG_IO_WORKERS"] = "1"
    os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYPTO_ENV_PREFIX"] = str(ROOT / "envs/pypto-nvidia")
    os.environ["PYPTO_WORKSPACE_ROOT"] = str(ROOT)
    os.environ["PYPTO_SGLANG_SOURCE_ROOT"] = str(ROOT / "upstream/sglang")
    os.environ["PYPTO_KERNEL_DSO_PATH"] = str(PYPTO_DSO)
    os.environ["PYPTO_KERNEL_PACKAGE_PATH"] = str(PYPTO_PACKAGE)
    os.environ["PYPTO_PLUGINS_PYPTO_DSO"] = str(PYPTO_DSO)
    os.environ["PYPTO_ALLOW_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    install_sgl_kernel_guard()


def install_sgl_kernel_guard() -> None:
    """Prevent import-time loading or accidental use of the 1.1 GiB AOT DSO."""

    if "sgl_kernel" in sys.modules:
        raise RuntimeError("sgl_kernel was imported before the strict PyPTO guard")
    package_root = SGLANG_OVERLAY / "sgl_kernel"
    if not package_root.is_dir():
        raise RuntimeError(f"missing pinned sgl_kernel package: {package_root}")
    module = ModuleType("sgl_kernel")
    module.__file__ = str(package_root / "__init__.py")
    module.__package__ = "sgl_kernel"
    module.__path__ = [str(package_root)]
    module.__version__ = "0.4.6.post1"

    def missing(name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def forbidden(*_args, **_kwargs):
            raise RuntimeError(
                f"strict PyPTO run forbids sgl_kernel provider call: {name}"
            )

        forbidden.__name__ = name
        setattr(module, name, forbidden)
        return forbidden

    module.__getattr__ = missing
    sys.modules["sgl_kernel"] = module


def output_path() -> pathlib.Path:
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    path = ROOT / "runs" / run_id / "qwen35-0p8b-smoke-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-torch-compile", action="store_true")
    args = parser.parse_args()
    configure_environment()
    report: dict[str, object] = {
        "schema": 1,
        "kind": "qwen35-0p8b-pypto-smoke",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "model_path": str(ROOT / "models/Qwen3.5-0.8B"),
        "prompt": PROMPT,
        "requested_new_tokens": 1,
        "torch_compile": not args.disable_torch_compile,
        "status": "starting",
    }
    engine = None
    try:
        import sglang as sgl

        report["sglang_version"] = sgl.__version__
        report["sglang_source"] = str(pathlib.Path(sgl.__file__).resolve())
        prompt_ids = list(PROMPT_TOKEN_IDS)
        report["prompt_token_ids"] = prompt_ids
        engine = sgl.Engine(
            model_path=str(ROOT / "models/Qwen3.5-0.8B"),
            tokenizer_path=str(ROOT / "models/Qwen3.5-0.8B"),
            skip_tokenizer_init=True,
            enable_multimodal=False,
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
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
            skip_server_warmup=True,
            log_level="error",
            watchdog_timeout=600,
            soft_watchdog_timeout=300,
        )
        report["status"] = "engine-ready"
        response = engine.generate(
            input_ids=prompt_ids,
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 1,
            },
            return_logprob=True,
        )
        report["status"] = "generated"
        report["response"] = response
        output_path().write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        output_path().write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
