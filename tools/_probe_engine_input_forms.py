"""Temporary SGLang Engine input-form and first-token diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.lanes import (  # noqa: E402
    prepare_worker_environment,
    server_kwargs,
)
from benchmarks.release import lanes  # noqa: E402
from benchmarks.release.sglang_compat import (  # noqa: E402
    install_sglang_release_compatibility,
)
from benchmarks.release.workload import verify_chat_workload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--zero-offload", action="store_true")
    parser.add_argument("--offload-gb", type=int)
    args = parser.parse_args()
    model_path = args.model_path.resolve(strict=True)
    workload, _ = verify_chat_workload(model_path)
    prepare_worker_environment("pypto")
    import sglang as sgl

    compatibility = install_sglang_release_compatibility()
    original_server_kwargs = lanes.server_kwargs

    def configured_server_kwargs(*values, **options):
        result = dict(original_server_kwargs(*values, **options))
        if args.disable_torch_compile:
            result["enable_torch_compile"] = False
        if args.zero_offload:
            result["cpu_offload_gb"] = 0
        if args.offload_gb is not None:
            if args.offload_gb < 0:
                parser.error("--offload-gb must be non-negative")
            result["cpu_offload_gb"] = args.offload_gb
        return result

    lanes.server_kwargs = configured_server_kwargs
    requested = configured_server_kwargs("pypto", model_path)
    engine = sgl.Engine(**requested)
    records = []
    try:
        sampling = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        }
        for label, input_ids in (
            ("flat", list(workload["prompt_token_ids"])),
            ("nested", [list(workload["prompt_token_ids"])]),
        ):
            result = engine.generate(
                input_ids=input_ids,
                sampling_params=dict(sampling),
                return_logprob=True,
                top_logprobs_num=8,
                rid=f"engine-input-{label}",
            )
            if isinstance(result, list):
                result_items = result
                result = result[0] if len(result) == 1 else {}
            else:
                result_items = None
            if not isinstance(result, dict):
                result = {"raw_type": type(result).__name__, "raw": repr(result)}
            meta = result.get("meta_info", {})
            if not isinstance(meta, dict):
                meta = {"raw_type": type(meta).__name__, "raw": repr(meta)}
            records.append(
                {
                    "label": label,
                    "result_container_type": type(result_items).__name__
                    if result_items is not None
                    else "dict",
                    "result_keys": sorted(result),
                    "output_ids": result.get("output_ids"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                    "output_token_logprobs": result.get("output_token_logprobs"),
                    "meta_info": meta,
                }
            )
        payload = {
            "requested": requested,
            "compatibility": compatibility,
            "records": records,
        }
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
