#!/usr/bin/env python3
"""Run an independent Transformers Qwen3.5 semantic oracle.

The oracle is intentionally separate from SGLang correctness and performance
workers.  It verifies the pinned chat-template input, records the first-step
logits and a short greedy prefix, and optionally compares those values with a
SGLang reference report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import (  # noqa: E402
    OUTPUT_TOKENS,
    ReleaseContractError,
    atomic_json,
    canonical_json_sha256,
    model_revision,
    require_path_below_runs,
    resolve_qwen35_model_spec,
    sha256_file,
    verify_chat_workload,
)


def _topk(torch, logits, tokenizer, k: int) -> dict[str, object]:
    values, indices = torch.topk(logits, k=k)
    ids = [int(value) for value in indices.cpu()]
    return {
        "token_ids": ids,
        "token_text": [
            tokenizer.decode([value], skip_special_tokens=False) for value in ids
        ],
        "logits": [float(value) for value in values.cpu()],
    }


def _semantic_smoke(tokenizer, output_ids: list[int]) -> dict[str, object]:
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    repeated_suffix = len(output_ids) >= 8 and len(set(output_ids[-8:])) == 1
    replacement_character = "\ufffd" in text
    return {
        "passed": not repeated_suffix and not replacement_character,
        "replacement_character": replacement_character,
        "repeated_last_eight_tokens": repeated_suffix,
        "output_text": text,
    }


def _compare_reference(torch, logits, output_ids, reference_path: Path) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("status") != "complete":
        raise ReleaseContractError("semantic oracle reference report is incomplete")
    logits_record = reference.get("logits")
    if not isinstance(logits_record, dict):
        raise ReleaseContractError("semantic oracle reference has no logits record")
    reference_tensor_path = Path(str(logits_record["path"])).resolve(strict=True)
    if sha256_file(reference_tensor_path) != logits_record.get("file_sha256"):
        raise ReleaseContractError("semantic oracle reference logits hash changed")
    reference_logits = torch.load(
        reference_tensor_path, map_location="cpu", weights_only=True
    ).float()
    if list(reference_logits.shape) != [OUTPUT_TOKENS, int(logits.shape[-1])]:
        raise ReleaseContractError("semantic oracle reference logits shape differs")
    candidate = logits.float().cpu().contiguous()
    first = reference_logits[0]
    difference = (candidate - first).abs()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.view(1, -1), first.view(1, -1)
    )[0]
    candidate_top = torch.topk(candidate, k=5).indices.tolist()
    reference_top = torch.topk(first, k=5).indices.tolist()
    return {
        "reference_report": str(reference_path),
        "reference_report_sha256": sha256_file(reference_path),
        "first_step_max_abs": float(difference.max()),
        "first_step_mean_abs": float(difference.mean()),
        "first_step_cosine": float(cosine),
        "first_step_top5_overlap": len(set(candidate_top) & set(reference_top)),
        "short_prefix_exact": output_ids == list(reference.get("output_token_ids", []))[: len(output_ids)],
        "reference_first_top5": reference_top,
        "oracle_first_top5": candidate_top,
    }


def run(args: argparse.Namespace) -> int:
    model_path = args.model_path.resolve(strict=True)
    spec = resolve_qwen35_model_spec(ROOT, model_path)
    workload, workload_resolution = verify_chat_workload(model_path, spec)
    report_path = require_path_below_runs(ROOT, args.output)
    logits_path = report_path.with_name(report_path.stem + "-first-logits.pt")
    report: dict[str, object] = {
        "schema": 1,
        "kind": "qwen35-transformers-semantic-oracle",
        "status": "starting",
        "model": {
            **spec.record(),
            "path": str(model_path),
            "revision": model_revision(model_path),
        },
        "workload": workload,
        "workload_resolution": workload_resolution,
        "device": args.device,
        "short_prefix_tokens": args.max_new_tokens,
        "top_k": args.top_k,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "cuda" and not torch.cuda.is_available():
            raise ReleaseContractError("semantic oracle requested CUDA without a GPU")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        input_ids = torch.tensor(
            [workload["prompt_token_ids"]], dtype=torch.int64, device=args.device
        )
        dtype = torch.bfloat16
        model_kwargs = {
            "local_files_only": True,
            "dtype": dtype,
            "low_cpu_mem_usage": False,
            "trust_remote_code": True,
        }
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        model = model.to(device=args.device).eval()
        output_ids: list[int] = []
        past_key_values = None
        first_logits = None
        current = input_ids
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                result = model(
                    input_ids=current,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = result.logits[0, -1].float()
                if first_logits is None:
                    first_logits = logits.detach().cpu().contiguous()
                token = int(torch.argmax(logits).item())
                output_ids.append(token)
                past_key_values = result.past_key_values
                current = torch.tensor([[token]], dtype=torch.int64, device=args.device)
        if first_logits is None:
            raise ReleaseContractError("semantic oracle produced no logits")
        torch.save(first_logits, logits_path)
        report.update(
            {
                "status": "complete",
                "input_token_ids": list(workload["prompt_token_ids"]),
                "first_logits": {
                    "path": str(logits_path),
                    "file_sha256": sha256_file(logits_path),
                    "raw_sha256": hashlib.sha256(first_logits.numpy().tobytes()).hexdigest(),
                    "shape": list(first_logits.shape),
                    "dtype": str(first_logits.dtype),
                },
                "first_step_top": _topk(torch, first_logits, tokenizer, args.top_k),
                "output_token_ids": output_ids,
                "output_sequence_sha256": canonical_json_sha256(output_ids),
                "semantic_smoke": _semantic_smoke(tokenizer, output_ids),
                "torch": {
                    "version": str(torch.__version__),
                    "cuda": str(torch.version.cuda),
                    "device": args.device,
                },
            }
        )
        if args.sglang_reference is not None:
            report["sglang_comparison"] = _compare_reference(
                torch, first_logits, output_ids, args.sglang_reference.resolve(strict=True)
            )
        if report["semantic_smoke"]["passed"] is not True:
            report["status"] = "failed"
            report["error"] = "semantic smoke detected replacement or pathological repetition"
            return_code = 1
        else:
            return_code = 0
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sglang-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.top_k <= 0:
        parser.error("max-new-tokens and top-k must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
