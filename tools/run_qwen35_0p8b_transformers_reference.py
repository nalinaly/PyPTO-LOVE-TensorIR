#!/usr/bin/env python3
"""Run an independent Transformers BF16 reference for the exact prompt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/Qwen3.5-0.8B"
TRANSFORMERS_OVERLAY = ROOT / "envs/sglang-runtime-py314"
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
    report_path = run_dir / "qwen35-0p8b-transformers-reference.json"
    logits_path = run_dir / "qwen35-0p8b-transformers-reference-logits.pt"
    report: dict[str, object] = {
        "kind": "qwen35-0p8b-transformers-reference",
        "model_path": str(MODEL_PATH),
        "prompt": PROMPT,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "run_id": run_id,
        "schema": 1,
        "status": "starting",
    }
    try:
        os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"
        sys.path.insert(0, str(TRANSFORMERS_OVERLAY))
        import torch
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
        )

        full_config = json.loads((MODEL_PATH / "config.json").read_text())
        config = Qwen3_5TextConfig.from_dict(full_config["text_config"])
        config._attn_implementation = "eager"
        config.use_cache = False
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config)

        checkpoint_path = next(MODEL_PATH.glob("*.safetensors"))
        checkpoint = load_file(checkpoint_path, device="cpu")
        state_dict = {}
        prefix = "model.language_model."
        for name, tensor in checkpoint.items():
            if name.startswith(prefix):
                state_dict["model." + name[len(prefix) :]] = tensor
            elif name == "lm_head.weight":
                state_dict[name] = tensor
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        missing = [name for name in incompatible.missing_keys if not name.endswith("rotary_emb.inv_freq")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "reference state dict mismatch: "
                f"missing={missing[:20]} unexpected={incompatible.unexpected_keys[:20]}"
            )
        inv_freq, attention_scaling = (
            model.model.rotary_emb.compute_default_rope_parameters(
                config, device=torch.device("cpu")
            )
        )
        model.model.rotary_emb.inv_freq = inv_freq
        model.model.rotary_emb.original_inv_freq = inv_freq.clone()
        model.model.rotary_emb.attention_scaling = attention_scaling
        del checkpoint, state_dict
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        input_ids = torch.tensor(
            [PROMPT_TOKEN_IDS], device="cuda", dtype=torch.int64
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        logits = output.logits[:, -1, :].float().cpu().contiguous()
        torch.save(logits, logits_path)
        top_values, top_indices = torch.topk(logits[0], 5)
        report.update(
            {
                "status": "complete",
                "logits_path": str(logits_path),
                "logits_shape": list(logits.shape),
                "logits_finite": bool(torch.isfinite(logits).all()),
                "logits_min": float(logits.min()),
                "logits_max": float(logits.max()),
                "logits_sha256": hashlib.sha256(
                    logits.numpy().tobytes()
                ).hexdigest(),
                "top5_token_ids": top_indices.tolist(),
                "top5_logits": top_values.tolist(),
                "next_token_ids": top_indices[:1].tolist(),
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


if __name__ == "__main__":
    raise SystemExit(main())
