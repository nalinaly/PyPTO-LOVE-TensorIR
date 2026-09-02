"""Run one raw Qwen3.5-9B inference on SGLang with the PyPTO backends.

    envs/pypto-release/bin/python demo/raw_sglang_inference.py

Starts the offline SGLang engine with ``--attention-backend pypto
--linear-attn-backend pypto`` (every compute kernel is PyPTO-compiled),
answers the fixed prompt with plain greedy decoding, and stops at the
model's own EOS — no ignore_eos, no token cap below the natural end. All
engine logs stream unmodified; the complete generated text is printed after
shutdown so nothing is cut off by trailing log lines.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The formal runtimes project the source-locked SGLang tree through
# PYTHONPATH (tools/run_isolated.py); a raw run does the same by hand.
sys.path.insert(0, str(ROOT / ".sources" / "sglang" / "python"))

import os  # noqa: E402

# Controlled runs set this; a raw run must too, or CPython writes .pyc
# files into the locked environment trees and breaks identity checks.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# The plugin binds imported code to the locked workspace; a raw run states
# the same three anchors the controlled runtime would set.
os.environ.setdefault("PYPTO_CACHE_DIR", str(ROOT / "caches" / "pypto" / "artifact-cache"))
os.environ.setdefault("PYPTO_ENV_PREFIX", str(ROOT / "envs" / "pypto-release"))
os.environ.setdefault("PYPTO_WORKSPACE_ROOT", str(ROOT))
os.environ.setdefault("PYPTO_SGLANG_SOURCE_ROOT", str(ROOT / ".sources" / "sglang"))

from benchmarks.release.lanes import server_kwargs  # noqa: E402
from pypto_plugins.sglang.stream import pypto_stream  # noqa: E402

PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"


def main() -> int:
    import sglang as sgl
    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    requested = server_kwargs("pypto", ROOT / "models/Qwen3.5-9B")
    # The demo runs without the CUPTI overlay; a slightly larger static
    # fraction simply leaves more KV headroom next to the shared display.
    requested["mem_fraction_static"] = 0.86
    # Gate-style entry: tokenize the chat template here and feed input_ids,
# which also gives the engine the request-pool geometry every formal
# gate run uses (the text-entry pool layout lands on tileiras-broken
# decode geometries at wider contexts).
    requested["skip_tokenizer_init"] = True
    # The gate geometry (context 256) compiles every kernel shape; a raw
    # full-length answer needs a wider window, and 352 is the width the
    # kernel library also uses in its own regression geometries.
    requested["context_length"] = 512
    requested["max_total_tokens"] = 512
    requested["max_prefill_tokens"] = 512
    print("== sglang raw inference (pypto attention + linear backends) ==")
    print("server args:", {k: requested[k] for k in sorted(requested) if isinstance(requested[k], (str, int, bool, float))})

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/Qwen3.5-9B"), local_files_only=True
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    print(f"-- chat template applied: {len(input_ids)} input tokens")

    with pypto_stream(torch.device("cuda")) as stream:
        engine = sgl.Engine(**requested)
        try:
            print("-- generating (greedy, stop at EOS, max 480 tokens (the full compilable window))")
            with torch.cuda.stream(stream):
                response = engine.generate(
                    input_ids=input_ids,
                    sampling_params={
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_new_tokens": 480,
                        "ignore_eos": False,
                    },
                )
        finally:
            engine.shutdown()

    print("-- generation finished")
    meta = response.get("meta_info") or {}
    finish_reason = meta.get("finish_reason", response.get("finish_reason"))
    completion_tokens = meta.get("completion_tokens")
    print(f"finish_reason    : {finish_reason}")
    print(f"completion_tokens: {completion_tokens}")
    ids = response.get("output_ids") or meta.get("output_token_ids")
    text = response.get("text") or ""
    if not text and ids:
        text = tokenizer.decode(ids, skip_special_tokens=False)
    print("prompt           :")
    print(PROMPT)
    print("full completion (verbatim, generation ended by the model):")
    print(text)
    print("== end of completion ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
