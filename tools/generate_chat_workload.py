#!/usr/bin/env python3
"""Generate or verify the pinned Qwen3.5 chat-template workload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import (  # noqa: E402
    PROMPT,
    RAW_PROMPT_TOKEN_IDS,
    atomic_json,
)


TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": True}
MODEL_IDS = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
}
TOKENIZER_FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_from_encoded(encoded: object) -> list[int]:
    ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise RuntimeError("tokenizer returned an invalid input ID sequence")
    return ids


def generate_manifest(root: Path = ROOT) -> dict[str, object]:
    from transformers import AutoTokenizer

    models: dict[str, object] = {}
    for name, model_id in MODEL_IDS.items():
        model_path = (root / "models" / name).resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=True,
            return_tensors=None,
            **TEMPLATE_KWARGS,
        )
        ids = _ids_from_encoded(encoded)
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
        models[name] = {
            "model_id": model_id,
            "input_token_ids": ids,
            "rendered_input": rendered,
            "tokenizer_files": {
                filename: _sha256(model_path / filename)
                for filename in TOKENIZER_FILES
            },
        }
    return {
        "schema": 1,
        "human_prompt": PROMPT,
        "template_kwargs": dict(TEMPLATE_KWARGS),
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/release/chat_workload.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = generate_manifest()
    if args.check:
        if not args.output.is_file():
            print(f"missing manifest: {args.output}", file=sys.stderr)
            return 1
        observed = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != generated:
            print("chat workload manifest differs from tokenizer rendering", file=sys.stderr)
            return 1
    else:
        atomic_json(args.output.resolve(), generated)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.resolve()),
                "models": {
                    name: len(record["input_token_ids"])
                    for name, record in generated["models"].items()
                },
                "raw_prompt_tokens": len(RAW_PROMPT_TOKEN_IDS),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
