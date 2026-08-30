#!/usr/bin/env python3
"""Generate the deterministic provenance manifest for the imported article demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARTICLE_URL = "https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ"
UPSTREAM_URL = "https://github.com/hw-native-sys/pypto-lib"
UPSTREAM_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
ARTICLE_PUBLISHED_AT = "2026-08-28T17:30:00+08:00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entrypoints(root: Path) -> list[dict[str, object]]:
    """Record every source file with a normal Python CLI entry point."""

    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if not (
            relative.startswith("examples/")
            or relative.startswith("models/qwen3_14b/")
            or relative.startswith("models/deepseek_v4_flash_mtp/")
        ):
            continue
        text = path.read_text(encoding="utf-8")
        if 'if __name__ == "__main__"' not in text:
            continue
        entry = {
            "path": relative,
            "help_args": ["--help"],
            "article_command": ["python", relative, "--help"],
            "execution_policy": (
                "excluded-draft"
                if path.name.endswith("_draft.py")
                else "ascend-cce-only"
                if path.name == "test_paged_attention_cce.py"
                else "runnable"
            ),
        }
        if relative == "examples/advanced/allreduce.py":
            entry["default_device"] = "0,1"
        result.append(
            entry
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    imported = []
    imported_roots = (
        "examples",
        "golden",
        "contract",
        "models/qwen3_14b",
        "models/deepseek_v4_flash_mtp",
    )
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or not any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in imported_roots
        ):
            continue
        imported.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    metadata = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"SOURCE_MANIFEST.json"}:
            metadata.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "schema": 1,
        "kind": "article-demo-provenance",
        "article": {
            "url": ARTICLE_URL,
            "published_at": ARTICLE_PUBLISHED_AT,
            "title": "让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！",
        },
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "copy_policy": "byte-for-byte source import; compatibility configuration stays outside imported files",
        },
        "license": {
            "name": "CANN Open Software License Agreement Version 2.0",
            "path": "LICENSE",
            "notice": "The imported files retain their upstream copyright headers and license notices.",
        },
        "files": imported,
        "metadata_files": metadata,
        "entrypoints": entrypoints(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
