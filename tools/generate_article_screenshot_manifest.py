#!/usr/bin/env python3
"""Bind article/demo terminal screenshots to their local evidence reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {
        "build": {
            "path": "docs/assets/screenshots/build-ctest.png",
            "command": "envs/pypto-release/bin/python tools/build_release.py --stage ctest --jobs 24",
            "status": "historical-pass",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：CTest 13/13 通过（当前截图绑定 ctest 阶段；完整四阶段 build 仍按发布 manifest 复核）。",
            "caption_en": "Ubuntu/PowerShell purple terminal: CTest 13/13 passed (this capture binds the ctest stage; the complete four-stage build remains a release-manifest gate).",
            "evidence": "runs/pypto-cpu-bounded-20260830T144910Z-2409462-115ae6/release-build-ctest.json",
        },
        "operator-correctness": {
            "path": "docs/assets/screenshots/operator-correctness.png",
            "command": "envs/pypto-release/bin/python -B -m pytest packages/pypto-kernels/tests/test_operators.py -q",
            "status": "historical-pass",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：pypto-kernels 结构/回归测试 38 passed。",
            "caption_en": "Ubuntu/PowerShell purple terminal: pypto-kernels regression tests, 38 passed.",
            "evidence": "state/evidence/operator-regression-current.json",
        },
        "performance": {
            "path": "docs/assets/screenshots/performance-ablation.png",
            "command": "python3 -B tools/print_inductor_ablation.py",
            "status": "historical-operator-scope",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：Qwen3.5-9B SwiGLU 算子级消融；数值来自 immutable evidence JSON，非整模结论。",
            "caption_en": "Ubuntu/PowerShell purple terminal: Qwen3.5-9B SwiGLU operator ablation; values come from immutable evidence JSON and are not a whole-model claim.",
            "evidence": "state/evidence/qwen35-9b-inductor-ablation-current.json",
        },
        "model-inference": {
            "path": "docs/assets/screenshots/model-inference.png",
            "command": "envs/pypto-release/bin/python -B tools/run_model_correctness.py all --model-path models/Qwen3.5-9B --semantic-oracle runs/semantic-oracle-qwen35-9b-chat-nonthinking.json",
            "status": "pending",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：Qwen3.5-9B chat-template 推理与 PyPTO/Inductor launch log；当前环境无法控制 Windows GUI，截图待捕获。",
            "caption_en": "Ubuntu/PowerShell purple terminal: Qwen3.5-9B chat-template inference with PyPTO/Inductor launch logs; Windows GUI capture is pending.",
            "evidence": "state/evidence/qwen35-9b-model-gate-current.json",
        },
        "article-demo-typical": {
            "path": "docs/assets/screenshots/article-demo-typical.png",
            "command": "envs/pypto-release/bin/python -B tools/run_article_demo.py --demo examples/beginner/hello_world.py --platform a2a3sim --output runs/article-demo-typical.json",
            "status": "pending",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：原样 hello_world.py 的 compile/golden/runtime 结果；当前 Ascend runtime blocker，截图待捕获。",
            "caption_en": "Ubuntu/PowerShell purple terminal: unchanged hello_world.py compile/golden/runtime result; Ascend runtime blocker remains, capture pending.",
            "evidence": "state/evidence/article-demo-typical-screenshot-pending-20260829.json",
        },
    }
    payload: dict[str, object] = {
        "schema": 1,
        "kind": "article-demo-screenshot-manifest",
        "status": "provisional",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "terminal": {"host": "Windows Terminal", "distro": "Ubuntu", "theme": "PowerShell purple"},
        "screenshots": {},
    }
    output = args.output.resolve()
    for role, raw in records.items():
        evidence = (ROOT / raw["evidence"]).resolve(strict=True)
        if not evidence.is_file():
            raise SystemExit(f"missing evidence for {role}")
        item = dict(raw)
        item["evidence_sha256"] = sha256(evidence)
        item["evidence"] = evidence.relative_to(ROOT).as_posix()
        image = (ROOT / raw["path"]).resolve()
        if image.is_file() and role != "article-demo-typical":
            item["sha256"] = sha256(image)
        else:
            item["capture_status"] = "pending"
        payload["screenshots"][role] = item
    write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
