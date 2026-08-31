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
            "command": "python3 -B tools/print_release_gate.py build --compact",
            "command_source": "tools/print_release_gate.py",
            "status": "current-evidence-replay",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：校验并回放已接受的 wheel build、install/pip-check、CTest 13/13、三阶段自然退出与同一 artifact set SHA；明确不是 live rerun。",
            "caption_en": "Ubuntu/PowerShell purple terminal: validates and replays accepted wheel build, install/pip-check, CTest 13/13, natural three-stage exit, and one artifact-set SHA identity; explicitly not a live rerun.",
            "evidence": "runs/pypto-cpu-bounded-20260830T144910Z-2409462-115ae6/release-build-ctest.json",
            "capture_evidence": "state/evidence/build-ctest-capture-current.json",
            "supporting_evidence": [
                "runs/pypto-cpu-bounded-20260830T091243Z-2320308-b57c23/release-build-wheels.json",
                "runs/pypto-cpu-bounded-20260830T091243Z-2320308-b57c23/process.json",
                "runs/pypto-cpu-bounded-20260830T091329Z-2320615-6a9134/release-build-install.json",
                "runs/pypto-cpu-bounded-20260830T091329Z-2320615-6a9134/process.json",
                "runs/pypto-cpu-bounded-20260830T144910Z-2409462-115ae6/process.json",
                "builds/qwen35-sm120-v1/native/Testing/Temporary/LastTest.log"
            ],
        },
        "operator-correctness": {
            "path": "docs/assets/screenshots/operator-correctness.png",
            "command": "python3 -B tools/print_release_gate.py operator --compact",
            "command_source": "tools/print_release_gate.py",
            "status": "current-evidence-replay",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：校验并回放已接受的 8/8 operator regression suite、case inventory 与 DSO identity；明确不是 live rerun。",
            "caption_en": "Ubuntu/PowerShell purple terminal: validates and replays the accepted 8/8 operator regression suites, case inventory, and DSO identity; explicitly not a live rerun.",
            "evidence": "state/evidence/operator-regression-current.json",
            "capture_evidence": "state/evidence/operator-correctness-capture-current.json",
        },
        "performance": {
            "path": "docs/assets/screenshots/performance-ablation.png",
            "command": "python3 -B tools/print_inductor_ablation.py --compact",
            "command_source": "tools/print_inductor_ablation.py",
            "status": "current-operator-scope",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：Qwen3.5-9B SwiGLU 算子级消融；数值来自 immutable evidence JSON，非整模结论。",
            "caption_en": "Ubuntu/PowerShell purple terminal: Qwen3.5-9B SwiGLU operator ablation; values come from immutable evidence JSON and are not a whole-model claim.",
            "evidence": "state/evidence/qwen35-9b-inductor-ablation-current.json",
            "capture_evidence": "state/evidence/performance-ablation-capture-current.json",
        },
        "model-inference": {
            "path": "docs/assets/screenshots/model-inference.png",
            "command": "python3 -B tools/print_qwen35_model_gate.py --compact",
            "command_source": "tools/print_qwen35_model_gate.py",
            "status": "current-evidence-replay",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：校验并回放已接受的 Qwen3.5-9B 三次 fresh-start 推理、输出文本与 100% PyPTO coverage；明确不是本轮 live rerun。",
            "caption_en": "Ubuntu/PowerShell purple terminal: validates and replays the accepted three-start Qwen3.5-9B inference, decoded output, and 100% PyPTO coverage; explicitly not a live rerun.",
            "evidence": "state/evidence/qwen35-9b-model-gate-current.json",
            "capture_evidence": "state/evidence/model-inference-capture-current.json",
        },
        "article-demo-typical": {
            "path": "docs/assets/screenshots/article-demo-typical.png",
            "command": "envs/pypto-release/bin/python -B tools/run_article_demo_nvidia.py --demo examples/beginner/hello_world.py --device 0 --run-id article-demo-nvidia-hello-screenshot --output state/evidence/article-demos-nvidia/011-hello_world-screenshot.json",
            "status": "pending",
            "caption_zh": "Ubuntu/PowerShell 紫色终端：未改写 hello_world.py 的严格 NVIDIA 兼容运行、PyPTO artifact 与 golden 结果；GUI capture 仍待补。",
            "caption_en": "Ubuntu/PowerShell purple terminal: strict NVIDIA compatibility run of unchanged hello_world.py with PyPTO artifact and golden result; GUI capture remains pending.",
            "evidence": "state/evidence/article-demo-hello-nvidia-screenshot-pending-current.json",
        },
    }
    payload: dict[str, object] = {
        "schema": 1,
        "kind": "article-demo-screenshot-manifest",
        "status": "provisional",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "terminal": {"host": "Windows Terminal", "distro": "Ubuntu", "theme": "PowerShell purple"},
        "capture_capability": {},
        "screenshots": {},
    }
    capability_path = (
        ROOT / "state/evidence/windows-terminal-capture-capability-current.json"
    ).resolve(strict=True)
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    if capability.get("status") != "pass":
        raise SystemExit("Windows Terminal capture capability is not verified")
    payload["capture_capability"] = {
        "evidence": capability_path.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256(capability_path),
        "status": "pass",
    }
    output = args.output.resolve()
    for role, raw in records.items():
        evidence = (ROOT / raw["evidence"]).resolve(strict=True)
        if not evidence.is_file():
            raise SystemExit(f"missing evidence for {role}")
        item = dict(raw)
        item["evidence_sha256"] = sha256(evidence)
        item["evidence"] = evidence.relative_to(ROOT).as_posix()
        command_source_relative = raw.get("command_source")
        if command_source_relative is not None:
            command_source = (ROOT / str(command_source_relative)).resolve(strict=True)
            item["command_source"] = command_source.relative_to(ROOT).as_posix()
            item["command_source_sha256"] = sha256(command_source)
        supporting = raw.get("supporting_evidence")
        if supporting is not None:
            item["supporting_evidence"] = []
            for relative in supporting:
                supporting_path = (ROOT / str(relative)).resolve(strict=True)
                item["supporting_evidence"].append(
                    {
                        "path": supporting_path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(supporting_path),
                    }
                )
        image = (ROOT / raw["path"]).resolve()
        capture_relative = raw.get("capture_evidence")
        if capture_relative is not None:
            capture_path = (ROOT / str(capture_relative)).resolve(strict=True)
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            if (
                capture.get("status") != "pass"
                or capture.get("command") != raw["command"]
                or capture.get("output_sha256") != sha256(image)
                or capture.get("exit_code") != 0
                or type(capture.get("window_width")) is not int
                or type(capture.get("window_height")) is not int
                or int(capture.get("visible_samples", 0)) < 16
            ):
                raise SystemExit(f"invalid capture evidence for {role}")
            item["capture_evidence"] = capture_path.relative_to(ROOT).as_posix()
            item["capture_evidence_sha256"] = sha256(capture_path)
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
