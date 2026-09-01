#!/usr/bin/env python3
"""Bind terminal screenshots to the live runs and capture sidecars that produced them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
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


def newest(pattern: str) -> Path:
    matches = sorted(Path(p) for p in glob.glob(str(ROOT / pattern)))
    if not matches:
        raise SystemExit(f"no evidence matches {pattern}")
    return matches[-1]


# Each role binds one live screenshot to the newest on-disk evidence of the run
# it shows and to the PowerShell capture sidecar written at capture time.
ROLES = {
    "build-release": {
        "path": "docs/assets/screenshots/build-release.png",
        "evidence_glob": "runs/pypto-cpu-bounded-*/release-build-install.json",
        "capture_evidence": "state/evidence/build-release-capture-current.json",
        "status": "current-live-run",
        "caption_zh": "PowerShell 中 wsl -d Ubuntu（紫色终端）真实执行四阶段构建：wheels/native/CTest/install 全部 return_code=0，status=complete。",
        "caption_en": "Live four-stage build (wheels/native/CTest/install, all return_code=0, status=complete) executed in Ubuntu entered via `wsl -d Ubuntu` from PowerShell.",
    },
    "operator-correctness": {
        "path": "docs/assets/screenshots/operator-correctness.png",
        "evidence_glob": "runs/*/operator-numerical-regression.json",
        "capture_evidence": "state/evidence/operator-correctness-capture-current.json",
        "status": "current-live-run",
        "caption_zh": "真实 GPU 算子正确性回归：8/8 套件、101 用例全部通过（all_correct=true）。",
        "caption_en": "Live GPU operator correctness regression: 8/8 suites, 101 cases, all_correct=true.",
    },
    "operator-performance": {
        "path": "docs/assets/screenshots/operator-performance.png",
        "evidence_glob": "runs/release-operator-ab-*/aggregation.json",
        "capture_evidence": "state/evidence/operator-performance-capture-current.json",
        "status": "current-live-run",
        "caption_zh": "真实算子级性能 A/B：7 个功能对齐算子，PyPTO vs SGLang stock，4+4 次独立冷启动。",
        "caption_en": "Live operator-level performance A/B: 7 aligned operators, PyPTO vs SGLang stock, 4+4 fresh starts.",
    },
    "model-inference": {
        "path": "docs/assets/screenshots/model-inference.png",
        "evidence_glob": "runs/*/qwen35-9b-correctness.json",
        "capture_evidence": "state/evidence/model-inference-capture-current.json",
        "status": "current-live-run",
        "caption_zh": "真实端到端推理：固定 prompt 的 64-token 贪心解码，逐 token 门禁与 100% PyPTO coverage。",
        "caption_en": "Live end-to-end inference: fixed-prompt 64-token greedy decoding with per-token gates and 100% PyPTO coverage.",
    },
    "article-demo-typical": {
        "path": "docs/assets/screenshots/article-demo-typical.png",
        "evidence_glob": "state/evidence/article-demos-nvidia/011-hello_world-screenshot.json",
        "capture_evidence": "state/evidence/article-demo-typical-capture-current.json",
        "status": "current-live-run",
        "caption_zh": "未改写 hello_world.py 的严格 NVIDIA 兼容运行、PyPTO artifact 与 golden 精度通过。",
        "caption_en": "Strict NVIDIA compatibility run of unchanged hello_world.py with a PyPTO artifact and passing golden comparison.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "schema": 2,
        "kind": "release-screenshot-manifest",
        "status": "provisional",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "terminal": {
            "host": "Windows Terminal",
            "distro": "Ubuntu",
            "theme": "Ubuntu purple profile, nested PowerShell prompt -> wsl -d Ubuntu",
            "capture_script": "tools/windows/capture_powershell.ps1",
        },
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
    for role, raw in ROLES.items():
        evidence = newest(raw["evidence_glob"]).resolve()
        image = (ROOT / raw["path"]).resolve()
        capture_path = (ROOT / raw["capture_evidence"]).resolve(strict=True)
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (
            capture.get("status") != "pass"
            or capture.get("output_sha256") != sha256(image)
            or capture.get("exit_code") != 0
            or type(capture.get("window_width")) is not int
            or type(capture.get("window_height")) is not int
            or int(capture.get("visible_samples", 0)) < 16
        ):
            raise SystemExit(f"invalid capture evidence for {role}")
        if not image.is_file():
            raise SystemExit(f"missing screenshot for {role}")
        item = {
            "path": raw["path"],
            "command": capture.get("command"),
            "status": raw["status"],
            "caption_zh": raw["caption_zh"],
            "caption_en": raw["caption_en"],
            "evidence": evidence.relative_to(ROOT).as_posix(),
            "evidence_sha256": sha256(evidence),
            "capture_evidence": capture_path.relative_to(ROOT).as_posix(),
            "capture_evidence_sha256": sha256(capture_path),
            "capture_exit_code": capture.get("exit_code"),
            "sha256": sha256(image),
        }
        payload["screenshots"][role] = item
    payload["status"] = "complete"
    write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
