from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools/finalize_gpt_image2_assets.py"
    spec = importlib.util.spec_from_file_location("finalize_gpt_image2_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "state/evidence/source.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"status":"complete"}\n', encoding="utf-8")
    outputs = tmp_path / "docs/assets/generated"
    outputs.mkdir(parents=True)
    assets = []
    for index in range(5):
        path = outputs / f"figure-{index}.png"
        Image.new("RGB", (1536, 1024), (240, 240 - index, 235)).save(path)
        assets.append(
            {
                "id": f"figure-{index}",
                "output": path.relative_to(tmp_path).as_posix(),
                "use_case": "infographic-diagram",
                "prompt": f"Use case: infographic-diagram. Figure {index}.",
                "evidence_scope": "test scope",
            }
        )
    prompt = tmp_path / "state/evidence/prompts.json"
    prompt.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "gpt-image-2-evidence-prompts",
                "status": "pending-api-key",
                "model": "gpt-image-2",
                "generation": {
                    "mode": "imagegen-skill-fallback-cli",
                    "cli": "${CODEX_HOME}/skills/.system/imagegen/scripts/image_gen.py",
                    "model": "gpt-image-2",
                    "quality": "high",
                    "size": "1536x1024",
                    "output_format": "png",
                    "augmentation": False,
                },
                "source_evidence": [
                    {
                        "path": evidence.relative_to(tmp_path).as_posix(),
                        "sha256": file_sha256(evidence),
                    }
                ],
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )
    inspection = tmp_path / "state/evidence/inspection.json"
    inspection.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "gpt-image-2-visual-inspection",
                "status": "pass",
                "model": "gpt-image-2",
                "prompt_manifest_sha256": file_sha256(prompt),
                "reviewed_at": "2026-09-01T00:00:00+00:00",
                "assets": {
                    asset["id"]: {
                        "status": "pass",
                        "image_sha256": file_sha256(tmp_path / asset["output"]),
                        **{name: True for name in tool.REQUIRED_VISUAL_CHECKS},
                        "notes": "Synthetic test image.",
                    }
                    for asset in assets
                },
            }
        ),
        encoding="utf-8",
    )
    return prompt, inspection


def test_finalizer_binds_prompts_sources_images_and_manual_review(
    tmp_path: Path,
) -> None:
    prompt, inspection = fixture(tmp_path)
    result = tool.finalize(
        tmp_path,
        prompt,
        inspection,
        finalized_at="2026-09-01T00:01:00+00:00",
    )
    assert result["status"] == "complete"
    assert result["execution"]["model"] == "gpt-image-2"
    assert result["execution"]["cli"]["sha256"]
    assert len(result["assets"]) == 5
    assert all(asset["width"] == 1536 for asset in result["assets"])
    assert all(asset["height"] == 1024 for asset in result["assets"])
    assert all(asset["visual_inspection"]["text_accuracy"] for asset in result["assets"])


def test_finalizer_rejects_unreviewed_or_changed_image(tmp_path: Path) -> None:
    prompt, inspection = fixture(tmp_path)
    review = json.loads(inspection.read_text(encoding="utf-8"))
    review["assets"]["figure-0"]["image_sha256"] = "0" * 64
    inspection.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(tool.ReleaseContractError, match="visual inspection failed"):
        tool.finalize(tmp_path, prompt, inspection)
