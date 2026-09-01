#!/usr/bin/env python3
"""Validate GPT-Image-2 figures and freeze their publication provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import (  # noqa: E402
    ReleaseContractError,
    atomic_json,
)


EXPECTED_MODEL = "gpt-image-2"
EXPECTED_ASSET_COUNT = 5
EXPECTED_SIZE = (1536, 1024)
REQUIRED_VISUAL_CHECKS = (
    "subject_and_scope",
    "layout_and_legibility",
    "text_accuracy",
    "no_fabricated_metrics",
    "no_logo_or_watermark",
    "adjacent_table_required",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{label} root must be an object: {path}")
    return value


def repository_file(root: Path, raw: object, label: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw:
        raise ReleaseContractError(f"{label} path is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseContractError(f"{label} path must be repository-relative: {raw}")
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ReleaseContractError(f"{label} escaped the repository: {raw}") from error
    if path.is_symlink() or not path.is_file():
        raise ReleaseContractError(f"{label} is not a regular file: {raw}")
    return path, normalized


def png_info(path: Path) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ReleaseContractError(f"figure is not PNG: {path}")
            width, height = image.size
    except ReleaseContractError:
        raise
    except Exception as error:
        raise ReleaseContractError(f"invalid PNG figure: {path}: {error}") from error
    if (width, height) != EXPECTED_SIZE:
        raise ReleaseContractError(
            f"figure has wrong dimensions: {path}: {width}x{height}"
        )
    return {"width": width, "height": height, "bytes": path.stat().st_size}


def _cli_record() -> dict[str, str]:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).resolve()
    path = codex_home / "skills/.system/imagegen/scripts/image_gen.py"
    if not path.is_file():
        raise ReleaseContractError(f"imagegen skill CLI is missing: {path}")
    return {
        "path": "${CODEX_HOME}/skills/.system/imagegen/scripts/image_gen.py",
        "sha256": sha256(path),
    }


def finalize(
    root: Path,
    prompt_manifest_path: Path,
    inspection_path: Path,
    *,
    finalized_at: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    prompt_manifest_path = prompt_manifest_path.resolve(strict=True)
    inspection_path = inspection_path.resolve(strict=True)
    prompts = read_json(prompt_manifest_path, "prompt manifest")
    inspection = read_json(inspection_path, "visual inspection")
    generation = prompts.get("generation")
    assets = prompts.get("assets")
    source_evidence = prompts.get("source_evidence")
    if (
        prompts.get("schema") != 1
        or prompts.get("kind") != "gpt-image-2-evidence-prompts"
        or prompts.get("model") != EXPECTED_MODEL
        or not isinstance(generation, dict)
        or generation.get("mode") != "imagegen-skill-fallback-cli"
        or generation.get("model") != EXPECTED_MODEL
        or generation.get("quality") != "high"
        or generation.get("size") != "1536x1024"
        or generation.get("output_format") != "png"
        or generation.get("augmentation") is not False
        or not isinstance(assets, list)
        or len(assets) != EXPECTED_ASSET_COUNT
        or not isinstance(source_evidence, list)
        or not source_evidence
    ):
        raise ReleaseContractError("GPT-Image-2 prompt manifest contract drifted")

    prompt_sha = sha256(prompt_manifest_path)
    inspected_assets = inspection.get("assets")
    if (
        inspection.get("schema") != 1
        or inspection.get("kind") != "gpt-image-2-visual-inspection"
        or inspection.get("status") != "pass"
        or inspection.get("model") != EXPECTED_MODEL
        or inspection.get("prompt_manifest_sha256") != prompt_sha
        or not isinstance(inspection.get("reviewed_at"), str)
        or not isinstance(inspected_assets, dict)
    ):
        raise ReleaseContractError("GPT-Image-2 visual inspection is incomplete")

    sources = []
    for record in source_evidence:
        if not isinstance(record, dict):
            raise ReleaseContractError("source evidence record is malformed")
        path, relative = repository_file(root, record.get("path"), "source evidence")
        digest = sha256(path)
        if record.get("sha256") != digest:
            raise ReleaseContractError(f"source evidence SHA drifted: {relative}")
        sources.append({"path": relative, "sha256": digest})

    finalized_assets = []
    observed_ids: set[str] = set()
    observed_outputs: set[str] = set()
    for record in assets:
        if not isinstance(record, dict):
            raise ReleaseContractError("image asset record is malformed")
        asset_id = record.get("id")
        prompt = record.get("prompt")
        output = record.get("output")
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or asset_id in observed_ids
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(output, str)
            or output in observed_outputs
        ):
            raise ReleaseContractError(f"image asset identity is invalid: {record!r}")
        path, relative = repository_file(root, output, f"image asset {asset_id}")
        if Path(relative).parent.as_posix() != "docs/assets/generated":
            raise ReleaseContractError(
                f"image asset is outside docs/assets/generated: {relative}"
            )
        digest = sha256(path)
        review = inspected_assets.get(asset_id)
        if (
            not isinstance(review, dict)
            or review.get("status") != "pass"
            or review.get("image_sha256") != digest
            or any(review.get(name) is not True for name in REQUIRED_VISUAL_CHECKS)
            or not isinstance(review.get("notes"), str)
        ):
            raise ReleaseContractError(f"visual inspection failed: {asset_id}")
        observed_ids.add(asset_id)
        observed_outputs.add(output)
        finalized_assets.append(
            {
                "id": asset_id,
                "output": relative,
                "use_case": record.get("use_case"),
                "evidence_scope": record.get("evidence_scope"),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "image_sha256": digest,
                **png_info(path),
                "visual_inspection": {
                    name: review[name] for name in REQUIRED_VISUAL_CHECKS
                },
                "inspection_notes": review["notes"],
            }
        )

    return {
        "schema": 1,
        "kind": "gpt-image-2-publication-assets",
        "status": "complete",
        "finalized_at": finalized_at or datetime.now(timezone.utc).isoformat(),
        "execution": {
            **generation,
            "cli": _cli_record(),
        },
        "prompt_manifest": {
            "path": prompt_manifest_path.relative_to(root).as_posix(),
            "sha256": prompt_sha,
        },
        "source_evidence": sources,
        "visual_inspection": {
            "path": inspection_path.relative_to(root).as_posix(),
            "sha256": sha256(inspection_path),
            "reviewed_at": inspection["reviewed_at"],
        },
        "assets": finalized_assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "state/evidence/gpt-image2-ablation-prompts-20260829.json",
    )
    parser.add_argument(
        "--inspection",
        type=Path,
        default=ROOT / "state/evidence/gpt-image2-visual-inspection-current.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "state/evidence/gpt-image2-assets-current.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = read_json(args.output, "asset manifest") if args.check else None
    result = finalize(
        ROOT,
        args.prompt_manifest,
        args.inspection,
        finalized_at=(
            str(observed.get("finalized_at")) if observed is not None else None
        ),
    )
    if args.check:
        if observed != result:
            raise ReleaseContractError("GPT-Image-2 asset manifest does not reproduce")
        print(f"GPT-Image-2 asset manifest validated: {args.output}")
    else:
        atomic_json(args.output, result)
        print(f"GPT-Image-2 asset manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
