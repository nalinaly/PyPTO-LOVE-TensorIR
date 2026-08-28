#!/usr/bin/env python3
"""Atomically replace controlled documentation markers from release evidence.

The screenshot manifest is a schema-1 ``qwen35-release-screenshots`` object
bound by ``release_summary_sha256``.  Its ``screenshots`` object has exactly
``build``, ``operator-correctness``, ``model-inference`` and ``performance``.
Each record contains repository-relative ``path``, ``sha256``, non-placeholder
``caption_zh``/``caption_en`` and one or more ``evidence`` records with a
repository-relative JSON ``path`` and ``sha256``.  Build evidence must cover
wheels/native/ctest/install; the other roles bind their formal control summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import (  # noqa: E402
    ReleaseContractError,
    read_json,
    sha256_file,
)


EXPECTED_SCREENSHOTS = {
    "build",
    "operator-correctness",
    "performance",
    "model-inference",
}
SCREENSHOT_EVIDENCE_KINDS = {
    "build": {"pypto-sm120-release-build"},
    "operator-correctness": {"pypto-release-operator-regression-control"},
    "performance": {"qwen35-9b-performance-matrix-control"},
    "model-inference": {"qwen35-9b-all-control"},
}
PLACEHOLDERS = (
    "待正式",
    "待补",
    "tbd",
    "todo",
    "placeholder",
    "pending formal",
    "xx%",
)
DOCUMENT_FRAGMENTS = {
    "readme_zh": {"SUMMARY": "SUMMARY_ZH"},
    "readme_en": {"SUMMARY": "SUMMARY_EN"},
    "blog": {
        "SUMMARY": "SUMMARY_ZH",
        "OPERATOR_CORRECTNESS": "OPERATOR_CORRECTNESS_ZH",
        "MODEL_CORRECTNESS": "MODEL_CORRECTNESS_ZH",
        "COVERAGE": "COVERAGE_ZH",
        "PERFORMANCE": "PERFORMANCE_ZH",
        "BREAKDOWN": "BREAKDOWN_ZH",
        "CONCLUSION": "CONCLUSION_ZH",
    },
}
DOCUMENT_SCREENSHOTS = {
    "readme_zh": {
        "SUMMARY": (
            "build",
            "operator-correctness",
            "model-inference",
            "performance",
        )
    },
    "readme_en": {
        "SUMMARY": (
            "build",
            "operator-correctness",
            "model-inference",
            "performance",
        )
    },
    "blog": {
        "SUMMARY": ("build",),
        "OPERATOR_CORRECTNESS": ("operator-correctness",),
        "MODEL_CORRECTNESS": ("model-inference",),
        "PERFORMANCE": ("performance",),
    },
}


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReleaseContractError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ReleaseContractError(f"{label} is not a regular file: {resolved}")
    return resolved


def _require_document_paths(
    readme_zh: Path, readme_en: Path, blog: Path
) -> dict[str, Path]:
    documents = {
        "readme_zh": _regular_file(readme_zh, "Chinese README"),
        "readme_en": _regular_file(readme_en, "English README"),
        "blog": _regular_file(blog, "Chinese blog"),
    }
    if documents["readme_zh"] != (ROOT / "README.md").resolve(strict=True):
        raise ReleaseContractError("Chinese README must be the repository README.md")
    if documents["readme_en"] != (ROOT / "README_EN.md").resolve(strict=True):
        raise ReleaseContractError("English README must be README_EN.md")
    blog_root = (ROOT / "reports/local-blog").resolve(strict=True)
    if blog_root not in documents["blog"].parents or documents["blog"].suffix != ".md":
        raise ReleaseContractError("blog must be a Markdown file below reports/local-blog")
    return documents


def _verify_png(path: Path) -> dict[str, int]:
    payload = path.read_bytes()
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ReleaseContractError(f"screenshot is not a valid PNG: {path}")
    offset = 8
    chunks = []
    width = height = 0
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ReleaseContractError(f"screenshot PNG is truncated: {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise ReleaseContractError(f"screenshot PNG chunk is truncated: {path}")
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ReleaseContractError(f"screenshot PNG CRC differs: {path}")
        chunks.append(kind)
        if kind == b"IHDR":
            if length != 13 or len(chunks) != 1:
                raise ReleaseContractError(f"screenshot PNG IHDR is invalid: {path}")
            width, height = struct.unpack(">II", data[:8])
        offset = end
        if kind == b"IEND":
            break
    if offset != len(payload) or chunks[-1:] != [b"IEND"] or b"IDAT" not in chunks:
        raise ReleaseContractError(f"screenshot PNG structure is incomplete: {path}")
    if width < 800 or height < 450 or len(payload) < 4096:
        raise ReleaseContractError(
            f"screenshot is too small to be release evidence: {path} ({width}x{height})"
        )
    return {"width": width, "height": height, "bytes": len(payload)}


def _verify_screenshots(
    manifest_path: Path, release_summary_sha256: str
) -> dict[str, object]:
    manifest_path = _regular_file(manifest_path, "screenshot manifest")
    manifest = read_json(manifest_path)
    screenshots = manifest.get("screenshots")
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "qwen35-release-screenshots"
        or manifest.get("status") != "complete"
        or manifest.get("release_summary_sha256") != release_summary_sha256
        or not isinstance(screenshots, dict)
        or set(screenshots) != EXPECTED_SCREENSHOTS
    ):
        raise ReleaseContractError("screenshot manifest is incomplete or unbound")
    verified = {}
    for role, record in screenshots.items():
        if (
            not isinstance(record, dict)
            or type(record.get("path")) is not str
            or type(record.get("caption_zh")) is not str
            or type(record.get("caption_en")) is not str
            or type(record.get("evidence")) is not list
            or not record["evidence"]
        ):
            raise ReleaseContractError(f"screenshot record is incomplete: {role}")
        for caption in (record["caption_zh"], record["caption_en"]):
            if not caption.strip() or any(
                token in caption.lower() for token in PLACEHOLDERS
            ):
                raise ReleaseContractError(f"screenshot caption is a placeholder: {role}")
        path = _regular_file(ROOT / str(record["path"]), f"{role} screenshot")
        screenshot_root = (ROOT / "docs/assets/screenshots").resolve()
        if screenshot_root not in path.parents:
            raise ReleaseContractError(
                "release screenshots must be publishable files below "
                f"docs/assets/screenshots/: {path}"
            )
        digest = sha256_file(path)
        if record.get("sha256") != digest:
            raise ReleaseContractError(f"screenshot SHA-256 differs: {path}")
        evidence = []
        evidence_payloads = []
        for item in record["evidence"]:
            if not isinstance(item, dict) or type(item.get("path")) is not str:
                raise ReleaseContractError(f"screenshot evidence is invalid: {role}")
            evidence_path = _regular_file(ROOT / item["path"], f"{role} evidence")
            if (ROOT / "runs").resolve() not in evidence_path.parents:
                raise ReleaseContractError(
                    f"screenshot evidence must be below runs/: {evidence_path}"
                )
            if item.get("sha256") != sha256_file(evidence_path):
                raise ReleaseContractError(
                    f"screenshot evidence SHA-256 differs: {evidence_path}"
                )
            payload = read_json(evidence_path)
            if payload.get("status") != "complete":
                raise ReleaseContractError(
                    f"screenshot evidence is not complete: {evidence_path}"
                )
            evidence_payloads.append(payload)
            evidence.append(evidence_path.relative_to(ROOT).as_posix())
        observed_kinds = {payload.get("kind") for payload in evidence_payloads}
        if not SCREENSHOT_EVIDENCE_KINDS[role].issubset(observed_kinds):
            raise ReleaseContractError(
                f"screenshot evidence kind is incomplete for {role}: {observed_kinds}"
            )
        if role == "build" and {
            payload.get("stage") for payload in evidence_payloads
        } != {"wheels", "native", "ctest", "install"}:
            raise ReleaseContractError(
                "build screenshot evidence must bind all four release build stages"
            )
        verified[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "caption_zh": record["caption_zh"].strip(),
            "caption_en": record["caption_en"].strip(),
            "evidence": evidence,
            **_verify_png(path),
        }
    return verified


def _load_fragments(
    summary_path: Path, manifest_path: Path
) -> tuple[dict[str, str], str]:
    summary_path = _regular_file(summary_path, "release summary")
    runs_root = (ROOT / "runs").resolve(strict=True)
    if runs_root not in summary_path.parents:
        raise ReleaseContractError("release summary must be below runs/")
    summary = read_json(summary_path)
    if (
        summary.get("schema") != 1
        or summary.get("kind") != "qwen35-9b-release-results"
        or summary.get("status") != "complete"
        or not summary.get("release_identity")
        or not summary.get("operator_correctness")
        or not summary.get("operator_performance")
        or not summary.get("model_correctness")
        or not summary.get("performance")
        or not summary.get("profile_reconciliation")
    ):
        raise ReleaseContractError("release summary is incomplete")
    summary_sha256 = sha256_file(summary_path)
    manifest_path = _regular_file(manifest_path, "marker fragment manifest")
    if runs_root not in manifest_path.parents or manifest_path.parent != summary_path.parent:
        raise ReleaseContractError(
            "marker fragment manifest must share the release-summary directory"
        )
    manifest = read_json(manifest_path)
    records = manifest.get("fragments")
    summary_record = manifest.get("release_summary")
    expected_names = {
        value for mapping in DOCUMENT_FRAGMENTS.values() for value in mapping.values()
    }
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "qwen35-release-marker-fragments"
        or manifest.get("status") != "complete"
        or not isinstance(records, dict)
        or set(records) != expected_names
        or not isinstance(summary_record, dict)
        or summary_record.get("sha256") != summary_sha256
        or type(summary_record.get("path")) is not str
        or (manifest_path.parent / str(summary_record["path"])).resolve()
        != summary_path
    ):
        raise ReleaseContractError("marker fragments are incomplete or unbound")
    fragments = {}
    for name, record in records.items():
        if not isinstance(record, dict) or type(record.get("path")) is not str:
            raise ReleaseContractError(f"marker fragment record is invalid: {name}")
        path = _regular_file(manifest_path.parent / str(record["path"]), name)
        if manifest_path.parent not in path.parents:
            raise ReleaseContractError(f"marker fragment escaped its evidence directory: {path}")
        if record.get("sha256") != sha256_file(path):
            raise ReleaseContractError(f"marker fragment SHA-256 differs: {path}")
        text = path.read_text(encoding="utf-8").strip()
        lowered = text.lower()
        if not text or any(token in lowered for token in PLACEHOLDERS):
            raise ReleaseContractError(f"marker fragment is a placeholder: {name}")
        if "<!-- RELEASE_RESULTS:" in text:
            raise ReleaseContractError(f"marker fragment contains nested markers: {name}")
        fragments[name] = text
    return fragments, summary_sha256


def _replace_marker(text: str, marker: str, fragment: str) -> str:
    begin = f"<!-- RELEASE_RESULTS:{marker}_BEGIN -->"
    end = f"<!-- RELEASE_RESULTS:{marker}_END -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ReleaseContractError(f"document must contain exactly one {marker} marker pair")
    prefix, remainder = text.split(begin, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{begin}\n\n{fragment.rstrip()}\n\n{end}{suffix}"


def _screenshot_markdown(
    document: Path,
    language: str,
    roles: tuple[str, ...],
    screenshots: dict[str, object],
) -> str:
    lines = []
    for role in roles:
        record = screenshots[role]
        screenshot = ROOT / str(record["path"])
        relative = os.path.relpath(screenshot, document.parent).replace(os.sep, "/")
        caption = record[f"caption_{language}"]
        lines.append(f"![{caption}]({relative})")
    return "\n\n".join(lines)


def _stage_files(updates: dict[Path, str]) -> dict[Path, Path]:
    staged = {}
    try:
        for destination, text in updates.items():
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".sync", dir=destination.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            staged[destination] = Path(temporary)
        return staged
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise


def _atomic_replace_many(updates: dict[Path, str]) -> None:
    """Stage all bytes first and restore originals if a replace fails."""

    originals = {path: path.read_bytes() for path in updates}
    staged = _stage_files(updates)
    replaced = []
    try:
        for destination, temporary in staged.items():
            os.replace(temporary, destination)
            replaced.append(destination)
        for directory in {path.parent for path in updates}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        rollback = {
            path: originals[path].decode("utf-8") for path in replaced
        }
        rollback_staged = _stage_files(rollback)
        for destination, temporary in rollback_staged.items():
            os.replace(temporary, destination)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _require_documents_synced(updates: dict[Path, str]) -> None:
    """Fail unless every controlled document already has its rendered bytes."""

    stale = []
    for destination, expected in updates.items():
        if destination.read_bytes() != expected.encode("utf-8"):
            try:
                stale.append(destination.relative_to(ROOT).as_posix())
            except ValueError:
                stale.append(str(destination))
    if stale:
        raise ReleaseContractError(
            "documents are not synchronized with release fragments and "
            f"screenshots: {', '.join(sorted(stale))}"
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--release-summary", type=Path, required=True)
    value.add_argument("--marker-fragments", type=Path, required=True)
    value.add_argument("--readme-zh", type=Path, required=True)
    value.add_argument("--readme-en", type=Path, required=True)
    value.add_argument("--blog", type=Path, required=True)
    value.add_argument("--screenshots-manifest", type=Path, required=True)
    value.add_argument("--check", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    documents = _require_document_paths(args.readme_zh, args.readme_en, args.blog)
    fragments, summary_sha256 = _load_fragments(
        args.release_summary, args.marker_fragments
    )
    screenshots = _verify_screenshots(args.screenshots_manifest, summary_sha256)
    updates = {}
    for document_name, path in documents.items():
        text = path.read_text(encoding="utf-8")
        for marker, fragment_name in DOCUMENT_FRAGMENTS[document_name].items():
            fragment = fragments[fragment_name]
            roles = DOCUMENT_SCREENSHOTS[document_name].get(marker, ())
            if roles:
                language = "en" if document_name == "readme_en" else "zh"
                fragment += "\n\n" + _screenshot_markdown(
                    path, language, roles, screenshots
                )
            text = _replace_marker(text, marker, fragment)
        updates[path] = text
    if args.check:
        _require_documents_synced(updates)
    else:
        _atomic_replace_many(updates)
    print(
        json.dumps(
            {
                "status": "validated" if args.check else "updated",
                "release_summary_sha256": summary_sha256,
                "documents": [path.relative_to(ROOT).as_posix() for path in updates],
                "screenshots": screenshots,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
