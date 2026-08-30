"""Provenance and byte-for-byte contracts for the imported article demos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "pypto-lib"
MANIFEST = DEMO / "SOURCE_MANIFEST.json"


def test_article_demo_manifest_is_article_time_locked() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["kind"] == "article-demo-provenance"
    assert payload["article"]["url"] == (
        "https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ"
    )
    assert payload["upstream"]["commit"] == (
        "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
    )
    assert len(payload["files"]) >= 90
    assert len(payload["entrypoints"]) == 66
    assert sum(
        item.get("execution_policy") == "runnable" for item in payload["entrypoints"]
    ) == 57


def test_imported_demo_files_match_manifest() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in payload["files"]:
        path = DEMO / record["path"]
        assert path.is_file(), record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_size == record["bytes"], record["path"]
        assert digest == record["sha256"], record["path"]


def test_required_article_demo_entrypoints_exist() -> None:
    expected = {
        "examples/beginner/hello_world.py",
        "examples/intermediate/rms_norm.py",
        "examples/intermediate/softmax.py",
        "examples/intermediate/gemm.py",
        "models/qwen3_14b/decode_fwd.py",
        "models/qwen3_14b/contract.py",
        "golden/__init__.py",
    }
    observed = {record["path"] for record in json.loads(MANIFEST.read_text())["files"]}
    assert expected <= observed


def test_entrypoint_inventory_covers_examples_and_named_models() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {record["path"] for record in payload["entrypoints"]}
    assert sum(path.startswith("examples/") for path in paths) == 11
    assert any(path.startswith("models/qwen3_14b/") for path in paths)
    assert any(path.startswith("models/deepseek_v4_flash_mtp/") for path in paths)
