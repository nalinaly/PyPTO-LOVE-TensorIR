"""Tests for the external NVIDIA article-demo policy and matrix contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import classify_article_demos


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "demo" / "pypto-lib" / "SOURCE_MANIFEST.json"


def test_policy_is_manifest_bound_and_has_explicit_modes() -> None:
    manifest = classify_article_demos.load_manifest()
    policy = classify_article_demos.build_policy(manifest)
    assert policy["entrypoint_count"] == 66
    assert policy["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert policy["corpus_sha256"] == classify_article_demos.corpus_sha256(manifest)
    assert policy["acceptance"]["imported_source_must_remain_byte_identical"] is True
    assert policy["counts"] == {
        "computational-cuda-reference": 9,
        "computational-unmapped": 31,
        "hardware-api-skipped": 17,
        "source-excluded": 8,
        "strict-pypto-nvidia": 1,
    }


def test_hardware_skip_contains_source_evidence() -> None:
    manifest = classify_article_demos.load_manifest()
    policy = classify_article_demos.build_policy(manifest)
    by_path = {entry["path"]: entry for entry in policy["entries"]}
    allreduce = by_path["examples/advanced/allreduce.py"]
    assert allreduce["compatibility_mode"] == "hardware-api-skipped"
    markers = {item["marker"] for item in allreduce["hardware_api_evidence"]}
    assert "distributed-language-api" in markers
    assert allreduce["hardware_api_evidence"][0]["line"] > 0


def test_strict_example_is_the_only_artifact_claim() -> None:
    manifest = classify_article_demos.load_manifest()
    policy = classify_article_demos.build_policy(manifest)
    strict = [
        entry
        for entry in policy["entries"]
        if entry["compatibility_mode"] == "strict-pypto-nvidia"
    ]
    assert [entry["path"] for entry in strict] == [
        "examples/beginner/hello_world.py"
    ]
    assert strict[0]["adapter"] == "hello_world_strict"


def test_persisted_policy_matches_generator() -> None:
    persisted = json.loads(
        (ROOT / "state/evidence/article-demo-compatibility-policy-current.json").read_text(
            encoding="utf-8"
        )
    )
    generated = classify_article_demos.build_policy(classify_article_demos.load_manifest())
    assert persisted == generated


def test_current_performance_pair_is_resource_accepted() -> None:
    pair = json.loads(
        (ROOT / "state/evidence/qwen35-9b-performance-pair-current.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = pair["acceptance"]
    assert pair["status"] == "complete"
    assert acceptance["accepted"] is True
    assert acceptance["starts_total"] == 8
    assert acceptance["starts_below_floor"] == 0
    assert acceptance["control_comparability"]["mismatches"] == []
    assert pair["comparison"]["pypto_percent_of_matched"] == 15.694959760741742
