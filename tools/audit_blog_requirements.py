#!/usr/bin/env python3
"""Fail-closed audit for release documents and their evidence sidecars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import workload_record  # noqa: E402


ARTICLE_URL = "https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ"
BILIBILI_URL = (
    "https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source="
    "f2f41aa7b5e3cc8e0a23942779ccea11"
)
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
PYPTO_HEAD = "c27629e993a52b47d41fb898c749279dce44221b"
ARTICLE_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
OPERATOR_BREAKDOWN_SHA256 = (
    "24d416b8c11b0806090b9b2e97055fa713a77322df9845cb166fc333de5f88ba"
)
EVIDENCE_REFERENCE = re.compile(
    r"state/evidence/[A-Za-z0-9_.\-/]+\.json"
)


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for attribute, value in attrs:
            if attribute in {"href", "poster", "src"} and value is not None:
                self.references.append((tag, attribute, value))


def _parse_references(value: str) -> list[tuple[str, str, str]]:
    parser = _ReferenceParser()
    parser.feed(value)
    parser.close()
    return parser.references


def _local_reference(document: Path, reference: str) -> Path | None:
    split = urlsplit(reference)
    if split.scheme or split.netloc or reference.startswith(("#", "//")):
        return None
    path = unquote(split.path)
    if not path:
        return None
    return (document.parent / path).resolve()


def check_markdown_references(
    document: Path, text: str, errors: list[str]
) -> None:
    try:
        import markdown
    except ImportError:
        errors.append("Python-Markdown is unavailable for document link audit")
        return
    rendered = markdown.markdown(
        text,
        extensions=("fenced_code", "tables", "toc", "sane_lists"),
        output_format="html5",
    )
    for tag, attribute, reference in _parse_references(rendered):
        target = _local_reference(document, reference)
        if target is None:
            continue
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            errors.append(
                f"document reference escapes repository: {document.name}: {reference}"
            )
            continue
        if not target.exists():
            errors.append(f"missing document reference: {document.name}: {relative}")
        elif tag == "img" and attribute == "src" and not target.is_file():
            errors.append(f"document image is not a file: {document.name}: {relative}")


def check_offline_html_resources(
    document: Path, text: str, errors: list[str]
) -> None:
    resource_attributes = {
        ("audio", "src"),
        ("iframe", "src"),
        ("img", "src"),
        ("link", "href"),
        ("script", "src"),
        ("source", "src"),
        ("video", "poster"),
        ("video", "src"),
    }
    for tag, attribute, reference in _parse_references(text):
        if (tag, attribute) in resource_attributes:
            if tag == "img" and reference.startswith("data:image/"):
                continue
            errors.append(
                f"offline HTML retains a non-embedded resource: {tag} {reference}"
            )
            continue
        if tag != "a" or attribute != "href":
            continue
        target = _local_reference(document, reference)
        if target is None:
            continue
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            errors.append(f"offline HTML link escapes repository: {reference}")
            continue
        if not target.exists():
            errors.append(f"offline HTML has a missing local link: {relative}")


def check_document_resources(
    texts: dict[str, str], blog_path: Path, html: str, errors: list[str]
) -> None:
    documents = {
        "README.md": ROOT / "README.md",
        "README_EN.md": ROOT / "README_EN.md",
        "blog": blog_path,
    }
    for name, document in documents.items():
        check_markdown_references(document, texts[name], errors)
    for relative in ("demo/README.md", "demo/README_EN.md"):
        document = ROOT / relative
        check_markdown_references(document, read(document, errors), errors)
    check_offline_html_resources(blog_path.with_suffix(".html"), html, errors)

    completed = subprocess.run(
        ["git", "ls-files", "--", "state/evidence"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        errors.append("cannot enumerate tracked evidence for document closure")
        return
    tracked = set(completed.stdout.splitlines())
    for name, text in texts.items():
        for reference in sorted(set(EVIDENCE_REFERENCE.findall(text))):
            if reference not in tracked:
                errors.append(
                    f"{name} references evidence unavailable in a fresh clone: {reference}"
                )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8", errors="strict")


def load(path: Path, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing evidence: {path.relative_to(ROOT)}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"JSON root is not an object: {path.relative_to(ROOT)}")
        return None
    return value


def check_demo(errors: list[str]) -> None:
    path = ROOT / "demo/pypto-lib/SOURCE_MANIFEST.json"
    value = load(path, errors)
    if value is None:
        return
    if value.get("upstream", {}).get("commit") != ARTICLE_COMMIT:
        errors.append("article demo commit is not article-time pinned")
    if len(value.get("files", [])) != 151:
        errors.append("article demo file count is not 151")
    if len(value.get("entrypoints", [])) != 66:
        errors.append("article demo entrypoint count is not 66")
    for record in value.get("files", []):
        try:
            file_path = ROOT / "demo/pypto-lib" / str(record["path"])
            if (
                not file_path.is_file()
                or file_path.stat().st_size != record["bytes"]
                or sha256(file_path) != record["sha256"]
            ):
                errors.append(f"article demo hash mismatch: {record.get('path')}")
        except (KeyError, TypeError, OSError):
            errors.append(f"invalid article demo record: {record!r}")


def check_demo_docs(errors: list[str]) -> None:
    for relative in ("demo/README.md", "demo/README_EN.md"):
        text = read(ROOT / relative, errors)
        if "article-demo-typical.png" not in text:
            errors.append(f"{relative} misses the typical demo screenshot")
        if "PENDING_SCREENSHOT" in text:
            errors.append(f"{relative} retains a stale pending screenshot marker")
        if (
            "41" not in text
            or "40" not in text
            or ("unmapped" not in text.lower() and "未映射" not in text)
        ):
            errors.append(f"{relative} misses the complete computational denominator")


def check_demo_compatibility(errors: list[str]) -> None:
    """Validate the external NVIDIA policy and its current matrix evidence."""
    policy_path = ROOT / "state/evidence/article-demo-compatibility-policy-current.json"
    policy = load(policy_path, errors)
    if policy is None:
        return
    manifest_path = ROOT / "demo/pypto-lib/SOURCE_MANIFEST.json"
    manifest_sha = sha256(manifest_path)
    corpus_digest = hashlib.sha256()
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in sorted(
        manifest_value.get("files", []), key=lambda item: str(item.get("path", ""))
    ):
        relative = str(record["path"])
        file_path = ROOT / "demo/pypto-lib" / relative
        corpus_digest.update(relative.encode("utf-8"))
        corpus_digest.update(b"\0")
        corpus_digest.update(file_path.read_bytes())
        corpus_digest.update(b"\0")
    corpus_sha = corpus_digest.hexdigest()
    if (
        policy.get("schema") != 2
        or policy.get("manifest_sha256") != manifest_sha
        or policy.get("corpus_sha256") != corpus_sha
        or policy.get("entrypoint_count") != 66
    ):
        errors.append("article demo compatibility policy is not bound to the current manifest")
        return
    counts = policy.get("counts", {})
    expected_counts = {
        "computational-cuda-reference": 40,
        "hardware-api-skipped": 17,
        "source-excluded": 8,
        "strict-pypto-nvidia": 1,
    }
    if counts != expected_counts:
        errors.append(f"article demo compatibility counts drifted: {counts!r}")
    for entry in policy.get("entries", []):
        if not isinstance(entry, dict):
            errors.append("article demo compatibility entry is not an object")
            continue
        if entry.get("compatibility_mode") == "hardware-api-skipped":
            if entry.get("original_execution_policy") == "ascend-cce-only":
                continue
            if not entry.get("hardware_api_evidence"):
                errors.append(f"hardware demo skip lacks source evidence: {entry.get('path')}")
    matrix_path = ROOT / "state/evidence/article-demo-matrix-nvidia-current.json"
    matrix = load(matrix_path, errors)
    if matrix is None:
        return
    if (
        matrix.get("schema") != 1
        or matrix.get("backend") != "nvidia"
        or matrix.get("status") != "complete"
        or matrix.get("entrypoint_count") != 66
        or matrix.get("manifest_sha256_before") != manifest_sha
        or matrix.get("manifest_sha256_after") != manifest_sha
        or matrix.get("corpus_sha256_before") != corpus_sha
        or matrix.get("corpus_sha256_after") != corpus_sha
        or matrix.get("compatibility_policy", {}).get("sha256") != sha256(policy_path)
        or matrix.get("strict_nvidia_pass_count") != 1
        or matrix.get("computational_reference_pass_count") != 40
        or matrix.get("hardware_api_skipped_count") != 17
        or matrix.get("computational_unmapped_count") != 0
        or matrix.get("compatibility_status") != "complete"
    ):
        errors.append("current NVIDIA article-demo matrix is incomplete or stale")
        return
    runner_path = ROOT / "tools/run_article_demo_nvidia.py"
    runner_sha = sha256(runner_path)
    computational_records = 0
    for record in matrix.get("results", []):
        if not isinstance(record, dict):
            errors.append("article-demo matrix result is not an object")
            continue
        mode = record.get("compatibility_mode")
        if mode not in {"strict-pypto-nvidia", "computational-cuda-reference"}:
            continue
        computational_records += 1
        child_path = ROOT / str(record.get("child_report", ""))
        child = load(child_path, errors)
        if child is None:
            continue
        strict = mode == "strict-pypto-nvidia"
        outputs = child.get("outputs")
        if (
            record.get("status") != "pass"
            or record.get("golden_pass") is not True
            or not child_path.is_file()
            or record.get("child_report_sha256") != sha256(child_path)
            or child.get("status") != "pass"
            or child.get("golden_pass") is not True
            or child.get("strict_compiler_evidence") is not strict
            or child.get("compatibility", {}).get("mode") != mode
            or child.get("source") != record.get("source")
            or child.get("adapter_source", {}).get("path")
            != "tools/run_article_demo_nvidia.py"
            or child.get("adapter_source", {}).get("sha256") != runner_sha
            or not isinstance(outputs, list)
            or not outputs
            or any(
                not isinstance(output, dict)
                or output.get("ok") is not True
                or not isinstance(output.get("rtol"), (int, float))
                or not isinstance(output.get("atol"), (int, float))
                or not isinstance(output.get("error_count"), int)
                or not isinstance(output.get("allowed_error_count"), int)
                for output in outputs
            )
        ):
            errors.append(f"article-demo child evidence drifted: {record.get('path')}")
        if strict:
            artifact = child.get("artifact")
            if (
                not isinstance(artifact, dict)
                or artifact.get("fallback_used") is not False
                or not artifact.get("artifact_sha256")
                or not artifact.get("cubin_sha256")
            ):
                errors.append("strict article-demo artifact evidence is incomplete")
        elif child.get("artifact") is not None:
            errors.append(
                f"CUDA reference was promoted to artifact evidence: {record.get('path')}"
            )
    if computational_records != 41:
        errors.append(
            f"article-demo computational child count drifted: {computational_records}"
        )


def check_screenshots(errors: list[str]) -> None:
    path = ROOT / "state/evidence/article-demo-screenshot-manifest-current.json"
    value = load(path, errors)
    if value is None:
        return
    if value.get("status") != "complete":
        errors.append("current screenshot manifest is not complete")
    expected = {"build", "operator-correctness", "performance", "model-inference", "article-demo-typical"}
    screenshots = value.get("screenshots")
    if not isinstance(screenshots, dict) or set(screenshots) != expected:
        errors.append("current screenshot manifest roles are incomplete")
        return
    capability = value.get("capture_capability")
    if not isinstance(capability, dict) or capability.get("status") != "pass":
        errors.append("Windows Terminal capture capability is not bound")
    else:
        capability_path = ROOT / str(capability.get("evidence", ""))
        if (
            not capability_path.is_file()
            or capability.get("evidence_sha256") != sha256(capability_path)
        ):
            errors.append("Windows Terminal capture capability hash mismatch")
        capability_value = load(capability_path, errors)
        if capability_value is not None:
            script_path = ROOT / str(capability_value.get("capture_script", ""))
            capture = capability_value.get("capture")
            if (
                not script_path.is_file()
                or capability_value.get("capture_script_sha256") != sha256(script_path)
                or not isinstance(capture, dict)
                or capture.get("method") != "PrintWindow"
                or int(capture.get("visible_samples", 0)) < 16
                or capture.get("exit_code") != 0
            ):
                errors.append("Windows Terminal capture capability metadata drifted")
    for role, record in screenshots.items():
        if not isinstance(record, dict):
            errors.append(f"invalid screenshot record: {role}")
            continue
        evidence = ROOT / str(record.get("evidence", ""))
        if not evidence.is_file():
            errors.append(f"missing screenshot evidence: {role}")
        image = ROOT / str(record.get("path", ""))
        if record.get("status") == "pending":
            if record.get("evidence_sha256") != sha256(evidence):
                errors.append(f"pending screenshot evidence hash mismatch: {role}")
            if role == "article-demo-typical":
                pending = load(evidence, errors)
                if pending is not None and (
                    pending.get("report_status") != "pass"
                    or pending.get("strict_compiler_evidence") is not True
                    or pending.get("golden_pass") is not True
                ):
                    errors.append("typical article demo pending slot lacks strict NVIDIA result evidence")
                if pending is not None:
                    report_path = ROOT / str(pending.get("report", ""))
                    if (
                        not report_path.is_file()
                        or pending.get("report_sha256") != sha256(report_path)
                    ):
                        errors.append("typical article demo pending slot report hash is stale")
            if record.get("capture_status") != "pending" and image.is_file() and role == "model-inference":
                errors.append(f"pending model screenshot is not marked pending: {role}")
            continue
        if not image.is_file() or image.stat().st_size < 4096:
            errors.append(f"screenshot missing or too small: {role}")
        elif record.get("sha256") != sha256(image):
            errors.append(f"screenshot hash mismatch: {role}")
        if record.get("evidence_sha256") != sha256(evidence):
            errors.append(f"screenshot evidence hash mismatch: {role}")
        if role in {
            "build",
            "operator-correctness",
            "performance",
            "model-inference",
            "article-demo-typical",
        }:
            command_source = ROOT / str(record.get("command_source", ""))
            if (
                not command_source.is_file()
                or record.get("command_source_sha256") != sha256(command_source)
            ):
                errors.append(f"current {role} command source hash mismatch")
            capture_path = ROOT / str(record.get("capture_evidence", ""))
            capture = load(capture_path, errors)
            if capture is None:
                continue
            expected_status = {
                "performance": "current-operator-scope",
                "article-demo-typical": "current-live-run",
            }.get(role, "current-evidence-replay")
            if (
                record.get("status") != expected_status
                or record.get("capture_evidence_sha256") != sha256(capture_path)
                or capture.get("command") != record.get("command")
                or capture.get("output_sha256") != record.get("sha256")
                or capture.get("status") != "pass"
                or capture.get("capture_method") != "PrintWindow"
                or capture.get("exit_code") != 0
                or int(capture.get("visible_samples", 0)) < 16
                or int(capture.get("window_width", 0)) < 320
                or int(capture.get("window_height", 0)) < 200
            ):
                errors.append(f"current {role} screenshot capture metadata drifted")
            for supporting in record.get("supporting_evidence", []):
                if not isinstance(supporting, dict):
                    errors.append(f"invalid supporting screenshot evidence: {role}")
                    continue
                supporting_path = ROOT / str(supporting.get("path", ""))
                if (
                    not supporting_path.is_file()
                    or supporting.get("sha256") != sha256(supporting_path)
                ):
                    errors.append(f"supporting screenshot evidence drifted: {role}")
            if role == "model-inference":
                if (
                    "not a live rerun"
                    not in (ROOT / "tools/print_qwen35_model_gate.py").read_text(
                        encoding="utf-8"
                    )
                    or record.get("evidence")
                    != "state/evidence/qwen35-9b-model-gate-current.json"
                ):
                    errors.append("model screenshot does not preserve replay boundary")
            if role == "article-demo-typical":
                demo = load(evidence, errors)
                if demo is not None and (
                    demo.get("status") != "pass"
                    or demo.get("strict_compiler_evidence") is not True
                    or demo.get("golden_pass") is not True
                    or demo.get("compatibility", {}).get("mode")
                    != "strict-pypto-nvidia"
                    or demo.get("artifact", {}).get("fallback_used") is not False
                    or demo.get("upstream_commit") != ARTICLE_COMMIT
                    or capture.get("role") != "article-demo-typical"
                ):
                    errors.append("typical article-demo screenshot lacks strict live-run evidence")


def check_operator(errors: list[str]) -> None:
    path = ROOT / "state/evidence/operator-regression-current.json"
    value = load(path, errors)
    if value is None:
        return
    if value.get("status") != "complete" or value.get("all_correct") is not True:
        errors.append("current operator evidence is not complete/all-correct")
    if value.get("source", {}).get("pypto_commit") != PYPTO_HEAD:
        errors.append("current operator evidence has a different PyPTO revision")
    report = ROOT / str(value.get("regression_report", ""))
    if not report.is_file() or value.get("regression_report_sha256") != sha256(report):
        errors.append("current operator report hash is stale")
    if value.get("all_suites_passed") is not True or value.get("total_suite_count") != 8:
        errors.append("current operator suite summary is incomplete")


def check_models(errors: list[str]) -> None:
    for model in ("qwen35-0.8b", "qwen35-9b"):
        value = load(ROOT / f"state/evidence/{model}-model-gate-current.json", errors)
        if value is None:
            continue
        if (
            value.get("status") != "complete"
            or value.get("accepted_candidate_start_count") != 3
        ):
            errors.append(f"{model} model gate is not complete")
        for candidate in value.get("candidates", []):
            coverage = candidate.get("coverage", {})
            if (
                coverage.get("total_calls") != coverage.get("covered_calls")
                or coverage.get("violation_count") != 0
                or coverage.get("strict_policy_passed") is not True
                or candidate.get("pypto_revision") != PYPTO_HEAD
            ):
                errors.append(f"{model} candidate coverage/identity is invalid")


def check_performance(errors: list[str]) -> None:
    release_path = ROOT / "state/evidence/qwen35-9b-release-results-current.json"
    release = load(release_path, errors)
    if release is not None:
        performance = release.get("performance", {})
        profile = release.get("profile_reconciliation", {})
        correctness = release.get("model_correctness", {})
        operator_correctness = release.get("operator_correctness", {})
        operator_performance = release.get("operator_performance", {})
        lanes = performance.get("lanes", {}) if isinstance(performance, dict) else {}
        comparisons = (
            performance.get("comparisons", {})
            if isinstance(performance, dict)
            else {}
        )
        profile_comparisons = (
            profile.get("comparisons", {}) if isinstance(profile, dict) else {}
        )
        inputs = release.get("inputs")
        if (
            release.get("status") != "complete"
            or release.get("kind") != "qwen35-9b-release-results"
            or set(lanes) != {"pypto", "sglang-matched", "sglang-optimized"}
            or any(lanes[lane].get("fresh_starts") != 4 for lane in lanes)
            or abs(
                float(
                    comparisons.get("sglang-matched", {}).get(
                        "pypto_percent_of_baseline", -1
                    )
                )
                - 15.62082854134726
            )
            > 1e-12
            or abs(
                float(
                    comparisons.get("sglang-optimized", {}).get(
                        "pypto_percent_of_baseline", -1
                    )
                )
                - 18.71425380089334
            )
            > 1e-12
            or lanes.get("pypto", {}).get("resources", {}).get(
                "gpu_free_floor_bytes"
            )
            != 4 * 1024**3
            or lanes.get("sglang-matched", {}).get("resources", {}).get(
                "gpu_free_floor_bytes"
            )
            != 4 * 1024**3
            or lanes.get("sglang-optimized", {}).get("resources", {}).get(
                "gpu_free_floor_bytes"
            )
            != 0
            or lanes.get("sglang-optimized", {}).get("resources", {}).get(
                "gpu_free_floor_mode"
            )
            != "disabled-completion-only"
            or profile.get("status") != "complete"
            or profile.get("profile_scope")
            != "hybrid-compiled-pypto-optimized-descriptive-matched"
            or profile.get("profile_inputs")
            != {
                lane: {"fresh_starts": 3, "profile_requests": 15}
                for lane in ("pypto", "sglang-matched", "sglang-optimized")
            }
            or profile_comparisons.get("sglang-matched", {}).get(
                "execution_scope"
            )
            != "descriptive-stock-noncompiled"
            or profile_comparisons.get("sglang-optimized", {}).get(
                "execution_scope"
            )
            != "strict-compiled"
            or correctness.get("fresh_starts") != 3
            or correctness.get("accepted_requests") != 30
            or correctness.get("teacher_forced_traces") != 3
            or correctness.get("coverage_calls", {}).get("p50") != 33448
            or operator_correctness.get("suite_count") != 8
            or operator_correctness.get("case_count") != 101
            or not isinstance(operator_performance, dict)
            or not isinstance(operator_performance.get("comparisons"), dict)
            or len(operator_performance["comparisons"]) != 7
            or not isinstance(inputs, list)
            or len(inputs) != 223
        ):
            errors.append("current unified release evidence drifted")
        elif any(
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or not (path := ROOT / record["path"]).is_file()
            or sha256(path) != record["sha256"]
            for record in inputs
        ):
            errors.append("current unified release input hash drifted")
    ablation = load(ROOT / "state/evidence/qwen35-9b-inductor-ablation-current.json", errors)
    if ablation is not None:
        if ablation.get("schema") != 2 or ablation.get("performance_only") is not True:
            errors.append("current ablation is not performance-only schema 2")
        if ablation.get("interpretation", {}).get("whole_model_speedup") is not None:
            errors.append("operator ablation claims whole-model speedup")
        for phase in ("prefill", "decode"):
            derived = ablation.get("phases", {}).get(phase, {}).get("derived", {})
            if abs(float(derived.get("compiled_launch_reduction_vs_eager_percent", -1)) - 83.33333333333334) > 1e-9:
                errors.append(f"{phase} launch-reduction denominator drifted")
    operator_path = (
        ROOT / "state/evidence/qwen35-9b-operator-performance-breakdown-current.json"
    )
    operator = load(operator_path, errors)
    if operator is not None:
        comparisons = operator.get("comparisons")
        lanes = operator.get("lanes")
        identity = operator.get("global_evidence_identity")
        source_lock = load(ROOT / "vendor/source-lock.json", errors)
        expected_cases = {
            "down-linear-decode-1x12288x4096",
            "down-linear-prefill-31x12288x4096",
            "fp32-lm-head-decode-and-pruned-prefill-1x4096x248320",
            "gate-up-linear-decode-1x4096x24576",
            "gate-up-linear-prefill-31x4096x24576",
            "swiglu-decode-1x24576",
            "swiglu-prefill-31x24576",
        }
        repositories = (
            source_lock.get("repositories", {})
            if isinstance(source_lock, dict)
            else {}
        )
        sources = (
            identity.get("sources", {}) if isinstance(identity, dict) else {}
        )
        candidate_dso = (
            identity.get("candidate_packages", {}).get("dso", {})
            if isinstance(identity, dict)
            else {}
        )
        if (
            sha256(operator_path) != OPERATOR_BREAKDOWN_SHA256
            or operator.get("schema") != 1
            or operator.get("kind")
            != "qwen35-9b-aligned-operator-ab-performance-summary"
            or operator.get("status") != "complete"
            or not isinstance(comparisons, dict)
            or set(comparisons) != expected_cases
            or not isinstance(lanes, dict)
            or set(lanes) != {"pypto", "sglang-matched"}
            or any(
                not isinstance(lanes.get(lane), dict)
                or lanes[lane].get("fresh_starts") != 4
                for lane in ("pypto", "sglang-matched")
            )
            or any(
                not isinstance(record, dict)
                or not isinstance(record.get("pypto_latency_percent_of_stock"), (int, float))
                or not isinstance(record.get("median_ratio_bootstrap_95ci_percent"), dict)
                or record.get("contract", {}).get("name") != case
                for case, record in comparisons.items()
            )
            or sources.get("pypto", {}).get("commit")
            != repositories.get("pypto", {}).get("head_commit")
            or sources.get("tensor_ir", {}).get("commit")
            != repositories.get("tensor_ir", {}).get("head_commit")
            or candidate_dso.get("sha256")
            != (load(
                ROOT / "state/evidence/operator-regression-current.json", errors
            ) or {}).get("source", {}).get("dso", {}).get("sha256")
        ):
            errors.append("checked-in operator performance breakdown drifted")
    pair = load(ROOT / "state/evidence/qwen35-9b-performance-pair-current.json", errors)
    if pair is not None:
        acceptance = pair.get("acceptance")
        if (
            pair.get("status") != "complete"
            or pair.get("performance_only") is not True
            or pair.get("comparison", {}).get("pypto_percent_of_matched") is None
            or not isinstance(acceptance, dict)
            or acceptance.get("accepted") is not True
            or acceptance.get("status") != "resource-and-control-compliant"
            or acceptance.get("required_gpu_free_bytes") != 4 * 1024**3
            or acceptance.get("minimum_observed_gpu_free_bytes", 0) < 4 * 1024**3
            or acceptance.get("starts_below_floor") != 0
            or acceptance.get("starts_total") != 8
            or acceptance.get("control_comparability", {}).get("accepted") is not True
            or acceptance.get("control_comparability", {}).get("mismatches") != []
            or pair.get("source", {}).get("summary_script_sha256")
            != sha256(ROOT / "tools/summarize_qwen_performance_pair.py")
        ):
            errors.append("current performance pair acceptance boundary drifted")
        observed: list[object] = []
        for lane in ("pypto", "sglang-matched"):
            lane_record = pair.get("lanes", {}).get(lane, {})
            starts = lane_record.get("starts", []) if isinstance(lane_record, dict) else []
            observed.extend(
                start.get("resources", {}).get("minimum_gpu_memory_free_bytes")
                for start in starts
                if isinstance(start, dict) and isinstance(start.get("resources"), dict)
            )
        if len(observed) != 8 or any(
            type(value) is not int or value < 4 * 1024**3 for value in observed
        ):
            errors.append("performance pair raw start floors are not accepted")
        lock = ROOT / "vendor/source-lock.json"
        if pair.get("source", {}).get("source_lock_sha256") != sha256(lock):
            errors.append("performance pair source-lock hash is stale")
    historical_pair_path = (
        ROOT / "state/evidence/qwen35-9b-performance-pair-invalidated-20260830.json"
    )
    historical_pair = load(historical_pair_path, errors)
    if historical_pair is not None:
        historical_acceptance = historical_pair.get("acceptance", {})
        if (
            historical_pair.get("status") != "invalidated-resource-and-control"
            or not isinstance(historical_acceptance, dict)
            or historical_acceptance.get("accepted") is not False
            or historical_acceptance.get("starts_below_floor") != 3
            or historical_acceptance.get("minimum_observed_gpu_free_bytes")
            != 4185067520
        ):
            errors.append("historical invalidated performance pair boundary drifted")
    qualification = load(
        ROOT / "state/evidence/matched-performance-qualification-current.json",
        errors,
    )
    if qualification is not None:
        attempts = qualification.get("attempts")
        if (
            qualification.get("status") != "complete"
            or qualification.get("accepted") is not True
            or not isinstance(attempts, list)
            or len(attempts) != 2
            or any(
                not isinstance(attempt, dict)
                or attempt.get("status") != "complete"
                or attempt.get("abort_reason") is not None
                or not attempt.get("performance_report")
                for attempt in attempts
            )
            or qualification.get("acceptance", {}).get("pair_summary_sha256")
            != sha256(ROOT / "state/evidence/qwen35-9b-performance-pair-current.json")
        ):
            errors.append("historical matched qualification boundary drifted")
    optimized = load(
        ROOT / "state/evidence/optimized-lane-diagnostic-current.json", errors
    )
    if optimized is not None:
        optimized_attempts = optimized.get("attempts")
        memory_relief = optimized.get("official_memory_relief_audit", {})
        latest = (
            optimized_attempts[-1]
            if isinstance(optimized_attempts, list) and optimized_attempts
            else {}
        )
        if (
            optimized.get("status") != "open"
            or optimized.get("accepted_for_performance") is not False
            or latest.get("run_id")
            != "pypto-gpu-bounded-20260831T034327Z-2531381-964f87"
            or latest.get("abort_reason") != "gpu-free-memory-floor"
            or latest.get("controller_observation", {}).get("latest_gpu_free_mib")
            != 4000
            or latest.get("performance_report") is not None
            or memory_relief.get("status")
            != "not-applicable-with-current-formal-contract"
            or memory_relief.get("torch_memory_saver", {}).get(
                "installed_in_sglang_baseline"
            )
            is not False
            or memory_relief.get("torch_memory_saver", {}).get(
                "adapter_source_sha256"
            )
            != sha256(
                ROOT
                / ".sources/sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py"
            )
            or memory_relief.get("post_capture_kv_sizing", {}).get("enabled")
            is not False
            or memory_relief.get("post_capture_kv_sizing", {}).get(
                "server_args_source_sha256"
            )
            != sha256(ROOT / ".sources/sglang/python/sglang/srt/server_args.py")
        ):
            errors.append("optimized lane current resource boundary drifted")
    source = load(ROOT / "state/evidence/qwen35-9b-inductor-source-current.json", errors)
    if source is not None:
        if source.get("status") != "complete" or set(source.get("cases", {})) != {"prefill", "decode"}:
            errors.append("current Inductor source evidence is incomplete")
    eager = load(ROOT / "state/evidence/qwen35-9b-eager-compile-ablation-current.json", errors)
    if eager is not None and (
        eager.get("status") != "complete"
        or eager.get("performance_only") is not True
        or eager.get("interpretation", {}).get("whole_model_torch_compile_speedup_percent") is not None
    ):
        errors.append("full-model eager control must remain explicitly non-causal")
    if eager is not None:
        eager_resource = eager.get("eager_control", {}).get("resource_boundary", {})
        matched_boundary = eager.get("matched_compile_requested", {}).get(
            "source_pair_boundary", {}
        )
        pair_path = ROOT / "state/evidence/qwen35-9b-performance-pair-current.json"
        if (
            eager_resource.get("accepted") is not True
            or eager_resource.get("gpu_free_floor_bytes") != 4 * 1024**3
            or matched_boundary.get("source_pair_status") != "complete"
            or matched_boundary.get("source_pair_accepted") is not True
            or matched_boundary.get("matched_subset_resource_accepted") is not True
            or eager.get("matched_compile_requested", {}).get("summary_sha256")
            != sha256(pair_path)
        ):
            errors.append("full-model eager control resource/source boundary drifted")
    descriptive = load(
        ROOT / "state/evidence/qwen35-9b-descriptive-stock-profile-breakdown-current.json",
        errors,
    )
    if descriptive is not None:
        acceptance = descriptive.get("acceptance")
        source = descriptive.get("source")
        inputs = descriptive.get("inputs")
        if (
            descriptive.get("status") != "complete"
            or descriptive.get("kind")
            != "qwen35-9b-descriptive-stock-profile-breakdown"
            or descriptive.get("profile_scope") != "descriptive-stock-noncompiled"
            or not isinstance(acceptance, dict)
            or acceptance.get("accepted") is not True
            or acceptance.get("fresh_starts_per_lane") != 3
            or acceptance.get("profile_requests_per_start") != 5
            or acceptance.get("matched_compilation_effective") is not False
            or acceptance.get("minimum_gpu_memory_free_bytes", 0) < 4 * 1024**3
            or acceptance.get("minimum_host_memory_available_kib", 0) < 12 * 1024**2
            or acceptance.get("thermal_throttle_observed") is not False
            or acceptance.get("control_mismatches") != []
            or not isinstance(source, dict)
            or not isinstance(inputs, dict)
            or inputs.get("reconciliation_sha256") is None
            or not isinstance(inputs.get("profiles"), list)
            or len(inputs["profiles"]) != 6
        ):
            errors.append("descriptive stock profile breakdown acceptance drifted")
        else:
            reconciliation = ROOT / str(inputs.get("reconciliation", ""))
            if reconciliation.is_file() and inputs.get("reconciliation_sha256") != sha256(reconciliation):
                errors.append("descriptive profile reconciliation hash is stale")
            for record in inputs["profiles"]:
                if not isinstance(record, dict):
                    errors.append("descriptive profile input record is malformed")
                    continue
                report = ROOT / str(record.get("report", ""))
                raw_trace = ROOT / str(record.get("raw_trace", ""))
                if not report.is_file() or record.get("report_sha256") != sha256(report):
                    errors.append(f"descriptive profile report hash is stale: {record.get('report')}")
                if not raw_trace.is_file() or record.get("raw_trace_sha256") != sha256(raw_trace):
                    errors.append(f"descriptive profile raw trace hash is stale: {record.get('raw_trace')}")
                report_value = load(report, errors)
                if report_value is None:
                    continue
                compilation = report_value.get("compilation")
                expected_scope = (
                    "descriptive-stock-noncompiled"
                    if record.get("lane") == "sglang-matched"
                    else "strict-compiled"
                )
                scope_ok = (
                    isinstance(compilation, dict)
                    and report_value.get("profile_scope", "strict-compiled")
                    == expected_scope
                    and compilation.get("acceptance_scope", "strict-compiled")
                    == expected_scope
                )
                if (
                    report_value.get("status") != "complete"
                    or report_value.get("lane") != record.get("lane")
                    or report_value.get("workload") != workload_record()
                    or report_value.get("profile_requests") != 5
                    or not scope_ok
                    or not isinstance(compilation, dict)
                    or compilation.get("requested") is not True
                    or compilation.get("effective")
                    != (expected_scope == "strict-compiled")
                ):
                    errors.append(f"descriptive profile report boundary drifted: {report}")
    for relative in (
        "benchmarks/release/performance_runtime.py",
        "benchmarks/release/operator_performance_runtime.py",
        "benchmarks/release/inductor_ablation.py",
        "tools/run_performance_regression.py",
    ):
        text = read(ROOT / relative, errors)
        for forbidden in ("correctness_runtime", "torch.allclose", "torch.equal", "reference_logits"):
            if forbidden in text:
                errors.append(f"performance source contains forbidden correctness hook: {relative}:{forbidden}")
        if "19+64" in text:
            errors.append(f"performance source uses the historical raw-token workload: {relative}")


def check_sources(errors: list[str]) -> None:
    lock = load(ROOT / "vendor/source-lock.json", errors)
    if lock is None:
        return
    if lock.get("repositories", {}).get("pypto", {}).get("head_commit") != PYPTO_HEAD:
        errors.append("source-lock PyPTO head differs from current compiler")
    nested = ROOT / ".sources/pypto"
    observed = subprocess.run(
        ["git", "-C", str(nested), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if observed.returncode != 0 or observed.stdout.strip() != PYPTO_HEAD:
        errors.append("working .sources/pypto revision differs from source-lock")
    for name, spec in lock.get("packages", {}).items():
        source_commit = str(spec.get("source_commit", ""))
        source_tree = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"{source_commit}^{{tree}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        prefix_tree = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"HEAD:{spec.get('path')}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if (
            source_tree.returncode != 0
            or prefix_tree.returncode != 0
            or source_tree.stdout.strip() != str(spec.get("source_tree"))
            or prefix_tree.stdout.strip() != str(spec.get("prefix_tree"))
            or source_tree.stdout.strip() != prefix_tree.stdout.strip()
        ):
            errors.append(f"package source lock mismatch: {name}")
    release_python = ROOT / "envs/pypto-release/bin/python"
    if release_python.is_file():
        result = subprocess.run(
            [str(release_python), "-c", "from pypto.compiler import get_nvidia_backend_build_info; print(get_nvidia_backend_build_info().pypto_revision)"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != PYPTO_HEAD:
            errors.append("installed PyPTO build-info differs from source-lock")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "state/evidence/blog-requirements-audit-current.json",
    )
    args = parser.parse_args()
    errors: list[str] = []
    checked: list[str] = []
    blog_path = ROOT / "reports/local-blog/pypto-tensorir-rtx5090-qwen35-9b.md"
    texts = {
        "README.md": read(ROOT / "README.md", errors),
        "README_EN.md": read(ROOT / "README_EN.md", errors),
        "blog": read(blog_path, errors),
        "LEGAL_NOTICE.md": read(ROOT / "LEGAL_NOTICE.md", errors),
    }
    for name, text in texts.items():
        if name != "LEGAL_NOTICE.md":
            if ARTICLE_URL not in text or PROMPT not in text:
                errors.append(f"{name} misses article URL or exact prompt")
            if "invalidated-resource-and-control" not in text:
                errors.append(f"{name} misses matched-pair invalidation boundary")
            if "--optimized-memory-mode matched" not in text:
                errors.append(f"{name} misses the corrected performance memory mode")
            if "--pair-matrix" not in text:
                errors.append(f"{name} misses the independent matched pair matrix")
            if "qwen35-9b-release-results-current.json" not in text:
                errors.append(f"{name} misses the unified current release evidence")
            if "qwen35-9b-operator-performance-breakdown-current.json" not in text:
                errors.append(f"{name} misses the checked-in operator breakdown evidence")
            if "article-demo-compatibility-policy-current.json" not in text:
                errors.append(f"{name} misses the article-demo compatibility policy")
            if "strict-pypto-nvidia" not in text:
                errors.append(f"{name} misses the strict computational demo mode")
            if "computational_unmapped_count=0" not in text:
                errors.append(f"{name} misses the closed computational demo denominator")
            if (
                "15.62" not in text
                or "18.71" not in text
                or "-84.379" not in text
                or "-81.285" not in text
            ):
                errors.append(f"{name} misses the current three-lane metrics")
            if "--optimized-memory-mode zero-offload" in text:
                errors.append(f"{name} retains the rejected performance memory mode")
        if BILIBILI_URL not in text:
            errors.append(f"{name} misses interview attribution URL")
        checked.append(name)
    for heading in ("# 一、", "# 二、", "# 三、", "# 四、", "# 五、", "# 六、", "# 七、", "# 八、", "# 九、", "# 十、"):
        if heading not in texts["blog"]:
            errors.append(f"blog misses heading {heading}")
    if any("PENDING_GPT_IMAGE2" not in texts[name] for name in ("README.md", "README_EN.md", "blog")):
        errors.append("GPT-Image-2 pending/provenance marker is missing")
    html = read(blog_path.with_suffix(".html"), errors)
    if html and html.count("data:image/") < 3:
        errors.append("offline HTML does not embed current local images")
    if html and (
        "invalidated-resource-and-control" not in html
        or "--optimized-memory-mode matched" not in html
        or "--optimized-memory-mode zero-offload" in html
    ):
        errors.append("offline HTML retains the stale performance boundary")
    check_document_resources(texts, blog_path, html, errors)
    checked.append("document-resource-closure")
    check_demo(errors)
    check_demo_docs(errors)
    check_demo_compatibility(errors)
    check_screenshots(errors)
    check_operator(errors)
    check_models(errors)
    check_performance(errors)
    check_sources(errors)
    matrix_text = read(ROOT / "docs/final_requirement_matrix.md", errors)
    if not all(
        marker in matrix_text
        for marker in (
            "End-to-end PyPTO versus optimized stock",
            "18.7143% of optimized",
            "hybrid three-lane evidence",
        )
    ):
        errors.append("final requirement matrix is missing current optimized/profile closure")
    blockers = [
        "GPT-Image-2 generation awaits local API authorization",
    ]
    screenshot_manifest = ROOT / "state/evidence/article-demo-screenshot-manifest-current.json"
    try:
        screenshot_status = json.loads(
            screenshot_manifest.read_text(encoding="utf-8")
        ).get("status")
    except (OSError, json.JSONDecodeError):
        screenshot_status = None
    if screenshot_status != "complete":
        blockers.insert(2, "article-demo PowerShell role capture is pending")
    result = {
        "schema": 2,
        "kind": "blog-requirements-audit",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "failed",
        "checked": checked,
        "errors": errors,
        "open_gates": blockers,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "error_count": len(errors), "open_gate_count": len(blockers)}))
    print(f"blog requirements audit: {output}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
