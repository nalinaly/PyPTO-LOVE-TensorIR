#!/usr/bin/env python3
"""Fail-closed audit for release documents and their evidence sidecars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_URL = "https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ"
BILIBILI_URL = (
    "https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source="
    "f2f41aa7b5e3cc8e0a23942779ccea11"
)
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
PYPTO_HEAD = "c27629e993a52b47d41fb898c749279dce44221b"
ARTICLE_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"


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
        "computational-cuda-reference": 9,
        "computational-unmapped": 31,
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
        or matrix.get("computational_reference_pass_count") != 9
        or matrix.get("hardware_api_skipped_count") != 17
        or matrix.get("computational_unmapped_count") != 31
    ):
        errors.append("current NVIDIA article-demo matrix is incomplete or stale")


def check_screenshots(errors: list[str]) -> None:
    path = ROOT / "state/evidence/article-demo-screenshot-manifest-current.json"
    value = load(path, errors)
    if value is None:
        return
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
        if role in {"build", "operator-correctness", "performance", "model-inference"}:
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
            if (
                record.get("status")
                != ("current-operator-scope" if role == "performance" else "current-evidence-replay")
                or record.get("capture_evidence_sha256") != sha256(capture_path)
                or capture.get("command") != record.get("command")
                or capture.get("output_sha256") != record.get("sha256")
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
    pair = load(ROOT / "state/evidence/qwen35-9b-performance-pair-current.json", errors)
    if pair is not None:
        acceptance = pair.get("acceptance")
        if (
            pair.get("status") != "invalidated-resource-and-control"
            or pair.get("performance_only") is not True
            or pair.get("comparison", {}).get("pypto_percent_of_matched") is None
            or not isinstance(acceptance, dict)
            or acceptance.get("accepted") is not False
            or acceptance.get("required_gpu_free_bytes") != 4 * 1024**3
            or acceptance.get("minimum_observed_gpu_free_bytes") != 4185067520
            or acceptance.get("starts_below_floor") != 3
            or acceptance.get("control_comparability", {}).get("accepted")
            is not False
            or acceptance.get("control_comparability", {}).get("mismatches")
            != [{"field": "cpu_offload_gb", "pypto": 0, "sglang_matched": 2}]
            or acceptance.get("current_validator_script_sha256")
            != sha256(ROOT / "tools/summarize_qwen_performance_pair.py")
        ):
            errors.append("current performance pair invalidation boundary drifted")
        pypto_lane = pair.get("lanes", {}).get("pypto", {})
        starts = pypto_lane.get("starts", []) if isinstance(pypto_lane, dict) else []
        observed = [
            start.get("resources", {}).get("minimum_gpu_memory_free_bytes")
            for start in starts
            if isinstance(start, dict) and isinstance(start.get("resources"), dict)
        ]
        if (
            len(observed) != 4
            or sum(type(value) is int and value < 4 * 1024**3 for value in observed)
            != 3
        ):
            errors.append("performance pair raw start floors do not match invalidation")
        lock = ROOT / "vendor/source-lock.json"
        if pair.get("source", {}).get("source_lock_sha256") != sha256(lock):
            errors.append("performance pair source-lock hash is stale")
    qualification = load(
        ROOT / "state/evidence/matched-performance-qualification-current.json",
        errors,
    )
    if qualification is not None:
        attempts = qualification.get("attempts")
        if (
            qualification.get("status") != "open"
            or qualification.get("accepted") is not False
            or not isinstance(attempts, list)
            or len(attempts) != 1
            or attempts[0].get("abort_reason") != "host-memory-emergency-floor"
            or attempts[0].get("performance_report") is not None
            or qualification.get("current_controller_policy", {}).get(
                "runtime_abort_reason"
            )
            != "protected-heavy-coexistence"
            or qualification.get("current_controller_policy", {}).get(
                "controller_sha256"
            )
            != sha256(ROOT / "tools/run_pypto_gpu_bounded.py")
        ):
            errors.append("matched performance qualification boundary drifted")
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
            or matched_boundary.get("source_pair_status")
            != "invalidated-resource-and-control"
            or matched_boundary.get("source_pair_accepted") is not False
            or matched_boundary.get("matched_subset_resource_accepted") is not True
            or eager.get("matched_compile_requested", {}).get("summary_sha256")
            != sha256(pair_path)
        ):
            errors.append("full-model eager control resource/source boundary drifted")
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
            if "matched-performance-qualification-current.json" not in text:
                errors.append(f"{name} misses the current qualification blocker")
            if "article-demo-compatibility-policy-current.json" not in text:
                errors.append(f"{name} misses the article-demo compatibility policy")
            if "strict-pypto-nvidia" not in text:
                errors.append(f"{name} misses the strict computational demo mode")
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
    check_demo(errors)
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
            "OPEN: historical attempts failed resource/telemetry qualification",
            "protected-heavy-free window is unavailable",
            "no percentage promoted",
        )
    ):
        errors.append("final requirement matrix is missing optimized-lane boundary")
    blockers = [
        "matched full-model performance pair needs a resource-compliant rerun",
        "optimized stock lane has no accepted sample",
        "full-model CUPTI/NVTX profile awaits an accepted pair and exclusive resources",
        "article-demo PowerShell role capture is pending",
        "31 model article-demo compute entries have no bounded NVIDIA adapter",
        "article demo device runtime is Ascend-only for hardware-facing entries in this environment",
        "GPT-Image-2 generation awaits local API authorization",
    ]
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
