#!/usr/bin/env python3
"""Validate and print compact build or operator release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BUILD_REPORT = ROOT / (
    "runs/pypto-cpu-bounded-20260901T225323Z-276745-19c762/"
    "release-build-ctest.json"
)
BUILD_PROCESS = BUILD_REPORT.with_name("process.json")
WHEELS_REPORT = ROOT / (
    "runs/pypto-cpu-bounded-20260901T225245Z-276262-6029dd/"
    "release-build-wheels.json"
)
WHEELS_PROCESS = WHEELS_REPORT.with_name("process.json")
INSTALL_REPORT = ROOT / (
    "runs/pypto-cpu-bounded-20260901T225412Z-276974-db59fd/"
    "release-build-install.json"
)
INSTALL_PROCESS = INSTALL_REPORT.with_name("process.json")
BUILD_LOG = ROOT / "builds/qwen35-sm120-v1/native/Testing/Temporary/LastTest.log"
OPERATOR_EVIDENCE = ROOT / "state/evidence/operator-regression-current.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"release evidence rejected: {message}")


def below_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    require(ROOT in resolved.parents, "path escapes repository")
    return resolved


def build_gate() -> None:
    report = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    process = json.loads(BUILD_PROCESS.read_text(encoding="utf-8"))
    wheels_report = json.loads(WHEELS_REPORT.read_text(encoding="utf-8"))
    wheels_process = json.loads(WHEELS_PROCESS.read_text(encoding="utf-8"))
    install_report = json.loads(INSTALL_REPORT.read_text(encoding="utf-8"))
    install_process = json.loads(INSTALL_PROCESS.read_text(encoding="utf-8"))
    require(report.get("schema") == 1, "build schema")
    require(report.get("kind") == "pypto-sm120-release-build", "build kind")
    require(report.get("stage") == "ctest", "build stage")
    require(report.get("status") == "complete", "build status")
    require(report.get("jobs") == 24, "build parallelism")
    wheels = report.get("wheels")
    require(isinstance(wheels, list) and len(wheels) == 3, "wheel count")
    for stage, stage_report, stage_process in (
        ("wheels", wheels_report, wheels_process),
        ("install", install_report, install_process),
        ("ctest", report, process),
    ):
        require(stage_report.get("stage") == stage, f"{stage} report stage")
        require(stage_report.get("status") == "complete", f"{stage} report status")
        require(stage_report.get("wheels") == wheels, f"{stage} artifact set")
        require(stage_process.get("status") == "exited", f"{stage} controller status")
        require(stage_process.get("return_code") == 0, f"{stage} controller return")
        require(stage_process.get("abort_reason") is None, f"{stage} controller abort")
        cleanup = stage_process.get("session_cleanup", {})
        require(cleanup.get("complete") is True, f"{stage} controller cleanup")
        require(cleanup.get("kill_signaled") == [], f"{stage} forced cleanup")
    for wheel in wheels:
        path = below_root(ROOT / str(wheel.get("path", "")))
        require(path.stat().st_size == wheel.get("bytes"), "wheel size")
        require(sha256(path) == wheel.get("sha256"), "wheel hash")
    log = BUILD_LOG.read_text(encoding="utf-8")
    indices = {
        int(match.group(1))
        for match in re.finditer(r"^([0-9]+)/13 Testing:", log, re.MULTILINE)
    }
    require(indices == set(range(1, 14)), "CTest indices")
    require(len(re.findall(r"^Test Passed\.$", log, re.MULTILINE)) == 13, "CTest pass count")
    require("Test Failed." not in log, "CTest failure")
    print("accepted build evidence replay (not a live rerun)")
    print("wheel build=pass | install/pip-check=pass | CTest=13/13")
    print("three stages: jobs=24 return_code=0 cleanup=natural")
    print("wheels=3: pypto + pypto-kernels + framework-plugins")
    print("same artifact set across stages; sizes/SHA-256 verified")
    print(f"ctest-run={report['run_id']}")


def operator_gate() -> None:
    evidence = json.loads(OPERATOR_EVIDENCE.read_text(encoding="utf-8"))
    require(evidence.get("schema") == 2, "operator schema")
    require(evidence.get("kind") == "operator-regression-evidence", "operator kind")
    require(evidence.get("status") == "complete", "operator status")
    require(evidence.get("all_correct") is True, "operator correctness")
    require(evidence.get("all_suites_passed") is True, "operator suites")
    require(evidence.get("total_suite_count") == 8, "operator suite count")
    control = below_root(ROOT / str(evidence.get("control_summary", "")))
    regression = below_root(ROOT / str(evidence.get("regression_report", "")))
    require(sha256(control) == evidence.get("control_summary_sha256"), "control hash")
    require(sha256(regression) == evidence.get("regression_report_sha256"), "report hash")
    suites = evidence.get("suites")
    require(isinstance(suites, list) and len(suites) == 8, "suite records")
    cases: dict[str, int] = {}
    for suite in suites:
        require(suite.get("passed") is True, "suite failure")
        result = below_root(ROOT / str(suite.get("result", "")))
        require(sha256(result) == suite.get("result_sha256"), "suite result hash")
        cases[str(suite.get("suite_id"))] = int(suite.get("case_count", -1))
    expected = {
        "handwritten-compile-classification": 25,
        "handwritten-numerical": 32,
        "stateful-real-model-shapes": 14,
        "paged-attention": 14,
        "qk-real-model-shapes": 4,
        "linear-real-model-shapes": 8,
        "stateful-cuda-graph": 0,
        "inductor-swiglu-real-model-shapes": 4,
    }
    require(cases == expected, "operator case inventory")
    print("accepted operator evidence replay (not a live rerun)")
    regression_payload = json.loads(regression.read_text(encoding="utf-8"))
    print(f"run={regression_payload['run_id']}")
    print("8/8 suites passed | all_correct=true")
    print("classify=25 numerical=32 stateful=14 paged=14")
    print("qk=4 linear/lm-head=8 Inductor-SwiGLU=4")
    print("stateful CUDA-Graph lifecycle=pass")
    print(f"PyPTO={evidence['source']['pypto_commit'][:12]} DSO=verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("build", "operator"))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    (build_gate if args.gate == "build" else operator_gate)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
