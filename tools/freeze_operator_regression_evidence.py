#!/usr/bin/env python3
"""Freeze a compact, hash-bound operator regression evidence record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return Path(value).resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve(strict=True)
    summary = args.summary.resolve(strict=True)
    payload = json.loads(report.read_text(encoding="utf-8"))
    control = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("all_correct") is not True:
        raise SystemExit("operator report is not a complete all-correct run")
    if control.get("status") != "complete":
        raise SystemExit("operator control summary is not complete")
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    suites = []
    for suite in payload.get("suites", []):
        suites.append(
            {
                "suite_id": suite.get("suite_id"),
                "passed": suite.get("passed") is True,
                "case_count": suite.get("case_count"),
                "result": relative_path(suite.get("result", "")),
                "result_sha256": suite.get("result_sha256"),
            }
        )
    compiler = payload.get("evidence_identity", {}).get("compiler", {})
    result = {
        "schema": 2,
        "kind": "operator-regression-evidence",
        "status": "complete",
        "all_correct": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "regression_report": report.relative_to(ROOT).as_posix(),
        "regression_report_sha256": sha256(report),
        "control_summary": summary.relative_to(ROOT).as_posix(),
        "control_summary_sha256": sha256(summary),
        "source": {
            "pypto_commit": compiler.get("pypto_revision"),
            "source_revisions": [
                {
                    **revision,
                    "repository": relative_path(revision.get("repository")),
                    "scope": relative_path(revision.get("scope")),
                }
                for revision in payload.get("source_revisions", [])
            ],
            "dso": {
                **(payload.get("dso") or {}),
                "path": relative_path((payload.get("dso") or {}).get("path")),
            },
            "package": relative_path(payload.get("pypto_package")),
        },
        "suites": suites,
        "total_suite_count": len(suites),
        "all_suites_passed": bool(suites) and all(item["passed"] for item in suites),
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
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
