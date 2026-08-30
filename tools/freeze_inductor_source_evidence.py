#!/usr/bin/env python3
"""Extract exact Qwen3.5-9B generated source evidence from a regression report."""

from __future__ import annotations

import argparse
from datetime import timezone
from datetime import datetime
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve(strict=True)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("all_native") is not True:
        raise SystemExit("Inductor report is not all_native")
    selected = [
        case
        for case in payload.get("cases", [])
        if case.get("model") == "Qwen3.5-9B" and case.get("rows") in (1, 19)
    ]
    if {case.get("rows") for case in selected} != {1, 19}:
        raise SystemExit("report does not contain both 9B decode and prefill cases")
    cases = {}
    for case in selected:
        evidence = case.get("inductor_source_evidence")
        if not isinstance(evidence, dict):
            raise SystemExit("case has no source evidence")
        wrappers = evidence.get("wrapper_launch_sources")
        if not isinstance(wrappers, list) or len(wrappers) != 1:
            raise SystemExit("case does not have exactly one wrapper source")
        cases["decode" if case["rows"] == 1 else "prefill"] = {
            "rows": case["rows"],
            "columns": case["columns"],
            "source_node": evidence["source_node"],
            "kernel_name": evidence["kernel_name"],
            "entry_name": evidence["entry_name"],
            "pypto_source": evidence["pypto_source"],
            "pypto_source_sha256": evidence["pypto_source_sha256"],
            "wrapper_launch_sources": wrappers,
            "artifact_sha256": evidence["artifact_sha256"],
            "cubin_sha256": evidence["cubin_sha256"],
            "dso_sha256": evidence["dso_sha256"],
        }
    result = {
        "schema": 1,
        "kind": "qwen35-9b-inductor-source-evidence",
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Qwen3.5-9B packed SwiGLU pointwise",
        "source_report": report.relative_to(ROOT).as_posix(),
        "source_report_sha256": sha256(report),
        "torch_compile": payload.get("torch_compile"),
        "cases": cases,
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
