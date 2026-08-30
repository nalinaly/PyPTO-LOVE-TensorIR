#!/usr/bin/env python3
"""Run the local operator regression smoke and persist its transcript metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "packages/pypto-kernels/tests/test_operators.py",
        "-q",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    payload = {
        "schema": 1,
        "kind": "article-demo-operator-smoke",
        "status": "complete" if completed.returncode == 0 else "failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "transcript_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "source": "packages/pypto-kernels/tests/test_operators.py",
    }
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
    sys.stdout.write(output)
    print(json.dumps({"status": payload["status"], "return_code": completed.returncode}, sort_keys=True))
    print(f"operator smoke report: {path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
