#!/usr/bin/env python3
"""Run an imported PyPTO-Lib article demo without modifying its source.

The launcher is deliberately a thin provenance wrapper. It does not rewrite
the upstream file or translate its platform arguments. A zero child exit code
is the only condition that produces ``status: pass``; compile-only or partial
golden output is recorded as a failure/blocker instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo" / "pypto-lib"
MANIFEST = DEMO_ROOT / "SOURCE_MANIFEST.json"
DEFAULT_PYTHON = ROOT / "envs" / "pypto-release" / "bin" / "python"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("kind") != "article-demo-provenance":
        raise ValueError("article demo manifest has an unexpected kind")
    if payload.get("upstream", {}).get("commit") != (
        "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
    ):
        raise ValueError("article demo manifest is not locked to the article-time commit")
    return payload


def verify_source(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    relative = path.relative_to(DEMO_ROOT).as_posix()
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("article demo manifest files are missing")
    record = next((item for item in records if item.get("path") == relative), None)
    if not isinstance(record, dict):
        raise ValueError(f"demo is not in the source manifest: {relative}")
    observed_bytes = path.stat().st_size
    observed_sha = file_sha256(path)
    if observed_bytes != record.get("bytes") or observed_sha != record.get("sha256"):
        raise ValueError(f"imported demo source changed: {relative}")
    return {
        "path": relative,
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "upstream_commit": manifest["upstream"]["commit"],
    }


def classify_blocker(output: str) -> str | None:
    lowered = output.lower()
    if "no module named 'simpler" in lowered or "no module named \"simpler" in lowered:
        return "missing Ascend simpler runtime extension"
    if "no module named '_task_interface'" in lowered:
        return "missing Ascend task-interface native extension"
    if "kerneltype" in lowered and "has no attribute" in lowered:
        return "upstream PyPTO API is newer/older than the imported article demo"
    if "cuda" in lowered and ("not available" in lowered or "driver" in lowered):
        return "CUDA runtime/device unavailable"
    if "a2a3" in lowered and "platform" in lowered and "invalid" in lowered:
        return "upstream platform/runtime is unavailable in this environment"
    return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True, help="path relative to demo/pypto-lib")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--platform", default="a2a3sim")
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    relative = Path(args.demo)
    if relative.is_absolute() or ".." in relative.parts:
        parser.error("--demo must stay below demo/pypto-lib")
    source = (DEMO_ROOT / relative).resolve()
    if DEMO_ROOT not in source.parents or not source.is_file() or source.suffix != ".py":
        parser.error("--demo must name an imported Python file")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    python = args.python.resolve()
    if not python.is_file():
        parser.error(f"selected Python does not exist: {python}")
    manifest = load_manifest()
    source_record = verify_source(source, manifest)
    run_id = os.environ.get("PYPTO_RUN_ID", "article-demo-" + str(int(time.time())))
    if not RUN_ID_RE.fullmatch(run_id):
        parser.error("PYPTO_RUN_ID contains unsupported characters")
    output = args.output or (
        ROOT / "state" / "evidence" / "article-demos" / f"{source.stem}-{run_id}.json"
    )

    command = [str(python), "-B", str(source), "-p", str(args.platform), "-d", str(args.device)]
    remainder = list(args.remainder)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    command.extend(remainder)
    payload: dict[str, object] = {
        "schema": 1,
        "kind": "article-demo-run",
        "run_id": run_id,
        "source": source_record,
        "article_url": manifest["article"]["url"],
        "command": command,
        "cwd": str(DEMO_ROOT),
        "audit_only": bool(args.audit_only),
        "status": "audited" if args.audit_only else "not_started",
    }
    if args.audit_only:
        write_json(output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(DEMO_ROOT), environment.get("PYTHONPATH", "")) if item
    )
    environment["PYPTO_ARTICLE_DEMO_SOURCE_SHA256"] = str(source_record["sha256"])
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=DEMO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        return_code = None
        timed_out = True
    elapsed = time.monotonic() - started
    combined = stdout + stderr
    payload.update(
        {
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            "golden_pass_observed": bool(
                re.search(r"(?i)(golden|allclose|validation).*(pass|ok|success)", combined)
            ),
            "blocker": classify_blocker(combined),
            "status": "pass" if return_code == 0 and not timed_out else "fail",
        }
    )
    write_json(output, payload)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    print(json.dumps({k: payload[k] for k in ("status", "return_code", "blocker", "output") if k in payload}, ensure_ascii=False, sort_keys=True))
    print(f"article demo report: {output}")
    return 0 if return_code == 0 and not timed_out else (return_code or 124)


if __name__ == "__main__":
    raise SystemExit(main())
