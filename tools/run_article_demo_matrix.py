#!/usr/bin/env python3
"""Audit or execute every CLI entry point imported from the PyPTO-Lib article.

The imported source is never rewritten.  ``audit`` verifies all source hashes;
``help`` checks that each CLI can at least be discovered without running a
device; ``run`` invokes the same source files with the compatibility launcher
and records every exit code and blocker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from run_article_demo import DEMO_ROOT, MANIFEST, DEFAULT_PYTHON, file_sha256


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "state" / "evidence" / "article-demo-compatibility-policy-current.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("kind") != "article-demo-provenance":
        raise ValueError("unexpected article demo manifest kind")
    if payload.get("upstream", {}).get("commit") != (
        "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
    ):
        raise ValueError("article demo manifest is not article-time locked")
    if not isinstance(payload.get("entrypoints"), list):
        raise ValueError("article demo manifest has no entrypoint inventory")
    return payload


def _corpus_sha256(manifest: dict[str, object]) -> str:
    digest = hashlib.sha256()
    records = manifest.get("files", [])
    if not isinstance(records, list):
        raise ValueError("article demo manifest files are missing")
    for record in sorted(records, key=lambda item: str(item.get("path", ""))):
        relative = str(record["path"])
        path = (DEMO_ROOT / relative).resolve()
        if DEMO_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"imported corpus file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _policy(manifest: dict[str, object]) -> dict[str, object]:
    """Load the external NVIDIA policy without changing the source manifest."""
    if not POLICY.is_file():
        raise ValueError(
            f"NVIDIA compatibility policy is missing; run tools/classify_article_demos.py: {POLICY}"
        )
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    if payload.get("kind") != "article-demo-compatibility-policy":
        raise ValueError("unexpected article demo compatibility policy kind")
    if payload.get("upstream_commit") != manifest.get("upstream", {}).get("commit"):
        raise ValueError("article demo compatibility policy commit differs from manifest")
    if (
        payload.get("manifest_sha256") != file_sha256(MANIFEST)
        or payload.get("corpus_sha256") != _corpus_sha256(manifest)
    ):
        raise ValueError("article demo compatibility policy manifest hash is stale")
    return payload


def _source_record(
    relative: str, manifest: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    source = (DEMO_ROOT / relative).resolve()
    if DEMO_ROOT not in source.parents or not source.is_file():
        raise ValueError(f"entrypoint escaped or is missing: {relative}")
    records = manifest.get("files", [])
    record = next((item for item in records if item.get("path") == relative), None)
    if not isinstance(record, dict):
        raise ValueError(f"entrypoint is not in source manifest: {relative}")
    observed = file_sha256(source)
    if observed != record.get("sha256") or source.stat().st_size != record.get("bytes"):
        raise ValueError(f"imported source changed: {relative}")
    return source, {
        "path": relative,
        "bytes": source.stat().st_size,
        "sha256": observed,
    }


def _blocker(output: str) -> str | None:
    lowered = output.lower()
    if "no module named" in lowered and "simpler" in lowered:
        return "missing Ascend simpler runtime extension"
    if "no module named" in lowered and "task_interface" in lowered:
        return "missing Ascend task-interface native extension"
    if "kerneltype" in lowered and "has no attribute" in lowered:
        return "upstream PyPTO API differs from the installed runtime"
    if "cuda" in lowered and ("not available" in lowered or "driver" in lowered):
        return "CUDA runtime/device unavailable"
    if "need exactly 2 devices" in lowered:
        return "distributed demo requires two device IDs"
    if "no chip-level tasks found" in lowered:
        return "distributed Ascend L3 runtime unavailable in NVIDIA checkout"
    if "npu" in lowered or "a2a3" in lowered or "ascend" in lowered:
        if "unavailable" in lowered or "not found" in lowered or "invalid" in lowered:
            return "upstream Ascend platform/runtime unavailable"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "help", "run"), default="audit")
    parser.add_argument("--platform", default="a2a3sim")
    parser.add_argument(
        "--backend",
        choices=("ascend", "nvidia"),
        default="ascend",
        help="ascend runs the unchanged upstream CLI; nvidia uses the external computational policy",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path-prefix",
        default="",
        help="restrict the matrix to manifest paths with this prefix",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="run draft/Ascend-CANN-only entries instead of recording them as skipped",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    python = args.python.resolve()
    if not python.is_file():
        parser.error(f"Python executable does not exist: {python}")
    manifest = _manifest()
    compatibility_policy = _policy(manifest) if args.backend == "nvidia" else None
    policy_by_path = {
        item["path"]: item
        for item in (compatibility_policy or {}).get("entries", [])
        if isinstance(item, dict)
    }
    manifest_sha_before = file_sha256(MANIFEST)
    corpus_sha_before = _corpus_sha256(manifest)
    entries = [
        item
        for item in manifest["entrypoints"]
        if not args.path_prefix or str(item.get("path", "")).startswith(args.path_prefix)
    ]
    if not entries:
        parser.error("--path-prefix did not select any manifest entrypoint")
    results: list[dict[str, object]] = []
    matrix_started = time.monotonic()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("entrypoint record has an invalid schema")
        relative = item["path"]
        source, source_record = _source_record(relative, manifest)
        help_args = item.get("help_args", ["--help"])
        if not isinstance(help_args, list) or any(not isinstance(x, str) for x in help_args):
            raise ValueError(f"entrypoint help args are invalid: {relative}")
        execution_policy = item.get("execution_policy", "runnable")
        compatibility = policy_by_path.get(relative, {})
        compatibility_mode = compatibility.get("compatibility_mode")
        if args.backend == "nvidia" and args.mode == "run":
            # Hardware-facing, draft, and unmapped entries are intentionally
            # skipped.  They stay in the denominator and retain a reason.
            if compatibility_mode not in {
                "strict-pypto-nvidia",
                "computational-cuda-reference",
            }:
                results.append(
                    {
                        "path": relative,
                        "source": source_record,
                        "status": "skipped",
                        "execution_policy": execution_policy,
                        "compatibility_mode": compatibility_mode,
                        "skip_reason": compatibility.get("reason"),
                        "hardware_api_evidence": compatibility.get("hardware_api_evidence", []),
                        "command": None,
                        "return_code": None,
                    }
                )
                continue
        if execution_policy != "runnable" and not args.include_excluded:
            results.append(
                {
                    "path": relative,
                    "source": source_record,
                    "status": "skipped",
                    "execution_policy": execution_policy,
                    "command": None,
                    "return_code": None,
                }
            )
            continue
        if args.mode == "audit":
            results.append(
                {
                    "path": relative,
                    "source": source_record,
                    "status": "audited",
                    "execution_policy": execution_policy,
                    "command": None,
                    "return_code": None,
                }
            )
            continue
        nvidia_output: Path | None = None
        if args.mode == "help":
            command = [str(python), "-B", str(source), *help_args]
        elif args.backend == "nvidia":
            nvidia_dir = ROOT / "state" / "evidence" / "article-demos-nvidia"
            nvidia_output = nvidia_dir / f"{len(results):03d}-{source.stem}.json"
            command = [
                str(python),
                "-B",
                str(ROOT / "tools" / "run_article_demo_nvidia.py"),
                "--demo",
                relative,
                "--device",
                str(args.device).split(",", 1)[0],
                "--run-id",
                f"article-demo-nvidia-{len(results):03d}",
                "--output",
                str(nvidia_output),
            ]
        else:
            device = str(item.get("default_device", args.device))
            command = [
                str(python),
                "-B",
                str(ROOT / "tools" / "run_article_demo.py"),
                "--python",
                str(python),
                "--demo",
                relative,
                "--platform",
                args.platform,
                "--device",
                device,
            ]
        environment = dict(**__import__("os").environ)
        environment["PYTHONPATH"] = ":".join(
            item for item in (str(DEMO_ROOT), environment.get("PYTHONPATH", "")) if item
        )
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
            stdout, stderr = completed.stdout, completed.stderr
            return_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            return_code = None
            timed_out = True
        combined = stdout + stderr
        result_record: dict[str, object] = {
                "path": relative,
                "source": source_record,
                "execution_policy": execution_policy,
                "compatibility_mode": compatibility_mode,
                "status": "pass" if return_code == 0 and not timed_out else "fail",
                "command": command,
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                "blocker": _blocker(combined),
                "stdout": stdout,
                "stderr": stderr,
            }
        if nvidia_output is not None:
            child_path = nvidia_output
            result_record["child_report"] = str(child_path.relative_to(ROOT))
            if child_path.is_file():
                try:
                    child = json.loads(child_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    child = None
                if isinstance(child, dict):
                    result_record["child_status"] = child.get("status")
                    result_record["child_report_sha256"] = file_sha256(child_path)
                    result_record["golden_pass"] = child.get("golden_pass")
                    result_record["strict_compiler_evidence"] = child.get(
                        "strict_compiler_evidence"
                    )
                    result_record["artifact"] = child.get("artifact")
                    result_record["status"] = (
                        "pass"
                        if child.get("status") == "pass"
                        else "fail"
                    )
        results.append(result_record)
    manifest_sha_after = file_sha256(MANIFEST)
    corpus_sha_after = _corpus_sha256(manifest)
    if manifest_sha_after != manifest_sha_before or corpus_sha_after != corpus_sha_before:
        raise RuntimeError("article demo imported corpus changed during matrix execution")
    passed = all(item["status"] in {"audited", "pass", "skipped"} for item in results)
    strict_passes = sum(
        args.mode == "run"
        and item.get("status") == "pass"
        and item.get("compatibility_mode") == "strict-pypto-nvidia"
        for item in results
    )
    reference_passes = sum(
        args.mode == "run"
        and item.get("status") == "pass"
        and item.get("compatibility_mode") == "computational-cuda-reference"
        for item in results
    )
    hardware_skips = sum(
        args.mode == "run" and item.get("compatibility_mode") == "hardware-api-skipped"
        for item in results
    )
    unmapped_skips = sum(
        args.mode == "run" and item.get("compatibility_mode") == "computational-unmapped"
        for item in results
    )
    payload = {
        "schema": 1,
        "kind": "article-demo-matrix",
        "mode": args.mode,
        "backend": args.backend,
        "path_prefix": args.path_prefix,
        "article_url": manifest["article"]["url"],
        "upstream_commit": manifest["upstream"]["commit"],
        "entrypoint_count": len(results),
        "skipped_count": sum(item["status"] == "skipped" for item in results),
        "help_pass_count": sum(
            args.mode == "help" and item["status"] == "pass" for item in results
        ),
        "strict_nvidia_pass_count": strict_passes,
        "computational_reference_pass_count": reference_passes,
        "hardware_api_skipped_count": hardware_skips,
        "computational_unmapped_count": unmapped_skips,
        "manifest_sha256_before": manifest_sha_before,
        "manifest_sha256_after": manifest_sha_after,
        "corpus_sha256_before": corpus_sha_before,
        "corpus_sha256_after": corpus_sha_after,
        "compatibility_policy": (
            {
                "path": str(POLICY.relative_to(ROOT)),
                "sha256": file_sha256(POLICY),
                "revision": compatibility_policy.get("policy_revision"),
            }
            if compatibility_policy is not None
            else None
        ),
        "status": "complete" if passed else "failed",
        "compatibility_status": (
            "not-applicable"
            if args.backend != "nvidia" or args.mode != "run"
            else "complete"
            if unmapped_skips == 0
            else "partial-computational-coverage"
        ),
        "elapsed_seconds": time.monotonic() - matrix_started,
        "results": results,
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "entrypoint_count": len(results)}, sort_keys=True))
    print(f"article demo matrix report: {args.output.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
