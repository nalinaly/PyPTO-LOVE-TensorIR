#!/usr/bin/env python3
"""Aggregate performance-only eager/Inductor/PyPTO ablation reports.

The input reports are intentionally independent from correctness gates.  This
tool rejects reports that contain an output comparison field so a timing
summary cannot accidentally become a numerical acceptance record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODES = ("eager", "inductor-nv", "pypto")
PHASES = {
    "prefill": {"rows": 19, "columns": 12_288},
    "decode": {"rows": 1, "columns": 12_288},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path, mode: str, phase: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("kind") != "qwen35-9b-inductor-ablation":
        raise ValueError(f"{resolved}: unexpected report kind")
    if payload.get("mode") != mode:
        raise ValueError(f"{resolved}: expected mode {mode!r}")
    expected = PHASES[phase]
    if payload.get("shape") != [expected["rows"], expected["columns"]]:
        raise ValueError(f"{resolved}: shape is not the frozen {phase} geometry")
    if "output_max_abs_vs_eager_formula" in payload:
        raise ValueError(
            f"{resolved}: output comparison makes the performance input non-pure"
        )
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{resolved}: missing profiler record")
    kernels = profile.get("kernel_names")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError(f"{resolved}: missing kernel names")
    return {
        "mode": mode,
        "phase": phase,
        "shape": payload["shape"],
        "warm_call_ms": float(payload["warm_call_ms"]),
        "cold_first_call_ms": float(payload["cold_first_call_ms"]),
        "kernel_event_count": int(profile["kernel_event_count"]),
        "kernel_names": [str(name) for name in kernels],
        "warmup_calls": int(payload["warmup_calls"]),
        "timed_calls": int(payload["timed_calls"]),
        "raw_report": resolved.relative_to(ROOT).as_posix(),
        "raw_report_sha256": sha256(resolved),
    }


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "cannot resolve git HEAD")
    return result.stdout.strip()


def summarize(records: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase in PHASES:
        values = records[phase]
        eager = values["eager"]
        nv = values["inductor-nv"]
        pypto = values["pypto"]
        eager_warm = float(eager["warm_call_ms"])
        nv_warm = float(nv["warm_call_ms"])
        pypto_warm = float(pypto["warm_call_ms"])
        eager_cold = float(eager["cold_first_call_ms"])
        nv_cold = float(nv["cold_first_call_ms"])
        pypto_cold = float(pypto["cold_first_call_ms"])
        eager_launches = int(eager["kernel_event_count"])
        compiled_launches = int(nv["kernel_event_count"])
        if eager_launches <= 0 or compiled_launches <= 0:
            raise ValueError(f"{phase}: launch counts must be positive")
        phases[phase] = {
            "geometry": PHASES[phase],
            "eager": eager,
            "inductor_nv": nv,
            "pypto": pypto,
            "derived": {
                "inductor_nv_speedup_vs_eager_percent": (
                    eager_warm / nv_warm - 1.0
                )
                * 100.0,
                "pypto_speedup_vs_eager_percent": (eager_warm / pypto_warm - 1.0)
                * 100.0,
                "pypto_compile_longer_than_inductor_nv_percent": (
                    pypto_cold / nv_cold - 1.0
                )
                * 100.0,
                "compiled_launch_reduction_vs_eager_percent": (
                    1.0 - compiled_launches / eager_launches
                )
                * 100.0,
            },
        }
    return {
        "schema": 2,
        "kind": "qwen35-9b-inductor-ablation",
        "status": "complete",
        "performance_only": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model": "Qwen3.5-9B",
            "operator": "SwiGLU pointwise",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "prefill_rows": PHASES["prefill"]["rows"],
            "decode_rows": PHASES["decode"]["rows"],
            "columns": PHASES["prefill"]["columns"],
            "denominator": "one profiled operator invocation",
        },
        "source": {
            "repository_head": git_head(),
            "runner": "benchmarks/release/inductor_ablation.py",
            "runner_sha256": sha256(ROOT / "benchmarks/release/inductor_ablation.py"),
            "source_lock_sha256": sha256(ROOT / "vendor/source-lock.json"),
        },
        "phases": phases,
        "interpretation": {
            "whole_model_speedup": None,
            "launch_reduction_scope": "SwiGLU operator only",
            "cold_time_scope": "first operator call wall time, including compilation",
            "formula_speedup_percent": "(eager warm / mode warm - 1) * 100",
            "formula_compile_overhead_percent": "(PyPTO cold / NV cold - 1) * 100",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        for mode in MODES:
            parser.add_argument(
                f"--{phase}-{mode.replace('-', '_')}",
                type=Path,
                required=True,
                dest=f"{phase}_{mode.replace('-', '_')}",
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: dict[str, dict[str, dict[str, object]]] = {}
    for phase in PHASES:
        records[phase] = {}
        for mode in MODES:
            records[phase][mode] = load_report(
                getattr(args, f"{phase}_{mode.replace('-', '_')}"), mode, phase
            )
    result = summarize(records)
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
