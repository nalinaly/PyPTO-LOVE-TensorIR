"""Immutable workload and report helpers for the Qwen3.5 release."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_SIZE = "9B"
MODEL_ID = "Qwen/Qwen3.5-9B"
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
PROMPT_TOKEN_IDS = (
    144277,
    103426,
    108169,
    95967,
    236,
    124094,
    26076,
    96212,
    103182,
    108076,
    96799,
    24273,
    95761,
    104224,
    109276,
    95726,
    111104,
    115110,
    10992,
)
PROMPT_TOKENS = 19
OUTPUT_TOKENS = 64
CONCURRENCY = 1
CPU_JOBS = 24
MEASURED_REQUESTS = 10
UNTIMED_WARMUPS = 2
COMPILE_WARMUPS = 1
SAMPLE_INTERVAL_MS = 100
LANES = ("pypto", "sglang-matched", "sglang-optimized")
PERFORMANCE_SCHEDULE = (
    "pypto",
    "sglang-matched",
    "sglang-matched",
    "pypto",
    "sglang-optimized",
    "pypto",
    "pypto",
    "sglang-optimized",
    "sglang-matched",
    "sglang-optimized",
    "sglang-optimized",
    "sglang-matched",
)
PROFILE_SCHEDULE = (
    "pypto",
    "sglang-matched",
    "sglang-optimized",
    "sglang-matched",
    "sglang-optimized",
    "pypto",
    "sglang-optimized",
    "pypto",
    "sglang-matched",
)

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


class ReleaseContractError(RuntimeError):
    """A release input or result violates the frozen public contract."""


def workload_record() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "model_size": MODEL_SIZE,
        "prompt": PROMPT,
        "prompt_token_ids": list(PROMPT_TOKEN_IDS),
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "concurrency": CONCURRENCY,
        "greedy": True,
        "ignore_eos": True,
    }


def validate_workload(value: Mapping[str, object]) -> None:
    expected = workload_record()
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ReleaseContractError(
                f"workload field {key!r} differs: "
                f"expected {expected_value!r}, got {value.get(key)!r}"
            )


def workspace_root(anchor: str | os.PathLike[str]) -> Path:
    return Path(anchor).resolve().parents[2]


def require_run_directory(root: Path) -> tuple[str, Path]:
    run_id = os.environ.get("PYPTO_RUN_ID")
    if run_id is None or _RUN_ID.fullmatch(run_id) is None:
        raise ReleaseContractError("worker requires a bounded-controller PYPTO_RUN_ID")
    runs = (root / "runs").resolve()
    run_dir = (runs / run_id).resolve()
    if run_dir.parent != runs:
        raise ReleaseContractError("run directory escaped the workspace runs root")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def require_path_below_runs(root: Path, raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).resolve()
    runs = (root / "runs").resolve()
    if path != runs and runs not in path.parents:
        raise ReleaseContractError(f"release output must remain below {runs}")
    return path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ReleaseContractError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_revision(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        if path.suffix in {".json", ".model"}:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ReleaseContractError("a percentile requires at least one sample")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def distribution(values: Iterable[float]) -> dict[str, float]:
    observed = [float(value) for value in values]
    if not observed:
        raise ReleaseContractError("a distribution requires at least one sample")
    return {
        "min": min(observed),
        "mean": statistics.fmean(observed),
        "p50": statistics.median(observed),
        "p90_nearest_rank": nearest_rank(observed, 0.90),
        "p99_nearest_rank": nearest_rank(observed, 0.99),
        "max": max(observed),
    }


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
