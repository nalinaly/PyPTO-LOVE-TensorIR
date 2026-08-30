"""Immutable workload and report helpers for the Qwen3.5 release."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_SIZE = "9B"
MODEL_ID = "Qwen/Qwen3.5-9B"
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
RAW_PROMPT_TOKEN_IDS = (
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
RAW_PROMPT_TOKENS = len(RAW_PROMPT_TOKEN_IDS)
CHAT_WORKLOAD_PATH = Path(__file__).with_name("chat_workload.json")
CHAT_WORKLOAD = json.loads(CHAT_WORKLOAD_PATH.read_text(encoding="utf-8"))
if CHAT_WORKLOAD.get("schema") != 1:
    raise RuntimeError("unknown chat workload schema")
if CHAT_WORKLOAD.get("human_prompt") != PROMPT:
    raise RuntimeError("chat workload prompt differs from the release prompt")
CHAT_TEMPLATE_KWARGS = dict(CHAT_WORKLOAD["template_kwargs"])
_DEFAULT_CHAT_RECORD = CHAT_WORKLOAD["models"]["Qwen3.5-9B"]
PROMPT_TOKEN_IDS = tuple(_DEFAULT_CHAT_RECORD["input_token_ids"])
PROMPT_TOKENS = len(PROMPT_TOKEN_IDS)
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
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_829

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


class ReleaseContractError(RuntimeError):
    """A release input or result violates the frozen public contract."""


@dataclass(frozen=True)
class Qwen35ModelSpec:
    """One supported text-model release target and its derived report contract."""

    manifest_name: str
    model_id: str
    model_size: str
    report_stem: str
    num_hidden_layers: int

    @property
    def expected_inductor_calls(self) -> int:
        """One fused Inductor pointwise launch per layer and generated token."""

        return self.num_hidden_layers * OUTPUT_TOKENS

    def record(self) -> dict[str, object]:
        return {
            "manifest_name": self.manifest_name,
            "model_id": self.model_id,
            "model_size": self.model_size,
            "report_stem": self.report_stem,
            "num_hidden_layers": self.num_hidden_layers,
            "expected_inductor_calls": self.expected_inductor_calls,
        }


QWEN35_MODEL_SPECS = {
    "Qwen3.5-0.8B": Qwen35ModelSpec(
        manifest_name="Qwen3.5-0.8B",
        model_id="Qwen/Qwen3.5-0.8B",
        model_size="0.8B",
        report_stem="qwen35-0.8b",
        num_hidden_layers=24,
    ),
    "Qwen3.5-9B": Qwen35ModelSpec(
        manifest_name="Qwen3.5-9B",
        model_id=MODEL_ID,
        model_size=MODEL_SIZE,
        report_stem="qwen35-9b",
        num_hidden_layers=32,
    ),
}
DEFAULT_MODEL_SPEC = QWEN35_MODEL_SPECS["Qwen3.5-9B"]


def resolve_qwen35_model_spec(root: Path, model_path: Path) -> Qwen35ModelSpec:
    """Resolve and verify a supported model from its manifest and config."""

    root = root.resolve()
    resolved_model = model_path.resolve(strict=True)
    manifest_path = (root / "models/MANIFEST.json").resolve(strict=True)
    manifest = read_json(manifest_path)
    models = manifest.get("models")
    if manifest.get("schema") != 1 or not isinstance(models, dict):
        raise ReleaseContractError("model MANIFEST has an unknown schema")
    matches = []
    for name, record in models.items():
        if not isinstance(record, dict) or type(record.get("destination")) is not str:
            continue
        destination = (root / str(record["destination"])).resolve()
        if destination == resolved_model:
            matches.append((name, record))
    if len(matches) != 1:
        raise ReleaseContractError(
            "model path must match exactly one MANIFEST destination: "
            f"{resolved_model}"
        )
    manifest_name, manifest_record = matches[0]
    spec = QWEN35_MODEL_SPECS.get(str(manifest_name))
    if spec is None:
        raise ReleaseContractError(f"unsupported Qwen3.5 release model: {manifest_name}")
    if manifest_record.get("repository_id") != spec.model_id:
        raise ReleaseContractError(
            f"model repository ID differs for {spec.manifest_name}"
        )
    config_path = resolved_model / "config.json"
    config = read_json(config_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ReleaseContractError(f"model text_config is missing: {config_path}")
    if text_config.get("num_hidden_layers") != spec.num_hidden_layers:
        raise ReleaseContractError(
            f"model layer count differs for {spec.manifest_name}: "
            f"expected {spec.num_hidden_layers}, "
            f"got {text_config.get('num_hidden_layers')!r}"
        )
    return spec


def workload_record(
    model_spec: Qwen35ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, object]:
    try:
        chat = CHAT_WORKLOAD["models"][model_spec.manifest_name]
    except KeyError as error:
        raise ReleaseContractError(
            f"chat workload has no record for {model_spec.manifest_name}"
        ) from error
    if chat.get("model_id") != model_spec.model_id:
        raise ReleaseContractError(
            f"chat workload model ID differs for {model_spec.manifest_name}"
        )
    return {
        "workload_kind": "qwen35-chat-template-thinking",
        "model_id": model_spec.model_id,
        "model_size": model_spec.model_size,
        "prompt": PROMPT,
        "prompt_token_ids": list(chat["input_token_ids"]),
        "prompt_tokens": len(chat["input_token_ids"]),
        "raw_prompt_token_ids": list(RAW_PROMPT_TOKEN_IDS),
        "raw_prompt_tokens": RAW_PROMPT_TOKENS,
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "rendered_input": chat["rendered_input"],
        "tokenizer_files": dict(chat["tokenizer_files"]),
        "output_tokens": OUTPUT_TOKENS,
        "concurrency": CONCURRENCY,
        "greedy": True,
        "ignore_eos": True,
    }


def verify_chat_workload(
    model_path: Path,
    model_spec: Qwen35ModelSpec | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Re-render and hash-check the pinned chat-template workload at runtime."""

    model_path = model_path.resolve(strict=True)
    if model_spec is None:
        model_spec = resolve_qwen35_model_spec(
            Path(__file__).resolve().parents[2], model_path
        )
    expected = workload_record(model_spec)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ReleaseContractError(
            "chat workload verification requires the pinned Transformers tokenizer"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        return_tensors=None,
        **dict(expected["chat_template_kwargs"]),
    )
    ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise ReleaseContractError("chat template returned an invalid input ID sequence")
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if ids != expected["prompt_token_ids"]:
        raise ReleaseContractError(
            f"chat template token IDs differ for {model_spec.manifest_name}"
        )
    if rendered != expected["rendered_input"]:
        raise ReleaseContractError(
            f"chat template rendering differs for {model_spec.manifest_name}"
        )
    observed_files = {}
    for name, expected_hash in dict(expected["tokenizer_files"]).items():
        path = (model_path / name).resolve(strict=True)
        if path.parent != model_path:
            raise ReleaseContractError(f"chat tokenizer file escaped model directory: {name}")
        observed_hash = sha256_file(path)
        observed_files[name] = observed_hash
        if observed_hash != expected_hash:
            raise ReleaseContractError(
                f"chat tokenizer file hash differs for {model_spec.manifest_name}: {name}"
            )
    resolution = {
        "verified": True,
        "model": model_spec.manifest_name,
        "model_path": str(model_path),
        "input_token_count": len(ids),
        "input_ids_sha256": canonical_json_sha256(ids),
        "rendered_input_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "tokenizer_files": observed_files,
    }
    return expected, resolution


def validate_workload(
    value: Mapping[str, object],
    model_spec: Qwen35ModelSpec = DEFAULT_MODEL_SPEC,
) -> None:
    expected = workload_record(model_spec)
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
    if any(not math.isfinite(value) for value in observed):
        raise ReleaseContractError("a distribution contains a non-finite sample")
    p90 = nearest_rank(observed, 0.90)
    p99 = nearest_rank(observed, 0.99)
    return {
        "min": min(observed),
        "mean": statistics.fmean(observed),
        "p50": statistics.median(observed),
        "p90": p90,
        "p90_nearest_rank": p90,
        "p99": p99,
        "p99_nearest_rank": p99,
        "max": max(observed),
    }


def _bootstrap_seed(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return BOOTSTRAP_SEED ^ int.from_bytes(digest[:8], "big")


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    salt: str,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Deterministic percentile-bootstrap CI over fresh process starts."""

    observed = [float(value) for value in values]
    if not observed:
        raise ReleaseContractError("a bootstrap interval requires at least one start")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    seed = _bootstrap_seed(salt)
    generator = random.Random(seed)
    count = len(observed)
    estimates = [
        statistics.median(observed[generator.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "lower": nearest_rank(estimates, alpha),
        "upper": nearest_rank(estimates, 1.0 - alpha),
        "method": "nonparametric percentile bootstrap over fresh process starts",
        "resamples": resamples,
        "seed": seed,
    }


def fresh_start_summary(values: Iterable[float], *, salt: str) -> dict[str, object]:
    """Summarize one scalar estimate from each independent fresh start."""

    observed = [float(value) for value in values]
    summary: dict[str, object] = distribution(observed)
    summary.update(
        {
            "sample_count": len(observed),
            "sample_unit": "fresh_process_start",
            "headline_estimator": "median_of_fresh_start_estimates",
            "median_bootstrap_95ci": bootstrap_median_ci(observed, salt=salt),
        }
    )
    return summary


def bootstrap_median_comparison_ci(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    operation: str,
    salt: str,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Bootstrap a difference or ratio of independent start-level medians."""

    candidate_values = [float(value) for value in candidate]
    baseline_values = [float(value) for value in baseline]
    if not candidate_values or not baseline_values:
        raise ReleaseContractError("a comparison interval requires both start sets")
    if operation not in {"difference", "ratio"}:
        raise ValueError(f"unsupported bootstrap comparison: {operation}")
    if operation == "ratio" and any(value == 0.0 for value in baseline_values):
        raise ReleaseContractError("a bootstrap ratio has a zero baseline start")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    seed = _bootstrap_seed(salt)
    generator = random.Random(seed)
    candidate_count = len(candidate_values)
    baseline_count = len(baseline_values)
    estimates = []
    for _ in range(resamples):
        candidate_median = statistics.median(
            candidate_values[generator.randrange(candidate_count)]
            for _ in range(candidate_count)
        )
        baseline_median = statistics.median(
            baseline_values[generator.randrange(baseline_count)]
            for _ in range(baseline_count)
        )
        estimates.append(
            candidate_median - baseline_median
            if operation == "difference"
            else candidate_median / baseline_median
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "lower": nearest_rank(estimates, alpha),
        "upper": nearest_rank(estimates, 1.0 - alpha),
        "method": (
            "independent nonparametric percentile bootstrap over fresh process starts"
        ),
        "operation": operation,
        "resamples": resamples,
        "seed": seed,
    }


def fresh_start_methodology(
    within_start_unit: str = "requests",
) -> dict[str, object]:
    if not within_start_unit:
        raise ValueError("within_start_unit must be nonempty")
    return {
        "experimental_unit": "fresh_process_start",
        "within_start_estimator": f"median across {within_start_unit}",
        "headline_estimator": "median across fresh-start estimates",
        "tail_estimators": ["p90_nearest_rank", "p99_nearest_rank"],
        "uncertainty": "95% percentile bootstrap CI resampling fresh starts",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pooling_requests_across_starts": False,
    }


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
