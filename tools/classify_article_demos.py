#!/usr/bin/env python3
"""Classify imported article entry points for the NVIDIA compatibility path.

The imported files under ``demo/pypto-lib`` are immutable.  This tool records
an external execution policy instead of adding compatibility conditionals to
the upstream source.  Hardware-facing APIs are skipped with source-line
evidence; bounded computational examples are assigned an adapter mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo" / "pypto-lib"
MANIFEST = DEMO_ROOT / "SOURCE_MANIFEST.json"
ARTICLE_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"
DEFAULT_OUTPUT = ROOT / "state/evidence/article-demo-compatibility-policy-current.json"

# These markers identify APIs whose semantics depend on an Ascend device,
# CANN/CCE, NPU runtime, or distributed window/communication runtime.  The
# marker is matched in the entry-point source itself; the shared golden
# harness is deliberately not treated as an algorithmic hardware dependency.
HARDWARE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("distributed-language-api", r"pypto\.language\.distributed|\bpld\."),
    ("distributed-program-api", r"pypto\.ir\.distributed_compiled_program|DistributedConfig"),
    ("ascend-runtime-api", r"pypto\.runtime|pypto\.backend|simpler_setup|_task_interface|task_interface"),
    ("npu-runtime-api", r"torch_npu|\bacl\b|\btbe\b|\bte\b"),
    ("cce-kernel-tree", r"paged_attention_cce|/vendor/|\\vendor\\"),
    ("collective-or-device-control", r"\ballreduce\b|\bget_comm_ctx\b|\bwindow_buffer\b"),
)

TEACHING_ADAPTERS: dict[str, dict[str, str]] = {
    "examples/advanced/gemm_eltwise.py": {
        "mode": "computational-cuda-reference",
        "adapter": "gemm_eltwise_reference",
    },
    "examples/advanced/multi_proj.py": {
        "mode": "computational-cuda-reference",
        "adapter": "multi_proj_reference",
    },
    "examples/advanced/topk.py": {
        "mode": "computational-cuda-reference",
        "adapter": "topk_reference",
    },
    "examples/beginner/hello_world.py": {
        "mode": "strict-pypto-nvidia",
        "adapter": "hello_world_strict",
    },
    "examples/beginner/matmul.py": {
        "mode": "computational-cuda-reference",
        "adapter": "matmul_reference",
    },
    "examples/intermediate/gemm.py": {
        "mode": "computational-cuda-reference",
        "adapter": "gemm_reference",
    },
    "examples/intermediate/layer_norm.py": {
        "mode": "computational-cuda-reference",
        "adapter": "layer_norm_reference",
    },
    "examples/intermediate/rms_norm.py": {
        "mode": "computational-cuda-reference",
        "adapter": "rms_norm_reference",
    },
    "examples/intermediate/rope.py": {
        "mode": "computational-cuda-reference",
        "adapter": "rope_reference",
    },
    "examples/intermediate/softmax.py": {
        "mode": "computational-cuda-reference",
        "adapter": "softmax_reference",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def corpus_sha256(manifest: dict[str, Any]) -> str:
    """Hash every imported file with path boundaries, not just the manifest."""
    digest = hashlib.sha256()
    for record in sorted(
        manifest.get("files", []), key=lambda item: str(item.get("path", ""))
    ):
        relative = str(record["path"])
        path = (DEMO_ROOT / relative).resolve()
        if DEMO_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"imported corpus file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("kind") != "article-demo-provenance":
        raise ValueError("unexpected article demo manifest kind")
    if payload.get("upstream", {}).get("commit") != ARTICLE_COMMIT:
        raise ValueError("article demo manifest is not article-time locked")
    if not isinstance(payload.get("entrypoints"), list):
        raise ValueError("article demo manifest has no entrypoint inventory")
    return payload


def _marker_hits(source: str) -> list[dict[str, object]]:
    lines = source.splitlines()
    hits: list[dict[str, object]] = []
    for marker, expression in HARDWARE_PATTERNS:
        pattern = re.compile(expression, re.IGNORECASE)
        for line_number, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append({"marker": marker, "line": line_number, "text": line.strip()[:240]})
                break
    return hits


def _source_record(relative: str, manifest: dict[str, Any]) -> dict[str, object]:
    path = (DEMO_ROOT / relative).resolve()
    if DEMO_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"entrypoint escaped or is missing: {relative}")
    record = next((item for item in manifest["files"] if item.get("path") == relative), None)
    if not isinstance(record, dict):
        raise ValueError(f"entrypoint is not in source manifest: {relative}")
    observed = sha256_file(path)
    if observed != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        raise ValueError(f"imported source changed: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def classify_entry(item: dict[str, Any], manifest: dict[str, Any]) -> dict[str, object]:
    relative = str(item["path"])
    path = DEMO_ROOT / relative
    source = path.read_text(encoding="utf-8")
    source_record = _source_record(relative, manifest)
    original_policy = str(item.get("execution_policy", "runnable"))
    hits = _marker_hits(source)

    if original_policy == "excluded-draft":
        mode = "source-excluded"
        reason = "draft entrypoint retained for provenance but not an article execution target"
        adapter = None
    elif original_policy == "ascend-cce-only":
        mode = "hardware-api-skipped"
        reason = "entrypoint is explicitly Ascend CCE-only in the imported manifest"
        adapter = None
    elif hits:
        mode = "hardware-api-skipped"
        reason = "entrypoint directly references an Ascend/NPU/CCE/distributed hardware API"
        adapter = None
    elif relative in TEACHING_ADAPTERS:
        config = TEACHING_ADAPTERS[relative]
        mode = config["mode"]
        reason = "bounded computational teaching example; compatibility adapter is outside imported source"
        adapter = config["adapter"]
    else:
        mode = "computational-unmapped"
        reason = "computational entrypoint has no bounded NVIDIA adapter; retained and reported explicitly"
        adapter = None

    return {
        "path": relative,
        "source": source_record,
        "original_execution_policy": original_policy,
        "compatibility_mode": mode,
        "adapter": adapter,
        "reason": reason,
        "hardware_api_evidence": hits,
    }


def build_policy(manifest: dict[str, Any]) -> dict[str, object]:
    entries = [classify_entry(item, manifest) for item in manifest["entrypoints"]]
    counts: dict[str, int] = {}
    for entry in entries:
        mode = str(entry["compatibility_mode"])
        counts[mode] = counts.get(mode, 0) + 1
    return {
        "schema": 2,
        "kind": "article-demo-compatibility-policy",
        "policy_revision": "nvidia-computational-v1-20260831",
        "article_url": manifest["article"]["url"],
        "upstream_commit": manifest["upstream"]["commit"],
        "manifest_path": "demo/pypto-lib/SOURCE_MANIFEST.json",
        "manifest_sha256": sha256_file(MANIFEST),
        "corpus_sha256": corpus_sha256(manifest),
        "source_copy_policy": manifest["upstream"]["copy_policy"],
        "entrypoint_count": len(entries),
        "counts": counts,
        "entries": entries,
        "acceptance": {
            "hardware_api_entries_are_skipped": True,
            "computational_reference_is_not_strict_compiler_evidence": True,
            "strict_pypto_entries_require_artifact_and_golden": True,
            "imported_source_must_remain_byte_identical": True,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(canonical_json(payload))
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_policy(load_manifest())
    write_json(args.output.resolve(), payload)
    print(json.dumps({"status": "complete", "counts": payload["counts"]}, ensure_ascii=False, sort_keys=True))
    print(f"article demo policy: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
