from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_release_models", ROOT / "tools/download_release_models.py"
)
assert SPEC is not None and SPEC.loader is not None
models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = models
SPEC.loader.exec_module(models)


def test_public_manifest_is_portable_and_pinned() -> None:
    manifest = models.load_manifest()
    assert set(manifest["models"]) == {"Qwen3.5-0.8B", "Qwen3.5-9B"}
    assert "/home/" not in models.MANIFEST.read_text(encoding="utf-8")


def test_verify_model_checks_bytes_hash_and_links(tmp_path: Path) -> None:
    payload = b"model"
    destination = tmp_path / "models/Qwen3.5-test"
    destination.mkdir(parents=True)
    model = destination / "model.safetensors"
    model.write_bytes(payload)
    spec = {
        "repository_id": "Qwen/Qwen3.5-test",
        "revision": "a" * 40,
        "destination": "models/Qwen3.5-test",
        "files": {
            model.name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    assert models.verify_model(tmp_path, spec)["status"] == "verified"
    model.write_bytes(b"changed")
    with pytest.raises(models.ModelReleaseError, match="identity differs"):
        models.verify_model(tmp_path, spec)


def test_verify_model_rejects_extra_file(tmp_path: Path) -> None:
    destination = tmp_path / "models/Qwen3.5-test"
    destination.mkdir(parents=True)
    payload = b"model"
    (destination / "model.safetensors").write_bytes(payload)
    (destination / "extra").write_text("unexpected")
    spec = {
        "repository_id": "Qwen/Qwen3.5-test",
        "revision": "a" * 40,
        "destination": "models/Qwen3.5-test",
        "files": {
            "model.safetensors": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    with pytest.raises(models.ModelReleaseError, match="untracked"):
        models.verify_model(tmp_path, spec)
