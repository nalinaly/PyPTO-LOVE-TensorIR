from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


from benchmarks.release.evidence_identity import collect_model_identity
from benchmarks.release.workload import ReleaseContractError


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_fixture(root: Path, model_name: str = "Qwen3.5-9B") -> Path:
    model = root / f"models/{model_name}"
    model.mkdir(parents=True)
    files = {"config.json": b'{"model":"qwen"}\n', "weights.bin": b"abcdefgh"}
    for name, value in files.items():
        (model / name).write_bytes(value)
    manifest = {
        "schema": 1,
        "models": {
            model_name: {
                "destination": f"models/{model_name}",
                "repository_id": f"Qwen/{model_name}",
                "revision": "a" * 40,
                "files": {
                    name: {"bytes": len(value), "sha256": _sha256(value)}
                    for name, value in files.items()
                },
            }
        },
    }
    (root / "models/MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return model


def test_model_identity_hashes_every_manifest_file(tmp_path: Path) -> None:
    model = _model_fixture(tmp_path)
    identity = collect_model_identity(tmp_path, model)
    assert [item["path"] for item in identity["files"]] == [
        "config.json",
        "weights.bin",
    ]
    assert identity["files"][1]["sha256"] == _sha256(b"abcdefgh")
    assert len(identity["identity_sha256"]) == 64


def test_model_identity_infers_small_model_from_manifest_destination(
    tmp_path: Path,
) -> None:
    model = _model_fixture(tmp_path, "Qwen3.5-0.8B")
    identity = collect_model_identity(tmp_path, model)
    assert identity["name"] == "Qwen3.5-0.8B"
    assert identity["repository_id"] == "Qwen/Qwen3.5-0.8B"


def test_model_identity_rejects_same_size_content_drift(tmp_path: Path) -> None:
    model = _model_fixture(tmp_path)
    (model / "weights.bin").write_bytes(b"ABCDEFGH")
    with pytest.raises(ReleaseContractError, match="SHA-256 differs"):
        collect_model_identity(tmp_path, model)


def test_model_identity_rejects_unmanifested_and_linked_files(tmp_path: Path) -> None:
    model = _model_fixture(tmp_path)
    (model / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ReleaseContractError, match="file set differs"):
        collect_model_identity(tmp_path, model)
    (model / "extra.bin").unlink()
    linked = tmp_path / "linked.bin"
    linked.hardlink_to(model / "weights.bin")
    with pytest.raises(ReleaseContractError, match="single-link"):
        collect_model_identity(tmp_path, model)
