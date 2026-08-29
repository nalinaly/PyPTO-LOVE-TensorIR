"""CPU-only persistent ArtifactCache contracts for Inductor pointwise codegen."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace
import threading

import pytest

from pypto_plugins.errors import StrictCoverageError
from pypto_plugins.torch import pointwise_codegen as pc


@pytest.fixture(autouse=True)
def _isolated_artifact_cache_state(monkeypatch):
    monkeypatch.setattr(pc, "_ARTIFACT_CACHE_LOCK", threading.Lock())
    monkeypatch.setattr(pc, "_ARTIFACT_CACHE_STATE", None)


def _configured_compiler(created: list[object]) -> SimpleNamespace:
    class FakeArtifactCache:
        def __init__(self, root: str) -> None:
            self.root = root
            created.append(self)

    return SimpleNamespace(
        ArtifactCache=FakeArtifactCache,
        compile_structured_strict_cached=lambda *_args: object(),
    )


def _private_cache_root(tmp_path):
    root = tmp_path / "artifact-cache"
    root.mkdir(mode=0o700)
    return str(root.resolve())


def test_lazy_cache_reuses_one_handle_and_rebuilds_after_fork(
    monkeypatch, tmp_path
) -> None:
    created: list[object] = []
    compiler = _configured_compiler(created)
    root = _private_cache_root(tmp_path)
    monkeypatch.setenv("PYPTO_CACHE_DIR", root)
    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "1")
    process_id = [101]
    monkeypatch.setattr(pc.os, "getpid", lambda: process_id[0])

    parent = pc._artifact_cache_for(compiler)
    assert pc._artifact_cache_for(compiler) is parent
    process_id[0] = 202
    child = pc._artifact_cache_for(compiler)
    assert child is not parent
    assert pc._artifact_cache_for(compiler) is child
    assert [cache.root for cache in created] == [root, root]


def test_missing_cache_is_uncached_only_outside_strict_coverage(monkeypatch) -> None:
    calls = []
    compiler = SimpleNamespace(
        compile_structured_strict=lambda *args: calls.append(args) or "uncached"
    )
    monkeypatch.delenv("PYPTO_CACHE_DIR", raising=False)
    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "0")
    assert pc._artifact_cache_for(compiler) is None
    assert pc._compile_structured(compiler, "p", "r", "s") == "uncached"
    assert calls == [("p", "r", "s")]

    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "1")
    with pytest.raises(StrictCoverageError, match="strict coverage requires"):
        pc._compile_structured(compiler, "p", "r", "s")
    assert calls == [("p", "r", "s")]


def test_configured_cache_rejects_legacy_api_without_fallback(
    monkeypatch, tmp_path
) -> None:
    uncached_calls = []
    compiler = SimpleNamespace(
        ArtifactCache=lambda _root: object(),
        compile_structured_strict=lambda *args: uncached_calls.append(args),
    )
    monkeypatch.setenv("PYPTO_CACHE_DIR", _private_cache_root(tmp_path))
    with pytest.raises(
        StrictCoverageError,
        match="requires compile_structured_strict_cached",
    ):
        pc._compile_structured(compiler, "p", "r", "s")
    assert uncached_calls == []


def test_configured_cache_paths_and_constructor_fail_closed(
    monkeypatch, tmp_path
) -> None:
    created: list[object] = []
    compiler = _configured_compiler(created)

    monkeypatch.setenv("PYPTO_CACHE_DIR", "relative/cache")
    with pytest.raises(StrictCoverageError, match="absolute canonical"):
        pc._artifact_cache_for(compiler)

    missing = (tmp_path / "missing").resolve()
    monkeypatch.setenv("PYPTO_CACHE_DIR", str(missing))
    with pytest.raises(StrictCoverageError, match="missing or inaccessible"):
        pc._artifact_cache_for(compiler)

    real_root = tmp_path / "real-cache"
    real_root.mkdir(mode=0o700)
    symlink_root = tmp_path / "cache-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv("PYPTO_CACHE_DIR", str(symlink_root.absolute()))
    with pytest.raises(StrictCoverageError, match="without symlinks"):
        pc._artifact_cache_for(compiler)

    rejected_root = tmp_path / "rejected-cache"
    rejected_root.mkdir(mode=0o700)
    rejecting_compiler = SimpleNamespace(
        ArtifactCache=lambda _root: (_ for _ in ()).throw(
            RuntimeError("owner or mode rejected")
        ),
        compile_structured_strict_cached=lambda *_args: object(),
    )
    monkeypatch.setenv("PYPTO_CACHE_DIR", str(rejected_root.resolve()))
    with pytest.raises(StrictCoverageError, match="owner or mode rejected"):
        pc._artifact_cache_for(rejecting_compiler)


def test_cache_root_and_compiler_identity_cannot_change_in_process(
    monkeypatch, tmp_path
) -> None:
    first_root = _private_cache_root(tmp_path)
    second_path = tmp_path / "second-cache"
    second_path.mkdir(mode=0o700)
    second_root = str(second_path.resolve())
    compiler = _configured_compiler([])
    monkeypatch.setenv("PYPTO_CACHE_DIR", first_root)
    pc._artifact_cache_for(compiler)

    monkeypatch.setenv("PYPTO_CACHE_DIR", second_root)
    with pytest.raises(StrictCoverageError, match="changed after"):
        pc._artifact_cache_for(compiler)

    monkeypatch.setenv("PYPTO_CACHE_DIR", first_root)
    with pytest.raises(StrictCoverageError, match="compiler module changed"):
        pc._artifact_cache_for(_configured_compiler([]))


def test_full_cache_key_and_dispositions_are_validated() -> None:
    cache_key = "a" * 64
    assert pc._persistent_cache_key(
        SimpleNamespace(cache_key_digest=cache_key)
    ) == cache_key
    with pytest.raises(StrictCoverageError, match="invalid full cache key"):
        pc._persistent_cache_key(SimpleNamespace(cache_key_digest="a" * 16))

    for disposition in sorted(pc._PERSISTENT_CACHE_DISPOSITIONS):
        assert pc._persistent_cache_disposition(
            SimpleNamespace(name=disposition)
        ) == disposition
    with pytest.raises(StrictCoverageError, match="invalid disposition"):
        pc._persistent_cache_disposition(
            SimpleNamespace(name="CacheHitAfterWait")
        )


def test_cache_observation_does_not_change_source_or_artifact_identity(
    monkeypatch,
) -> None:
    source = "@pl.jit\ndef generated_pointwise_kernel():\n    pass\n"
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    common = dict(
        kernel_name="pypto_inductor_4444444444444444",
        entry_name="entry",
        build_spec_sha256="1" * 64,
        artifact_sha256="2" * 64,
        cubin_sha256="3" * 64,
        cubin_bytes=128,
        grid=(1, 1, 1),
        argument_count=3,
        workspace_bytes=0,
        fallback_used=False,
        pypto_source=source,
        pypto_source_sha256=source_sha,
        cache_identity_sha256="4" * 64,
        artifact_cache_key_sha256="5" * 64,
        source_node="torch-inductor:4444444444444444",
        dso_sha256="6" * 64,
    )
    cold = pc.PointwiseArtifact(
        **common,
        artifact_cache_disposition="CompiledAndPublished",
    )
    warm = replace(cold, artifact_cache_disposition="CacheHit")
    assert cold.kernel_name == warm.kernel_name
    assert cold.source_node == warm.source_node
    assert cold.cache_identity_sha256 == warm.cache_identity_sha256
    assert cold.pypto_source_sha256 == warm.pypto_source_sha256
    assert cold.artifact_sha256 == warm.artifact_sha256

    monkeypatch.setattr(pc, "_COMPILE_CACHE", {("cold",): cold})
    cold_snapshot = pc.artifact_cache_snapshot()
    monkeypatch.setattr(pc, "_COMPILE_CACHE", {("warm",): warm})
    warm_snapshot = pc.artifact_cache_snapshot()
    assert cold_snapshot[0][:2] == warm_snapshot[0][:2]
    assert cold_snapshot[0][2] == "CompiledAndPublished"
    assert warm_snapshot[0][2] == "CacheHit"

    evidence_common = dict(
        kernel_name=cold.kernel_name,
        entry_name=cold.entry_name,
        source_node=cold.source_node,
        cache_identity_sha256=cold.cache_identity_sha256,
        artifact_cache_key_sha256=cold.artifact_cache_key_sha256,
        build_spec_sha256=cold.build_spec_sha256,
        artifact_id="pypto-artifact-v1:" + "7" * 64,
        artifact_sha256=cold.artifact_sha256,
        cubin_sha256=cold.cubin_sha256,
        dso_sha256=cold.dso_sha256,
        pypto_source=cold.pypto_source,
        pypto_source_sha256=cold.pypto_source_sha256,
        wrapper_launch_sources=(),
    )
    cold_evidence = pc.PointwiseSourceEvidence(
        **evidence_common,
        artifact_cache_disposition="CompiledAndPublished",
    ).to_dict()
    warm_evidence = pc.PointwiseSourceEvidence(
        **evidence_common,
        artifact_cache_disposition="CacheHit",
    ).to_dict()
    cold_observation = cold_evidence.pop("artifact_cache_observation")
    warm_observation = warm_evidence.pop("artifact_cache_observation")
    assert cold_evidence == warm_evidence
    assert cold_observation["cache_key_sha256"] == warm_observation[
        "cache_key_sha256"
    ]
    assert cold_observation["disposition"] == "CompiledAndPublished"
    assert warm_observation["disposition"] == "CacheHit"
