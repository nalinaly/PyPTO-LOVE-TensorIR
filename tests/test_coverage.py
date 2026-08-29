from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import pypto_plugins.coverage as coverage_module
from pypto_plugins.coverage import (
    ALLOWED_PYPTO_PROVIDERS,
    ActivityKind,
    ArtifactRecord,
    CoverageAuditor,
    CoverageMode,
    EventScope,
    KernelEvent,
    KernelProvenance,
    ProvenanceOrigin,
    TRACE_COLLECTOR,
    TRACE_COLLECTOR_REVISION,
    TraceManifest,
    compute_artifact_registry_digest,
    compute_trace_digest,
)
from pypto_plugins.errors import StrictCoverageError
from pypto_plugins.sglang.inventory import CoverageProvider


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def artifact(provider: str, suffix: str = "test") -> ArtifactRecord:
    artifact_id = f"artifact:{provider}:{suffix}"
    return ArtifactRecord(
        artifact_id=artifact_id,
        artifact_sha256=_sha(artifact_id),
        provider=provider,
        kernel_name=f"pypto_{provider.split('.')[-1]}_{suffix}",
        source_node=f"fx:{suffix}",
        ir_region=f"region:{suffix}",
        compiler_revision="compiler-commit",
        kernels_revision=(
            "kernels-commit"
            if provider in {"pypto.matmul", "pypto.attention", "pypto.gdn"}
            else None
        ),
    )


def registered_event(
    artifact_record: ArtifactRecord,
    *,
    activity_id: str | None = None,
    calls: int = 1,
    gpu_time_ns: int = 100,
) -> KernelEvent:
    return KernelEvent(
        activity_id=activity_id or f"activity:{artifact_record.artifact_id}",
        scope=EventScope.MODEL_FORWARD,
        activity=ActivityKind.COMPUTE,
        provider=artifact_record.provider,
        kernel_name=artifact_record.kernel_name,
        call_count=calls,
        gpu_time_ns=gpu_time_ns,
        provenance=KernelProvenance(
            origin=ProvenanceOrigin.PYPTO_ARTIFACT_REGISTRY,
            artifact_id=artifact_record.artifact_id,
            artifact_sha256=artifact_record.artifact_sha256,
            compiler_revision=artifact_record.compiler_revision,
            kernels_revision=artifact_record.kernels_revision,
        ),
    )


def manifest(
    events: list[KernelEvent], artifacts: list[ArtifactRecord]
) -> TraceManifest:
    return TraceManifest(
        run_id="run:test",
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="model-commit",
        device_fingerprint="nvidia:sm120:gpu0:test",
        collector=TRACE_COLLECTOR,
        collector_revision=TRACE_COLLECTOR_REVISION,
        framework_profile="pypto",
        artifact_registry_digest=compute_artifact_registry_digest(artifacts),
        trace_digest=compute_trace_digest(events),
        activity_count=len(events),
        closed_world=True,
    )


def auditor(
    tmp_path: Path,
    events: list[KernelEvent],
    artifacts: list[ArtifactRecord],
    *,
    mode: CoverageMode = CoverageMode.STRICT,
    manifest_override: TraceManifest | None = None,
    name: str = "coverage.json",
) -> CoverageAuditor:
    return CoverageAuditor(
        mode=mode,
        report_path=tmp_path / name,
        manifest=manifest_override or manifest(events, artifacts),
        artifacts=artifacts,
    )


def record_all(instance: CoverageAuditor, events: list[KernelEvent]) -> None:
    for event in events:
        instance.record(event)


def test_allowed_provider_policy_is_fixed_and_complete() -> None:
    assert ALLOWED_PYPTO_PROVIDERS == {
        "pypto.generic",
        "pypto.tensorir",
        "pypto.matmul",
        "pypto.attention",
        "pypto.gdn",
    }


def test_static_sglang_inventory_provider_strings_are_policy_subset() -> None:
    inventory_providers = {
        provider.value
        for provider in CoverageProvider
        if provider is not CoverageProvider.HOST_ONLY
    }
    assert inventory_providers == ALLOWED_PYPTO_PROVIDERS - {"pypto.tensorir"}


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_kernel_event_rejects_non_exact_integer_call_count(bad: object) -> None:
    with pytest.raises(TypeError, match="call_count"):
        KernelEvent(
            "event",
            EventScope.MODEL_FORWARD,
            ActivityKind.COMPUTE,
            "external",
            "kernel",
            bad,  # type: ignore[arg-type]
            1,
        )


def test_kernel_event_validates_strings_enums_and_ranges() -> None:
    with pytest.raises(TypeError, match="scope"):
        KernelEvent("event", "model-forward", ActivityKind.COMPUTE, "x", "k", 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        KernelEvent("event", EventScope.MODEL_FORWARD, ActivityKind.COMPUTE, "x", "k", 0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        KernelEvent("event", EventScope.MODEL_FORWARD, ActivityKind.COMPUTE, "x", "k", 1, -1)
    with pytest.raises(ValueError, match="trimmed"):
        KernelEvent(" event ", EventScope.MODEL_FORWARD, ActivityKind.COMPUTE, "x", "k", 1, 1)


def test_provenance_and_artifact_validation() -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        KernelProvenance(ProvenanceOrigin.EXTERNAL, "external", "ABC")
    with pytest.raises(ValueError, match="kernels_revision"):
        ArtifactRecord(
            "artifact",
            _sha("artifact"),
            "pypto.gdn",
            "pypto_gdn_test",
            "fx",
            "ir",
            "compiler",
        )
    generic = artifact("pypto.generic")
    assert generic.kernels_revision is None
    assert artifact("pypto.gdn").kernels_revision == "kernels-commit"


def test_valid_strict_trace_passes_with_exact_non_vacuous_totals(tmp_path: Path) -> None:
    artifacts = [artifact(provider) for provider in sorted(ALLOWED_PYPTO_PROVIDERS)]
    events = [
        registered_event(item, calls=index + 1, gpu_time_ns=(index + 1) * 101)
        for index, item in enumerate(artifacts)
    ]
    instance = auditor(tmp_path, events, artifacts)
    record_all(instance, events)
    summary = instance.finalize(event_stream_complete=True)
    assert summary.strict_policy_passed is True
    assert summary.covered_calls == summary.total_calls == 15
    assert summary.covered_gpu_time_ns == summary.total_gpu_time_ns == 1515
    payload = json.loads(instance.report_path.read_text())
    assert payload["strict_policy_passed"] is True
    assert payload["observed_trace"]["trace_digest"] == payload["manifest"]["trace_digest"]
    assert payload["fallbacks"] == []


def test_development_fallback_is_retained_with_exact_arithmetic(tmp_path: Path) -> None:
    item = artifact("pypto.matmul")
    good = registered_event(item, calls=3, gpu_time_ns=600)
    fallback = KernelEvent(
        "activity:fallback",
        EventScope.MODEL_FORWARD,
        ActivityKind.COMPUTE,
        "cublasLt",
        "cublasLtMatmul",
        1,
        200,
        KernelProvenance(ProvenanceOrigin.EXTERNAL, "external:cublasLt"),
    )
    events = [good, fallback]
    instance = auditor(tmp_path, events, [item], mode=CoverageMode.DEVELOPMENT)
    record_all(instance, events)
    summary = instance.finalize(event_stream_complete=True)
    assert summary.strict_policy_passed is False
    assert (summary.covered_calls, summary.total_calls) == (3, 4)
    assert (summary.covered_gpu_time_ns, summary.total_gpu_time_ns) == (600, 800)
    payload = json.loads(instance.report_path.read_text())
    assert [entry["provider"] for entry in payload["fallbacks"]] == ["cublasLt"]


def test_strict_fallback_is_latched_and_writes_partial_report(tmp_path: Path) -> None:
    fallback = KernelEvent(
        "activity:triton",
        EventScope.MODEL_FORWARD,
        ActivityKind.COMPUTE,
        "triton",
        "triton_red_fused",
        1,
        200,
        KernelProvenance(ProvenanceOrigin.EXTERNAL, "external:triton"),
    )
    instance = auditor(tmp_path, [fallback], [])
    with pytest.raises(StrictCoverageError, match="triton") as caught:
        instance.record(fallback)
    assert caught.value.code == "fallback-provider"
    payload = json.loads(instance.report_path.read_text())
    assert payload["finalized"] is False
    assert payload["strict_policy_passed"] is False
    assert payload["fallbacks"][0]["provider"] == "triton"
    with pytest.raises(StrictCoverageError, match="poisoned"):
        instance.record(fallback)
    with pytest.raises(StrictCoverageError, match="durable report"):
        instance.finalize(event_stream_complete=True)


@pytest.mark.parametrize(
    "provenance",
    [None, KernelProvenance(ProvenanceOrigin.EXTERNAL, "external:forged")],
)
def test_allowed_provider_without_registry_proof_is_uncovered(
    tmp_path: Path, provenance: KernelProvenance | None
) -> None:
    item = artifact("pypto.generic")
    event = replace(registered_event(item), provenance=provenance)
    instance = auditor(tmp_path, [event], [item])
    with pytest.raises(StrictCoverageError, match="provenance"):
        instance.record(event)
    payload = json.loads(instance.report_path.read_text())
    assert payload["events"][0]["disposition"] == "fallback"


def test_registry_fields_must_correlate_exactly(tmp_path: Path) -> None:
    item = artifact("pypto.gdn")
    event = registered_event(item)
    assert event.provenance is not None
    forged = replace(
        event,
        provenance=replace(event.provenance, kernels_revision="different-kernels"),
    )
    instance = auditor(tmp_path, [forged], [item])
    with pytest.raises(StrictCoverageError, match="exactly match"):
        instance.record(forged)


def test_runtime_memcpy_memset_and_sampling_are_visible_but_excluded(tmp_path: Path) -> None:
    item = artifact("pypto.gdn")
    model = registered_event(item, calls=2, gpu_time_ns=400)
    memcpy = KernelEvent(
        "activity:memcpy",
        EventScope.RUNTIME,
        ActivityKind.MEMCPY,
        "cuda.runtime",
        "cudaMemcpyAsync",
        10,
        30,
    )
    memset = KernelEvent(
        "activity:memset",
        EventScope.RUNTIME,
        ActivityKind.MEMSET,
        "cuda.runtime",
        "cudaMemsetAsync",
        2,
        10,
    )
    sampling = KernelEvent(
        "activity:sampling",
        EventScope.SAMPLING,
        ActivityKind.COMPUTE,
        "sglang.sampling",
        "sampling_topk",
        2,
        50,
    )
    framework = KernelEvent(
        "activity:framework-copy",
        EventScope.FRAMEWORK,
        ActivityKind.COMPUTE,
        "sglang.framework",
        "multi_tensor_apply_kernel",
        1,
        20,
        KernelProvenance(
            origin=ProvenanceOrigin.EXTERNAL,
            artifact_id="framework:sglang.input-buffer-staging",
        ),
    )
    events = [model, memcpy, memset, sampling, framework]
    instance = auditor(tmp_path, events, [item])
    record_all(instance, events)
    summary = instance.finalize(event_stream_complete=True)
    assert (summary.covered_calls, summary.total_calls) == (2, 2)
    assert (summary.covered_gpu_time_ns, summary.total_gpu_time_ns) == (400, 400)
    assert summary.excluded_event_groups == 4
    payload = json.loads(instance.report_path.read_text())
    assert {entry["activity_id"] for entry in payload["excluded"]} == {
        "activity:memcpy",
        "activity:memset",
        "activity:sampling",
        "activity:framework-copy",
    }


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            KernelEvent(
                "activity:runtime-compute",
                EventScope.RUNTIME,
                ActivityKind.COMPUTE,
                "cuda.runtime",
                "hidden_compute",
                1,
                1,
            ),
            "memcpy or memset",
        ),
        (
            KernelEvent(
                "activity:model-memcpy",
                EventScope.MODEL_FORWARD,
                ActivityKind.MEMCPY,
                "cuda.runtime",
                "cudaMemcpyAsync",
                1,
                1,
            ),
            "classified as compute",
        ),
        (
            KernelEvent(
                "activity:hidden-sampling",
                EventScope.SAMPLING,
                ActivityKind.COMPUTE,
                "triton",
                "hidden_compute",
                1,
                1,
            ),
            "not recognized",
        ),
    ],
)
def test_scope_cannot_hide_unrecognized_compute(
    tmp_path: Path, event: KernelEvent, message: str
) -> None:
    instance = auditor(tmp_path, [event], [])
    with pytest.raises(StrictCoverageError, match=message):
        instance.record(event)
    assert instance.report_path.exists()


@pytest.mark.parametrize("case", ["empty", "excluded", "zero-time", "incomplete"])
def test_non_vacuous_and_complete_strict_requirements(tmp_path: Path, case: str) -> None:
    artifacts: list[ArtifactRecord] = []
    events: list[KernelEvent] = []
    complete = True
    if case == "excluded":
        events = [
            KernelEvent(
                "activity:memcpy",
                EventScope.RUNTIME,
                ActivityKind.MEMCPY,
                "cuda.runtime",
                "cudaMemcpyAsync",
                1,
                10,
            )
        ]
    elif case in {"zero-time", "incomplete"}:
        item = artifact("pypto.generic")
        artifacts = [item]
        events = [registered_event(item, gpu_time_ns=0 if case == "zero-time" else 10)]
        complete = case != "incomplete"
    instance = auditor(tmp_path, events, artifacts)
    record_all(instance, events)
    with pytest.raises(StrictCoverageError, match="durable report"):
        instance.finalize(event_stream_complete=complete)
    payload = json.loads(instance.report_path.read_text())
    assert payload["finalized"] is True
    assert payload["strict_policy_passed"] is False
    assert payload["violations"]


def test_manifest_count_and_trace_digest_are_reconciled(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    expected = manifest([event], [item])
    broken = replace(expected, activity_count=2, trace_digest="0" * 64)
    instance = auditor(tmp_path, [event], [item], manifest_override=broken)
    instance.record(event)
    with pytest.raises(StrictCoverageError):
        instance.finalize(event_stream_complete=True)
    codes = {entry["code"] for entry in json.loads(instance.report_path.read_text())["violations"]}
    assert {"activity-count-mismatch", "trace-digest-mismatch"} <= codes


def test_collector_revision_is_strictly_pinned(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    broken = replace(manifest([event], [item]), collector_revision="untrusted-revision")
    instance = auditor(tmp_path, [event], [item], manifest_override=broken)
    instance.record(event)
    with pytest.raises(StrictCoverageError):
        instance.finalize(event_stream_complete=True)
    codes = {entry["code"] for entry in json.loads(instance.report_path.read_text())["violations"]}
    assert "unknown-collector-revision" in codes


def test_artifact_registry_digest_is_checked_before_recording(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    broken = replace(manifest([event], [item]), artifact_registry_digest="0" * 64)
    with pytest.raises(ValueError, match="registry digest"):
        auditor(tmp_path, [event], [item], manifest_override=broken)


def test_duplicate_activity_id_and_identity_are_rejected(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    first = registered_event(item, activity_id="activity:first")
    expected = manifest([first], [item])
    instance = auditor(tmp_path, [first], [item], manifest_override=expected)
    instance.record(first)
    with pytest.raises(StrictCoverageError, match="duplicate activity_id"):
        instance.record(first)

    second = replace(first, activity_id="activity:second")
    other = auditor(tmp_path, [first], [item], manifest_override=expected, name="other.json")
    other.record(first)
    with pytest.raises(StrictCoverageError, match="duplicate normalized"):
        other.record(second)


def test_event_order_permutations_produce_byte_identical_reports(tmp_path: Path) -> None:
    artifacts = [artifact("pypto.gdn"), artifact("pypto.matmul")]
    events = [registered_event(artifacts[0]), registered_event(artifacts[1])]
    expected = manifest(events, artifacts)
    first = auditor(tmp_path, events, artifacts, manifest_override=expected, name="first.json")
    second = auditor(tmp_path, events, list(reversed(artifacts)), manifest_override=expected, name="second.json")
    record_all(first, events)
    record_all(second, list(reversed(events)))
    first.finalize(event_stream_complete=True)
    second.finalize(event_stream_complete=True)
    assert first.report_path.read_bytes() == second.report_path.read_bytes()


def test_successful_finalization_is_idempotent_and_closes_recording(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    instance = auditor(tmp_path, [event], [item])
    instance.record(event)
    first = instance.finalize(event_stream_complete=True)
    first_bytes = instance.report_path.read_bytes()
    second = instance.finalize(event_stream_complete=True)
    assert first == second
    assert instance.report_path.read_bytes() == first_bytes
    with pytest.raises(RuntimeError, match="after coverage finalization"):
        instance.record(event)


def test_atomic_replacement_preserves_existing_report_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    instance = auditor(tmp_path, [event], [item])
    instance.record(event)
    instance._publish(  # exercise the owning partial-to-final transition
        finalized=False,
        event_stream_complete=False,
        strict_policy_passed=False,
    )
    previous = instance.report_path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(coverage_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        instance.finalize(event_stream_complete=True)
    assert instance.report_path.read_bytes() == previous
    assert list(tmp_path.glob(".coverage.json.*.tmp")) == []


def test_report_path_cannot_be_taken_over_by_another_run(tmp_path: Path) -> None:
    item = artifact("pypto.generic")
    event = registered_event(item)
    first = auditor(tmp_path, [event], [item])
    first.record(event)
    first.finalize(event_stream_complete=True)
    report_bytes = first.report_path.read_bytes()

    other_manifest = replace(manifest([event], [item]), run_id="run:other")
    with pytest.raises((BlockingIOError, FileExistsError)):
        auditor(
            tmp_path,
            [event],
            [item],
            manifest_override=other_manifest,
        )
    first.close()
    with pytest.raises(FileExistsError, match="already owned"):
        auditor(
            tmp_path,
            [event],
            [item],
            manifest_override=other_manifest,
        )
    assert (tmp_path / "coverage.json").read_bytes() == report_bytes


def test_relative_report_path_stays_bound_after_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_directory = tmp_path / "original"
    other_directory = tmp_path / "other"
    original_directory.mkdir()
    other_directory.mkdir()
    monkeypatch.chdir(original_directory)

    item = artifact("pypto.generic")
    event = registered_event(item)
    expected = manifest([event], [item])
    instance = CoverageAuditor(
        mode=CoverageMode.STRICT,
        report_path="reports/coverage.json",
        manifest=expected,
        artifacts=[item],
    )
    assert instance.report_path == original_directory / "reports/coverage.json"
    monkeypatch.chdir(other_directory)
    instance.record(event)
    instance.finalize(event_stream_complete=True)

    assert (original_directory / "reports/coverage.json").exists()
    assert not (other_directory / "reports/coverage.json").exists()


def test_coverage_module_imports_no_framework_or_profiler(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
before = set(sys.modules)
import pypto_plugins.coverage
new = set(sys.modules) - before
forbidden = ('torch', 'sglang', 'cupy', 'pypto.compiler', 'cuda', 'triton')
assert not any(name == prefix or name.startswith(prefix + '.') for name in new for prefix in forbidden), sorted(new)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
