from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import pypto_plugins.activity_trace as trace
from pypto_plugins.coverage import ActivityKind, EventScope, ProvenanceOrigin
from pypto_plugins.errors import StrictCoverageError


class _Artifact:
    identity_digest = "identity-digest"
    kernel_abi = SimpleNamespace(entry_function_name="pypto_test_kernel")
    identities = SimpleNamespace(source_ir_digest="source-ir-digest")
    producer_identity = SimpleNamespace(
        toolchain_identity=SimpleNamespace(pypto_revision="compiler-revision")
    )

    def serialize(self) -> bytes:
        return b"serialized-artifact"


class _Monitor:
    def __init__(self, *, active: bool = True, popped_delta: int = 0) -> None:
        self.active = active
        self.popped_delta = popped_delta
        self.annotations: list[str] = []

    def stats(self) -> dict[str, bool]:
        return {"trace_window_active": self.active}

    def push_user_annotation(self, value: str) -> int:
        self.annotations.append(value)
        return len(self.annotations)

    def pop_user_annotation(self) -> int:
        return len(self.annotations) + self.popped_delta


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    trace.clear_artifact_registry_for_testing()


def _record():
    return trace.artifact_record_from_runtime(
        _Artifact(),
        provider="pypto.generic",
        source_node="fx:test",
    )


def _annotation(record) -> str:
    return json.dumps(
        {
            "artifact": record.to_dict(),
            "kind": trace.ANNOTATION_KIND,
            "schema_version": trace.ANNOTATION_SCHEMA_VERSION,
        }
    )


def test_artifact_record_comes_from_native_artifact() -> None:
    record = _record()
    assert record.artifact_id == "pypto-artifact-v1:identity-digest"
    assert record.kernel_name == "pypto_test_kernel"
    assert record.ir_region == "tensorir:source-ir-digest"
    assert record.compiler_revision == "compiler-revision"
    assert trace.artifact_registry_snapshot() == (record,)


def test_launch_annotation_uses_active_cupti_window(monkeypatch) -> None:
    monitor = _Monitor()
    monkeypatch.setattr(trace, "_active_monitor", lambda: monitor)
    record = _record()
    with trace.annotate_artifact_launch(record):
        pass
    payload = json.loads(monitor.annotations[0])
    assert payload["kind"] == trace.ANNOTATION_KIND
    assert payload["artifact"] == record.to_dict()


def test_launch_annotation_is_noop_without_active_window(monkeypatch) -> None:
    monitor = _Monitor(active=False)
    monkeypatch.setattr(trace, "_active_monitor", lambda: monitor)
    with trace.annotate_artifact_launch(_record()):
        pass
    assert monitor.annotations == []


def test_launch_annotation_rejects_unbalanced_stack(monkeypatch) -> None:
    monitor = _Monitor(popped_delta=1)
    monkeypatch.setattr(trace, "_active_monitor", lambda: monitor)
    with pytest.raises(StrictCoverageError, match="unbalanced"):
        with trace.annotate_artifact_launch(_record()):
            pass


def test_normalize_cupti_window_joins_artifact_and_runtime_activities() -> None:
    record = _record()
    window = {
        "start_ns": 10,
        "user_annotations": {"7": _annotation(record)},
        "events": [
            {
                "kind": "external_correlation",
                "external_id": 7,
                "correlation_id": 41,
            },
            {
                "kind": "kernel",
                "name": "pypto_test_kernel",
                "correlation_id": 41,
                "start_ns": 20,
                "end_ns": 70,
            },
            {
                "kind": "gpu_memcpy",
                "copy_kind": 1,
                "src_kind": 1,
                "dst_kind": 2,
                "start_ns": 71,
                "end_ns": 81,
            },
        ],
    }
    normalized = trace.normalize_cupti_window(window, dropped_records=0)
    assert normalized.closed_world is True
    assert normalized.artifacts == (record,)
    compute = next(event for event in normalized.events if event.activity is ActivityKind.COMPUTE)
    assert compute.scope is EventScope.MODEL_FORWARD
    assert compute.provider == "pypto.generic"
    assert compute.gpu_time_ns == 50
    assert compute.provenance is not None
    assert compute.provenance.origin is ProvenanceOrigin.PYPTO_ARTIFACT_REGISTRY
    memcpy = next(event for event in normalized.events if event.activity is ActivityKind.MEMCPY)
    assert memcpy.scope is EventScope.RUNTIME


def test_normalize_marks_uncorrelated_kernel_as_external() -> None:
    normalized = trace.normalize_cupti_window(
        {
            "start_ns": 10,
            "user_annotations": {},
            "events": [
                {
                    "kind": "kernel",
                    "name": "at::native::fallback",
                    "correlation_id": 4,
                    "start_ns": 20,
                    "end_ns": 21,
                }
            ],
        },
        dropped_records=0,
    )
    event = normalized.events[0]
    assert event.provider == "cuda.external"
    assert event.provenance is not None
    assert event.provenance.origin is ProvenanceOrigin.EXTERNAL


def test_normalize_rejects_drops_name_mismatch_and_unused_annotations() -> None:
    record = _record()
    with pytest.raises(StrictCoverageError, match="dropped"):
        trace.normalize_cupti_window({}, dropped_records=1)
    with pytest.raises(StrictCoverageError, match="does not match"):
        trace.normalize_cupti_window(
            {
                "start_ns": 1,
                "user_annotations": {"1": _annotation(record)},
                "events": [
                    {
                        "kind": "external_correlation",
                        "external_id": 1,
                        "correlation_id": 2,
                    },
                    {
                        "kind": "kernel",
                        "name": "wrong_kernel",
                        "correlation_id": 2,
                        "start_ns": 2,
                        "end_ns": 3,
                    },
                ],
            },
            dropped_records=0,
        )
    with pytest.raises(StrictCoverageError, match="no matching"):
        trace.normalize_cupti_window(
            {
                "start_ns": 1,
                "user_annotations": {"1": _annotation(record)},
                "events": [
                    {
                        "kind": "kernel",
                        "name": "fallback",
                        "correlation_id": 2,
                        "start_ns": 2,
                        "end_ns": 3,
                    }
                ],
            },
            dropped_records=0,
        )
