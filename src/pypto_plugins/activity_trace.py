"""CUPTI launch annotations and fail-closed activity normalization.

The launch boundary publishes the exact PyPTO ``Artifact`` identity through
CUPTI external correlation.  The collector then joins each GPU activity to
that annotation; kernel names alone are never accepted as provenance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Any, Iterator, Mapping

from .coverage import (
    ActivityKind,
    ArtifactRecord,
    EventScope,
    KernelEvent,
    KernelProvenance,
    ProvenanceOrigin,
)
from .errors import StrictCoverageError


ANNOTATION_KIND = "pypto.artifact-launch.v1"
ANNOTATION_SCHEMA_VERSION = 1
FRAMEWORK_ANNOTATION_KIND = "pypto.framework-bookkeeping.v1"
FRAMEWORK_ANNOTATION_SCHEMA_VERSION = 1
FRAMEWORK_PROVIDER = "sglang.framework"

_lock = RLock()
_artifact_registry: dict[str, ArtifactRecord] = {}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def artifact_record_from_runtime(
    artifact: Any,
    *,
    provider: str,
    source_node: str,
    ir_region: str | None = None,
    kernels_revision: str | None = None,
) -> ArtifactRecord:
    """Build the coverage identity from the immutable native Artifact."""

    serialized = bytes(artifact.serialize())
    identity_digest = str(artifact.identity_digest)
    compiler_revision = str(
        artifact.producer_identity.toolchain_identity.pypto_revision
    )
    source_ir_digest = str(artifact.identities.source_ir_digest)
    record = ArtifactRecord(
        artifact_id=f"pypto-artifact-v1:{identity_digest}",
        artifact_sha256=hashlib.sha256(serialized).hexdigest(),
        provider=provider,
        kernel_name=str(artifact.kernel_abi.entry_function_name),
        source_node=source_node,
        ir_region=ir_region or f"tensorir:{source_ir_digest}",
        compiler_revision=compiler_revision,
        kernels_revision=kernels_revision,
    )
    register_artifact(record)
    return record


def register_artifact(record: ArtifactRecord) -> None:
    """Register one exact Artifact identity and reject conflicting reuse."""

    if type(record) is not ArtifactRecord:
        raise TypeError("record must be an exact ArtifactRecord")
    with _lock:
        previous = _artifact_registry.setdefault(record.artifact_id, record)
        if previous != record:
            raise StrictCoverageError(
                f"conflicting PyPTO artifact registration: {record.artifact_id}"
            )


def artifact_registry_snapshot() -> tuple[ArtifactRecord, ...]:
    with _lock:
        return tuple(_artifact_registry[key] for key in sorted(_artifact_registry))


def clear_artifact_registry_for_testing() -> None:
    with _lock:
        _artifact_registry.clear()


def _active_monitor() -> Any | None:
    try:
        from torch.profiler import _cupti_monitor
    except ImportError:
        return None
    return _cupti_monitor.get_monitor()


def trace_window_active() -> bool:
    monitor = _active_monitor()
    return monitor is not None and bool(monitor.stats().get("trace_window_active"))


def _annotation_payload(record: ArtifactRecord) -> str:
    return _canonical_json(
        {
            "artifact": record.to_dict(),
            "kind": ANNOTATION_KIND,
            "schema_version": ANNOTATION_SCHEMA_VERSION,
        }
    )


@dataclass(frozen=True, slots=True)
class FrameworkAnnotation:
    provider: str
    source_node: str


def _framework_annotation_payload(source_node: str) -> str:
    if type(source_node) is not str or not source_node or source_node != source_node.strip():
        raise ValueError("framework source_node must be a non-empty trimmed string")
    return _canonical_json(
        {
            "kind": FRAMEWORK_ANNOTATION_KIND,
            "provider": FRAMEWORK_PROVIDER,
            "schema_version": FRAMEWORK_ANNOTATION_SCHEMA_VERSION,
            "source_node": source_node,
        }
    )


@contextmanager
def annotate_artifact_launch(record: ArtifactRecord) -> Iterator[None]:
    """Correlate one native launch when an active trace window exists."""

    register_artifact(record)
    monitor = _active_monitor()
    if monitor is None or not bool(monitor.stats().get("trace_window_active")):
        yield
        return
    external_id = monitor.push_user_annotation(_annotation_payload(record))
    if type(external_id) is not int or external_id <= 0:
        raise StrictCoverageError("CUPTI rejected the PyPTO artifact annotation")
    try:
        yield
    finally:
        popped_id = monitor.pop_user_annotation()
        if popped_id != external_id:
            raise StrictCoverageError(
                "CUPTI PyPTO annotation stack is unbalanced: "
                f"pushed={external_id} popped={popped_id}"
            )


@contextmanager
def annotate_framework_activity(source_node: str) -> Iterator[None]:
    """Correlate explicitly excluded SGLang framework bookkeeping compute."""

    monitor = _active_monitor()
    if monitor is None or not bool(monitor.stats().get("trace_window_active")):
        yield
        return
    external_id = monitor.push_user_annotation(
        _framework_annotation_payload(source_node)
    )
    if type(external_id) is not int or external_id <= 0:
        raise StrictCoverageError("CUPTI rejected the framework annotation")
    try:
        yield
    finally:
        popped_id = monitor.pop_user_annotation()
        if popped_id != external_id:
            raise StrictCoverageError(
                "CUPTI framework annotation stack is unbalanced: "
                f"pushed={external_id} popped={popped_id}"
            )


def _artifact_record_from_payload(payload: object) -> ArtifactRecord:
    if type(payload) is not dict:
        raise StrictCoverageError("PyPTO CUPTI artifact payload must be an object")
    expected = {
        "artifact_id",
        "artifact_sha256",
        "compiler_revision",
        "ir_region",
        "kernel_name",
        "kernels_revision",
        "provider",
        "source_node",
    }
    if set(payload) != expected:
        raise StrictCoverageError("PyPTO CUPTI artifact payload has unknown fields")
    try:
        return ArtifactRecord(**payload)
    except (TypeError, ValueError) as error:
        raise StrictCoverageError(
            f"invalid PyPTO CUPTI artifact payload: {error}"
        ) from error


def _decode_annotation(value: object) -> ArtifactRecord | FrameworkAnnotation | None:
    if type(value) is not str:
        raise StrictCoverageError("CUPTI user annotation must be a string")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if type(payload) is not dict:
        return None
    kind = payload.get("kind")
    if kind == FRAMEWORK_ANNOTATION_KIND:
        if set(payload) != {"kind", "provider", "schema_version", "source_node"}:
            raise StrictCoverageError(
                "framework CUPTI annotation has unknown fields"
            )
        if (
            payload["schema_version"] != FRAMEWORK_ANNOTATION_SCHEMA_VERSION
            or payload["provider"] != FRAMEWORK_PROVIDER
            or type(payload["source_node"]) is not str
            or not payload["source_node"]
            or payload["source_node"] != payload["source_node"].strip()
        ):
            raise StrictCoverageError("invalid framework CUPTI annotation")
        return FrameworkAnnotation(
            provider=payload["provider"], source_node=payload["source_node"]
        )
    if kind != ANNOTATION_KIND:
        if isinstance(kind, str) and kind.startswith("pypto."):
            raise StrictCoverageError(f"unknown PyPTO CUPTI annotation kind: {kind}")
        return None
    if set(payload) != {"artifact", "kind", "schema_version"}:
        raise StrictCoverageError("PyPTO CUPTI annotation has unknown fields")
    if payload["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise StrictCoverageError("unsupported PyPTO CUPTI annotation schema")
    return _artifact_record_from_payload(payload["artifact"])


def _raw_kernel_matches_artifact(raw_name: str, expected: str) -> bool:
    if raw_name == expected or raw_name.startswith(expected + "("):
        return True
    symbol = raw_name.split("(", 1)[0].rsplit(" ", 1)[-1]
    return symbol == expected


@dataclass(frozen=True, slots=True)
class NormalizedActivityTrace:
    artifacts: tuple[ArtifactRecord, ...]
    events: tuple[KernelEvent, ...]
    closed_world: bool


def normalize_cupti_window(
    window: Mapping[str, object],
    *,
    dropped_records: int,
    scope: EventScope = EventScope.MODEL_FORWARD,
) -> NormalizedActivityTrace:
    """Normalize one exact CUPTI trace window into the coverage schema."""

    if type(scope) is not EventScope:
        raise TypeError("scope must be an exact EventScope")
    if type(dropped_records) is not int or dropped_records < 0:
        raise ValueError("dropped_records must be a non-negative integer")
    if dropped_records:
        raise StrictCoverageError(
            f"CUPTI dropped {dropped_records} activity records"
        )
    if type(window) is not dict:
        raise TypeError("window must be an exact dict")
    if type(window.get("start_ns")) is not int or int(window["start_ns"]) <= 0:
        raise StrictCoverageError("CUPTI trace window has no valid start timestamp")
    raw_events = window.get("events")
    raw_annotations = window.get("user_annotations")
    if type(raw_events) is not list or type(raw_annotations) is not dict:
        raise StrictCoverageError("CUPTI trace window is missing events or annotations")

    annotations: dict[int, ArtifactRecord | FrameworkAnnotation] = {}
    for raw_id, value in raw_annotations.items():
        try:
            external_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise StrictCoverageError("invalid CUPTI external annotation id") from error
        record = _decode_annotation(value)
        if record is not None:
            previous = annotations.setdefault(external_id, record)
            if previous != record:
                raise StrictCoverageError(
                    f"conflicting CUPTI annotation id: {external_id}"
                )

    correlations: dict[int, int] = {}
    for raw in raw_events:
        if type(raw) is not dict or raw.get("kind") != "external_correlation":
            continue
        correlation_id = int(raw.get("correlation_id", 0))
        external_id = int(raw.get("external_id", 0))
        if correlation_id <= 0 or external_id <= 0:
            raise StrictCoverageError("invalid CUPTI external correlation record")
        previous = correlations.setdefault(correlation_id, external_id)
        if previous != external_id:
            raise StrictCoverageError(
                f"conflicting CUPTI correlation id: {correlation_id}"
            )

    aggregates: dict[
        tuple[EventScope, ActivityKind, str, str, KernelProvenance | None],
        list[int],
    ] = {}
    used_annotations: set[int] = set()
    gpu_activity_count = 0
    for raw in raw_events:
        if type(raw) is not dict:
            raise StrictCoverageError("CUPTI event must be an object")
        kind = raw.get("kind")
        if kind not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        gpu_activity_count += 1
        start_ns = raw.get("start_ns")
        end_ns = raw.get("end_ns")
        if type(start_ns) is not int or type(end_ns) is not int or end_ns < start_ns:
            raise StrictCoverageError("CUPTI GPU activity has invalid timestamps")
        duration_ns = end_ns - start_ns

        if kind == "kernel":
            raw_name = raw.get("name")
            correlation_id = raw.get("correlation_id")
            if type(raw_name) is not str or not raw_name:
                raise StrictCoverageError("CUPTI kernel activity has no name")
            if type(correlation_id) is not int or correlation_id <= 0:
                raise StrictCoverageError("CUPTI kernel has no correlation id")
            external_id = correlations.get(correlation_id)
            record = annotations.get(external_id) if external_id is not None else None
            if record is None:
                provider = "cuda.external"
                kernel_name = raw_name
                provenance = KernelProvenance(
                    origin=ProvenanceOrigin.EXTERNAL,
                    artifact_id=f"cupti-kernel:{correlation_id}:{raw_name}",
                )
                event_scope = scope
            elif isinstance(record, FrameworkAnnotation):
                used_annotations.add(external_id)
                provider = record.provider
                kernel_name = raw_name
                provenance = KernelProvenance(
                    origin=ProvenanceOrigin.EXTERNAL,
                    artifact_id=f"framework:{record.source_node}",
                )
                event_scope = EventScope.FRAMEWORK
            else:
                if not _raw_kernel_matches_artifact(raw_name, record.kernel_name):
                    raise StrictCoverageError(
                        "CUPTI kernel name does not match its PyPTO Artifact: "
                        f"raw={raw_name!r} artifact={record.kernel_name!r}"
                    )
                used_annotations.add(external_id)
                provider = record.provider
                kernel_name = record.kernel_name
                provenance = KernelProvenance(
                    origin=ProvenanceOrigin.PYPTO_ARTIFACT_REGISTRY,
                    artifact_id=record.artifact_id,
                    artifact_sha256=record.artifact_sha256,
                    compiler_revision=record.compiler_revision,
                    kernels_revision=record.kernels_revision,
                )
                event_scope = scope
            key = (
                event_scope,
                ActivityKind.COMPUTE,
                provider,
                kernel_name,
                provenance,
            )
        elif kind == "gpu_memcpy":
            copy_name = "Memcpy:" + ":".join(
                str(int(raw.get(field, 0)))
                for field in ("copy_kind", "src_kind", "dst_kind")
            )
            key = (
                EventScope.RUNTIME,
                ActivityKind.MEMCPY,
                "cuda.runtime",
                copy_name,
                None,
            )
        else:
            memset_name = f"Memset:{int(raw.get('memory_kind', 0))}"
            key = (
                EventScope.RUNTIME,
                ActivityKind.MEMSET,
                "cuda.runtime",
                memset_name,
                None,
            )
        aggregate = aggregates.setdefault(key, [0, 0])
        aggregate[0] += 1
        aggregate[1] += duration_ns

    if gpu_activity_count == 0:
        raise StrictCoverageError("CUPTI trace window contains no GPU activities")
    unused = sorted(set(annotations) - used_annotations)
    if unused:
        raise StrictCoverageError(
            f"PyPTO CUPTI annotations have no matching kernel activities: {unused}"
        )

    events: list[KernelEvent] = []
    for key, (call_count, gpu_time_ns) in sorted(
        aggregates.items(), key=lambda item: repr(item[0])
    ):
        event_scope, activity, provider, kernel_name, provenance = key
        identity = _canonical_json(
            {
                "activity": activity.value,
                "kernel_name": kernel_name,
                "provenance": None if provenance is None else provenance.to_dict(),
                "provider": provider,
                "scope": event_scope.value,
            }
        )
        events.append(
            KernelEvent(
                activity_id="cupti:" + hashlib.sha256(identity.encode("ascii")).hexdigest(),
                scope=event_scope,
                activity=activity,
                provider=provider,
                kernel_name=kernel_name,
                call_count=call_count,
                gpu_time_ns=gpu_time_ns,
                provenance=provenance,
            )
        )
    artifacts = tuple(
        sorted(
            {
                annotations[item].artifact_id: annotations[item]
                for item in used_annotations
                if isinstance(annotations[item], ArtifactRecord)
            }.values(),
            key=lambda record: record.artifact_id,
        )
    )
    return NormalizedActivityTrace(
        artifacts=artifacts,
        events=tuple(events),
        closed_world=True,
    )


__all__ = (
    "ANNOTATION_KIND",
    "ANNOTATION_SCHEMA_VERSION",
    "NormalizedActivityTrace",
    "annotate_artifact_launch",
    "annotate_framework_activity",
    "artifact_record_from_runtime",
    "artifact_registry_snapshot",
    "clear_artifact_registry_for_testing",
    "normalize_cupti_window",
    "register_artifact",
    "trace_window_active",
)
