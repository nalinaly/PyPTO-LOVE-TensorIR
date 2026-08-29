"""Fail-closed audit of adapter-normalized model-forward GPU activities.

This module deliberately has no Torch, SGLang, CUDA-profiler, or compiler
dependency.  A framework adapter supplies a closed activity trace and the
PyPTO artifact registry used to launch it.  The auditor verifies the exact
trace and registry digests, classifies every activity, and publishes a durable
machine-readable policy result.

It does not collect activities or prove that an adapter's closed-world claim
is true.  See the project README for the evidence boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any, Iterable, Mapping, NoReturn, Sequence

from .errors import StrictCoverageError


class CoverageMode(str, Enum):
    DEVELOPMENT = "development"
    STRICT = "strict"


class EventScope(str, Enum):
    MODEL_FORWARD = "model-forward"
    FRAMEWORK = "framework"
    RUNTIME = "runtime"
    SAMPLING = "sampling"


class ActivityKind(str, Enum):
    COMPUTE = "compute"
    MEMCPY = "memcpy"
    MEMSET = "memset"


class ProvenanceOrigin(str, Enum):
    PYPTO_ARTIFACT_REGISTRY = "pypto-artifact-registry"
    EXTERNAL = "external"


ALLOWED_PYPTO_PROVIDERS = frozenset(
    {
        "pypto.generic",
        "pypto.tensorir",
        "pypto.matmul",
        "pypto.attention",
        "pypto.gdn",
    }
)
"""The fixed strict-coverage provider policy; callers cannot weaken it."""

OPERATOR_LIBRARY_PROVIDERS = frozenset(
    {"pypto.matmul", "pypto.attention", "pypto.gdn"}
)
ALLOWED_SAMPLING_PROVIDERS = frozenset({"pypto.sampling", "sglang.sampling"})
ALLOWED_FRAMEWORK_PROVIDERS = frozenset({"sglang.framework"})
TRACE_COLLECTOR = "pypto.activity-trace.v1"
TRACE_COLLECTOR_REVISION = "pypto.activity-trace.v1-normalizer-schema-1"
FRAMEWORK_PROFILE = "pypto"
SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_exact_enum(value: object, enum_type: type[Enum], field: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{field} must be an exact {enum_type.__name__}")


def _require_nonempty_string(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty, trimmed string")


def _require_optional_nonempty_string(value: object, field: str) -> None:
    if value is not None:
        _require_nonempty_string(value, field)


def _require_sha256(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_payloads(payloads: Sequence[Mapping[str, Any]]) -> str:
    encoded = _canonical_json(list(payloads)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class KernelProvenance:
    origin: ProvenanceOrigin
    artifact_id: str
    artifact_sha256: str | None = None
    compiler_revision: str | None = None
    kernels_revision: str | None = None

    def __post_init__(self) -> None:
        _require_exact_enum(self.origin, ProvenanceOrigin, "origin")
        _require_nonempty_string(self.artifact_id, "artifact_id")
        if self.artifact_sha256 is not None:
            _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_optional_nonempty_string(self.compiler_revision, "compiler_revision")
        _require_optional_nonempty_string(self.kernels_revision, "kernels_revision")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "compiler_revision": self.compiler_revision,
            "kernels_revision": self.kernels_revision,
            "origin": self.origin.value,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    artifact_sha256: str
    provider: str
    kernel_name: str
    source_node: str
    ir_region: str
    compiler_revision: str
    kernels_revision: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.artifact_id, "artifact_id")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_nonempty_string(self.provider, "provider")
        if self.provider not in ALLOWED_PYPTO_PROVIDERS:
            raise ValueError(f"artifact provider is not an allowed PyPTO provider: {self.provider}")
        _require_nonempty_string(self.kernel_name, "kernel_name")
        if not self.kernel_name.startswith("pypto_"):
            raise ValueError("PyPTO artifact kernel_name must start with 'pypto_'")
        _require_nonempty_string(self.source_node, "source_node")
        _require_nonempty_string(self.ir_region, "ir_region")
        _require_nonempty_string(self.compiler_revision, "compiler_revision")
        _require_optional_nonempty_string(self.kernels_revision, "kernels_revision")
        if self.provider in OPERATOR_LIBRARY_PROVIDERS and self.kernels_revision is None:
            raise ValueError(f"{self.provider} artifacts require kernels_revision")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "compiler_revision": self.compiler_revision,
            "ir_region": self.ir_region,
            "kernel_name": self.kernel_name,
            "kernels_revision": self.kernels_revision,
            "provider": self.provider,
            "source_node": self.source_node,
        }


@dataclass(frozen=True, slots=True)
class KernelEvent:
    activity_id: str
    scope: EventScope
    activity: ActivityKind
    provider: str
    kernel_name: str
    call_count: int
    gpu_time_ns: int
    provenance: KernelProvenance | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.activity_id, "activity_id")
        _require_exact_enum(self.scope, EventScope, "scope")
        _require_exact_enum(self.activity, ActivityKind, "activity")
        _require_nonempty_string(self.provider, "provider")
        _require_nonempty_string(self.kernel_name, "kernel_name")
        if type(self.call_count) is not int:
            raise TypeError("call_count must be an exact integer")
        if self.call_count <= 0:
            raise ValueError("call_count must be positive")
        if type(self.gpu_time_ns) is not int:
            raise TypeError("gpu_time_ns must be an exact integer")
        if self.gpu_time_ns < 0:
            raise ValueError("gpu_time_ns must be non-negative")
        if self.provenance is not None and type(self.provenance) is not KernelProvenance:
            raise TypeError("provenance must be an exact KernelProvenance or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "activity": self.activity.value,
            "activity_id": self.activity_id,
            "call_count": self.call_count,
            "gpu_time_ns": self.gpu_time_ns,
            "kernel_name": self.kernel_name,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
            "provider": self.provider,
            "scope": self.scope.value,
        }

    def identity_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        del payload["activity_id"]
        return payload


@dataclass(frozen=True, slots=True)
class TraceManifest:
    run_id: str
    model_id: str
    model_revision: str
    device_fingerprint: str
    collector: str
    collector_revision: str
    framework_profile: str
    artifact_registry_digest: str
    trace_digest: str
    activity_count: int
    closed_world: bool

    def __post_init__(self) -> None:
        for field in (
            "run_id",
            "model_id",
            "model_revision",
            "device_fingerprint",
            "collector",
            "collector_revision",
            "framework_profile",
        ):
            _require_nonempty_string(getattr(self, field), field)
        _require_sha256(self.artifact_registry_digest, "artifact_registry_digest")
        _require_sha256(self.trace_digest, "trace_digest")
        if type(self.activity_count) is not int:
            raise TypeError("activity_count must be an exact integer")
        if self.activity_count < 0:
            raise ValueError("activity_count must be non-negative")
        if type(self.closed_world) is not bool:
            raise TypeError("closed_world must be an exact bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "activity_count": self.activity_count,
            "artifact_registry_digest": self.artifact_registry_digest,
            "closed_world": self.closed_world,
            "collector": self.collector,
            "collector_revision": self.collector_revision,
            "device_fingerprint": self.device_fingerprint,
            "framework_profile": self.framework_profile,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "run_id": self.run_id,
            "trace_digest": self.trace_digest,
        }


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    mode: CoverageMode
    finalized: bool
    event_stream_complete: bool
    strict_policy_passed: bool
    model_forward_event_groups: int
    covered_calls: int
    total_calls: int
    covered_gpu_time_ns: int
    total_gpu_time_ns: int
    fallback_event_groups: int
    excluded_event_groups: int
    violation_count: int


def _sorted_event_payloads(events: Iterable[KernelEvent]) -> list[dict[str, object]]:
    materialized = list(events)
    ids: set[str] = set()
    identities: set[str] = set()
    payloads: list[dict[str, object]] = []
    for event in materialized:
        if type(event) is not KernelEvent:
            raise TypeError("events must contain exact KernelEvent instances")
        if event.activity_id in ids:
            raise ValueError(f"duplicate activity_id: {event.activity_id}")
        ids.add(event.activity_id)
        identity = _canonical_json(event.identity_dict())
        if identity in identities:
            raise ValueError("duplicate normalized activity identity; aggregate its call_count instead")
        identities.add(identity)
        payloads.append(event.to_dict())
    return sorted(payloads, key=_canonical_json)


def compute_trace_digest(events: Iterable[KernelEvent]) -> str:
    """Return the order-independent digest of a complete normalized trace."""

    return _digest_payloads(_sorted_event_payloads(events))


def _sorted_artifact_payloads(
    artifacts: Iterable[ArtifactRecord],
) -> list[dict[str, object]]:
    ids: set[str] = set()
    payloads: list[dict[str, object]] = []
    for artifact in artifacts:
        if type(artifact) is not ArtifactRecord:
            raise TypeError("artifacts must contain exact ArtifactRecord instances")
        if artifact.artifact_id in ids:
            raise ValueError(f"duplicate artifact_id: {artifact.artifact_id}")
        ids.add(artifact.artifact_id)
        payloads.append(artifact.to_dict())
    return sorted(payloads, key=_canonical_json)


def compute_artifact_registry_digest(artifacts: Iterable[ArtifactRecord]) -> str:
    """Return the order-independent digest of an artifact registry snapshot."""

    return _digest_payloads(_sorted_artifact_payloads(artifacts))


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    ) + "\n"
    serialized_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return serialized_digest


class CoverageAuditor:
    """Audit one manifest-bound, normalized GPU activity stream."""

    def __init__(
        self,
        *,
        mode: CoverageMode,
        report_path: str | Path,
        manifest: TraceManifest,
        artifacts: Iterable[ArtifactRecord],
    ) -> None:
        _require_exact_enum(mode, CoverageMode, "mode")
        if type(manifest) is not TraceManifest:
            raise TypeError("manifest must be an exact TraceManifest")
        if not isinstance(report_path, (str, Path)):
            raise TypeError("report_path must be str or Path")
        artifact_tuple = tuple(artifacts)
        artifact_payloads = _sorted_artifact_payloads(artifact_tuple)
        registry_digest = _digest_payloads(artifact_payloads)
        if registry_digest != manifest.artifact_registry_digest:
            raise ValueError("artifact registry digest does not match trace manifest")

        self._mode = mode
        self._report_path = Path(report_path).expanduser().resolve()
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_lock_path = self._report_path.with_name(
            f"{self._report_path.name}.lock"
        )
        report_lock_fd = os.open(
            self._report_lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(report_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self._report_path.exists():
                raise FileExistsError(
                    f"coverage report path is already owned: {self._report_path}"
                )
        except BaseException:
            os.close(report_lock_fd)
            raise
        self._report_lock_fd: int | None = report_lock_fd
        self._last_report_sha256: str | None = None
        self._manifest = manifest
        self._artifacts = artifact_tuple
        self._artifact_payloads = artifact_payloads
        self._artifact_by_id = {artifact.artifact_id: artifact for artifact in artifact_tuple}
        self._events: list[KernelEvent] = []
        self._event_ids: set[str] = set()
        self._event_identities: set[str] = set()
        self._dispositions: dict[str, str] = {}
        self._violations: list[dict[str, str]] = []
        self._violation_keys: set[tuple[str, str | None]] = set()
        self._poisoned = False
        self._finalized = False
        self._summary: CoverageSummary | None = None
        self._final_error: tuple[str, str] | None = None
        self._closed = False
        self._lock = RLock()

    @property
    def report_path(self) -> Path:
        return self._report_path

    def close(self) -> None:
        """Release this auditor's process lock; the report remains immutable."""

        with self._lock:
            if self._closed:
                return
            descriptor = self._report_lock_fd
            self._report_lock_fd = None
            self._closed = True
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def __enter__(self) -> CoverageAuditor:
        if self._closed:
            raise RuntimeError("coverage auditor is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        descriptor = getattr(self, "_report_lock_fd", None)
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            except OSError:
                pass
            self._report_lock_fd = None

    def _add_violation(
        self, code: str, message: str, activity_id: str | None = None
    ) -> None:
        key = (code, activity_id)
        if key in self._violation_keys:
            return
        self._violation_keys.add(key)
        payload = {"code": code, "message": message}
        if activity_id is not None:
            payload["activity_id"] = activity_id
        self._violations.append(payload)

    def _strict_failure(self, code: str, message: str) -> NoReturn:
        self._poisoned = True
        self._publish(
            finalized=False,
            event_stream_complete=False,
            strict_policy_passed=False,
        )
        raise StrictCoverageError(
            message,
            code=code,
            report_path=self._report_path,
        )

    def _event_violation(self, event: KernelEvent, code: str, message: str) -> None:
        self._add_violation(code, message, event.activity_id)
        self._dispositions[event.activity_id] = "invalid"
        if self._mode is CoverageMode.STRICT:
            self._strict_failure(code, message)

    def _provenance_error(self, event: KernelEvent) -> str | None:
        provenance = event.provenance
        if provenance is None:
            return "missing PyPTO artifact provenance"
        if provenance.origin is not ProvenanceOrigin.PYPTO_ARTIFACT_REGISTRY:
            return "provenance origin is not the PyPTO artifact registry"
        artifact = self._artifact_by_id.get(provenance.artifact_id)
        if artifact is None:
            return "artifact_id is absent from the manifest-bound registry"
        expected = (
            artifact.provider,
            artifact.kernel_name,
            artifact.artifact_sha256,
            artifact.compiler_revision,
            artifact.kernels_revision,
        )
        observed = (
            event.provider,
            event.kernel_name,
            provenance.artifact_sha256,
            provenance.compiler_revision,
            provenance.kernels_revision,
        )
        if observed != expected:
            return "event provenance does not exactly match its artifact registry record"
        return None

    def record(self, event: KernelEvent) -> None:
        if type(event) is not KernelEvent:
            raise TypeError("event must be an exact KernelEvent")
        with self._lock:
            if self._closed:
                raise RuntimeError("coverage auditor is closed")
            if self._finalized:
                raise RuntimeError("cannot record after coverage finalization")
            if self._poisoned and self._mode is CoverageMode.STRICT:
                raise StrictCoverageError(
                    "strict coverage auditor is poisoned by an earlier violation",
                    code="auditor-poisoned",
                    report_path=self._report_path,
                )

            identity = _canonical_json(event.identity_dict())
            if event.activity_id in self._event_ids:
                self._add_violation(
                    "duplicate-activity-id",
                    f"duplicate activity_id: {event.activity_id}",
                    event.activity_id,
                )
                if self._mode is CoverageMode.STRICT:
                    self._strict_failure(
                        "duplicate-activity-id",
                        f"duplicate activity_id: {event.activity_id}",
                    )
                return
            if identity in self._event_identities:
                self._add_violation(
                    "duplicate-activity-identity",
                    "duplicate normalized activity identity; aggregate call_count instead",
                    event.activity_id,
                )
                if self._mode is CoverageMode.STRICT:
                    self._strict_failure(
                        "duplicate-activity-identity",
                        "duplicate normalized activity identity; aggregate call_count instead",
                    )
                return

            self._events.append(event)
            self._event_ids.add(event.activity_id)
            self._event_identities.add(identity)

            if event.scope is EventScope.MODEL_FORWARD:
                if event.activity is not ActivityKind.COMPUTE:
                    self._event_violation(
                        event,
                        "model-forward-non-compute",
                        "model-forward activities must be classified as compute",
                    )
                    return
                if event.provider not in ALLOWED_PYPTO_PROVIDERS:
                    self._dispositions[event.activity_id] = "fallback"
                    self._add_violation(
                        "fallback-provider",
                        f"model-forward compute provider is not PyPTO: {event.provider}",
                        event.activity_id,
                    )
                    if self._mode is CoverageMode.STRICT:
                        self._strict_failure(
                            "fallback-provider",
                            f"strict coverage rejected provider {event.provider}",
                        )
                    return
                provenance_error = self._provenance_error(event)
                if provenance_error is not None:
                    self._dispositions[event.activity_id] = "fallback"
                    self._add_violation(
                        "unverified-pypto-provenance",
                        provenance_error,
                        event.activity_id,
                    )
                    if self._mode is CoverageMode.STRICT:
                        self._strict_failure(
                            "unverified-pypto-provenance",
                            f"strict coverage rejected {event.kernel_name}: {provenance_error}",
                        )
                    return
                self._dispositions[event.activity_id] = "covered"
                return

            if event.scope is EventScope.RUNTIME:
                if event.activity not in (ActivityKind.MEMCPY, ActivityKind.MEMSET):
                    self._event_violation(
                        event,
                        "runtime-compute",
                        "runtime scope may contain only CUDA memcpy or memset activities",
                    )
                    return
                if event.provider != "cuda.runtime":
                    self._event_violation(
                        event,
                        "runtime-provider",
                        "runtime memcpy/memset provider must be cuda.runtime",
                    )
                    return
                self._dispositions[event.activity_id] = "excluded"
                return

            if event.scope is EventScope.FRAMEWORK:
                provenance = event.provenance
                if event.activity is not ActivityKind.COMPUTE:
                    self._event_violation(
                        event,
                        "framework-non-compute",
                        "framework bookkeeping must be classified as compute",
                    )
                    return
                if event.provider not in ALLOWED_FRAMEWORK_PROVIDERS:
                    self._event_violation(
                        event,
                        "framework-provider",
                        f"framework provider is not recognized: {event.provider}",
                    )
                    return
                if (
                    provenance is None
                    or provenance.origin is not ProvenanceOrigin.EXTERNAL
                    or not provenance.artifact_id.startswith("framework:")
                ):
                    self._event_violation(
                        event,
                        "framework-provenance",
                        "framework compute requires explicit bookkeeping provenance",
                    )
                    return
                self._dispositions[event.activity_id] = "excluded"
                return

            if event.scope is EventScope.SAMPLING:
                if event.provider not in ALLOWED_SAMPLING_PROVIDERS:
                    self._event_violation(
                        event,
                        "sampling-provider",
                        f"sampling scope provider is not recognized: {event.provider}",
                    )
                    return
                self._dispositions[event.activity_id] = "excluded"
                return

            raise AssertionError(f"unhandled EventScope: {event.scope}")

    def _summary_for(
        self,
        *,
        finalized: bool,
        event_stream_complete: bool,
        strict_policy_passed: bool,
    ) -> CoverageSummary:
        model_events = [
            event
            for event in self._events
            if event.scope is EventScope.MODEL_FORWARD
            and event.activity is ActivityKind.COMPUTE
        ]
        covered = [
            event
            for event in model_events
            if self._dispositions.get(event.activity_id) == "covered"
        ]
        fallbacks = [
            event
            for event in model_events
            if self._dispositions.get(event.activity_id) != "covered"
        ]
        excluded = [
            event
            for event in self._events
            if self._dispositions.get(event.activity_id) == "excluded"
        ]
        return CoverageSummary(
            mode=self._mode,
            finalized=finalized,
            event_stream_complete=event_stream_complete,
            strict_policy_passed=strict_policy_passed,
            model_forward_event_groups=len(model_events),
            covered_calls=sum(event.call_count for event in covered),
            total_calls=sum(event.call_count for event in model_events),
            covered_gpu_time_ns=sum(event.gpu_time_ns for event in covered),
            total_gpu_time_ns=sum(event.gpu_time_ns for event in model_events),
            fallback_event_groups=len(fallbacks),
            excluded_event_groups=len(excluded),
            violation_count=len(self._violations),
        )

    def _build_payload(
        self,
        *,
        finalized: bool,
        event_stream_complete: bool,
        strict_policy_passed: bool,
    ) -> tuple[CoverageSummary, dict[str, object]]:
        summary = self._summary_for(
            finalized=finalized,
            event_stream_complete=event_stream_complete,
            strict_policy_passed=strict_policy_passed,
        )
        event_payloads: list[dict[str, object]] = []
        fallback_payloads: list[dict[str, object]] = []
        excluded_payloads: list[dict[str, object]] = []
        for event in self._events:
            payload = event.to_dict()
            payload["disposition"] = self._dispositions.get(event.activity_id, "invalid")
            event_payloads.append(payload)
            if payload["disposition"] in ("fallback", "invalid"):
                fallback_payloads.append(payload)
            elif payload["disposition"] == "excluded":
                excluded_payloads.append(payload)
        event_payloads.sort(key=_canonical_json)
        fallback_payloads.sort(key=_canonical_json)
        excluded_payloads.sort(key=_canonical_json)
        violations = sorted(self._violations, key=_canonical_json)
        observed_payloads = _sorted_event_payloads(self._events)
        payload: dict[str, object] = {
            "allowed_providers": sorted(ALLOWED_PYPTO_PROVIDERS),
            "artifact_registry": self._artifact_payloads,
            "events": event_payloads,
            "excluded": excluded_payloads,
            "fallbacks": fallback_payloads,
            "finalized": finalized,
            "manifest": self._manifest.to_dict(),
            "mode": self._mode.value,
            "model_forward_compute": {
                "calls": {"covered": summary.covered_calls, "total": summary.total_calls},
                "event_groups": summary.model_forward_event_groups,
                "gpu_time_ns": {
                    "covered": summary.covered_gpu_time_ns,
                    "total": summary.total_gpu_time_ns,
                },
            },
            "observed_trace": {
                "activity_count": len(self._events),
                "trace_digest": _digest_payloads(observed_payloads),
            },
            "schema_version": SCHEMA_VERSION,
            "strict_policy_passed": strict_policy_passed,
            "event_stream_complete": event_stream_complete,
            "violations": violations,
        }
        return summary, payload

    def _publish(
        self,
        *,
        finalized: bool,
        event_stream_complete: bool,
        strict_policy_passed: bool,
    ) -> CoverageSummary:
        summary, payload = self._build_payload(
            finalized=finalized,
            event_stream_complete=event_stream_complete,
            strict_policy_passed=strict_policy_passed,
        )
        if self._closed or self._report_lock_fd is None:
            raise RuntimeError("coverage auditor is closed")
        if self._last_report_sha256 is None:
            if self._report_path.exists():
                raise FileExistsError(
                    f"coverage report appeared after ownership was acquired: {self._report_path}"
                )
        else:
            try:
                current_digest = hashlib.sha256(self._report_path.read_bytes()).hexdigest()
            except FileNotFoundError as error:
                raise RuntimeError("owned coverage report disappeared before update") from error
            if current_digest != self._last_report_sha256:
                raise RuntimeError("owned coverage report changed outside this auditor")
        self._last_report_sha256 = _atomic_write_json(self._report_path, payload)
        return summary

    def finalize(self, *, event_stream_complete: bool) -> CoverageSummary:
        if type(event_stream_complete) is not bool:
            raise TypeError("event_stream_complete must be an exact bool")
        with self._lock:
            if self._closed:
                raise RuntimeError("coverage auditor is closed")
            if self._finalized:
                if self._final_error is not None:
                    code, message = self._final_error
                    raise StrictCoverageError(
                        message,
                        code=code,
                        report_path=self._report_path,
                    )
                assert self._summary is not None
                return self._summary

            if not event_stream_complete:
                self._add_violation(
                    "incomplete-event-stream",
                    "adapter did not assert that the normalized event stream is complete",
                )
            if not self._manifest.closed_world:
                self._add_violation(
                    "open-world-manifest",
                    "trace manifest is not a closed-world activity claim",
                )
            if self._manifest.collector != TRACE_COLLECTOR:
                self._add_violation(
                    "unknown-collector",
                    f"strict policy requires collector {TRACE_COLLECTOR}",
                )
            if self._manifest.collector_revision != TRACE_COLLECTOR_REVISION:
                self._add_violation(
                    "unknown-collector-revision",
                    f"strict policy requires collector revision {TRACE_COLLECTOR_REVISION}",
                )
            if self._manifest.framework_profile != FRAMEWORK_PROFILE:
                self._add_violation(
                    "wrong-framework-profile",
                    f"strict policy requires framework profile {FRAMEWORK_PROFILE}",
                )
            if len(self._events) != self._manifest.activity_count:
                self._add_violation(
                    "activity-count-mismatch",
                    "observed activity count does not match trace manifest",
                )
            observed_trace_digest = compute_trace_digest(self._events)
            if observed_trace_digest != self._manifest.trace_digest:
                self._add_violation(
                    "trace-digest-mismatch",
                    "observed normalized trace digest does not match trace manifest",
                )

            provisional = self._summary_for(
                finalized=True,
                event_stream_complete=event_stream_complete,
                strict_policy_passed=False,
            )
            if provisional.model_forward_event_groups == 0 or provisional.total_calls == 0:
                self._add_violation(
                    "no-model-forward-evidence",
                    "strict coverage requires non-empty model-forward compute evidence",
                )
            if provisional.total_gpu_time_ns == 0:
                self._add_violation(
                    "zero-model-forward-gpu-time",
                    "strict coverage requires positive model-forward GPU timing evidence",
                )
            if provisional.covered_calls != provisional.total_calls:
                self._add_violation(
                    "incomplete-call-coverage",
                    "covered model-forward calls do not equal total calls",
                )
            if provisional.covered_gpu_time_ns != provisional.total_gpu_time_ns:
                self._add_violation(
                    "incomplete-time-coverage",
                    "covered model-forward GPU time does not equal total GPU time",
                )

            strict_pass = (
                self._mode is CoverageMode.STRICT
                and event_stream_complete
                and not self._violations
                and provisional.model_forward_event_groups > 0
                and provisional.total_calls > 0
                and provisional.total_gpu_time_ns > 0
                and provisional.covered_calls == provisional.total_calls
                and provisional.covered_gpu_time_ns == provisional.total_gpu_time_ns
            )
            summary = self._publish(
                finalized=True,
                event_stream_complete=event_stream_complete,
                strict_policy_passed=strict_pass,
            )
            self._summary = summary
            self._finalized = True

            if self._mode is CoverageMode.STRICT and not strict_pass:
                code = "strict-policy-failed"
                message = "strict model-forward coverage policy failed; see durable report"
                self._final_error = (code, message)
                raise StrictCoverageError(
                    message,
                    code=code,
                    report_path=self._report_path,
                )
            return summary


__all__ = [
    "ALLOWED_FRAMEWORK_PROVIDERS",
    "ALLOWED_PYPTO_PROVIDERS",
    "ALLOWED_SAMPLING_PROVIDERS",
    "ActivityKind",
    "ArtifactRecord",
    "CoverageAuditor",
    "CoverageMode",
    "CoverageSummary",
    "EventScope",
    "FRAMEWORK_PROFILE",
    "KernelEvent",
    "KernelProvenance",
    "OPERATOR_LIBRARY_PROVIDERS",
    "ProvenanceOrigin",
    "TRACE_COLLECTOR",
    "TRACE_COLLECTOR_REVISION",
    "TraceManifest",
    "compute_artifact_registry_digest",
    "compute_trace_digest",
]
