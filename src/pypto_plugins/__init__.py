"""PyPTO framework compatibility plugins."""

from .versions import (
    EXPECTED_SGLANG_COMMIT,
    EXPECTED_SGLANG_VERSION,
    EXPECTED_TORCH_COMMIT,
    EXPECTED_TORCH_VERSION,
)
from .coverage import (
    ALLOWED_PYPTO_PROVIDERS,
    ActivityKind,
    ArtifactRecord,
    CoverageAuditor,
    CoverageMode,
    CoverageSummary,
    EventScope,
    KernelEvent,
    KernelProvenance,
    ProvenanceOrigin,
    TRACE_COLLECTOR_REVISION,
    TraceManifest,
    compute_artifact_registry_digest,
    compute_trace_digest,
)

__all__ = [
    "EXPECTED_SGLANG_COMMIT",
    "EXPECTED_SGLANG_VERSION",
    "EXPECTED_TORCH_COMMIT",
    "EXPECTED_TORCH_VERSION",
    "ALLOWED_PYPTO_PROVIDERS",
    "ActivityKind",
    "ArtifactRecord",
    "CoverageAuditor",
    "CoverageMode",
    "CoverageSummary",
    "EventScope",
    "KernelEvent",
    "KernelProvenance",
    "ProvenanceOrigin",
    "TRACE_COLLECTOR_REVISION",
    "TraceManifest",
    "compute_artifact_registry_digest",
    "compute_trace_digest",
]
