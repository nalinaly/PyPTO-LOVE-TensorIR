"""Pinned SGLang state-lifecycle inventory for future StateBundle adaptation."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType


class LifecycleAction(str, Enum):
    ROUTE_SELECTION = "route-selection"
    SLOT_ALLOCATION = "slot-allocation"
    SLOT_TRANSLATION = "slot-translation"
    SLOT_REUSE = "slot-reuse"
    NEW_SLOT_ZERO = "new-slot-zero"
    RESTORE_COW = "restore-copy-on-write"
    CHECKPOINT_SNAPSHOT = "checkpoint-snapshot"
    CHECKPOINT_DONATE = "checkpoint-donate"
    CHECKPOINT_COMMIT = "checkpoint-commit"
    FORWARD_SNAPSHOT = "forward-metadata-snapshot"
    COMPLETION_HANDOFF = "completion-handoff"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LifecycleSite:
    source: str
    symbol: str
    action: LifecycleAction
    rationale: str


@dataclass(frozen=True, slots=True)
class StateLifecycleAudit:
    sources: tuple[tuple[str, str], ...]
    sites: tuple[tuple[str, str, str, tuple[int, ...]], ...]
    scope_digest: str


STATE_LIFECYCLE_SCOPE_V1 = MappingProxyType(
    {
        "precision": "full-precision",
        "mamba_radix_strategies": (
            "no_buffer",
            "extra_buffer",
            "extra_buffer_lazy",
        ),
        "unified_radix_selected": True,
        "legacy_mamba_radix_selected": False,
        "unified_slot_translation": True,
        "page_major_envelope": True,
        "overlap_schedule": True,
        "speculative": False,
        "mtp": False,
        "replayssm": False,
        "int8_checkpoint": False,
        "hicache": False,
        "disaggregation": False,
        "capture_supported": False,
        "registration_ready": False,
        "requires_single_dso": True,
        "requires_target_info": True,
        "requires_compile_request": True,
        "requires_current_stream_runtime": True,
        "requires_statebundle_runtime": True,
        "implementation_order": (
            "single-dso",
            "target-info",
            "compile-request-current-stream",
            "statebundle-runtime",
            "sglang-plugin-adapter",
        ),
    }
)


def _scope_payload(value) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("state lifecycle scope must be a mapping")
    payload = dict(value)
    expected = dict(STATE_LIFECYCLE_SCOPE_V1)
    if payload != expected:
        missing = sorted(set(expected) - set(payload))
        unknown = sorted(set(payload) - set(expected))
        raise RuntimeError(
            "state lifecycle scope mismatch: "
            f"missing={missing}, unknown={unknown}, expected={expected}, got={payload}"
        )
    return payload


def validate_state_lifecycle_scope(value) -> str:
    """Fail closed unless every normalized lifecycle capability matches v1."""

    payload = _scope_payload(value)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


SGLANG_STATE_SOURCE_SPECS = (
    SourceSpec(
        "memory_pool",
        "python/sglang/srt/mem_cache/memory_pool.py",
        "41e87128dbdf9178dad65f695e7a98428ecfeebc1e3407fee57df5067f4205c3",
    ),
    SourceSpec(
        "slot_allocator",
        "python/sglang/srt/mem_cache/allocator/mamba.py",
        "b3d4561b7f96aa55f615c491b3d5a8c8e72e156fd7e6cf639ee7f87cfd80674c",
    ),
    SourceSpec(
        "legacy_mamba_radix_cache",
        "python/sglang/srt/mem_cache/mamba_radix_cache.py",
        "b70a931986791a07b02e85645e1faa2b6071e8aba1ccce630ae4f69072fcfd12",
    ),
    SourceSpec(
        "radix_registry",
        "python/sglang/srt/mem_cache/registry.py",
        "b60134d1ba8c6fdad7b30f268b5d96264102bdfe9881e2401264cb7f99fd31cd",
    ),
    SourceSpec(
        "unified_radix_cache",
        "python/sglang/srt/mem_cache/unified_radix_cache.py",
        "4a395a6b652135753f9b92f0b01de63e497adbcdd44401f844e552b12346f57c",
    ),
    SourceSpec(
        "mamba_component",
        "python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
        "9bfcb7a20af48c2fce8cb63537db54cb4231456d9c04896370e290611e0ba799",
    ),
    SourceSpec(
        "unified_memory_pool",
        "python/sglang/srt/mem_cache/unified_memory_pool.py",
        "9875cd709458de684243a802f2da6d0ba6ca3ed6a61dd35d69800cd9d97cf1f9",
    ),
    SourceSpec(
        "linear_backend",
        "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
        "23636d986589dcd73de7c2cace59589a8b1e6c62b441d9ec78d76939bcec4a4c",
    ),
    SourceSpec(
        "schedule_batch",
        "python/sglang/srt/managers/schedule_batch.py",
        "348cd231d705a618c3d4a80a356470f92a430a941c97b429a8b5d16336fd035c",
    ),
    SourceSpec(
        "forward_batch",
        "python/sglang/srt/model_executor/forward_batch_info.py",
        "33755cc114c2b7aec62a57688c92507aa86f12bcc7fd3e1ea0c674ed64600875",
    ),
    SourceSpec(
        "model_runner",
        "python/sglang/srt/model_executor/model_runner.py",
        "2bbe91845a9482d04ce0d1ea870a41397be1b4fbe436e0a2a4f555ae7cc67adb",
    ),
)


STATE_LIFECYCLE_SITES = (
    LifecycleSite(
        "radix_registry",
        "default_radix_cache_factory",
        LifecycleAction.ROUTE_SELECTION,
        "pinned hybrid-SSM route selects UnifiedRadixCache",
    ),
    LifecycleSite(
        "radix_registry",
        "_create_unified_radix_cache",
        LifecycleAction.ROUTE_SELECTION,
        "constructs the selected unified tree and Mamba component",
    ),
    LifecycleSite(
        "unified_radix_cache",
        "UnifiedRadixCache",
        LifecycleAction.ROUTE_SELECTION,
        "active pinned hybrid-SSM prefix-cache implementation",
    ),
    LifecycleSite(
        "unified_radix_cache",
        "match_prefix",
        LifecycleAction.RESTORE_COW,
        "active prefix match invokes component COW finalization",
    ),
    LifecycleSite(
        "unified_radix_cache",
        "cache_finished_req",
        LifecycleAction.CHECKPOINT_COMMIT,
        "finished request checkpoint commit",
    ),
    LifecycleSite(
        "unified_radix_cache",
        "cache_unfinished_req",
        LifecycleAction.CHECKPOINT_COMMIT,
        "chunked unfinished request checkpoint commit",
    ),
    LifecycleSite(
        "mamba_component",
        "MambaComponent",
        LifecycleAction.ROUTE_SELECTION,
        "selected unified-tree Mamba lifecycle component",
    ),
    LifecycleSite(
        "mamba_component",
        "finalize_match_result_in_cache",
        LifecycleAction.RESTORE_COW,
        "records deferred COW source and destination",
    ),
    LifecycleSite(
        "mamba_component",
        "_alloc_mamba_slot",
        LifecycleAction.SLOT_ALLOCATION,
        "allocates checkpoint or active destination slot",
    ),
    LifecycleSite(
        "mamba_component",
        "prepare_for_caching_req",
        LifecycleAction.CHECKPOINT_SNAPSHOT,
        "chooses snapshot/donate source at the tracked boundary",
    ),
    LifecycleSite(
        "mamba_component",
        "commit_insert_component_data",
        LifecycleAction.CHECKPOINT_COMMIT,
        "publishes the complete Mamba component generation into the tree",
    ),
    LifecycleSite(
        "mamba_component",
        "cleanup_after_caching_req",
        LifecycleAction.SLOT_REUSE,
        "free/keep handoff after cache insertion",
    ),
    LifecycleSite(
        "memory_pool",
        "MambaPool",
        LifecycleAction.COMPLETION_HANDOFF,
        "physical full-state pool and component views",
    ),
    LifecycleSite(
        "memory_pool",
        "clear_slots",
        LifecycleAction.NEW_SLOT_ZERO,
        "current implementation clears every conv/temporal component",
    ),
    LifecycleSite(
        "memory_pool",
        "copy_from",
        LifecycleAction.RESTORE_COW,
        "current exact full-state slot clone",
    ),
    LifecycleSite(
        "memory_pool",
        "HybridReqToTokenPool",
        LifecycleAction.SLOT_ALLOCATION,
        "request-to-physical-state pool ownership",
    ),
    LifecycleSite(
        "memory_pool",
        "alloc",
        LifecycleAction.SLOT_ALLOCATION,
        "sets active slot and deferred-clear lifecycle",
    ),
    LifecycleSite(
        "memory_pool",
        "translate_mamba_indices",
        LifecycleAction.SLOT_TRANSLATION,
        "base virtual-to-physical slot translation contract",
    ),
    LifecycleSite(
        "memory_pool",
        "free_mamba_cache",
        LifecycleAction.SLOT_REUSE,
        "slot reuse handoff after request/checkpoint completion",
    ),
    LifecycleSite(
        "memory_pool",
        "_alloc_ping_pong_buffer",
        LifecycleAction.SLOT_ALLOCATION,
        "extra-buffer ping-pong slot ownership",
    ),
    LifecycleSite(
        "memory_pool",
        "set_mamba_ping_pong_slot",
        LifecycleAction.SLOT_ALLOCATION,
        "replaces one tracked ping-pong destination slot",
    ),
    LifecycleSite(
        "memory_pool",
        "donate_mamba_ping_pong_slot",
        LifecycleAction.CHECKPOINT_DONATE,
        "donates a complete tracked generation and installs its replacement",
    ),
    LifecycleSite(
        "unified_memory_pool",
        "UnifiedMambaPool",
        LifecycleAction.SLOT_TRANSLATION,
        "unified/page-major component views preserve physical slot layout",
    ),
    LifecycleSite(
        "unified_memory_pool",
        "UnifiedHybridReqToTokenPool",
        LifecycleAction.SLOT_TRANSLATION,
        "selected unified-memory pool specialization",
    ),
    LifecycleSite(
        "unified_memory_pool",
        "translate_mamba_indices",
        LifecycleAction.SLOT_TRANSLATION,
        "unified virtual-to-physical state-slot translation",
    ),
    LifecycleSite(
        "slot_allocator",
        "MambaSlotAllocator",
        LifecycleAction.SLOT_ALLOCATION,
        "request-level slots with reserved dummy slot zero",
    ),
    LifecycleSite(
        "slot_allocator",
        "free",
        LifecycleAction.SLOT_REUSE,
        "free-list reuse must follow completion handoff",
    ),
    LifecycleSite(
        "linear_backend",
        "_track_mamba_state_extend",
        LifecycleAction.CHECKPOINT_SNAPSHOT,
        "prefill/extend checkpoint state tracking",
    ),
    LifecycleSite(
        "linear_backend",
        "_track_mamba_state_decode",
        LifecycleAction.CHECKPOINT_SNAPSHOT,
        "decode checkpoint state tracking",
    ),
    LifecycleSite(
        "schedule_batch",
        "Req",
        LifecycleAction.SLOT_ALLOCATION,
        "request carries deferred COW source and clear intent",
    ),
    LifecycleSite(
        "schedule_batch",
        "_mamba_radix_cache_v2_req_prepare_for_extend",
        LifecycleAction.CHECKPOINT_SNAPSHOT,
        "selects exact tracked token boundary",
    ),
    LifecycleSite(
        "schedule_batch",
        "_collect_deferred_mamba_cow_and_clear",
        LifecycleAction.FORWARD_SNAPSHOT,
        "snapshots deferred zero/copy mappings once per forward",
    ),
    LifecycleSite(
        "schedule_batch",
        "mamba_lazy_prealloc_at_boundary",
        LifecycleAction.SLOT_ALLOCATION,
        "lazy extra-buffer allocation at the tracked checkpoint boundary",
    ),
    LifecycleSite(
        "forward_batch",
        "ForwardBatch",
        LifecycleAction.FORWARD_SNAPSHOT,
        "owns forward-resolved COW/clear mapping tensors",
    ),
    LifecycleSite(
        "model_runner",
        "_maybe_execute_deferred_mamba_cow_and_clear",
        LifecycleAction.COMPLETION_HANDOFF,
        "current forward-stream execution point before state readers",
    ),
)


def _symbol_lines(tree: ast.AST) -> dict[str, tuple[int, ...]]:
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, []).append(node.lineno)
    return {name: tuple(sorted(lines)) for name, lines in found.items()}


def audit_state_lifecycle_inventory(sglang_root: str | Path) -> StateLifecycleAudit:
    """Verify pinned lifecycle sources/symbols without importing SGLang."""

    root = Path(sglang_root).resolve()
    sources: dict[str, dict[str, tuple[int, ...]]] = {}
    source_results: list[tuple[str, str]] = []
    for spec in SGLANG_STATE_SOURCE_SPECS:
        path = root / spec.relative_path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.sha256:
            raise RuntimeError(
                f"SGLang state source fingerprint mismatch for {spec.name}: "
                f"expected {spec.sha256}, got {digest}"
            )
        sources[spec.name] = _symbol_lines(ast.parse(data, filename=str(path)))
        source_results.append((spec.name, digest))

    site_results: list[tuple[str, str, str, tuple[int, ...]]] = []
    for site in STATE_LIFECYCLE_SITES:
        lines = sources[site.source].get(site.symbol, ())
        if not lines:
            raise RuntimeError(
                f"required SGLang state lifecycle site "
                f"{site.source}:{site.symbol} is absent"
            )
        site_results.append((site.source, site.symbol, site.action.value, lines))
    return StateLifecycleAudit(
        tuple(source_results),
        tuple(site_results),
        validate_state_lifecycle_scope(STATE_LIFECYCLE_SCOPE_V1),
    )


__all__ = (
    "LifecycleAction",
    "LifecycleSite",
    "SGLANG_STATE_SOURCE_SPECS",
    "STATE_LIFECYCLE_SCOPE_V1",
    "STATE_LIFECYCLE_SITES",
    "SourceSpec",
    "StateLifecycleAudit",
    "audit_state_lifecycle_inventory",
    "validate_state_lifecycle_scope",
)
