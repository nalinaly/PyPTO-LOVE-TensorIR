# DECISIONS

## D-0001: Three-project implementation boundary

`pypto` owns compiler/runtime infrastructure; `pypto-kernels` owns every
custom high-performance operator; `pypto-framework-plugins` owns only pinned
framework adaptation.

## D-0002: Official framework source remains zero-diff

Torch and SGLang changes are installed through OOT registration and SGLang's
plugin/HookRegistry mechanisms. Compatibility code is exact-SHA gated and is
not described as a stable upstream ABI.

## D-0003: RTX 5090 Laptop is the final target

All capacity and performance claims refer to the local 24 GB/82-SM Laptop GPU,
not the 32 GB desktop RTX 5090.

## D-0004: 9B capacity does not justify hidden scope relaxation

Minimal BF16 9B failure stops the project for user direction. CPU offload,
quantization, or external compute fallback are not automatic alternatives.

## D-0005: Protected AMD/zcode scopes are never managed by this project

This project observes their resource use only. It never edits their trees,
reuses their processes, shares caches/endpoints, or signals their PIDs.

## D-0006: Runtime coverage is a manifest-bound evidence decision

Provider names alone never prove coverage. A strict decision requires a fixed
collector protocol, closed normalized trace, immutable artifact registry,
exact digest/provenance reconciliation, non-vacuous call and GPU-time totals,
latched violations, and single-owner durable reports. Collector completeness,
model correctness, and performance remain separate acceptance evidence.

## D-0007: Operator ABI and payload identity are producer-owned

`pypto-kernels` publishes the only framework-adapter ABI manifest. Plugins pin
and independently recompute it, validate live bindings, and separately prove
wheel or editable source ownership. Package version equality alone is never a
compatibility claim. Native executable payloads stay rejected until their own
digest-bound manifest and readiness gate exist.

## D-0008: Paired-state transfer is generic runtime infrastructure

GDN owns the semantic fact that BF16 conv and FP32 recurrent state form one
generation. Exact zero/copy/checkpoint, leases, generations, stream enqueue and
completion belong to the generic PyPTO compiler/runtime. The framework plugin
translates Radix lifecycle but does not execute copies. State transfer is not a
GDN operator, operator artifact or operator tuning record.
