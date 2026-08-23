"""Commit-pinned static inventory of Qwen3.5 CUDA text-forward compute sites."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CoverageProvider(str, Enum):
    """Required strict-mode owner after framework adaptation."""

    GENERIC = "pypto.generic"
    MATMUL = "pypto.matmul"
    ATTENTION = "pypto.attention"
    GDN = "pypto.gdn"
    HOST_ONLY = "host-only"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ComputeSite:
    source: str
    symbol: str
    provider: CoverageProvider
    rationale: str


@dataclass(frozen=True, slots=True)
class InventoryAudit:
    sources: tuple[tuple[str, str], ...]
    sites: tuple[tuple[str, str, str, tuple[int, ...]], ...]


SGLANG_SOURCE_SPECS = (
    SourceSpec(
        "qwen35",
        "python/sglang/srt/models/qwen3_5.py",
        "cd580d4cd0dff194890e7b58bbc0a80d69bab1aea4ff5eb6c0b272e524bca782",
    ),
    SourceSpec(
        "gdn_backend",
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "250554948366d8c67285d538591d593cbc86844594e54caa44a1e13ed7d4250c",
    ),
    SourceSpec(
        "attention_registry",
        "python/sglang/srt/layers/attention/attention_registry.py",
        "36bb36a0de1597de6adfaad1a478732e2751c3dd0f2edbc632a82c0bc6394b68",
    ),
    SourceSpec(
        "dense_mlp",
        "python/sglang/srt/models/qwen2_moe.py",
        "20467a642243f2160f34381c31468b85c60c33316e4a33b38c07be3ac53424df",
    ),
    SourceSpec(
        "qwen3_vl_wrapper",
        "python/sglang/srt/models/qwen3_vl.py",
        "0dbf764335424f3fdb67c34178aae5ddf4f1b6455fdf4de7225917785becf460",
    ),
    SourceSpec(
        "logits_processor",
        "python/sglang/srt/layers/logits_processor.py",
        "43a0674df59067223fb793d78bb9f947049d2f983704a09a3c67282bf654cad0",
    ),
    SourceSpec(
        "layernorm",
        "python/sglang/srt/layers/layernorm.py",
        "cca629524c065d0b298167fb21c4da3786b380f0771799e8f3d0a1a4b31084bb",
    ),
    SourceSpec(
        "radix_attention",
        "python/sglang/srt/layers/radix_attention.py",
        "9e51244758e1c0923fc03b08045039900fc8ca3138485a6d6c47926f0cd423d1",
    ),
    SourceSpec(
        "radix_linear_attention",
        "python/sglang/srt/layers/radix_linear_attention.py",
        "4af975b5e0ea2b17505920a43509372200fffc62eebed2e4b04346175710a179",
    ),
    SourceSpec(
        "activation",
        "python/sglang/srt/layers/activation.py",
        "231a49e1f2f3f2237415e77d85d36c159c2ea3424f0197100f469252164bab7e",
    ),
)


QWEN35_COMPUTE_SITES = (
    ComputeSite(
        "qwen35",
        "MergedColumnParallelLinear",
        CoverageProvider.MATMUL,
        "GDN QKVZ/BA and dense MLP merged projections",
    ),
    ComputeSite(
        "qwen35",
        "QKVParallelLinear",
        CoverageProvider.MATMUL,
        "full-attention QKV plus output-gate projection",
    ),
    ComputeSite(
        "qwen35",
        "RowParallelLinear",
        CoverageProvider.MATMUL,
        "attention/GDN/MLP output projections",
    ),
    ComputeSite(
        "qwen35",
        "VocabParallelEmbedding",
        CoverageProvider.GENERIC,
        "embedding gather; TP=1 text path",
    ),
    ComputeSite(
        "qwen35",
        "GemmaRMSNorm",
        CoverageProvider.GENERIC,
        "residual/input/final and QK normalization",
    ),
    ComputeSite(
        "qwen35",
        "RMSNormGated",
        CoverageProvider.GENERIC,
        "initially generic pointwise/reduction; later GDN epilogue fusion",
    ),
    ComputeSite(
        "qwen35",
        "RadixAttention",
        CoverageProvider.ATTENTION,
        "paged full-attention backend abstraction",
    ),
    ComputeSite(
        "qwen35",
        "RadixLinearAttention",
        CoverageProvider.GDN,
        "stateful linear-attention backend abstraction",
    ),
    ComputeSite(
        "qwen35",
        "fused_qkvzba_split_reshape_cat_contiguous",
        CoverageProvider.GDN,
        "packed GDN projection split/layout preprocessing",
    ),
    ComputeSite(
        "qwen35",
        "fused_qk_gemma_rmsnorm_rope_gate",
        CoverageProvider.GENERIC,
        "express as native ops first; promote only a generic template if needed",
    ),
    ComputeSite(
        "qwen35",
        "fused_qk_gemma_rmsnorm_with_gate",
        CoverageProvider.GENERIC,
        "non-NVIDIA branch remains inventoried to prevent hidden mixed paths",
    ),
    ComputeSite(
        "qwen35",
        "fused_qk_gemma_rmsnorm",
        CoverageProvider.GENERIC,
        "Q/K row reductions and scale",
    ),
    ComputeSite(
        "qwen35",
        "fused_sigmoid_mul",
        CoverageProvider.GENERIC,
        "attention output gate epilogue",
    ),
    ComputeSite(
        "qwen35",
        "triton.cdiv",
        CoverageProvider.HOST_ONLY,
        "host shape arithmetic, not a compute kernel; retain zero GPU launches",
    ),
    ComputeSite(
        "qwen35",
        "torch.cat",
        CoverageProvider.GDN,
        "unfused packed GDN Q/K/V layout path",
    ),
    ComputeSite(
        "qwen35",
        "torch.zeros_like",
        CoverageProvider.GENERIC,
        "DP-attention padding path; TP=1 still inventory-audited",
    ),
    ComputeSite(
        "gdn_backend",
        "causal_conv1d_update",
        CoverageProvider.GDN,
        "decode convolution-state update",
    ),
    ComputeSite(
        "gdn_backend",
        "causal_conv1d_fn",
        CoverageProvider.GDN,
        "chunked prefill/extend causal convolution",
    ),
    ComputeSite(
        "gdn_backend",
        "fused_gdn_gating",
        CoverageProvider.GDN,
        "prefill decay/gate/beta preprocessing",
    ),
    ComputeSite(
        "gdn_backend",
        "fused_qkv_split_gdn_prefill",
        CoverageProvider.GDN,
        "prefill packed projection split/reshape",
    ),
    ComputeSite(
        "gdn_backend",
        "GDNKernelDispatcher",
        CoverageProvider.GDN,
        "decode/prefill/verify kernel-provider selection",
    ),
    ComputeSite(
        "dense_mlp",
        "MergedColumnParallelLinear",
        CoverageProvider.MATMUL,
        "dense MLP gate/up projection",
    ),
    ComputeSite(
        "dense_mlp",
        "RowParallelLinear",
        CoverageProvider.MATMUL,
        "dense MLP down projection",
    ),
    ComputeSite(
        "dense_mlp",
        "SiluAndMul",
        CoverageProvider.GENERIC,
        "dense SwiGLU activation between PyPTO matmuls",
    ),
    ComputeSite(
        "qwen3_vl_wrapper",
        "ParallelLMHead",
        CoverageProvider.MATMUL,
        "text-only wrapper LM-head projection when weights are not tied",
    ),
    ComputeSite(
        "qwen3_vl_wrapper",
        "LogitsProcessor",
        CoverageProvider.GENERIC,
        "logit selection/postprocessing wrapper after the LM-head matmul",
    ),
    ComputeSite(
        "logits_processor",
        "torch.matmul",
        CoverageProvider.MATMUL,
        "BF16 LM-head projection paths",
    ),
    ComputeSite(
        "logits_processor",
        "torch.ops.sgl_kernel.weight_packed_linear",
        CoverageProvider.MATMUL,
        "quantized branch is outside v1 scope but must never become a hidden strict fallback",
    ),
    ComputeSite(
        "logits_processor",
        "fused_softcap",
        CoverageProvider.GENERIC,
        "optional logits soft-cap remains a generic pointwise coverage obligation",
    ),
    ComputeSite(
        "layernorm",
        "gemma_fused_add_rmsnorm",
        CoverageProvider.GENERIC,
        "SGLang CUDA fused residual Gemma RMSNorm must become compiler-generated",
    ),
    ComputeSite(
        "layernorm",
        "gemma_rmsnorm",
        CoverageProvider.GENERIC,
        "SGLang CUDA Gemma RMSNorm must become compiler-generated",
    ),
    ComputeSite(
        "layernorm",
        "rms_norm_batch_invariant",
        CoverageProvider.GENERIC,
        "optional deterministic norm path remains a row-reduction obligation",
    ),
    ComputeSite(
        "radix_attention",
        "unified_attention_with_output",
        CoverageProvider.ATTENTION,
        "piecewise-compile custom-op boundary for full attention",
    ),
    ComputeSite(
        "radix_attention",
        "unified_attention_with_output_and_lse",
        CoverageProvider.ATTENTION,
        "attention custom-op variant with explicit LSE output",
    ),
    ComputeSite(
        "radix_attention",
        "breakable_unified_attention_with_output",
        CoverageProvider.ATTENTION,
        "breakable CUDA Graph full-attention boundary",
    ),
    ComputeSite(
        "radix_attention",
        "breakable_unified_attention_with_output_and_lse",
        CoverageProvider.ATTENTION,
        "breakable CUDA Graph full-attention boundary with LSE output",
    ),
    ComputeSite(
        "radix_attention",
        "attention_with_output_extra_kwargs",
        CoverageProvider.ATTENTION,
        "full-attention boundary carrying non-schema auxiliary tensors",
    ),
    ComputeSite(
        "radix_attention",
        "breakable_attention_with_output_extra_kwargs",
        CoverageProvider.ATTENTION,
        "breakable full-attention boundary carrying auxiliary tensors",
    ),
    ComputeSite(
        "radix_linear_attention",
        "unified_linear_attention_with_output",
        CoverageProvider.GDN,
        "piecewise-compile custom-op boundary for GDN linear attention",
    ),
    ComputeSite(
        "radix_linear_attention",
        "bcg_unified_linear_attention_with_output",
        CoverageProvider.GDN,
        "breakable CUDA Graph GDN custom-op boundary",
    ),
    ComputeSite(
        "activation",
        "silu_and_mul",
        CoverageProvider.GENERIC,
        "CUDA SwiGLU helper must be replaced by fused PyPTO pointwise codegen",
    ),
)


def _call_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parent = _call_name(function.value)
        return function.attr if parent is None else f"{parent}.{function.attr}"
    return None


def _symbol_lines(tree: ast.AST) -> dict[str, tuple[int, ...]]:
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name is not None:
                found.setdefault(name, []).append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, []).append(node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.lineno)
    return {name: tuple(sorted(lines)) for name, lines in found.items()}


def audit_qwen35_inventory(sglang_root: str | Path) -> InventoryAudit:
    """Verify every frozen source and required call site without importing SGLang."""

    root = Path(sglang_root).resolve()
    sources: dict[str, tuple[SourceSpec, dict[str, tuple[int, ...]]]] = {}
    source_results: list[tuple[str, str]] = []
    for spec in SGLANG_SOURCE_SPECS:
        path = root / spec.relative_path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.sha256:
            raise RuntimeError(
                f"SGLang source fingerprint mismatch for {spec.name}: "
                f"expected {spec.sha256}, got {digest}"
            )
        sources[spec.name] = (
            spec,
            _symbol_lines(ast.parse(data, filename=str(path))),
        )
        source_results.append((spec.name, digest))

    site_results: list[tuple[str, str, str, tuple[int, ...]]] = []
    for site in QWEN35_COMPUTE_SITES:
        calls = sources[site.source][1]
        lines = calls.get(site.symbol, ())
        if not lines:
            raise RuntimeError(
                f"required Qwen3.5 compute site {site.source}:{site.symbol} is absent"
            )
        site_results.append((site.source, site.symbol, site.provider.value, lines))
    return InventoryAudit(tuple(source_results), tuple(site_results))


__all__ = (
    "ComputeSite",
    "CoverageProvider",
    "InventoryAudit",
    "QWEN35_COMPUTE_SITES",
    "SGLANG_SOURCE_SPECS",
    "SourceSpec",
    "audit_qwen35_inventory",
)
