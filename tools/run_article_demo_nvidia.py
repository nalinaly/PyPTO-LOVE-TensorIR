#!/usr/bin/env python3
"""Run computational article examples through the NVIDIA compatibility path.

The upstream article files are imported and hashed but never edited.  The
compatibility implementations live in this file: ``hello_world`` uses the
strict PyPTO/TensorIR/CUDA-Tile bridge; the remaining small teaching examples
use independent CUDA Torch references until a corresponding strict adapter is
available.  Hardware-facing and unmapped entries are rejected by policy rather
than silently falling back.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo" / "pypto-lib"
MANIFEST = DEMO_ROOT / "SOURCE_MANIFEST.json"
POLICY_PATH = (
    ROOT / "state" / "evidence" / "article-demo-compatibility-policy-current.json"
)
ARTICLE_COMMIT = "6c292d30ccc787ee4e1fe61541fd3faec0dafa65"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def corpus_sha256(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        manifest.get("files", []), key=lambda item: str(item.get("path", ""))
    ):
        relative = str(record["path"])
        path = (DEMO_ROOT / relative).resolve()
        if DEMO_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"imported corpus file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    temporary.replace(path)


def load_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("kind") != "article-demo-compatibility-policy":
        raise ValueError("unexpected article demo compatibility policy")
    if policy.get("upstream_commit") != ARTICLE_COMMIT:
        raise ValueError("compatibility policy is not article-time locked")
    manifest = load_manifest()
    if (
        policy.get("manifest_sha256") != sha256_file(MANIFEST)
        or policy.get("corpus_sha256") != corpus_sha256(manifest)
    ):
        raise ValueError("compatibility policy manifest hash is stale")
    return policy


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("kind") != "article-demo-provenance":
        raise ValueError("unexpected article demo manifest")
    if manifest.get("upstream", {}).get("commit") != ARTICLE_COMMIT:
        raise ValueError("article demo manifest is not article-time locked")
    return manifest


def source_record(relative: str, manifest: dict[str, Any]) -> dict[str, object]:
    path = (DEMO_ROOT / relative).resolve()
    if DEMO_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"demo path is outside the imported tree: {relative}")
    record = next((item for item in manifest["files"] if item.get("path") == relative), None)
    if not isinstance(record, dict):
        raise ValueError(f"demo is absent from SOURCE_MANIFEST.json: {relative}")
    observed = sha256_file(path)
    if observed != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        raise ValueError(f"imported demo source changed: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def policy_entry(relative: str, policy: dict[str, Any]) -> dict[str, Any]:
    entry = next((item for item in policy["entries"] if item.get("path") == relative), None)
    if not isinstance(entry, dict):
        raise ValueError(f"demo has no compatibility policy: {relative}")
    return entry


def import_demo(relative: str) -> Any:
    path = (DEMO_ROOT / relative).resolve()
    module_name = "_article_demo_" + hashlib.sha256(relative.encode()).hexdigest()[:16]
    # The article runs each model entry from its own directory, where sibling
    # modules such as ``config.py`` and ``golden.py`` are importable by their
    # original short names.  Keep the parent directory available to preserve
    # that source-level import contract; matrix entries execute in fresh
    # subprocesses, so sibling-name collisions cannot cross model families.
    for import_root in (DEMO_ROOT, path.parent):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _builder(module: Any) -> Callable[[], list[Any]]:
    for name in ("build_specs", "build_tensor_specs"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise ValueError("article demo does not expose build_specs/build_tensor_specs")


def _build_specs(module: Any, relative: str) -> list[Any]:
    """Call an article builder with a bounded, documented decode fixture."""

    builder = _builder(module)
    required = [
        parameter
        for parameter in inspect.signature(builder).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not required:
        return list(builder())
    names = tuple(parameter.name for parameter in required)
    if names == ("B", "S"):
        return list(builder(int(module.DECODE_BATCH), int(module.DECODE_SEQ)))
    if names == ("token_count", "vocab_size"):
        return list(builder(int(module.DECODE_TOKENS), 256))
    raise ValueError(
        f"no bounded NVIDIA fixture for {relative} builder parameters {names}"
    )


GOLDEN_FUNCTIONS: dict[str, str] = {
    "models/deepseek_v4_flash_mtp/rmsnorm.py": "golden_rms_norm_test",
    "models/deepseek_v4_flash_mtp/hc_post.py": "golden_hc_post",
    "models/deepseek_v4_flash_mtp/prefill_indexer.py": "golden_prefill_indexer",
    "models/deepseek_v4_flash_mtp/prefill_swa.py": "golden_prefill_attention_swa",
    "models/deepseek_v4_flash_mtp/prefill_hca.py": "golden_prefill_attention_hca",
    "models/deepseek_v4_flash_mtp/prefill_csa.py": "golden_prefill_attention_csa",
}


def _golden_name(module: Any, relative: str | None = None) -> str:
    if relative is not None and relative in GOLDEN_FUNCTIONS:
        name = GOLDEN_FUNCTIONS[relative]
        if not callable(getattr(module, name, None)):
            raise ValueError(f"declared golden function is missing for {relative}: {name}")
        return name
    candidates = [
        name
        for name, value in vars(module).items()
        if name.startswith("golden_") and callable(value)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one golden function, found {len(candidates)}")
    return candidates[0]


def _golden(
    module: Any, relative: str | None = None
) -> Callable[[dict[str, Any]], object]:
    return getattr(module, _golden_name(module, relative))


def _spec_values(
    module: Any, relative: str, seed: int
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    specs = _build_specs(module, relative)
    if not specs:
        raise ValueError("article demo returned no tensor/scalar specs")
    values = {
        spec.name: spec.create_tensor()
        if hasattr(spec, "create_tensor")
        else spec.value.clone()
        if isinstance(spec.value, torch.Tensor)
        else spec.value
        for spec in specs
    }
    expected = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }
    _golden(module, relative)(expected)
    return specs, values, expected


def _sparse_attention_output(
    values: dict[str, Any], module: Any, rows_by_token: list[list[Any]]
):
    """Independent CUDA reference for the shared sparse-attention tail."""

    import torch

    tokens = int(module.T)
    heads = int(module.H)
    head_dim = int(module.HEAD_DIM)
    padded_topk = int(
        module.PADDED_TOPK
        if hasattr(module, "PADDED_TOPK")
        else module.PREFILL_SPARSE_PAD
    )
    tile_width = int(
        module.ATTN_K_TILE
        if hasattr(module, "ATTN_K_TILE")
        else module.PREFILL_ATTN_TILE
    )
    scale = float(module.SOFTMAX_SCALE)
    negative_infinity = float(
        module.NEG_INF if hasattr(module, "NEG_INF") else module.FP32_NEG_INF
    )
    nope_dim = int(module.NOPE_DIM)
    half_rope = int(
        module.HALF_ROPE if hasattr(module, "HALF_ROPE") else module.ROPE_HALF
    )
    batches = int(module.B)
    sequence = tokens // batches
    groups = int(module.O_GROUPS)
    group_input = int(module.O_GROUP_IN)
    lora = int(module.O_LORA)
    d = int(module.D)
    scale_max = float(module.INT8_SCALE_MAX)
    amax_eps = float(module.INT8_AMAX_EPS)
    q = values["q"].float()
    sink = values["attn_sink"].float()
    output = torch.zeros(
        tokens, heads, head_dim, dtype=torch.float32, device=q.device
    )

    for token in range(tokens):
        rows = rows_by_token[token]
        if len(rows) > padded_topk:
            raise ValueError(
                f"sparse row count exceeds padded top-k: {len(rows)} > {padded_topk}"
            )
        valid = [row is not None for row in rows]
        rows = list(rows) + [None] * (padded_topk - len(rows))
        valid.extend([False] * (padded_topk - len(valid)))
        if not any(valid):
            continue
        maxima = []
        denominators = []
        numerators = []
        for start in range(0, padded_topk, tile_width):
            tile_rows = rows[start : start + tile_width]
            tile = torch.zeros(
                tile_width, head_dim, dtype=torch.float32, device=q.device
            )
            tile_valid = torch.tensor(
                valid[start : start + tile_width], dtype=torch.bool, device=q.device
            )
            for index, row in enumerate(tile_rows):
                if row is not None:
                    tile[index] = row.float()
            scores = torch.matmul(q[token], tile.transpose(0, 1)) * scale
            scores = scores.masked_fill(~tile_valid.unsqueeze(0), negative_infinity)
            maximum = scores.max(dim=-1, keepdim=True).values
            exponent = torch.exp(scores - maximum).masked_fill(
                ~tile_valid.unsqueeze(0), 0.0
            )
            maxima.append(maximum)
            denominators.append(exponent.sum(dim=-1, keepdim=True))
            numerators.append(
                torch.matmul(
                    exponent.to(torch.bfloat16).float(),
                    tile.to(torch.bfloat16).float(),
                )
            )
        maximum = maxima[0]
        denominator = denominators[0]
        numerator = numerators[0]
        for current_max, current_denominator, current_numerator in zip(
            maxima[1:], denominators[1:], numerators[1:]
        ):
            next_max = torch.maximum(maximum, current_max)
            alpha = torch.exp(maximum - next_max)
            beta = torch.exp(current_max - next_max)
            denominator = alpha * denominator + beta * current_denominator
            numerator = alpha * numerator + beta * current_numerator
            maximum = next_max
        denominator = denominator + torch.exp(sink.unsqueeze(-1) - maximum)
        output[token] = numerator / denominator

    pair = output[..., nope_dim:].unflatten(-1, (-1, 2))
    even, odd = pair[..., 0], pair[..., 1]
    cos = values["freqs_cos"].float()[:, :half_rope].unsqueeze(1)
    sin = values["freqs_sin"].float()[:, :half_rope].unsqueeze(1)
    inverse_even = (even * cos + odd * sin).to(torch.bfloat16).float()
    inverse_odd = (odd * cos - even * sin).to(torch.bfloat16).float()
    output = torch.cat(
        [
            output[..., :nope_dim],
            torch.stack([inverse_even, inverse_odd], dim=-1).flatten(-2),
        ],
        dim=-1,
    ).to(torch.bfloat16)

    model_output = output.float().reshape(
        batches, sequence, groups, group_input
    )
    projected = torch.einsum(
        "bsgd,grd->bsgr", model_output, values["wo_a"].float()
    ).reshape(tokens, groups, lora)
    amax = projected.abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
    quant_scale = scale_max / amax
    quantized = (
        torch.round(projected * quant_scale)
        .to(torch.int32)
        .to(torch.float16)
        .to(torch.int8)
    )
    dequant_scale = 1.0 / quant_scale
    weight = values["wo_b"].reshape(d, groups, lora)
    result = torch.zeros(tokens, d, dtype=torch.float32, device=q.device)
    for group in range(groups):
        lhs = quantized[:, group]
        padded_rows = max(32, tokens)
        padded = torch.zeros(
            padded_rows, lora, dtype=torch.int8, device=q.device
        )
        padded[:tokens] = lhs
        partial = torch._int_mm(padded, weight[:, group].transpose(0, 1))[
            :tokens
        ].float()
        result += partial * dequant_scale[:, group]
    return (result * values["wo_b_scale"].float().unsqueeze(0)).to(
        torch.bfloat16
    )


def _prefill_frontend(values: dict[str, Any], module: Any):
    """Run the independent HC-pre, RMSNorm and QKV formulas for a wrapper."""

    import torch

    prefix = "models/deepseek_v4_flash_mtp/"
    hc_pre_module = import_demo(prefix + "hc_pre.py")
    qkv_module = import_demo(prefix + "qkv_proj_rope.py")
    tokens = int(module.T)
    d = int(module.D)
    hc_mult = int(module.HC_MULT)
    heads = int(module.H)
    head_dim = int(module.HEAD_DIM)
    q_lora = int(module.Q_LORA)
    x_hc = values["x_hc"].reshape(tokens, hc_mult, d)
    pre_values = {
        "x": x_hc,
        "hc_fn": values["hc_attn_fn"],
        "hc_scale": values["hc_attn_scale"],
        "hc_base": values["hc_attn_base"],
        "x_mixed": torch.zeros(
            tokens, d, dtype=torch.bfloat16, device=x_hc.device
        ),
        "post": torch.zeros(
            tokens, hc_mult, dtype=torch.float32, device=x_hc.device
        ),
        "comb": torch.zeros(
            tokens, hc_mult * hc_mult, dtype=torch.float32, device=x_hc.device
        ),
    }
    _reference_adapter(prefix + "hc_pre.py", pre_values, hc_pre_module)
    x_mixed = pre_values["x_mixed"].float()
    epsilon = float(module.EPS if hasattr(module, "EPS") else module.M.rms_norm_eps)
    x_normed = (
        x_mixed
        * torch.rsqrt(
            x_mixed.square().mean(dim=-1, keepdim=True) + epsilon
        )
        * values["attn_norm_w"].float()
    ).to(torch.bfloat16)
    positions = values["position_ids"].long()
    rope_cos = values["freqs_cos"].index_select(0, positions)
    rope_sin = values["freqs_sin"].index_select(0, positions)
    qkv_values = {
        "x": x_normed,
        "wq_a": values["wq_a"],
        "wq_b": values["wq_b"],
        "wq_b_scale": values["wq_b_scale"],
        "wkv": values["wkv"],
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "gamma_cq": values["gamma_cq"],
        "gamma_ckv": values["gamma_ckv"],
        "q": torch.zeros(
            tokens, heads, head_dim, dtype=torch.bfloat16, device=x_hc.device
        ),
        "kv": torch.zeros(
            tokens, head_dim, dtype=torch.bfloat16, device=x_hc.device
        ),
        "qr": torch.zeros(tokens, q_lora, dtype=torch.int8, device=x_hc.device),
        "qr_scale": torch.zeros(
            tokens, 1, dtype=torch.float32, device=x_hc.device
        ),
    }
    _reference_adapter(prefix + "qkv_proj_rope.py", qkv_values, qkv_module)
    return pre_values, x_normed, positions, rope_cos, rope_sin, qkv_values


def _prefill_hc_post(
    values: dict[str, Any], module: Any, pre_values: dict[str, Any], attention
):
    """Run HC-post and enforce the active-prefix zero-tail contract."""

    import torch

    prefix = "models/deepseek_v4_flash_mtp/"
    hc_post_module = import_demo(prefix + "hc_post.py")
    tokens = int(module.T)
    hc_mult = int(module.HC_MULT)
    d = int(module.D)
    post_values = {
        "x": attention,
        "residual": pre_values["x"],
        "post": pre_values["post"],
        "comb": pre_values["comb"],
        "y": torch.zeros(
            tokens, hc_mult, d, dtype=torch.float32, device=attention.device
        ),
    }
    _reference_adapter(prefix + "hc_post.py", post_values, hc_post_module)
    active = int(values["num_tokens"].item())
    if active < tokens:
        post_values["y"][active:] = 0
    return post_values["y"]


def _reference_adapter(
    relative: str, values: dict[str, Any], module: Any | None = None
) -> dict[str, Any]:
    """Compute the same mathematical result independently of the imported golden."""

    import torch

    if relative.endswith("gemm_eltwise.py"):
        values["resid"] = values["attn_out"].float() @ values["wo"].float()
        values["resid"] = values["resid"] + values["hidden_states"].float()
    elif relative.endswith("multi_proj.py"):
        x = values["x"].float()
        values["q_out"] = x @ values["wq"].float()
        values["k_out"] = x @ values["wk"].float()
        values["v_out"] = x @ values["wv"].float()
    elif relative.endswith("topk.py"):
        # The imported example's tie policy is checked by the paired values;
        # deterministic random inputs make this path exact in normal runs.
        vals, indices = torch.topk(values["scores"].float(), 16, dim=-1, sorted=True)
        values["topk_vals"] = vals
        values["topk_idx"] = indices.to(torch.int32)
    elif relative.endswith("matmul.py") or relative.endswith("gemm.py"):
        output = values["a"].float() @ values["b"].float()
        values["c"] = output
    elif relative.endswith("layer_norm.py"):
        x = values["x"].float()
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        values["y"] = (x - mean) / torch.sqrt(variance + 1e-5)
        values["y"] = values["y"] * values["gamma"].float() + values["beta"].float()
    elif relative.endswith("rms_norm.py"):
        x = values["x"].float()
        values["y"] = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        values["y"] = values["y"] * values["gamma"].float()
    elif relative == "examples/intermediate/rope.py":
        x = values["x"].float()
        half = x.shape[-1] // 2
        values["y"] = torch.cat(
            [
                x[..., :half] * values["cos"][..., :half].float()
                - x[..., half:] * values["sin"][..., :half].float(),
                x[..., half:] * values["cos"][..., half:].float()
                + x[..., :half] * values["sin"][..., half:].float(),
            ],
            dim=-1,
        )
    elif relative.endswith("softmax.py"):
        values["y"] = torch.softmax(values["x"].float(), dim=-1)
    elif relative == "models/deepseek_v4_flash_mtp/rmsnorm.py":
        if module is None:
            raise ValueError("DeepSeek RMSNorm adapter requires imported constants")
        x = values["x"].float()
        norm_w = values["norm_w"].float()
        eps = float(module.EPS)
        values["x_normed"] = (
            x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * norm_w
        ).to(torch.bfloat16)
    elif relative == "models/deepseek_v4_flash_mtp/lookup_embedding.py":
        values["hidden_states"] = values["embed_weight"].index_select(
            0, values["input_ids"].long()
        )
    elif relative == "models/deepseek_v4_flash_mtp/hc_post.py":
        if module is None:
            raise ValueError("DeepSeek HC-post adapter requires imported constants")
        hc_mult = int(module.HC_MULT)
        d = int(module.D)
        x = values["x"].float()
        residual = values["residual"].float()
        post = values["post"].float()
        comb = values["comb"].float().reshape(-1, hc_mult, hc_mult)
        y = torch.zeros(
            x.shape[0], hc_mult, d, dtype=torch.float32, device=x.device
        )
        for out_h in range(hc_mult):
            row = x * post[:, out_h : out_h + 1]
            for in_h in range(hc_mult):
                row = row + residual[:, in_h, :] * comb[:, in_h, out_h : out_h + 1]
            y[:, out_h, :] = row
        values["y"] = y
    elif relative == "models/deepseek_v4_flash_mtp/hc_head.py":
        if module is None:
            raise ValueError("DeepSeek HC-head adapter requires imported constants")
        hc_mult = int(module.HC_MULT)
        hc_dim = int(module.HC_DIM)
        d = int(module.D)
        eps = float(module.EPS)
        hc_eps = float(module.HC_EPS)
        x = values["x_hc"].float()
        x_flat = x.reshape(x.shape[0], hc_dim)
        fn = values["hc_head_fn"].float()
        sq_sum = x_flat.square().sum(dim=1, keepdim=True)
        inv = torch.rsqrt(sq_sum / hc_dim + eps)
        mixes = []
        for h in range(hc_mult):
            mix = torch.zeros(x.shape[0], 1, dtype=torch.float32, device=x.device)
            for k0 in range(0, hc_dim, 256):
                mix = mix + (
                    x_flat[:, k0 : k0 + 256]
                    * fn[h : h + 1, k0 : k0 + 256]
                ).sum(dim=1, keepdim=True)
            mixes.append(mix * inv)
        mix = torch.cat(mixes, dim=1)
        pre = torch.sigmoid(
            mix * values["hc_head_scale"].float()
            + values["hc_head_base"].float()
        ) + hc_eps
        values["y"] = sum(
            x[:, h, :] * pre[:, h : h + 1] for h in range(hc_mult)
        ).to(torch.bfloat16)
    elif relative == "models/deepseek_v4_flash_mtp/hc_pre.py":
        if module is None:
            raise ValueError("DeepSeek HC-pre adapter requires imported constants")
        hc_mult = int(module.HC_MULT)
        mix_hc = int(module.MIX_HC)
        hc_dim = int(module.HC_DIM)
        eps = float(module.NORM_EPS)
        hc_eps = float(module.HC_EPS)
        sinkhorn_iters = int(module.HC_SINKHORN_ITER)
        x = values["x"].float()
        x_flat = x.reshape(x.shape[0], hc_dim)
        fn = values["hc_fn"].float()
        inv = torch.rsqrt(x_flat.square().sum(dim=1, keepdim=True) / hc_dim + eps)
        mixes = torch.zeros(
            x.shape[0], mix_hc, dtype=torch.float32, device=x.device
        )
        split_width = hc_dim // 4
        for split in range(4):
            k0 = split * split_width
            partial = torch.matmul(
                x_flat[:, k0 : k0 + split_width].double(),
                fn[:, k0 : k0 + split_width].double().transpose(0, 1),
            )
            mixes = (mixes.double() + partial).float()
        mixes = mixes * inv
        scale = values["hc_scale"].float()
        base = values["hc_base"].float()
        pre = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + hc_eps
        post = 2 * torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * scale[1]
            + base[hc_mult : 2 * hc_mult]
        )
        comb = mixes[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult)
        comb = torch.softmax(
            comb * scale[2] + base[2 * hc_mult:].reshape(1, hc_mult, hc_mult), dim=-1
        ) + hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        mixed = sum(
            x[:, h, :] * pre[:, h : h + 1] for h in range(hc_mult)
        ).to(torch.bfloat16)
        values["x_mixed"] = mixed.reshape(x.shape[0], int(module.D))
        values["post"] = post
        values["comb"] = comb.reshape(x.shape[0], hc_mult * hc_mult)
    elif relative == "models/deepseek_v4_flash_mtp/sample.py":
        if module is None:
            raise ValueError("DeepSeek sampler adapter requires imported constants")
        import numpy as np

        vocab = int(module.VOCAB)
        rows = int(module.SAMPLE_ROWS)
        eps = float(module.SAMPLING_EPS)
        modulus = int(module.RANDOM_KEY_MODULUS)
        multiplier = np.uint32(int(module.HASH_MULTIPLIER))
        position_multiplier = int(module.POSITION_MULTIPLIER)
        uint23_scale = np.float32(module.UINT23_SCALE)
        logits = values["logits"].float()
        sampled_ids = values["sampled_ids"]
        sampled_ids.zero_()
        for row in range(rows):
            row_logits = logits[row]
            temperature = float(values["temperatures"][row].item())
            if temperature < eps:
                selected = torch.argmax(row_logits)
            else:
                seed = int(values["seeds"][row].item())
                position = int(values["positions"][row].item())
                counters = np.arange(vocab, dtype=np.uint32)
                random_key = np.uint32(
                    (seed + position * position_multiplier) % modulus
                )
                random_bits = counters ^ random_key
                random_bits ^= random_bits >> np.uint32(16)
                random_bits *= multiplier
                random_bits &= np.uint32(0x7FFFFFFF)
                random_bits ^= random_bits >> np.uint32(16)
                random_bits *= multiplier
                random_bits &= np.uint32(0x7FFFFFFF)
                random_bits ^= random_bits >> np.uint32(16)
                uniform_bits = (random_bits >> np.uint32(8)).astype(np.float32)
                uniform = (uniform_bits + np.float32(0.5)) * uint23_scale
                noise = torch.from_numpy(-np.log(-np.log(uniform))).to(
                    row_logits.device
                )
                scaled = row_logits / temperature
                top_k = int(values["top_ks"][row].item())
                if 0 < top_k < vocab:
                    boundary = torch.topk(scaled, top_k).values[-1]
                    greater = scaled > boundary
                    boundary_indices = torch.nonzero(scaled == boundary).flatten()
                    boundary_keep = top_k - int(greater.sum().item())
                    keep = greater.clone()
                    keep[boundary_indices[:boundary_keep]] = True
                    filtered = torch.full_like(scaled, -torch.inf)
                    filtered[keep] = scaled[keep]
                    scaled = filtered
                selected = torch.argmax(scaled + noise)
            sampled_ids[row, 0] = selected.to(torch.int32)
    elif relative == "models/deepseek_v4_flash_mtp/mtp_projection.py":
        if module is None:
            raise ValueError("DeepSeek MTP adapter requires imported constants")
        d = int(module.D)
        hc_mult = int(module.HC_MULT)
        eps = float(module.EPS)
        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)

        def rms_norm(input_value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            shape = input_value.shape
            flattened = input_value.reshape(-1, d).float()
            sq_sum = flattened.square().sum(dim=1, keepdim=True)
            return (
                flattened
                * torch.rsqrt(sq_sum / d + eps)
                * weight.float().reshape(1, d)
            ).reshape(shape)

        def quantize_rows(input_value: torch.Tensor):
            amax = input_value.float().abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
            quant_scale = scale_max / amax
            quantized = torch.round(input_value.float() * quant_scale).clamp(
                -int(scale_max), int(scale_max)
            )
            return quantized.to(torch.int8), 1.0 / quant_scale

        def int8_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            """Use the exact CUDA integer GEMM path, padding its M dimension."""

            lhs_2d = lhs.reshape(-1, lhs.shape[-1])
            rows = lhs_2d.shape[0]
            if hasattr(torch, "_int_mm"):
                padded_rows = max(32, rows)
                if padded_rows != rows:
                    padded = torch.zeros(
                        padded_rows,
                        lhs_2d.shape[1],
                        dtype=torch.int8,
                        device=lhs_2d.device,
                    )
                    padded[:rows] = lhs_2d
                    lhs_2d = padded
                return torch._int_mm(lhs_2d, rhs)[:rows].float()
            return torch.matmul(lhs_2d.float(), rhs.float()).round()

        hidden = rms_norm(values["hidden_states"], values["enorm_w"])
        hidden = hidden * values["e_proj_smooth"].float()
        previous = rms_norm(values["prev_hidden_states"], values["hnorm_w"])
        previous = previous * values["h_proj_smooth"].float()
        hidden_i8, hidden_scale = quantize_rows(hidden)
        previous_i8, previous_scale = quantize_rows(previous)
        e_acc = int8_mm(hidden_i8, values["e_proj_w"].transpose(0, 1))
        e_out = e_acc * hidden_scale * values["e_proj_w_scale"].float().reshape(1, d)
        h_acc = int8_mm(previous_i8, values["h_proj_w"].transpose(0, 1)).reshape(
            previous_i8.shape[0], previous_i8.shape[1], d
        )
        h_out = h_acc * previous_scale * values["h_proj_w_scale"].float().reshape(1, 1, d)
        values["hidden_states_out"] = (e_out.unsqueeze(1) + h_out).reshape(
            -1, hc_mult, d
        ).float()
    elif relative == "models/deepseek_v4_flash_mtp/qkv_proj_rope.py":
        if module is None:
            raise ValueError("DeepSeek QKV adapter requires imported constants")
        d = int(module.D)
        q_lora = int(module.Q_LORA)
        heads = int(module.H)
        head_dim = int(module.HEAD_DIM)
        nope_dim = int(module.NOPE_DIM)
        rope_half = int(module.ROPE_HALF)
        eps = float(module.EPS)
        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)

        def rms(input_value: torch.Tensor, gamma: torch.Tensor | None = None):
            result = input_value.float() * torch.rsqrt(
                input_value.float().square().mean(dim=-1, keepdim=True) + eps
            )
            return result if gamma is None else result * gamma.float()

        def quantize(input_value: torch.Tensor):
            amax = input_value.float().abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
            quant_scale = scale_max / amax
            quantized = torch.round(input_value.float() * quant_scale).clamp(
                -int(scale_max), int(scale_max)
            )
            return quantized.to(torch.int8), (1.0 / quant_scale).float()

        def int8_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            rows = lhs.shape[0]
            padded_rows = max(32, rows)
            if padded_rows != rows:
                padded = torch.zeros(
                    padded_rows,
                    lhs.shape[1],
                    dtype=torch.int8,
                    device=lhs.device,
                )
                padded[:rows] = lhs
                lhs = padded
            return torch._int_mm(lhs, rhs)[:rows].float()

        def rope(input_value: torch.Tensor) -> torch.Tensor:
            pair = input_value.unflatten(-1, (-1, 2))
            even = pair[..., 0]
            odd = pair[..., 1]
            cos = values["rope_cos"].float()[..., :rope_half]
            sin = values["rope_sin"].float()[..., :rope_half]
            while cos.ndim < even.ndim:
                cos = cos.unsqueeze(-2)
                sin = sin.unsqueeze(-2)
            rotated_even = (even * cos - odd * sin).to(torch.bfloat16)
            rotated_odd = (even * sin + odd * cos).to(torch.bfloat16)
            return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

        x = values["x"].float().reshape(-1, d)
        q_lora_value = torch.matmul(x, values["wq_a"].float()).reshape(-1, q_lora)
        q_lora_value = rms(q_lora_value, values["gamma_cq"])
        qr_i8, qr_scale = quantize(q_lora_value)
        q_acc = int8_mm(qr_i8, values["wq_b"])
        q_full = (
            q_acc
            * qr_scale
            * values["wq_b_scale"].float().reshape(1, heads * head_dim)
        ).reshape(-1, heads, head_dim)
        q_full = rms(q_full)
        values["q"] = torch.cat(
            [q_full[..., :nope_dim], rope(q_full[..., nope_dim:])], dim=-1
        ).to(torch.bfloat16)

        kv_full = torch.matmul(x, values["wkv"].float())
        kv_full = rms(kv_full, values["gamma_ckv"])
        kv_rope = rope(kv_full[..., nope_dim:].unsqueeze(1)).squeeze(1)
        values["kv"] = torch.cat([kv_full[..., :nope_dim], kv_rope], dim=-1).to(
            torch.bfloat16
        )
        values["qr"] = qr_i8
        values["qr_scale"] = qr_scale
    elif relative == "models/deepseek_v4_flash_mtp/gate.py":
        if module is None:
            raise ValueError("DeepSeek gate adapter requires imported constants")
        tokens = int(module.T)
        d = int(module.D)
        experts = int(module.N_EXPERTS)
        topk = int(module.TOPK)
        norm_eps = float(module.NORM_EPS)
        amax_eps = float(module.INT8_AMAX_EPS)
        scale_max = float(module.INT8_SCALE_MAX)
        route_scale = float(module.ROUTE_SCALE)
        hash_layers = int(module.N_HASH_LAYERS)
        active = max(0, min(tokens, int(values["num_tokens"].item())))

        x = values["x_mixed"].float().reshape(tokens, d)
        norm_w = values["norm_w"].float().reshape(1, d)
        inv = torch.rsqrt(x.square().sum(dim=-1, keepdim=True) / d + norm_eps)
        xg = x * norm_w
        amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
        quant_scale = scale_max / amax
        x_i8 = (
            torch.round(xg * quant_scale)
            .to(torch.int32)
            .to(torch.float16)
            .to(torch.int8)
        )
        x_scale = (1.0 / quant_scale) * inv
        logits = inv * torch.matmul(xg, values["gate_w"].float().transpose(0, 1))
        softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
        scores = torch.sqrt(softplus)
        biased = scores + values["gate_bias"].float().reshape(1, experts)
        layer_id = int(values["layer_id"].item())
        if layer_id < hash_layers:
            indices = values["tid2eid"][values["input_ids"].long()]
        else:
            indices = torch.argsort(-biased, dim=-1, stable=True)[..., :topk]
        selected = torch.gather(scores, dim=-1, index=indices.long())
        weights = selected / selected.sum(dim=-1, keepdim=True) * route_scale
        if active < tokens:
            x_scale[active:] = 0
            indices[active:] = 0
            weights[active:] = 0
        values["x_norm_i8"] = x_i8
        values["x_norm_scale"] = x_scale.float()
        values["indices"] = indices.to(torch.int32)
        values["weights"] = weights.float()
    elif relative == "models/deepseek_v4_flash_mtp/decode_compressor_ratio128.py":
        if module is None:
            raise ValueError("ratio-128 decode adapter requires imported constants")
        ratio = int(module.COMPRESS_RATIO)
        state_block = int(module.COMPRESS_STATE_BLOCK_SIZE)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        eps = float(module.EPS)
        sequence = int(module.S)

        x = values["x"].float()
        position_ids = values["position_ids"].long()
        state_slots = values["state_slot_mapping"].long()
        cache_slots = values["cmp_slot_mapping"].long()
        state = values["compress_state"]
        state_table = values["compress_state_block_table"]
        cache = values["cmp_kv_cache"]
        kv_projection = torch.matmul(x, values["wkv"].float().transpose(0, 1))
        score_projection = torch.matmul(
            x, values["wgate"].float().transpose(0, 1)
        )
        pooled = torch.zeros(
            x.shape[0], 1, head_dim, dtype=torch.float32, device=x.device
        )
        boundaries: list[int | None] = []

        def read_state(batch: int, position: int):
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[batch, logical].item())
            if physical < 0:
                return (
                    torch.zeros(out_dim, dtype=torch.float32, device=x.device),
                    torch.full(
                        (out_dim,), -torch.inf, dtype=torch.float32, device=x.device
                    ),
                )
            return (
                state[physical, intra, :out_dim],
                state[physical, intra, out_dim : 2 * out_dim],
            )

        for batch in range(x.shape[0]):
            boundary = None
            for token in range(sequence):
                position = int(position_ids[batch, token].item())
                score_projection[batch, token] += values["ape"][position % ratio]
                slot = int(state_slots[batch, token].item())
                if slot >= 0:
                    physical = slot // state_block
                    intra = slot % state_block
                    state[physical, intra, :out_dim] = kv_projection[batch, token]
                    state[physical, intra, out_dim : 2 * out_dim] = score_projection[
                        batch, token
                    ]
                if (position + 1) % ratio == 0:
                    boundary = token
            boundaries.append(boundary)
            if boundary is not None:
                end = int(position_ids[batch, boundary].item())
                kv_rows = []
                score_rows = []
                for position in range(end - ratio + 1, end + 1):
                    kv_row, score_row = read_state(batch, position)
                    kv_rows.append(kv_row)
                    score_rows.append(score_row)
                kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
                score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
                pooled[batch : batch + 1] = (
                    kv_state * torch.softmax(score_state, dim=1)
                ).sum(dim=1, keepdim=True)

        for batch, boundary in enumerate(boundaries):
            if boundary is None:
                continue
            normalized = pooled[batch : batch + 1]
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float()
            pair = normalized[..., -rope_dim:].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            cos = values["cos"][batch].float().reshape(-1)
            sin = values["sin"][batch].float().reshape(-1)
            rotated = torch.stack(
                [even * cos - odd * sin, even * sin + odd * cos], dim=-1
            ).flatten(-2)
            normalized = torch.cat([normalized[..., :-rope_dim], rotated], dim=-1)
            cache_row = int(cache_slots[batch, boundary].item())
            if cache_row >= 0:
                values["kv"][batch : batch + 1, 0:1] = normalized
                cache[
                    cache_row // cache_block,
                    cache_row % cache_block,
                    0,
                ] = normalized[0, 0]
        values["compress_state"] = state
        values["cmp_kv_cache"] = cache
    elif relative == "models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py":
        if module is None:
            raise ValueError("ratio-4 decode adapter requires imported constants")
        ratio = int(module.COMPRESS_RATIO)
        state_block = int(module.COMPRESS_STATE_BLOCK_SIZE)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        eps = float(module.EPS)
        sequence = int(module.S)

        x = values["x"].float()
        positions = values["position_ids"].long()
        state_slots = values["state_slot_mapping"].long()
        cache_slots = values["cmp_slot_mapping"].long()
        state = values["compress_state"]
        state_table = values["compress_state_block_table"]
        cache = values["cmp_kv_cache"]
        kv_projection = torch.matmul(x, values["wkv"].float().transpose(0, 1))
        score_projection = torch.matmul(
            x, values["wgate"].float().transpose(0, 1)
        )
        pooled = torch.zeros(
            x.shape[0], 1, head_dim, dtype=torch.float32, device=x.device
        )
        boundaries: list[int | None] = []

        def read_half(batch: int, position: int, back: bool):
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[batch, logical].item())
            if physical < 0:
                return (
                    torch.zeros(head_dim, dtype=torch.float32, device=x.device),
                    torch.full(
                        (head_dim,), -torch.inf, dtype=torch.float32, device=x.device
                    ),
                )
            column = head_dim if back else 0
            return (
                state[physical, intra, column : column + head_dim],
                state[
                    physical,
                    intra,
                    out_dim + column : out_dim + column + head_dim,
                ],
            )

        for batch in range(x.shape[0]):
            first = int(positions[batch, 0].item())
            pre_tokens = min(sequence, ratio - first % ratio)
            boundary = ratio - 1 - first % ratio
            should_compress = 0 <= boundary < sequence
            boundary_end = first + pre_tokens - 1
            current_start = boundary_end - ratio + 1
            previous_start = current_start - ratio
            boundaries.append(boundary if should_compress else None)

            for token in range(sequence):
                position = int(positions[batch, token].item())
                score_projection[batch, token] += values["ape"][position % ratio]
                slot = int(state_slots[batch, token].item())
                if slot >= 0:
                    physical = slot // state_block
                    intra = slot % state_block
                    state[physical, intra, :out_dim] = kv_projection[batch, token]
                    state[physical, intra, out_dim:] = score_projection[batch, token]

            if should_compress:
                kv_rows = []
                score_rows = []
                for offset in range(ratio):
                    position = previous_start + offset
                    if position < 0:
                        kv_rows.append(
                            torch.zeros(
                                head_dim, dtype=torch.float32, device=x.device
                            )
                        )
                        score_rows.append(
                            torch.full(
                                (head_dim,),
                                -torch.inf,
                                dtype=torch.float32,
                                device=x.device,
                            )
                        )
                    else:
                        kv_row, score_row = read_half(batch, position, False)
                        kv_rows.append(kv_row)
                        score_rows.append(score_row)
                for offset in range(ratio):
                    kv_row, score_row = read_half(
                        batch, current_start + offset, True
                    )
                    kv_rows.append(kv_row)
                    score_rows.append(score_row)
                kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
                score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
                pooled[batch : batch + 1] = (
                    kv_state * torch.softmax(score_state, dim=1)
                ).sum(dim=1, keepdim=True)

        for batch, boundary in enumerate(boundaries):
            if boundary is None:
                continue
            normalized = pooled[batch : batch + 1]
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float()
            pair = normalized[..., -rope_dim:].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            cos = values["cos"][batch].float().reshape(-1)
            sin = values["sin"][batch].float().reshape(-1)
            rotated = torch.stack(
                [even * cos - odd * sin, even * sin + odd * cos], dim=-1
            ).flatten(-2)
            normalized = torch.cat([normalized[..., :-rope_dim], rotated], dim=-1)
            cache_row = int(cache_slots[batch, boundary].item())
            if cache_row >= 0:
                values["kv"][batch : batch + 1, 0:1] = normalized
                cache[
                    cache_row // cache_block,
                    cache_row % cache_block,
                    0,
                ] = normalized[0, 0]
        values["compress_state"] = state
        values["cmp_kv_cache"] = cache
    elif relative == "models/deepseek_v4_flash_mtp/decode_indexer_compressor.py":
        if module is None:
            raise ValueError("decode indexer-compressor adapter requires constants")
        ratio = int(module.COMPRESS_RATIO)
        state_block = int(module.COMPRESS_STATE_BLOCK_SIZE)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        cache_block = int(module.IDX_STORAGE_BLOCK_SIZE)
        eps = float(module.EPS)
        sequence = int(module.S)
        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)

        x = values["x"].float()
        positions = values["position_ids"].long()
        state_slots = values["inner_state_slot_mapping"].long()
        cache_slots = values["idx_slot_mapping"].long()
        state = values["compress_state"]
        state_table = values["compress_state_block_table"]
        cache = values["idx_kv_cache"]
        cache_scale = values["idx_kv_scale"]
        kv_projection = torch.matmul(x, values["wkv"].float().transpose(0, 1))
        score_projection = torch.matmul(
            x, values["wgate"].float().transpose(0, 1)
        )
        pooled = torch.zeros(
            x.shape[0], 1, head_dim, dtype=torch.float32, device=x.device
        )
        boundaries: list[int | None] = []

        def read_half(batch: int, position: int, back: bool):
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[batch, logical].item())
            if physical < 0:
                return (
                    torch.zeros(head_dim, dtype=torch.float32, device=x.device),
                    torch.full(
                        (head_dim,), -torch.inf, dtype=torch.float32, device=x.device
                    ),
                )
            column = head_dim if back else 0
            return (
                state[physical, intra, column : column + head_dim],
                state[
                    physical,
                    intra,
                    out_dim + column : out_dim + column + head_dim,
                ],
            )

        for batch in range(x.shape[0]):
            first = int(positions[batch, 0].item())
            pre_tokens = min(sequence, ratio - first % ratio)
            boundary = ratio - 1 - first % ratio
            should_compress = 0 <= boundary < sequence
            current_start = first + pre_tokens - ratio
            previous_start = current_start - ratio
            boundaries.append(boundary if should_compress else None)
            for token in range(sequence):
                position = int(positions[batch, token].item())
                score_projection[batch, token] += values["ape"][position % ratio]
                slot = int(state_slots[batch, token].item())
                if slot >= 0:
                    physical = slot // state_block
                    intra = slot % state_block
                    state[physical, intra, :out_dim] = kv_projection[batch, token]
                    state[physical, intra, out_dim:] = score_projection[batch, token]
            if should_compress:
                kv_rows = []
                score_rows = []
                for offset in range(ratio):
                    position = previous_start + offset
                    if position < 0:
                        kv_rows.append(
                            torch.zeros(head_dim, dtype=torch.float32, device=x.device)
                        )
                        score_rows.append(
                            torch.full(
                                (head_dim,), -torch.inf, dtype=torch.float32, device=x.device
                            )
                        )
                    else:
                        kv_row, score_row = read_half(batch, position, False)
                        kv_rows.append(kv_row)
                        score_rows.append(score_row)
                for offset in range(ratio):
                    kv_row, score_row = read_half(
                        batch, current_start + offset, True
                    )
                    kv_rows.append(kv_row)
                    score_rows.append(score_row)
                kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
                score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
                pooled[batch : batch + 1] = (
                    kv_state * torch.softmax(score_state, dim=1)
                ).sum(dim=1, keepdim=True)

        for batch, boundary in enumerate(boundaries):
            if boundary is None:
                continue
            normalized = pooled[batch : batch + 1]
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float()
            pair = normalized[..., -rope_dim:].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            cos = values["cos"][batch].float().reshape(-1)
            sin = values["sin"][batch].float().reshape(-1)
            normalized = torch.cat(
                [
                    normalized[..., :-rope_dim],
                    torch.stack(
                        [even * cos - odd * sin, even * sin + odd * cos],
                        dim=-1,
                    ).flatten(-2),
                ],
                dim=-1,
            )
            transformed = torch.matmul(
                normalized.to(torch.bfloat16).float(), values["hadamard"].float()
            )
            row = int(cache_slots[batch, boundary].item())
            if row >= 0:
                values["kv"][batch : batch + 1, 0:1] = transformed
                rounded = transformed[0, 0].to(torch.bfloat16).float()
                amax = rounded.abs().amax().clamp_min(amax_eps)
                quant_scale = scale_max / amax
                cache[
                    row // cache_block, row % cache_block, 0
                ] = torch.round(rounded * quant_scale).to(torch.int32).to(
                    torch.float16
                ).to(torch.int8)
                cache_scale[row // cache_block, row % cache_block, 0, 0] = (
                    1.0 / quant_scale
                )
        values["compress_state"] = state
        values["idx_kv_cache"] = cache
        values["idx_kv_scale"] = cache_scale
    elif relative == "models/deepseek_v4_flash_mtp/prefill_compressor_ratio128.py":
        if module is None:
            raise ValueError("ratio-128 prefill adapter requires imported constants")
        ratio = int(module.COMPRESS_RATIO)
        state_block = int(module.HCA_STATE_BLOCK_SIZE)
        state_blocks = int(module.HCA_STATE_BLOCK_NUM)
        state_dim = int(module.COMPRESS_STATE_DIM)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        nope_dim = int(module.NOPE_HEAD_DIM)
        rope_half = int(module.ROPE_HALF)
        cache_blocks = int(module.HCA_CMP_BLOCK_NUM)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        eps = float(module.EPS)
        max_seq = int(module.MAX_SEQ_LEN)
        active = int(values["num_tokens"].item())

        kv_projection = torch.matmul(
            values["x"].float(), values["wkv"].float().transpose(0, 1)
        )
        score_projection = torch.matmul(
            values["x"].float(), values["wgate"].float().transpose(0, 1)
        )
        state = values["compress_state"]
        state_flat = state.reshape(state_blocks * state_block, state_dim)
        kv_state = state_flat[:, :out_dim]
        score_state = state_flat[:, out_dim:]
        state_table = values["compress_state_block_table"]
        cache = values["cmp_kv"]
        cache_flat = cache.reshape(cache_blocks * cache_block, head_dim)

        def state_row(position: int) -> int:
            if position < 0 or position >= max_seq:
                return -1
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[logical].item())
            return -1 if physical < 0 else physical * state_block + intra

        for token in range(active):
            cache_row = int(values["cmp_slot_mapping"][token].item())
            if cache_row < 0:
                continue
            write_position = int(values["position_ids"][token].item())
            pool_kv = torch.zeros(
                ratio, out_dim, dtype=torch.float32, device=state.device
            )
            pool_score = torch.zeros_like(pool_kv)
            for slot in range(ratio):
                row = state_row(write_position + 1 - ratio + slot)
                if row >= 0:
                    pool_kv[slot] = kv_state[row]
                    pool_score[slot] = score_state[row]
            for source_token in range(active):
                position = int(values["position_ids"][source_token].item())
                if position > write_position:
                    continue
                slot = position % ratio
                pool_kv[slot] = kv_projection[source_token]
                pool_score[slot] = (
                    score_projection[source_token] + values["ape"][slot]
                )
            pooled = (pool_kv * torch.softmax(pool_score, dim=0)).sum(
                dim=0, keepdim=True
            )
            normalized = pooled * torch.rsqrt(
                pooled.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float().reshape(1, head_dim)
            pair = normalized[..., nope_dim:].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            compressed_position = write_position + 1 - ratio
            cos = values["freqs_cos"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            sin = values["freqs_sin"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            normalized[:, nope_dim:] = torch.stack(
                [even * cos - odd * sin, even * sin + odd * cos], dim=-1
            ).flatten(-2)
            cache_flat[cache_row] = normalized[0]

        for token in range(active):
            position = int(values["position_ids"][token].item())
            row = int(values["state_slot_mapping"][token].item())
            if row < 0:
                continue
            slot = position % ratio
            kv_state[row] = kv_projection[token]
            score_state[row] = score_projection[token] + values["ape"][slot]
        values["compress_state"] = state
        values["cmp_kv"] = cache
    elif relative == "models/deepseek_v4_flash_mtp/prefill_compressor_ratio4.py":
        if module is None:
            raise ValueError("ratio-4 prefill adapter requires imported constants")
        ratio = int(module.COMPRESS_RATIO)
        state_len = int(module.STATE_LEN)
        state_block = int(module.CSA_STATE_BLOCK_SIZE)
        state_blocks = int(module.CSA_STATE_BLOCK_NUM)
        state_dim = int(module.COMPRESS_STATE_DIM)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        nope_dim = int(module.NOPE_HEAD_DIM)
        rope_half = int(module.ROPE_HEAD_DIM) // 2
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        eps = float(module.EPS)
        max_seq = int(module.MAX_SEQ_LEN)
        active = int(values["num_tokens"].item())

        x = values["x"].reshape(-1, int(module.D)).float()
        kv_projection = torch.matmul(x, values["wkv"].float().transpose(0, 1))
        score_projection = torch.matmul(
            x, values["wgate"].float().transpose(0, 1)
        )
        state = values["compress_state"]
        state_flat = state.reshape(state_blocks * state_block, state_dim)
        kv_state = state_flat[:, :out_dim]
        score_state = state_flat[:, out_dim:]
        state_table = values["compress_state_block_table"]
        cache = values["cmp_kv"]
        cache_flat = cache.reshape(-1, 1, head_dim)[:, 0, :]

        def state_row(position: int) -> int:
            if position < 0 or position >= max_seq:
                return -1
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[logical].item())
            return -1 if physical < 0 else physical * state_block + intra

        for token in range(active):
            cache_row = int(values["cmp_slot_mapping"][token].item())
            if cache_row < 0:
                continue
            write_position = int(values["position_ids"][token].item())
            current_start = write_position + 1 - ratio
            previous_start = current_start - ratio
            pool_kv = torch.zeros(
                state_len, head_dim, dtype=torch.float32, device=state.device
            )
            pool_score = torch.full_like(pool_kv, -torch.inf)
            for offset in range(ratio):
                previous_position = previous_start + offset
                if write_position >= 2 * ratio - 1:
                    row = state_row(previous_position)
                    if row >= 0:
                        pool_kv[offset] = kv_state[row, :head_dim]
                        pool_score[offset] = score_state[row, :head_dim]
                current_position = current_start + offset
                row = state_row(current_position)
                if row >= 0:
                    pool_kv[ratio + offset] = kv_state[row, head_dim:out_dim]
                    pool_score[ratio + offset] = score_state[
                        row, head_dim:out_dim
                    ]
            for source_token in range(active):
                position = int(values["position_ids"][source_token].item())
                if position < previous_start or position > write_position:
                    continue
                if position < current_start:
                    slot = position - previous_start
                    column = 0
                else:
                    slot = ratio + position - current_start
                    column = head_dim
                ape_slot = position % ratio
                pool_kv[slot] = kv_projection[
                    source_token, column : column + head_dim
                ]
                pool_score[slot] = (
                    score_projection[source_token, column : column + head_dim]
                    + values["ape"][ape_slot, column : column + head_dim]
                )
            initial = state_len - 1
            maximum = pool_score[initial : initial + 1].clone()
            denominator = torch.ones_like(maximum)
            numerator = pool_kv[initial : initial + 1].clone()
            for slot in range(state_len - 1):
                if slot < ratio and write_position < 2 * ratio - 1:
                    continue
                score = pool_score[slot : slot + 1]
                kv = pool_kv[slot : slot + 1]
                next_maximum = torch.maximum(maximum, score)
                alpha = torch.exp(maximum - next_maximum)
                beta = torch.exp(score - next_maximum)
                denominator = alpha * denominator + beta
                numerator = numerator * alpha + kv * beta
                maximum = next_maximum
            pooled = numerator / denominator
            normalized = pooled * torch.rsqrt(
                pooled.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float().reshape(1, head_dim)
            pair = normalized[..., nope_dim:head_dim].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            compressed_position = write_position + 1 - ratio
            cos = values["freqs_cos"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            sin = values["freqs_sin"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            normalized[:, nope_dim:head_dim] = torch.stack(
                [even * cos - odd * sin, even * sin + odd * cos], dim=-1
            ).flatten(-2)
            cache_flat[cache_row] = normalized.to(torch.bfloat16)[0]

        for token in range(active):
            position = int(values["position_ids"][token].item())
            row = int(values["state_slot_mapping"][token].item())
            if row < 0:
                continue
            ape_slot = position % ratio
            kv_state[row] = kv_projection[token]
            score_state[row] = score_projection[token] + values["ape"][ape_slot]
        values["compress_state"] = state
        values["cmp_kv"] = cache
    elif relative == "models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py":
        if module is None:
            raise ValueError("prefill indexer-compressor adapter requires constants")
        ratio = int(module.COMPRESS_RATIO)
        state_len = int(module.STATE_LEN)
        state_block = int(module.INNER_STATE_BLOCK_SIZE)
        state_blocks = int(module.INNER_STATE_BLOCK_NUM)
        state_dim = int(module.COMPRESS_STATE_DIM)
        out_dim = int(module.OUT_DIM)
        head_dim = int(module.HEAD_DIM)
        nope_dim = int(module.NOPE_HEAD_DIM)
        rope_half = int(module.ROPE_HEAD_DIM) // 2
        cache_block = int(module.IDX_STORAGE_BLOCK_SIZE)
        max_writes = int(module.MAX_CMP_WRITES)
        eps = float(module.EPS)
        max_seq = int(module.MAX_SEQ_LEN)
        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)
        active = int(values["num_tokens"].item())

        kv_projection = torch.matmul(
            values["x"].float(), values["wkv"].float().transpose(0, 1)
        )
        score_projection = torch.matmul(
            values["x"].float(), values["wgate"].float().transpose(0, 1)
        )
        state = values["compress_state"]
        state_flat = state.reshape(state_blocks * state_block, state_dim)
        kv_state = state_flat[:, :out_dim]
        score_state = state_flat[:, out_dim:]
        state_table = values["inner_compress_state_block_table"]
        cache = values["idx_kv_cache"]
        cache_scale = values["idx_kv_scale"]
        cache_rows = cache.reshape(-1, 1, head_dim)[:, 0, :]
        scale_rows = cache_scale.reshape(-1, 1, 1)[:, 0, 0]
        extracted = torch.zeros(
            max_writes, head_dim, dtype=torch.int8, device=state.device
        )

        def state_row(position: int) -> int:
            if position < 0 or position >= max_seq:
                return -1
            logical = position // state_block
            intra = position % state_block
            physical = int(state_table[logical].item())
            return -1 if physical < 0 else physical * state_block + intra

        write_index = 0
        for token in range(active):
            cache_row = int(values["idx_slot_mapping"][token].item())
            if cache_row < 0:
                continue
            write_position = int(values["position_ids"][token].item())
            current_start = write_position + 1 - ratio
            previous_start = current_start - ratio
            pool_kv = torch.zeros(
                state_len, head_dim, dtype=torch.float32, device=state.device
            )
            pool_score = torch.full_like(pool_kv, -torch.inf)
            for offset in range(ratio):
                previous_position = previous_start + offset
                if write_position >= 2 * ratio - 1:
                    row = state_row(previous_position)
                    if row >= 0:
                        pool_kv[offset] = kv_state[row, :head_dim]
                        pool_score[offset] = score_state[row, :head_dim]
                row = state_row(current_start + offset)
                if row >= 0:
                    pool_kv[ratio + offset] = kv_state[row, head_dim:out_dim]
                    pool_score[ratio + offset] = score_state[
                        row, head_dim:out_dim
                    ]
            for source_token in range(active):
                position = int(values["position_ids"][source_token].item())
                if position < previous_start or position > write_position:
                    continue
                if position < current_start:
                    slot = position - previous_start
                    column = 0
                else:
                    slot = ratio + position - current_start
                    column = head_dim
                ape_slot = position % ratio
                pool_kv[slot] = kv_projection[
                    source_token, column : column + head_dim
                ]
                pool_score[slot] = (
                    score_projection[source_token, column : column + head_dim]
                    + values["ape"][ape_slot, column : column + head_dim]
                )
            initial = state_len - 1
            maximum = pool_score[initial : initial + 1].clone()
            denominator = torch.ones_like(maximum)
            numerator = pool_kv[initial : initial + 1].clone()
            for slot in range(state_len - 1):
                if slot < ratio and write_position < 2 * ratio - 1:
                    continue
                score = pool_score[slot : slot + 1]
                kv = pool_kv[slot : slot + 1]
                next_maximum = torch.maximum(maximum, score)
                alpha = torch.exp(maximum - next_maximum)
                beta = torch.exp(score - next_maximum)
                denominator = alpha * denominator + beta
                numerator = numerator * alpha + kv * beta
                maximum = next_maximum
            pooled = numerator / denominator
            normalized_fp32 = pooled * torch.rsqrt(
                pooled.square().mean(dim=-1, keepdim=True) + eps
            ) * values["norm_w"].float().reshape(1, head_dim)
            normalized = normalized_fp32.clone()
            normalized[:, :nope_dim] = normalized_fp32[:, :nope_dim].to(
                torch.bfloat16
            ).float()
            pair = normalized_fp32[:, nope_dim:head_dim].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            compressed_position = write_position + 1 - ratio
            cos = values["freqs_cos"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            sin = values["freqs_sin"][
                compressed_position : compressed_position + 1, :rope_half
            ].float()
            normalized[:, nope_dim:head_dim] = torch.stack(
                [even * cos - odd * sin, even * sin + odd * cos], dim=-1
            ).flatten(-2).to(torch.bfloat16).float()
            final = torch.matmul(
                normalized.to(torch.bfloat16).float(), values["hadamard"].float()
            )
            rounded = final.to(torch.bfloat16)[0].float()
            amax = rounded.abs().amax().clamp_min(amax_eps)
            quant_scale = scale_max / amax
            row_i8 = torch.round(rounded * quant_scale).to(torch.int32).to(
                torch.float16
            ).to(torch.int8)
            cache_rows[cache_row] = row_i8
            scale_rows[cache_row] = 1.0 / quant_scale
            if write_index < max_writes:
                extracted[write_index] = row_i8
            write_index += 1

        for token in range(active):
            position = int(values["position_ids"][token].item())
            row = int(values["inner_state_slot_mapping"][token].item())
            if row < 0:
                continue
            ape_slot = position % ratio
            kv_state[row] = kv_projection[token]
            score_state[row] = score_projection[token] + values["ape"][ape_slot]
        values["kv"] = extracted
        values["compress_state"] = state
        values["idx_kv_cache"] = cache
        values["idx_kv_scale"] = cache_scale
    elif relative == "models/deepseek_v4_flash_mtp/decode_indexer.py":
        if module is None:
            raise ValueError("decode indexer adapter requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        compressor_module = import_demo(prefix + "decode_indexer_compressor.py")
        batch = int(module.B)
        sequence = int(module.S)
        heads = int(module.IDX_N_HEADS)
        head_dim = int(module.IDX_HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        ratio = int(module.COMPRESS_RATIO)
        score_len = int(module.SCORE_LEN)
        topk = int(module.IDX_TOPK)
        reduce_tile = int(module.REDUCE_TILE)
        weight_scale = float(module.WEIGHTS_SCALE)
        negative_infinity = float(module.FP32_NEG_INF)

        qr = values["qr"]
        rows = qr.shape[0]
        padded = torch.zeros(
            max(32, rows), qr.shape[1], dtype=torch.int8, device=qr.device
        )
        padded[:rows] = qr
        q_acc = torch._int_mm(padded, values["wq_b"])[:rows].float()
        q = (
            q_acc
            * values["qr_scale"].float()
            * values["wq_b_scale"].float().reshape(1, heads * head_dim)
        ).reshape(batch, sequence, heads, head_dim)
        pair = q[..., -rope_dim:].unflatten(-1, (-1, 2))
        even, odd = pair[..., 0], pair[..., 1]
        cos = values["cos"].reshape(batch, 1, 1, -1).float()
        sin = values["sin"].reshape(batch, 1, 1, -1).float()
        q = torch.cat(
            [
                q[..., :-rope_dim],
                torch.stack(
                    [
                        (even * cos - odd * sin).to(torch.bfloat16),
                        (even * sin + odd * cos).to(torch.bfloat16),
                    ],
                    dim=-1,
                ).flatten(-2),
            ],
            dim=-1,
        )
        q = torch.matmul(q.to(torch.bfloat16).float(), values["hadamard"].float())

        inner_values = {
            "x": values["x"],
            "kv": values["inner_kv"],
            "compress_state": values["inner_compress_state"],
            "compress_state_block_table": values[
                "inner_compress_state_block_table"
            ],
            "wkv": values["inner_wkv"],
            "wgate": values["inner_wgate"],
            "ape": values["inner_ape"],
            "norm_w": values["inner_norm_w"],
            "cos": values["cos"],
            "sin": values["sin"],
            "hadamard": values["hadamard"],
            "idx_kv_cache": values["idx_kv_cache"],
            "idx_kv_scale": values["idx_kv_scale"],
            "position_ids": values["position_ids"],
            "idx_slot_mapping": values["idx_slot_mapping"],
            "inner_state_slot_mapping": values["inner_state_slot_mapping"],
        }
        _reference_adapter(
            prefix + "decode_indexer_compressor.py",
            inner_values,
            compressor_module,
        )
        weights = torch.matmul(
            values["x"].float(), values["weights_proj"].float()
        ) * weight_scale
        q_flat = q.reshape(batch * sequence * heads, head_dim)
        amax = q_flat.abs().amax(dim=-1, keepdim=True).clamp_min(
            float(module.INT8_AMAX_EPS)
        )
        quant_scale = float(module.INT8_SCALE_MAX) / amax
        q_i8 = torch.round(q_flat * quant_scale).to(torch.int32).to(
            torch.float16
        ).to(torch.int8).reshape(batch, sequence, heads, head_dim)
        q_scale = (1.0 / quant_scale).reshape(batch, sequence, heads, 1)
        cache = inner_values["idx_kv_cache"]
        cache_scale = inner_values["idx_kv_scale"].float()
        score_output = torch.full(
            (batch, sequence, score_len),
            negative_infinity,
            dtype=torch.float32,
            device=q.device,
        )
        topk_output = torch.full(
            (batch, sequence, score_len),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        offset = int(values["offset"].item())
        for batch_index in range(batch):
            cache_len = int(values["kv_seq_lens"][batch_index].item()) // ratio
            if cache_len <= 0:
                continue
            cache_rows = []
            scale_rows = []
            for slot in range(cache_len):
                block = int(
                    values["idx_block_table"][
                        batch_index, slot // reduce_tile
                    ].item()
                )
                cache_rows.append(cache[block, slot % reduce_tile, 0])
                scale_rows.append(cache_scale[block, slot % reduce_tile, 0, 0])
            kv_i8 = torch.stack(cache_rows, dim=0)
            kv_scale = torch.stack(scale_rows).reshape(1, cache_len)
            query_rows = q_i8[batch_index].reshape(sequence * heads, head_dim)
            padded_query = torch.zeros(
                max(32, query_rows.shape[0]),
                head_dim,
                dtype=torch.int8,
                device=q.device,
            )
            padded_query[: query_rows.shape[0]] = query_rows
            padded_cache_len = ((cache_len + 7) // 8) * 8
            cache_rhs = torch.zeros(
                head_dim,
                padded_cache_len,
                dtype=torch.int8,
                device=q.device,
            )
            cache_rhs[:, :cache_len] = kv_i8.transpose(0, 1)
            score_i32 = torch._int_mm(padded_query, cache_rhs)[
                : query_rows.shape[0], :cache_len
            ].reshape(sequence, heads, cache_len)
            score = score_i32.float() * q_scale[batch_index]
            score = (
                torch.relu(score) * weights[batch_index].unsqueeze(-1)
            ).sum(dim=1) * kv_scale
            for token in range(sequence):
                visible = min(
                    cache_len,
                    (int(values["position_ids"][batch_index, token].item()) + 1)
                    // ratio,
                    score_len,
                )
                if visible <= 0:
                    continue
                score_output[batch_index, token, :visible] = score[token, :visible]
                count = min(topk, visible)
                indices = torch.topk(score[token, :visible], count).indices
                topk_output[batch_index, token, :count] = indices.to(
                    torch.int32
                ) + offset
        values["idx_kv_cache"] = cache
        values["idx_kv_scale"] = inner_values["idx_kv_scale"]
        values["score"] = score_output
        values["topk_idxs"] = topk_output
    elif relative == "models/deepseek_v4_flash_mtp/prefill_indexer.py":
        if module is None:
            raise ValueError("prefill indexer adapter requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        compressor_module = import_demo(prefix + "prefill_indexer_compressor.py")
        tokens = int(module.T)
        heads = int(module.IDX_N_HEADS)
        head_dim = int(module.IDX_HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        ratio = int(module.COMPRESS_RATIO)
        score_cap = int(module.INDEXER_SCORE_CAP)
        topk_cap = int(module.INDEXER_TOPK_CAP)
        cache_tile = int(module.CACHE_TILE)
        weight_scale = float(module.WEIGHTS_SCALE)
        negative_infinity = float(module.FP32_NEG_INF)
        active = int(values["num_tokens"].item())

        compressor_values = {
            "x": values["x"],
            "kv": torch.zeros(
                int(module.MAX_CMP_WRITES),
                head_dim,
                dtype=torch.int8,
                device=values["x"].device,
            ),
            "compress_state": values["inner_compress_state"],
            "inner_compress_state_block_table": values[
                "inner_compress_state_block_table"
            ],
            "wkv": values["inner_wkv"],
            "wgate": values["inner_wgate"],
            "ape": values["inner_ape"],
            "norm_w": values["inner_norm_w"],
            "freqs_cos": values["freqs_cos"],
            "freqs_sin": values["freqs_sin"],
            "hadamard": values["hadamard"],
            "idx_kv_cache": values["idx_kv_cache"],
            "idx_kv_scale": values["idx_kv_scale"],
            "idx_block_table": values["idx_block_table"],
            "position_ids": values["position_ids"],
            "num_tokens": values["num_tokens"],
            "idx_slot_mapping": values["idx_slot_mapping"],
            "inner_state_slot_mapping": values["inner_state_slot_mapping"],
        }
        _reference_adapter(
            prefix + "prefill_indexer_compressor.py",
            compressor_values,
            compressor_module,
        )
        visible = ((values["position_ids"].long() + 1) // ratio).clamp(
            max=score_cap
        )
        max_visible = int(visible[:active].max().item()) if active > 0 else 0
        score_output = torch.full(
            (tokens, score_cap),
            negative_infinity,
            dtype=torch.float32,
            device=values["x"].device,
        )
        topk_output = torch.full(
            (tokens, topk_cap),
            -1,
            dtype=torch.int32,
            device=values["x"].device,
        )
        if max_visible > 0:
            qr = values["qr"]
            padded_qr = torch.zeros(
                max(32, tokens), qr.shape[1], dtype=torch.int8, device=qr.device
            )
            padded_qr[:tokens] = qr
            q_acc = torch._int_mm(padded_qr, values["wq_b"])[:tokens].float()
            q = (
                q_acc
                * values["qr_scale"].float()
                * values["wq_b_scale"].float().reshape(1, heads * head_dim)
            ).reshape(tokens, heads, head_dim)
            pair = q[..., -rope_dim:].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            cos = values["cos"].float().reshape(tokens, 1, -1)
            sin = values["sin"].float().reshape(tokens, 1, -1)
            q = torch.cat(
                [
                    q[..., :-rope_dim],
                    torch.stack(
                        [
                            (even * cos - odd * sin).to(torch.bfloat16),
                            (even * sin + odd * cos).to(torch.bfloat16),
                        ],
                        dim=-1,
                    ).flatten(-2),
                ],
                dim=-1,
            )
            q = torch.matmul(
                q.to(torch.bfloat16).float(), values["hadamard"].float()
            )
            weights = torch.matmul(
                values["x"].float(), values["weights_proj"].float()
            ) * weight_scale
            cache_flat = compressor_values["idx_kv_cache"].reshape(-1, head_dim)
            scale_flat = compressor_values["idx_kv_scale"].float().reshape(-1, 1)
            rows = [
                int(values["idx_block_table"][index // cache_tile].item())
                * int(module.IDX_STORAGE_BLOCK_SIZE)
                + index % cache_tile
                for index in range(max_visible)
            ]
            kv_i8 = torch.stack([cache_flat[row] for row in rows])
            kv_scale = torch.stack([scale_flat[row] for row in rows]).reshape(
                1, 1, max_visible
            )
            q_flat = q.reshape(tokens * heads, head_dim)
            amax = q_flat.abs().amax(dim=-1, keepdim=True).clamp_min(
                float(module.INT8_AMAX_EPS)
            )
            quant_scale = float(module.INT8_SCALE_MAX) / amax
            q_i8 = torch.round(q_flat * quant_scale).to(torch.int32).to(
                torch.float16
            ).to(torch.int8).reshape(tokens, heads, head_dim)
            q_scale = (1.0 / quant_scale).reshape(tokens, heads, 1)
            query_rows = q_i8.reshape(tokens * heads, head_dim)
            padded_query = torch.zeros(
                max(32, query_rows.shape[0]),
                head_dim,
                dtype=torch.int8,
                device=q.device,
            )
            padded_query[: query_rows.shape[0]] = query_rows
            padded_visible = ((max_visible + 7) // 8) * 8
            cache_rhs = torch.zeros(
                head_dim, padded_visible, dtype=torch.int8, device=q.device
            )
            cache_rhs[:, :max_visible] = kv_i8.transpose(0, 1)
            score_i32 = torch._int_mm(padded_query, cache_rhs)[
                : query_rows.shape[0], :max_visible
            ].reshape(tokens, heads, max_visible)
            score = score_i32.float() * q_scale * kv_scale
            score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)
            columns = torch.arange(max_visible, device=q.device).unsqueeze(0)
            score = score.masked_fill(columns >= visible.unsqueeze(1), negative_infinity)
            score_output[:, :max_visible] = score
            for token in range(active):
                count = min(topk_cap, int(visible[token].item()))
                if count > 0:
                    topk_output[token, :count] = torch.topk(
                        score[token], count
                    ).indices.to(torch.int32)
        values["idx_kv_cache"] = compressor_values["idx_kv_cache"]
        values["idx_kv_scale"] = compressor_values["idx_kv_scale"]
        values["score"] = score_output
        values["topk_idxs"] = topk_output
    elif relative == "models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py":
        if module is None:
            raise ValueError("decode SWA adapter requires imported constants")
        flat = values["ori_kv"].reshape(-1, int(module.HEAD_DIM))
        rows_by_token = []
        for token in range(int(module.T)):
            valid = int(values["swa_lens"][token].item())
            rows = []
            for column, raw in enumerate(values["swa_indices"][token].tolist()):
                slot = int(raw)
                rows.append(flat[slot] if column < valid and slot >= 0 else None)
            rows_by_token.append(rows)
        values["attn_out"] = _sparse_attention_output(
            values, module, rows_by_token
        )
    elif relative == "models/deepseek_v4_flash_mtp/decode_sparse_attn_hca.py":
        if module is None:
            raise ValueError("decode HCA adapter requires imported constants")
        head_dim = int(module.HEAD_DIM)
        sequence = int(module.S)
        ori = values["ori_kv"].reshape(-1, head_dim)
        compressed = values["cmp_kv"].reshape(-1, head_dim)
        rows_by_token = []
        for token in range(int(module.T)):
            batch = token // sequence
            rows = []
            for raw in values["window_swa_indices"][token].tolist():
                slot = int(raw)
                rows.append(ori[slot] if slot >= 0 else None)
            for raw in values["cmp_sparse_indices"][token].tolist():
                slot = int(raw)
                if slot < 0:
                    rows.append(None)
                else:
                    row = int(values["cmp_block_table"][batch, slot].item())
                    rows.append(compressed[row] if row >= 0 else None)
            rows_by_token.append(rows)
        values["attn_out"] = _sparse_attention_output(
            values, module, rows_by_token
        )
    elif relative == "models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py":
        if module is None:
            raise ValueError("decode CSA adapter requires imported constants")
        head_dim = int(module.HEAD_DIM)
        sequence = int(module.S)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        compress_ratio = int(module.COMPRESS_RATIO)
        cmp_topk = int(module.CMP_TOPK)
        ori = values["ori_kv"].reshape(-1, head_dim)
        compressed = values["cmp_kv"].reshape(-1, head_dim)
        raw_topk = values["idx_topk"][:, :cmp_topk].long()
        bound = (
            (values["position_ids"][:, 0].long() + 1) // compress_ratio
        ).unsqueeze(1)
        compressed_indices = torch.where(
            (raw_topk >= 0) & (raw_topk < bound),
            raw_topk,
            torch.full_like(raw_topk, -1),
        )
        rows_by_token = []
        for token in range(int(module.T)):
            batch = token // sequence
            rows = []
            for raw in values["window_swa_indices"][token].tolist():
                slot = int(raw)
                rows.append(ori[slot] if slot >= 0 else None)
            for raw in compressed_indices[token].tolist():
                slot = int(raw)
                if slot < 0:
                    rows.append(None)
                else:
                    block = int(
                        values["cmp_block_table"][batch, slot // cache_block].item()
                    )
                    row = block * cache_block + slot % cache_block
                    rows.append(compressed[row] if block >= 0 else None)
            rows_by_token.append(rows)
        values["attn_out"] = _sparse_attention_output(
            values, module, rows_by_token
        )
    elif relative == "models/deepseek_v4_flash_mtp/prefill_sparse_attn.py":
        if module is None:
            raise ValueError("prefill sparse adapter requires imported constants")
        head_dim = int(module.HEAD_DIM)
        cache_block = int(values["cmp_storage_block_size"].item())
        max_slots = int(module.CMP_MAX_BLOCKS) * cache_block
        active = int(values["num_tokens"].item())
        ori = values["ori_kv"].reshape(-1, head_dim)
        compressed = values["cmp_kv"].reshape(-1, head_dim)
        rows_by_token = []
        for token in range(int(module.T)):
            rows = []
            if token < active:
                for raw in values["swa_indices"][token].tolist():
                    row = int(raw)
                    if row >= 0:
                        rows.append(ori[row])
                for raw in values["cmp_indices"][token].tolist():
                    slot = int(raw)
                    if slot < 0 or slot >= max_slots:
                        continue
                    block = int(
                        values["cmp_block_table"][slot // cache_block].item()
                    )
                    if block >= 0:
                        rows.append(
                            compressed[block * cache_block + slot % cache_block]
                        )
            rows_by_token.append(rows)
        values["attn_out"] = _sparse_attention_output(
            values, module, rows_by_token
        )
    elif relative == "models/deepseek_v4_flash_mtp/decode_swa.py":
        if module is None:
            raise ValueError("decode SWA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        hc_pre_module = import_demo(prefix + "hc_pre.py")
        qkv_module = import_demo(prefix + "qkv_proj_rope.py")
        sparse_module = import_demo(prefix + "decode_sparse_attn_swa.py")
        hc_post_module = import_demo(prefix + "hc_post.py")
        tokens = int(module.T)
        d = int(module.D)
        hc_mult = int(module.HC_MULT)
        head_dim = int(module.HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        q_lora = int(module.Q_LORA)
        heads = int(module.H)

        pre_values = {
            "x": values["x_hc"],
            "hc_fn": values["hc_attn_fn"],
            "hc_scale": values["hc_attn_scale"],
            "hc_base": values["hc_attn_base"],
            "x_mixed": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=values["x_hc"].device
            ),
            "post": torch.zeros(
                tokens, hc_mult, dtype=torch.float32, device=values["x_hc"].device
            ),
            "comb": torch.zeros(
                tokens,
                hc_mult * hc_mult,
                dtype=torch.float32,
                device=values["x_hc"].device,
            ),
        }
        _reference_adapter(prefix + "hc_pre.py", pre_values, hc_pre_module)
        x_mixed = pre_values["x_mixed"].float()
        x_normed = (
            x_mixed
            * torch.rsqrt(
                x_mixed.square().mean(dim=-1, keepdim=True)
                + float(module.EPS)
            )
            * values["attn_norm_w"].float()
        ).to(torch.bfloat16)
        positions = values["position_ids"].long()
        rope_cos = values["freqs_cos"].index_select(0, positions)
        rope_sin = values["freqs_sin"].index_select(0, positions)
        qkv_values = {
            "x": x_normed,
            "wq_a": values["wq_a"],
            "wq_b": values["wq_b"],
            "wq_b_scale": values["wq_b_scale"],
            "wkv": values["wkv"],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "gamma_cq": values["gamma_cq"],
            "gamma_ckv": values["gamma_ckv"],
            "q": torch.zeros(
                tokens,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=x_mixed.device,
            ),
            "kv": torch.zeros(
                tokens, head_dim, dtype=torch.bfloat16, device=x_mixed.device
            ),
            "qr": torch.zeros(
                tokens, q_lora, dtype=torch.int8, device=x_mixed.device
            ),
            "qr_scale": torch.zeros(
                tokens, 1, dtype=torch.float32, device=x_mixed.device
            ),
        }
        _reference_adapter(prefix + "qkv_proj_rope.py", qkv_values, qkv_module)
        cache = values["kv_cache"]
        block_size = int(module.BLOCK_SIZE)
        for token in range(tokens):
            row = int(values["swa_slot_mapping"][token].item())
            if row >= 0:
                cache[row // block_size, row % block_size, 0] = qkv_values["kv"][
                    token
                ]
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "swa_indices": values["swa_indices"],
            "swa_lens": values["swa_lens"],
            "attn_sink": values["attn_sink"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=x_mixed.device
            ),
        }
        _reference_adapter(
            prefix + "decode_sparse_attn_swa.py", sparse_values, sparse_module
        )
        post_values = {
            "x": sparse_values["attn_out"],
            "residual": values["x_hc"],
            "post": pre_values["post"],
            "comb": pre_values["comb"],
            "y": torch.zeros(
                tokens,
                hc_mult,
                d,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
        }
        _reference_adapter(prefix + "hc_post.py", post_values, hc_post_module)
        values["kv_cache"] = cache
        values["x_out"] = post_values["y"]
    elif relative == "models/deepseek_v4_flash_mtp/decode_hca.py":
        if module is None:
            raise ValueError("decode HCA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        hc_pre_module = import_demo(prefix + "hc_pre.py")
        qkv_module = import_demo(prefix + "qkv_proj_rope.py")
        compressor_module = import_demo(prefix + "decode_compressor_ratio128.py")
        sparse_module = import_demo(prefix + "decode_sparse_attn_hca.py")
        hc_post_module = import_demo(prefix + "hc_post.py")
        tokens = int(module.T)
        batch = int(module.B)
        sequence = int(module.S)
        d = int(module.D)
        hc_mult = int(module.HC_MULT)
        heads = int(module.H)
        head_dim = int(module.HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        q_lora = int(module.Q_LORA)
        ratio = int(module.COMPRESS_RATIO)

        pre_values = {
            "x": values["x_hc"],
            "hc_fn": values["hc_attn_fn"],
            "hc_scale": values["hc_attn_scale"],
            "hc_base": values["hc_attn_base"],
            "x_mixed": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=values["x_hc"].device
            ),
            "post": torch.zeros(
                tokens, hc_mult, dtype=torch.float32, device=values["x_hc"].device
            ),
            "comb": torch.zeros(
                tokens,
                hc_mult * hc_mult,
                dtype=torch.float32,
                device=values["x_hc"].device,
            ),
        }
        _reference_adapter(prefix + "hc_pre.py", pre_values, hc_pre_module)
        x_mixed = pre_values["x_mixed"].float()
        x_normed = (
            x_mixed
            * torch.rsqrt(
                x_mixed.square().mean(dim=-1, keepdim=True)
                + float(module.EPS)
            )
            * values["attn_norm_w"].float()
        ).to(torch.bfloat16)
        positions = values["position_ids"].long()
        rope_cos = values["freqs_cos"].index_select(0, positions)
        rope_sin = values["freqs_sin"].index_select(0, positions)
        qkv_values = {
            "x": x_normed,
            "wq_a": values["wq_a"],
            "wq_b": values["wq_b"],
            "wq_b_scale": values["wq_b_scale"],
            "wkv": values["wkv"],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "gamma_cq": values["gamma_cq"],
            "gamma_ckv": values["gamma_ckv"],
            "q": torch.zeros(
                tokens,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=x_mixed.device,
            ),
            "kv": torch.zeros(
                tokens, head_dim, dtype=torch.bfloat16, device=x_mixed.device
            ),
            "qr": torch.zeros(
                tokens, q_lora, dtype=torch.int8, device=x_mixed.device
            ),
            "qr_scale": torch.zeros(
                tokens, 1, dtype=torch.float32, device=x_mixed.device
            ),
        }
        _reference_adapter(prefix + "qkv_proj_rope.py", qkv_values, qkv_module)

        half_rope = rope_dim // 2
        compressor_cos = torch.empty(
            batch, half_rope, dtype=torch.float32, device=x_mixed.device
        )
        compressor_sin = torch.empty_like(compressor_cos)
        for batch_index in range(batch):
            first = int(positions[batch_index * sequence].item())
            compressed_position = first + (ratio - first % ratio) - ratio
            compressor_cos[batch_index] = values["freqs_cos"][
                compressed_position, :half_rope
            ].float()
            compressor_sin[batch_index] = values["freqs_sin"][
                compressed_position, :half_rope
            ].float()
        compressor_values = {
            "x": x_normed.reshape(batch, sequence, d),
            "kv": torch.zeros(
                batch,
                sequence,
                head_dim,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
            "compress_state": values["compress_state"],
            "compress_state_block_table": values[
                "compress_state_block_table"
            ],
            "wkv": values["cmp_wkv"],
            "wgate": values["cmp_wgate"],
            "ape": values["cmp_ape"],
            "norm_w": values["cmp_norm_w"],
            "cos": compressor_cos,
            "sin": compressor_sin,
            "cmp_kv_cache": values["cmp_kv"],
            "position_ids": positions.reshape(batch, sequence).to(torch.int32),
            "cmp_slot_mapping": values["cmp_slot_mapping"].reshape(
                batch, sequence
            ),
            "state_slot_mapping": values["state_slot_mapping"].reshape(
                batch, sequence
            ),
        }
        _reference_adapter(
            prefix + "decode_compressor_ratio128.py",
            compressor_values,
            compressor_module,
        )
        cache = values["kv_cache"]
        block_size = int(module.BLOCK_SIZE)
        for token in range(tokens):
            row = int(values["ori_slot_mapping"][token].item())
            if row >= 0:
                cache[row // block_size, row % block_size, 0] = qkv_values["kv"][
                    token
                ]
        compressed_topk = torch.full(
            (tokens, int(module.HCA_CMP_TOPK)),
            -1,
            dtype=torch.int32,
            device=x_mixed.device,
        )
        for token in range(tokens):
            batch_index = token // sequence
            position = int(positions[token].item())
            valid = min(
                int(module.HCA_TOPK_LIMIT),
                (position + 1) // ratio,
                int(values["kv_seq_lens"][batch_index].item()) // ratio,
            )
            if valid:
                compressed_topk[token, :valid] = torch.arange(
                    valid, dtype=torch.int32, device=x_mixed.device
                )
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "window_swa_indices": values["window_swa_indices"],
            "cmp_kv": compressor_values["cmp_kv_cache"],
            "cmp_block_table": values["cmp_block_table"],
            "cmp_sparse_indices": compressed_topk,
            "attn_sink": values["attn_sink"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=x_mixed.device
            ),
        }
        _reference_adapter(
            prefix + "decode_sparse_attn_hca.py", sparse_values, sparse_module
        )
        post_values = {
            "x": sparse_values["attn_out"],
            "residual": values["x_hc"],
            "post": pre_values["post"],
            "comb": pre_values["comb"],
            "y": torch.zeros(
                tokens,
                hc_mult,
                d,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
        }
        _reference_adapter(prefix + "hc_post.py", post_values, hc_post_module)
        values["kv_cache"] = cache
        values["x_out"] = post_values["y"]
    elif relative == "models/deepseek_v4_flash_mtp/decode_csa.py":
        if module is None:
            raise ValueError("decode CSA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        hc_pre_module = import_demo(prefix + "hc_pre.py")
        qkv_module = import_demo(prefix + "qkv_proj_rope.py")
        compressor_module = import_demo(prefix + "decode_compressor_ratio4.py")
        indexer_module = import_demo(prefix + "decode_indexer.py")
        sparse_module = import_demo(prefix + "decode_sparse_attn_csa.py")
        hc_post_module = import_demo(prefix + "hc_post.py")
        tokens = int(module.T)
        batch = int(module.B)
        sequence = int(module.S)
        d = int(module.D)
        hc_mult = int(module.HC_MULT)
        heads = int(module.H)
        head_dim = int(module.HEAD_DIM)
        idx_head_dim = int(module.IDX_HEAD_DIM)
        rope_dim = int(module.ROPE_HEAD_DIM)
        q_lora = int(module.Q_LORA)
        ratio = int(module.COMPRESS_RATIO)

        pre_values = {
            "x": values["x_hc"],
            "hc_fn": values["hc_attn_fn"],
            "hc_scale": values["hc_attn_scale"],
            "hc_base": values["hc_attn_base"],
            "x_mixed": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=values["x_hc"].device
            ),
            "post": torch.zeros(
                tokens, hc_mult, dtype=torch.float32, device=values["x_hc"].device
            ),
            "comb": torch.zeros(
                tokens,
                hc_mult * hc_mult,
                dtype=torch.float32,
                device=values["x_hc"].device,
            ),
        }
        _reference_adapter(prefix + "hc_pre.py", pre_values, hc_pre_module)
        x_mixed = pre_values["x_mixed"].float()
        x_normed = (
            x_mixed
            * torch.rsqrt(
                x_mixed.square().mean(dim=-1, keepdim=True)
                + float(module.EPS)
            )
            * values["attn_norm_w"].float()
        ).to(torch.bfloat16)
        positions = values["position_ids"].long()
        rope_cos = values["freqs_cos"].index_select(0, positions)
        rope_sin = values["freqs_sin"].index_select(0, positions)
        first_positions = positions.reshape(batch, sequence)[:, 0]
        step_cos = values["freqs_cos"][first_positions, : rope_dim // 2].float()
        step_sin = values["freqs_sin"][first_positions, : rope_dim // 2].float()
        compressed_positions = (
            first_positions + (ratio - first_positions % ratio) - ratio
        )
        compressed_cos = values["freqs_cos"][
            compressed_positions, : rope_dim // 2
        ].float()
        compressed_sin = values["freqs_sin"][
            compressed_positions, : rope_dim // 2
        ].float()
        qkv_values = {
            "x": x_normed,
            "wq_a": values["wq_a"],
            "wq_b": values["wq_b"],
            "wq_b_scale": values["wq_b_scale"],
            "wkv": values["wkv"],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "gamma_cq": values["gamma_cq"],
            "gamma_ckv": values["gamma_ckv"],
            "q": torch.zeros(
                tokens,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=x_mixed.device,
            ),
            "kv": torch.zeros(
                tokens, head_dim, dtype=torch.bfloat16, device=x_mixed.device
            ),
            "qr": torch.zeros(
                tokens, q_lora, dtype=torch.int8, device=x_mixed.device
            ),
            "qr_scale": torch.zeros(
                tokens, 1, dtype=torch.float32, device=x_mixed.device
            ),
        }
        _reference_adapter(prefix + "qkv_proj_rope.py", qkv_values, qkv_module)
        main_compressor = {
            "x": x_normed.reshape(batch, sequence, d),
            "kv": torch.zeros(
                batch,
                sequence,
                head_dim,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
            "compress_state": values["compress_state"],
            "compress_state_block_table": values[
                "compress_state_block_table"
            ],
            "wkv": values["cmp_wkv"],
            "wgate": values["cmp_wgate"],
            "ape": values["cmp_ape"],
            "norm_w": values["cmp_norm_w"],
            "cos": compressed_cos,
            "sin": compressed_sin,
            "cmp_kv_cache": values["cmp_kv"],
            "position_ids": positions.reshape(batch, sequence).to(torch.int32),
            "cmp_slot_mapping": values["cmp_slot_mapping"].reshape(
                batch, sequence
            ),
            "state_slot_mapping": values["state_slot_mapping"].reshape(
                batch, sequence
            ),
        }
        _reference_adapter(
            prefix + "decode_compressor_ratio4.py",
            main_compressor,
            compressor_module,
        )
        indexer_values = {
            "x": x_normed.reshape(batch, sequence, d),
            "qr": qkv_values["qr"],
            "qr_scale": qkv_values["qr_scale"],
            "wq_b": values["idx_wq_b"],
            "wq_b_scale": values["idx_wq_b_scale"],
            "weights_proj": values["weights_proj"],
            "cos": step_cos,
            "sin": step_sin,
            "hadamard": values["hadamard_idx"],
            "inner_kv": torch.zeros(
                batch,
                sequence,
                idx_head_dim,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
            "inner_compress_state": values["inner_compress_state"],
            "inner_compress_state_block_table": values[
                "inner_compress_state_block_table"
            ],
            "inner_wkv": values["inner_wkv"],
            "inner_wgate": values["inner_wgate"],
            "inner_ape": values["inner_ape"],
            "inner_norm_w": values["inner_norm_w"],
            "idx_kv_cache": values["idx_kv_cache"],
            "idx_kv_scale": values["idx_kv_scale"],
            "idx_block_table": values["idx_block_table"],
            "score": torch.zeros(
                batch,
                sequence,
                int(module.INDEXER_SCORE_LEN),
                dtype=torch.float32,
                device=x_mixed.device,
            ),
            "topk_idxs": torch.full(
                (batch, sequence, int(module.INDEXER_SCORE_LEN)),
                -1,
                dtype=torch.int32,
                device=x_mixed.device,
            ),
            "position_ids": positions.reshape(batch, sequence).to(torch.int32),
            "idx_slot_mapping": values["idx_slot_mapping"].reshape(
                batch, sequence
            ),
            "inner_state_slot_mapping": values[
                "inner_state_slot_mapping"
            ].reshape(batch, sequence),
            "kv_seq_lens": values["kv_seq_lens"],
            "offset": torch.tensor(0, dtype=torch.int32, device=x_mixed.device),
        }
        _reference_adapter(
            prefix + "decode_indexer.py", indexer_values, indexer_module
        )
        cache = values["kv_cache"]
        block_size = int(module.BLOCK_SIZE)
        for token in range(tokens):
            row = int(values["ori_slot_mapping"][token].item())
            if row >= 0:
                cache[row // block_size, row % block_size, 0] = qkv_values["kv"][
                    token
                ]
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "window_swa_indices": values["window_swa_indices"],
            "cmp_kv": main_compressor["cmp_kv_cache"],
            "cmp_block_table": values["cmp_block_table"],
            "idx_topk": indexer_values["topk_idxs"].reshape(
                tokens, int(module.INDEXER_SCORE_LEN)
            ),
            "position_ids": positions.reshape(tokens, 1),
            "attn_sink": values["attn_sink"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=x_mixed.device
            ),
        }
        _reference_adapter(
            prefix + "decode_sparse_attn_csa.py", sparse_values, sparse_module
        )
        post_values = {
            "x": sparse_values["attn_out"],
            "residual": values["x_hc"],
            "post": pre_values["post"],
            "comb": pre_values["comb"],
            "y": torch.zeros(
                tokens,
                hc_mult,
                d,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
        }
        _reference_adapter(prefix + "hc_post.py", post_values, hc_post_module)
        values["kv_cache"] = cache
        values["x_out"] = post_values["y"]
    elif relative == "models/deepseek_v4_flash_mtp/prefill_swa.py":
        if module is None:
            raise ValueError("prefill SWA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        hc_pre_module = import_demo(prefix + "hc_pre.py")
        qkv_module = import_demo(prefix + "qkv_proj_rope.py")
        sparse_module = import_demo(prefix + "prefill_sparse_attn.py")
        hc_post_module = import_demo(prefix + "hc_post.py")
        tokens = int(module.T)
        d = int(module.D)
        hc_mult = int(module.HC_MULT)
        heads = int(module.H)
        head_dim = int(module.HEAD_DIM)
        q_lora = int(module.Q_LORA)
        block_size = int(module.BLOCK_SIZE)
        active = int(values["num_tokens"].item())

        pre_values = {
            "x": values["x_hc"].reshape(tokens, hc_mult, d),
            "hc_fn": values["hc_attn_fn"],
            "hc_scale": values["hc_attn_scale"],
            "hc_base": values["hc_attn_base"],
            "x_mixed": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=values["x_hc"].device
            ),
            "post": torch.zeros(
                tokens, hc_mult, dtype=torch.float32, device=values["x_hc"].device
            ),
            "comb": torch.zeros(
                tokens,
                hc_mult * hc_mult,
                dtype=torch.float32,
                device=values["x_hc"].device,
            ),
        }
        _reference_adapter(prefix + "hc_pre.py", pre_values, hc_pre_module)
        x_mixed = pre_values["x_mixed"].float()
        x_normed = (
            x_mixed
            * torch.rsqrt(
                x_mixed.square().mean(dim=-1, keepdim=True)
                + float(module.EPS)
            )
            * values["attn_norm_w"].float()
        ).to(torch.bfloat16)
        positions = values["position_ids"].long()
        rope_cos = values["freqs_cos"].index_select(0, positions)
        rope_sin = values["freqs_sin"].index_select(0, positions)
        qkv_values = {
            "x": x_normed,
            "wq_a": values["wq_a"],
            "wq_b": values["wq_b"],
            "wq_b_scale": values["wq_b_scale"],
            "wkv": values["wkv"],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "gamma_cq": values["gamma_cq"],
            "gamma_ckv": values["gamma_ckv"],
            "q": torch.zeros(
                tokens,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=x_mixed.device,
            ),
            "kv": torch.zeros(
                tokens, head_dim, dtype=torch.bfloat16, device=x_mixed.device
            ),
            "qr": torch.zeros(
                tokens, q_lora, dtype=torch.int8, device=x_mixed.device
            ),
            "qr_scale": torch.zeros(
                tokens, 1, dtype=torch.float32, device=x_mixed.device
            ),
        }
        _reference_adapter(prefix + "qkv_proj_rope.py", qkv_values, qkv_module)
        cache = values["kv_cache"]
        cache_flat = cache.reshape(-1, head_dim)
        for token in range(active):
            row = int(values["ori_slot_mapping"][token].item())
            if row >= 0:
                cache_flat[row] = qkv_values["kv"][token]
        window_indices = torch.full(
            (tokens, int(module.WIN)),
            -1,
            dtype=torch.int32,
            device=x_mixed.device,
        )
        for token in range(active):
            position = int(positions[token].item())
            valid = min(int(module.WIN), position + 1)
            start = position + 1 - valid
            for column, key_position in enumerate(range(start, position + 1)):
                block = int(values["block_table"][key_position // block_size].item())
                if block >= 0:
                    window_indices[token, column] = block * block_size + (
                        key_position % block_size
                    )
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "swa_indices": window_indices,
            "cmp_kv": torch.zeros(
                int(module.CMP_BLOCK_NUM),
                block_size,
                1,
                head_dim,
                dtype=torch.bfloat16,
                device=x_mixed.device,
            ),
            "cmp_block_table": torch.zeros(
                int(module.SPARSE_CMP_MAX_BLOCKS),
                dtype=torch.int32,
                device=x_mixed.device,
            ),
            "cmp_storage_block_size": torch.tensor(
                block_size, dtype=torch.int32, device=x_mixed.device
            ),
            "cmp_indices": torch.full(
                (tokens, int(module.IDX_TOPK)),
                -1,
                dtype=torch.int32,
                device=x_mixed.device,
            ),
            "valid_block_mask": torch.zeros(
                tokens,
                int(sparse_module.VALID_BLOCK_MASK_COLS),
                dtype=torch.int32,
                device=x_mixed.device,
            ),
            "attn_sink": values["attn_sink"],
            "num_tokens": values["num_tokens"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=x_mixed.device
            ),
        }
        _reference_adapter(
            prefix + "prefill_sparse_attn.py", sparse_values, sparse_module
        )
        post_values = {
            "x": sparse_values["attn_out"],
            "residual": pre_values["x"],
            "post": pre_values["post"],
            "comb": pre_values["comb"],
            "y": torch.zeros(
                tokens,
                hc_mult,
                d,
                dtype=torch.float32,
                device=x_mixed.device,
            ),
        }
        _reference_adapter(prefix + "hc_post.py", post_values, hc_post_module)
        if active < tokens:
            post_values["y"][active:] = 0
        values["kv_cache"] = cache
        values["x_out"] = post_values["y"]
    elif relative == "models/deepseek_v4_flash_mtp/prefill_hca.py":
        if module is None:
            raise ValueError("prefill HCA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        compressor_module = import_demo(prefix + "prefill_compressor_ratio128.py")
        sparse_module = import_demo(prefix + "prefill_sparse_attn.py")
        (
            pre_values,
            x_normed,
            positions,
            rope_cos,
            rope_sin,
            qkv_values,
        ) = _prefill_frontend(values, module)
        tokens = int(module.T)
        head_dim = int(module.HEAD_DIM)
        d = int(module.D)
        block_size = int(module.BLOCK_SIZE)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        ratio = int(module.COMPRESS_RATIO)
        active = int(values["num_tokens"].item())
        cache = values["kv_cache"]
        cache_flat = cache.reshape(-1, head_dim)
        for token in range(active):
            row = int(values["ori_slot_mapping"][token].item())
            if row >= 0:
                cache_flat[row] = qkv_values["kv"][token]
        compressor_values = {
            "x": x_normed.reshape(tokens, d),
            "compress_state": values["compress_state"],
            "compress_state_block_table": values[
                "compress_state_block_table"
            ],
            "wkv": values["cmp_wkv"],
            "wgate": values["cmp_wgate"],
            "ape": values["cmp_ape"],
            "norm_w": values["cmp_norm_w"],
            "freqs_cos": values["freqs_cos"],
            "freqs_sin": values["freqs_sin"],
            "cmp_kv": values["cmp_kv"],
            "position_ids": values["position_ids"],
            "num_tokens": values["num_tokens"],
            "cmp_slot_mapping": values["cmp_slot_mapping"],
            "state_slot_mapping": values["state_slot_mapping"],
        }
        _reference_adapter(
            prefix + "prefill_compressor_ratio128.py",
            compressor_values,
            compressor_module,
        )
        window_indices = torch.full(
            (tokens, int(module.WIN)),
            -1,
            dtype=torch.int32,
            device=cache.device,
        )
        compressed_indices = torch.full(
            (tokens, int(module.IDX_TOPK)),
            -1,
            dtype=torch.int32,
            device=cache.device,
        )
        compressed_cap = int(module.SPARSE_CMP_MAX_BLOCKS) * cache_block
        for token in range(active):
            position = int(positions[token].item())
            valid_window = min(int(module.WIN), position + 1)
            start = position + 1 - valid_window
            for column, key_position in enumerate(range(start, position + 1)):
                block = int(
                    values["ori_block_table"][key_position // block_size].item()
                )
                if block >= 0:
                    window_indices[token, column] = block * block_size + (
                        key_position % block_size
                    )
            visible = min(
                (position + 1) // ratio,
                int(module.IDX_TOPK),
                compressed_cap,
            )
            if visible > 0:
                compressed_indices[token, :visible] = torch.arange(
                    visible, dtype=torch.int32, device=cache.device
                )
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "swa_indices": window_indices,
            "cmp_kv": compressor_values["cmp_kv"],
            "cmp_block_table": values["cmp_block_table"],
            "cmp_storage_block_size": torch.tensor(
                cache_block, dtype=torch.int32, device=cache.device
            ),
            "cmp_indices": compressed_indices,
            "valid_block_mask": torch.zeros(
                tokens,
                int(sparse_module.VALID_BLOCK_MASK_COLS),
                dtype=torch.int32,
                device=cache.device,
            ),
            "attn_sink": values["attn_sink"],
            "num_tokens": values["num_tokens"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=cache.device
            ),
        }
        _reference_adapter(
            prefix + "prefill_sparse_attn.py", sparse_values, sparse_module
        )
        values["kv_cache"] = cache
        values["cmp_kv"] = compressor_values["cmp_kv"]
        values["x_out"] = _prefill_hc_post(
            values, module, pre_values, sparse_values["attn_out"]
        )
    elif relative == "models/deepseek_v4_flash_mtp/prefill_csa.py":
        if module is None:
            raise ValueError("prefill CSA wrapper requires imported constants")
        prefix = "models/deepseek_v4_flash_mtp/"
        compressor_module = import_demo(prefix + "prefill_compressor_ratio4.py")
        indexer_module = import_demo(prefix + "prefill_indexer.py")
        sparse_module = import_demo(prefix + "prefill_sparse_attn.py")
        (
            pre_values,
            x_normed,
            positions,
            rope_cos,
            rope_sin,
            qkv_values,
        ) = _prefill_frontend(values, module)
        tokens = int(module.T)
        d = int(module.D)
        head_dim = int(module.HEAD_DIM)
        idx_head_dim = int(module.IDX_HEAD_DIM)
        block_size = int(module.BLOCK_SIZE)
        cache_block = int(module.CMP_STORAGE_BLOCK_SIZE)
        active = int(values["num_tokens"].item())

        main_compressor = {
            "x": x_normed.reshape(tokens, d),
            "compress_state": values["compress_state"],
            "compress_state_block_table": values[
                "compress_state_block_table"
            ],
            "wkv": values["cmp_wkv"],
            "wgate": values["cmp_wgate"],
            "ape": values["cmp_ape"],
            "norm_w": values["cmp_norm_w"],
            "freqs_cos": values["freqs_cos"],
            "freqs_sin": values["freqs_sin"],
            "cmp_kv": values["cmp_kv"],
            "position_ids": values["position_ids"],
            "num_tokens": values["num_tokens"],
            "cmp_slot_mapping": values["cmp_slot_mapping"],
            "state_slot_mapping": values["state_slot_mapping"],
        }
        _reference_adapter(
            prefix + "prefill_compressor_ratio4.py",
            main_compressor,
            compressor_module,
        )
        indexer_values = {
            "x": x_normed.reshape(tokens, d),
            "qr": qkv_values["qr"],
            "qr_scale": qkv_values["qr_scale"],
            "wq_b": values["idx_wq_b"],
            "wq_b_scale": values["idx_wq_b_scale"],
            "weights_proj": values["idx_weights_proj"],
            "cos": rope_cos[:, : int(module.HALF_ROPE)].float(),
            "sin": rope_sin[:, : int(module.HALF_ROPE)].float(),
            "freqs_cos": values["freqs_cos"],
            "freqs_sin": values["freqs_sin"],
            "hadamard": values["hadamard_idx"],
            "inner_compress_state": values["inner_compress_state"],
            "inner_compress_state_block_table": values[
                "inner_compress_state_block_table"
            ],
            "inner_wkv": values["inner_wkv"],
            "inner_wgate": values["inner_wgate"],
            "inner_ape": values["inner_ape"],
            "inner_norm_w": values["inner_norm_w"],
            "idx_kv_cache": values["idx_kv_cache"],
            "idx_kv_scale": values["idx_kv_scale"],
            "idx_block_table": values["idx_block_table"],
            "position_ids": values["position_ids"],
            "num_tokens": values["num_tokens"],
            "idx_slot_mapping": values["idx_slot_mapping"],
            "inner_state_slot_mapping": values["inner_state_slot_mapping"],
            "score": torch.full(
                (tokens, int(module.INDEXER_SCORE_CAP)),
                float(indexer_module.FP32_NEG_INF),
                dtype=torch.float32,
                device=x_normed.device,
            ),
            "topk_idxs": torch.full(
                (tokens, int(module.INDEXER_TOPK_CAP)),
                -1,
                dtype=torch.int32,
                device=x_normed.device,
            ),
        }
        _reference_adapter(
            prefix + "prefill_indexer.py", indexer_values, indexer_module
        )
        cache = values["kv_cache"]
        cache_flat = cache.reshape(-1, head_dim)
        for token in range(active):
            row = int(values["ori_slot_mapping"][token].item())
            if row >= 0:
                cache_flat[row] = qkv_values["kv"][token]
        window_indices = torch.full(
            (tokens, int(module.WIN)),
            -1,
            dtype=torch.int32,
            device=cache.device,
        )
        for token in range(active):
            position = int(positions[token].item())
            valid_window = min(int(module.WIN), position + 1)
            start = position + 1 - valid_window
            for column, key_position in enumerate(range(start, position + 1)):
                block = int(
                    values["ori_block_table"][key_position // block_size].item()
                )
                if block >= 0:
                    window_indices[token, column] = block * block_size + (
                        key_position % block_size
                    )
        sparse_values = {
            "q": qkv_values["q"],
            "ori_kv": cache,
            "swa_indices": window_indices,
            "cmp_kv": main_compressor["cmp_kv"],
            "cmp_block_table": values["cmp_block_table"],
            "cmp_storage_block_size": torch.tensor(
                cache_block, dtype=torch.int32, device=cache.device
            ),
            "cmp_indices": indexer_values["topk_idxs"].clone(),
            "valid_block_mask": torch.zeros(
                tokens,
                int(sparse_module.VALID_BLOCK_MASK_COLS),
                dtype=torch.int32,
                device=cache.device,
            ),
            "attn_sink": values["attn_sink"],
            "num_tokens": values["num_tokens"],
            "freqs_cos": rope_cos,
            "freqs_sin": rope_sin,
            "wo_a": values["wo_a"],
            "wo_b": values["wo_b"],
            "wo_b_scale": values["wo_b_scale"],
            "attn_out": torch.zeros(
                tokens, d, dtype=torch.bfloat16, device=cache.device
            ),
        }
        _reference_adapter(
            prefix + "prefill_sparse_attn.py", sparse_values, sparse_module
        )
        values["kv_cache"] = cache
        values["cmp_kv"] = main_compressor["cmp_kv"]
        values["idx_kv_cache"] = indexer_values["idx_kv_cache"]
        values["idx_kv_scale"] = indexer_values["idx_kv_scale"]
        values["x_out"] = _prefill_hc_post(
            values, module, pre_values, sparse_values["attn_out"]
        )
    elif relative == "models/deepseek_v4_flash_mtp/expert_shared.py":
        if module is None:
            raise ValueError("DeepSeek shared-expert adapter requires constants")
        import torch.nn.functional as functional

        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)
        limit = float(module.SWIGLU_LIMIT)

        def dequant_weight(weight: torch.Tensor, scale: torch.Tensor):
            return weight.float() * scale.float().unsqueeze(-1)

        def quantize_rows(input_value: torch.Tensor):
            amax = input_value.abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
            quant_scale = scale_max / amax
            quantized = torch.round(input_value * quant_scale).clamp(
                -int(scale_max), int(scale_max)
            )
            return quantized.to(torch.int8), 1.0 / quant_scale

        x = values["x_local_i8"].float() * values["x_local_scale_dq"].float()
        w1 = dequant_weight(values["shared_w1"], values["shared_w1_scale"])
        w3 = dequant_weight(values["shared_w3"], values["shared_w3_scale"])
        w2 = dequant_weight(values["shared_w2"], values["shared_w2_scale"])
        gate = torch.matmul(x, w1.transpose(0, 1))
        up = torch.matmul(x, w3.transpose(0, 1))
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        hidden = functional.silu(gate) * up
        hidden_i8, hidden_scale = quantize_rows(hidden)
        hidden = hidden_i8.float() * hidden_scale
        values["sh"] = torch.matmul(hidden, w2.transpose(0, 1)).to(torch.bfloat16)
    elif relative == "models/deepseek_v4_flash_mtp/expert_routed.py":
        if module is None:
            raise ValueError("DeepSeek routed-expert adapter requires constants")
        import torch.nn.functional as functional

        experts = int(module.N_LOCAL_EXPERTS)
        recv_max = int(module.RECV_MAX)
        d = int(module.D)
        scale_max = float(module.INT8_SCALE_MAX)
        amax_eps = float(module.INT8_AMAX_EPS)
        limit = float(module.SWIGLU_LIMIT)

        def dequant_weight(weight: torch.Tensor, scale: torch.Tensor):
            return weight.float() * scale.float().unsqueeze(-1)

        def quantize_rows(input_value: torch.Tensor):
            amax = input_value.abs().amax(dim=-1, keepdim=True).clamp_min(amax_eps)
            quant_scale = scale_max / amax
            quantized = torch.round(input_value * quant_scale).clamp(
                -int(scale_max), int(scale_max)
            )
            return quantized.to(torch.int8), 1.0 / quant_scale

        w1 = dequant_weight(values["routed_w1"], values["routed_w1_scale"])
        w3 = dequant_weight(values["routed_w3"], values["routed_w3_scale"])
        w2 = dequant_weight(values["routed_w2"], values["routed_w2_scale"])
        output = torch.zeros(
            experts, recv_max, d, dtype=torch.bfloat16, device=w1.device
        )
        for expert in range(experts):
            rows = int(values["recv_expert_count"][expert, 0].item())
            if rows == 0:
                continue
            x = (
                values["recv_x"][expert, :rows].float()
                * values["recv_scale_dq"][expert, :rows].float().reshape(-1, 1)
            )
            route_weight = values["recv_weights"][expert, :rows].float().reshape(-1, 1)
            gate = torch.matmul(x, w1[expert].transpose(0, 1))
            up = torch.matmul(x, w3[expert].transpose(0, 1))
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            hidden = functional.silu(gate) * up
            hidden_i8, hidden_scale = quantize_rows(hidden)
            hidden = hidden_i8.float() * (hidden_scale * route_weight)
            output[expert, :rows] = torch.matmul(
                hidden, w2[expert].transpose(0, 1)
            ).to(torch.bfloat16)
        values["recv_y"] = output
    elif relative == "models/qwen3_14b/greedy_sample.py":
        if module is None:
            raise ValueError("qwen3 greedy adapter requires imported module constants")
        real_vocab = int(module.REAL_VOCAB)
        logits = values["logits"].float()
        token_ids = torch.argmax(logits[:, :real_vocab], dim=-1).to(torch.int32)
        sampled_ids = values["sampled_ids"]
        sampled_ids.zero_()
        sampled_ids[:, :1] = token_ids[:, None]
    elif relative == "models/qwen3_14b/topk_select.py":
        if module is None:
            raise ValueError("qwen3 topk adapter requires imported module constants")
        real_vocab = int(module.REAL_VOCAB)
        topk = int(module.TOPK)
        neg_inf = float(module.FP32_NEG_INF)
        control = values["sampling_control"]
        active_batch = int(control[0].item())
        selection_k = int(control[1].item())
        topk_values = values["topk_values"]
        topk_indices = values["topk_indices"]
        topk_values.fill_(neg_inf)
        topk_indices.zero_()
        logits = values["logits"][:active_batch, :real_vocab].float()
        if selection_k == 1:
            token_ids = torch.argmax(logits, dim=-1)
            topk_values[:active_batch, :1] = torch.gather(
                logits, dim=-1, index=token_ids[:, None]
            )
            topk_indices[:active_batch, :1] = token_ids[:, None].to(torch.int32)
        else:
            vals, indices = torch.topk(
                logits, topk, dim=-1, largest=True, sorted=True
            )
            topk_values[:active_batch] = vals
            topk_indices[:active_batch] = indices.to(torch.int32)
    else:
        raise ValueError(f"no computational reference adapter for {relative}")
    return values


def _run_strict_hello(
    values: dict[str, Any], device: Any, source_node: str
) -> dict[str, Any]:
    import torch

    from pypto_plugins.torch import pointwise_codegen as pc
    from pypto_plugins.torch.runtime_bridge import pypto_launch
    from pypto_plugins.torch.scheduling import REGISTRY

    x = values["x"].to(device)
    output = torch.empty_like(x)
    scalar = float(values["a"].item())
    flat_shape = (x.numel(),)
    builder = pc.PointwiseProgramBuilder(flat_shape, "float32")
    input_value = builder.add_input("x")
    result_value = builder.emit("tensor.adds", [input_value, builder.scalar(scalar)])
    builder.mark_output(result_value)
    program = builder.build()
    registry_name = "article_demo_hello_" + source_node.replace("/", "_").replace(".", "_")
    artifact = pc.compile_pointwise(program, tile=128, registry_name=registry_name)
    REGISTRY.register(registry_name, artifact)
    stream = torch.cuda.current_stream(device)
    pypto_launch(registry_name, (x.reshape(-1), output.reshape(-1)), stream.cuda_stream)
    stream.synchronize()
    return {
        "outputs": {"y": output},
        "adapter": "strict-pypto-nvidia",
        "artifact": {
            "kernel_name": artifact.kernel_name,
            "entry_name": artifact.entry_name,
            "source_sha256": artifact.pypto_source_sha256,
            "artifact_sha256": artifact.artifact_sha256,
            "cubin_sha256": artifact.cubin_sha256,
            "cache_key_sha256": artifact.artifact_cache_key_sha256,
            "source_node": artifact.source_node,
            "fallback_used": artifact.fallback_used,
            "tile": [128],
            "flattened_shape": list(flat_shape),
            "specialized_scalar_a": scalar,
        },
    }


COMPARISON_TOLERANCES: dict[str, dict[str, float]] = {
    "models/deepseek_v4_flash_mtp/mtp_projection.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/gate.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/decode_compressor_ratio128.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/prefill_compressor_ratio128.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/prefill_compressor_ratio4.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/decode_swa.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
    "models/deepseek_v4_flash_mtp/decode_indexer_compressor.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/decode_indexer.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/prefill_indexer.py": {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    "models/deepseek_v4_flash_mtp/decode_hca.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
    "models/deepseek_v4_flash_mtp/decode_csa.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
    "models/deepseek_v4_flash_mtp/prefill_swa.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
    "models/deepseek_v4_flash_mtp/prefill_hca.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
    "models/deepseek_v4_flash_mtp/prefill_csa.py": {
        "rtol": 1e-2,
        "atol": 1e-2,
    },
}

OUTPUT_COMPARISON_TOLERANCES: dict[tuple[str, str], dict[str, float]] = {
    (
        "models/deepseek_v4_flash_mtp/decode_compressor_ratio128.py",
        "kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_compressor_ratio128.py",
        "cmp_kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py",
        "kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py",
        "cmp_kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/prefill_compressor_ratio128.py",
        "cmp_kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/prefill_compressor_ratio4.py",
        "cmp_kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_swa.py",
        "kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer_compressor.py",
        "kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer_compressor.py",
        "idx_kv_cache",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer_compressor.py",
        "idx_kv_scale",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py",
        "kv",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py",
        "idx_kv_cache",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py",
        "idx_kv_scale",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer.py",
        "score",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer.py",
        "idx_kv_cache",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/decode_indexer.py",
        "idx_kv_scale",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer.py",
        "score",
    ): {"rtol": 1.0 / 128, "atol": 1e-4},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer.py",
        "idx_kv_cache",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/prefill_indexer.py",
        "idx_kv_scale",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.01},
    (
        "models/deepseek_v4_flash_mtp/decode_hca.py",
        "kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/decode_csa.py",
        "kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/prefill_swa.py",
        "kv_cache",
    ): {"rtol": 1e-2, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/prefill_hca.py",
        "kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/prefill_hca.py",
        "cmp_kv",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/prefill_csa.py",
        "kv_cache",
    ): {"rtol": 1.0 / 128, "atol": 1e-4, "max_error_ratio": 0.005},
    (
        "models/deepseek_v4_flash_mtp/prefill_csa.py",
        "idx_kv_cache",
    ): {"rtol": 0.0, "atol": 1.0, "max_error_ratio": 0.0},
}


def _compare(
    relative: str,
    outputs: dict[str, Any],
    expected: dict[str, Any],
    specs: list[Any],
    module: Any,
) -> tuple[bool, list[dict[str, Any]]]:
    import torch

    metrics: list[dict[str, Any]] = []
    all_ok = True
    for spec in specs:
        if not getattr(spec, "is_output", False):
            continue
        actual = outputs[spec.name].detach().cpu()
        reference = expected[spec.name].detach().cpu()
        if relative in {
            "models/deepseek_v4_flash_mtp/decode_indexer.py",
            "models/deepseek_v4_flash_mtp/prefill_indexer.py",
        } and spec.name == "topk_idxs":
            count = int(
                module.IDX_TOPK
                if relative.endswith("decode_indexer.py")
                else module.INDEXER_TOPK_CAP
            )
            offset = int(module.OFFSET) if hasattr(module, "OFFSET") else 0
            actual_top = actual[..., :count]
            expected_top = reference[..., :count]
            score = outputs["score"].detach().cpu().float()
            source_indices = (actual_top.long() - offset).clamp(
                min=0, max=score.shape[-1] - 1
            )
            paired = torch.gather(score, dim=-1, index=source_indices)
            mismatch = actual_top != expected_top
            pair_ok = paired[..., :-1] >= paired[..., 1:]
            position_ok = torch.ones_like(mismatch)
            position_ok[..., 1:] &= pair_ok
            position_ok[..., :-1] &= pair_ok
            failures = mismatch & ~position_ok
            mismatch_count = int(mismatch.sum().item())
            error_count = int(failures.sum().item())
            ok = error_count == 0
            all_ok = all_ok and ok
            metrics.append(
                {
                    "name": spec.name,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "ok": ok,
                    "comparison_mode": "topk-paired-score-order",
                    "compared_topk": count,
                    "mismatch_count": mismatch_count,
                    "error_count": error_count,
                    "allowed_error_count": 0,
                    "max_abs_diff": float(
                        (actual_top.long() - expected_top.long()).abs().max().item()
                    ),
                    "rtol": 0.0,
                    "atol": 0.0,
                    "max_error_ratio": 0.0,
                }
            )
            continue
        path_tolerance = COMPARISON_TOLERANCES.get(relative, {})
        output_tolerance = OUTPUT_COMPARISON_TOLERANCES.get(
            (relative, spec.name), path_tolerance
        )
        if actual.dtype.is_floating_point:
            default = 1e-2 if actual.dtype is torch.bfloat16 else 1e-4
            tolerance = output_tolerance or {"rtol": default, "atol": default}
            rtol = float(tolerance["rtol"])
            atol = float(tolerance["atol"])
            max_error_ratio = float(tolerance.get("max_error_ratio", 0.0))
            actual_float = actual.float()
            reference_float = reference.float()
            finite = torch.isfinite(actual_float) & torch.isfinite(reference_float)
            difference = (actual_float - reference_float).abs()
            threshold = atol + rtol * reference_float.abs()
            bad = (~finite) | (difference > threshold)
            error_count = int(bad.sum().item())
            allowed_errors = round(max_error_ratio * actual.numel())
            ok = error_count <= allowed_errors
            max_abs = float(difference.max().item())
        else:
            rtol = float(output_tolerance.get("rtol", 0.0))
            atol = float(output_tolerance.get("atol", 0.0))
            max_error_ratio = float(
                output_tolerance.get("max_error_ratio", 0.0)
            )
            difference = (
                actual.to(torch.int64) - reference.to(torch.int64)
            ).abs()
            bad = difference > atol
            error_count = int(bad.sum().item())
            allowed_errors = round(max_error_ratio * actual.numel())
            ok = error_count <= allowed_errors
            max_abs = (
                0.0 if actual.numel() == 0 else float(difference.max().item())
            )
        all_ok = all_ok and ok
        metrics.append({
            "name": spec.name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "ok": ok,
            "max_abs_diff": max_abs,
            "rtol": rtol,
            "atol": atol,
            "max_error_ratio": max_error_ratio,
            "error_count": error_count,
            "allowed_error_count": allowed_errors,
        })
    return all_ok, metrics


def run(
    relative: str, *, device_index: int, seed: int, run_id: str
) -> dict[str, object]:
    import torch

    manifest = load_manifest()
    policy = load_policy()
    entry = policy_entry(relative, policy)
    source = source_record(relative, manifest)
    mode = str(entry["compatibility_mode"])
    started = time.monotonic()
    payload: dict[str, object] = {
        "schema": 2,
        "kind": "article-demo-nvidia-run",
        "article_url": manifest["article"]["url"],
        "upstream_commit": manifest["upstream"]["commit"],
        "run_id": run_id,
        "source": source,
        "compatibility": {
            "mode": mode,
            "adapter": entry.get("adapter"),
            "reason": entry["reason"],
            "policy_sha256": sha256_file(POLICY_PATH),
        },
        "seed": seed,
        "device_index": device_index,
        "adapter_source": {
            "path": "tools/run_article_demo_nvidia.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "status": "not_started",
    }
    if mode not in {"strict-pypto-nvidia", "computational-cuda-reference"}:
        payload.update(
            {
                "status": "skipped",
                "skip_reason": entry["reason"],
                "elapsed_seconds": 0.0,
                "stages": {
                    "source_audit": "pass",
                    "compatibility_compile": "skipped",
                    "device_execution": "skipped",
                    "precision_compare": "skipped",
                },
            }
        )
        return payload
    if not torch.cuda.is_available():
        payload.update(
            {
                "status": "blocked",
                "blocker": "CUDA runtime/device unavailable",
                "stages": {
                    "source_audit": "pass",
                    "compatibility_compile": "not_started",
                    "device_execution": "blocked",
                    "precision_compare": "not_started",
                },
            }
        )
        return payload
    device = torch.device("cuda", device_index)
    module = import_demo(relative)
    specs, cpu_values, expected = _spec_values(module, relative, seed)
    payload["inputs"] = [
        {
            "name": spec.name,
            "shape": list(getattr(spec, "shape", [])),
            "dtype": str(getattr(spec, "dtype", "")),
            "is_output": bool(getattr(spec, "is_output", False)),
        }
        for spec in specs
    ]
    payload["golden"] = {
        "function": _golden_name(module, relative),
        "outputs": [spec.name for spec in specs if getattr(spec, "is_output", False)],
    }
    device_values = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in cpu_values.items()
    }
    if mode == "strict-pypto-nvidia":
        result = _run_strict_hello(device_values, device, relative)
    else:
        result_values = _reference_adapter(relative, device_values, module)
        result = {
            "outputs": {
                spec.name: result_values[spec.name]
                for spec in specs
                if getattr(spec, "is_output", False)
            },
            "adapter": "computational-cuda-reference",
        }
    passed, metrics = _compare(
        relative, result["outputs"], expected, specs, module
    )
    payload.update(
        {
            "adapter": result["adapter"],
            "outputs": metrics,
            "golden_pass": passed,
            "strict_compiler_evidence": mode == "strict-pypto-nvidia",
            "elapsed_seconds": time.monotonic() - started,
            "status": "pass" if passed else "fail",
            "stages": {
                "source_audit": "pass",
                "input_generation": "pass",
                "golden_reference": "pass",
                "compatibility_compile": (
                    "strict-pypto-nvidia"
                    if mode == "strict-pypto-nvidia"
                    else "independent-cuda-reference"
                ),
                "device_execution": "pass",
                "precision_compare": "pass" if passed else "fail",
            },
        }
    )
    if "artifact" in result:
        payload["artifact"] = result["artifact"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, help="path relative to demo/pypto-lib")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--run-id",
        default=None,
        help="stable evidence identifier; defaults to article-demo-nvidia-<stem>",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    relative = Path(args.demo).as_posix()
    if relative.startswith("/") or ".." in Path(relative).parts:
        parser.error("--demo must stay below demo/pypto-lib")
    try:
        run_id = args.run_id or f"article-demo-nvidia-{Path(relative).stem}"
        allowed_run_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
        if not run_id or any(ch not in allowed_run_id for ch in run_id):
            parser.error("--run-id contains unsupported characters")
        payload = run(relative, device_index=args.device, seed=args.seed, run_id=run_id)
    except Exception as error:  # noqa: BLE001 - preserve exact blocker in evidence
        source: dict[str, object] = {"path": relative}
        try:
            source = source_record(relative, load_manifest())
        except Exception:  # noqa: BLE001 - retain the original failure detail
            pass
        payload = {
            "schema": 2,
            "kind": "article-demo-nvidia-run",
            "source": source,
            "run_id": args.run_id,
            "adapter_source": {
                "path": "tools/run_article_demo_nvidia.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_json(args.output.resolve(), payload)
    print(f"[ARTICLE-NVIDIA] source={relative}")
    print(f"[ARTICLE-NVIDIA] mode={payload.get('compatibility', {}).get('mode', 'unknown')}")
    print(
        "[ARTICLE-NVIDIA] adapter="
        f"{payload.get('adapter', payload.get('compatibility', {}).get('adapter'))}"
    )
    print(f"[ARTICLE-NVIDIA] status={payload.get('status')} golden_pass={payload.get('golden_pass')}")
    if payload.get("artifact"):
        artifact = payload["artifact"]
        print(
            f"[ARTICLE-NVIDIA] artifact={artifact.get('kernel_name')} "
            f"fallback_used={artifact.get('fallback_used')}"
        )
    print(f"article demo NVIDIA report: {args.output.resolve()}")
    return 0 if payload.get("status") in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
