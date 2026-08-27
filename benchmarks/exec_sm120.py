#!/usr/bin/env python3
"""Execution acceptance for the Qwen3.5 PyPTO operators on SM120."""

import json
import hashlib
import math
import os
import pathlib
import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import torch

from pypto_kernels._boot import DSO_PATH, bootstrap
from pypto_kernels import (
    attention,
    causal_conv1d,
    embedding,
    fused_add_rmsnorm,
    gdn,
    gated_rmsnorm,
    linear,
    qk_rmsnorm_rope,
    rmsnorm,
    rope,
    sigmoid_mul,
    silu_and_mul,
)


def main() -> int:
    torch.manual_seed(3)
    stream = torch.cuda.Stream()
    cases = []
    for m, n in ((256, 1024), (4096, 1024), (1, 3584)):
        g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 2
        u = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 2
        out = silu_and_mul.silu_and_mul(g, u, stream=stream)
        stream.synchronize()
        ref = torch.nn.functional.silu(g.float()) * u.float()
        cases.append(
            {
                "case": f"silu_and_mul {m}x{n}",
                "implementation": "native-tile-dsl",
                "launches": 1,
                "max_abs_diff": float((out.float() - ref).abs().max()),
                "correct": bool(torch.allclose(out.float(), ref, rtol=5e-2, atol=5e-2)),
            }
        )
        value = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        out2 = sigmoid_mul.sigmoid_mul(value, gate, stream=stream)
        stream.synchronize()
        ref2 = value.float() * torch.sigmoid(gate.float())
        cases.append(
            {
                "case": f"sigmoid_mul {m}x{n}",
                "implementation": "native-tile-dsl",
                "launches": 1,
                "max_abs_diff": float((out2.float() - ref2).abs().max()),
                "correct": bool(
                    torch.allclose(out2.float(), ref2, rtol=5e-2, atol=5e-2)
                ),
            }
        )
    # Every broadcast-dependent former B-class operator below is one compile
    # and one launch. Launch arguments follow builder input discovery order.
    embedding_tokens, embedding_vocab, embedding_hidden = 32, 248320, 1024
    token_ids = torch.randint(
        0,
        embedding_vocab,
        (embedding_tokens,),
        device="cuda",
        dtype=torch.int64,
    )
    embedding_weight = torch.randn(
        embedding_vocab,
        embedding_hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    embedding_out = embedding.embedding(token_ids, embedding_weight, stream=stream)
    stream.synchronize()
    embedding_ref = embedding_weight[token_ids]
    cases.append(
        {
            "case": "embedding_bf16 32x248320x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float(
                (embedding_out.float() - embedding_ref.float()).abs().max()
            ),
            "correct": bool(torch.equal(embedding_out, embedding_ref)),
        }
    )

    # Qwen3.5-0.8B full-attention preparation: per-head Q/Gate interleave,
    # Gemma RMSNorm, partial NeoX RoPE and gate deinterleave in one graph.
    qk_tokens, q_heads, kv_heads = 2, 8, 2
    qk_head_dim, qk_rotary_dim, qk_max_positions = 256, 64, 262144
    q_gate = torch.randn(
        qk_tokens,
        2 * q_heads * qk_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    qk_key = torch.randn(
        qk_tokens,
        kv_heads * qk_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_weight = torch.randn(
        qk_head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    k_weight = torch.randn(
        qk_head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    rope_angles = torch.randn(
        qk_max_positions,
        qk_rotary_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    cos_sin_cache = torch.cat(
        (torch.cos(rope_angles), torch.sin(rope_angles)), dim=1
    ).to(torch.bfloat16)
    positions = torch.tensor([0, 17], device="cuda", dtype=torch.int64)
    q_out, k_out, gate_out = qk_rmsnorm_rope.qk_rmsnorm_rope_gate(
        q_gate,
        qk_key,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_heads=q_heads,
        kv_heads=kv_heads,
        stream=stream,
    )
    stream.synchronize()

    q_gate_heads = q_gate.view(qk_tokens, q_heads, 2 * qk_head_dim)
    q_input = q_gate_heads[..., :qk_head_dim]
    gate_ref = q_gate_heads[..., qk_head_dim:]
    k_input = qk_key.view(qk_tokens, kv_heads, qk_head_dim)

    def qk_norm_reference(value, weight):
        normalized = (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1.0e-6)
            * (1.0 + weight.float())
        )
        return normalized.to(torch.bfloat16).float()

    def partial_neox_reference(value):
        half = qk_rotary_dim // 2
        low = value[..., :half]
        high = value[..., half:qk_rotary_dim]
        tail = value[..., qk_rotary_dim:]
        selected = cos_sin_cache[positions].float()
        cos = selected[:, :half].unsqueeze(1)
        sin = selected[:, half:].unsqueeze(1)
        return torch.cat(
            (low * cos - high * sin, high * cos + low * sin, tail), dim=-1
        ).to(torch.bfloat16)

    q_ref = partial_neox_reference(qk_norm_reference(q_input, q_weight))
    k_ref = partial_neox_reference(qk_norm_reference(k_input, k_weight))
    q_diff = float(
        (q_out.view_as(q_ref).float() - q_ref.float()).abs().max()
    )
    k_diff = float(
        (k_out.view_as(k_ref).float() - k_ref.float()).abs().max()
    )
    cases.append(
        {
            "case": "qk_rmsnorm_partial_rope_gate_bf16 2x8x2x256x64",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "q_max_abs_diff": q_diff,
            "k_max_abs_diff": k_diff,
            "gate_exact": bool(torch.equal(gate_out, gate_ref)),
            "max_abs_diff": max(q_diff, k_diff),
            "correct": bool(
                torch.equal(gate_out, gate_ref)
                and torch.allclose(
                    q_out.view_as(q_ref).float(),
                    q_ref.float(),
                    rtol=5e-2,
                    atol=5e-2,
                )
                and torch.allclose(
                    k_out.view_as(k_ref).float(),
                    k_ref.float(),
                    rtol=5e-2,
                    atol=5e-2,
                )
            ),
        }
    )

    rows, cols = 256, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    rms_weight = torch.randn(cols, device="cuda", dtype=torch.bfloat16) * 0.1
    rms_out = rmsnorm.rmsnorm(x, rms_weight, stream=stream)
    stream.synchronize()
    rms_ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * (1.0 + rms_weight.float())
    )
    cases.append(
        {
            "case": "rmsnorm_bf16 256x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((rms_out.float() - rms_ref).abs().max()),
            "correct": bool(
                torch.allclose(rms_out.float(), rms_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rms_residual = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.5
    fused_norm_out, residual_out = fused_add_rmsnorm.fused_add_rmsnorm(
        x, rms_residual, rms_weight, stream=stream
    )
    stream.synchronize()
    residual_ref = x + rms_residual
    fused_norm_ref = (
        residual_ref.float()
        * torch.rsqrt(residual_ref.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * (1.0 + rms_weight.float())
    )
    fused_norm_diff = float((fused_norm_out.float() - fused_norm_ref).abs().max())
    residual_diff = float((residual_out.float() - residual_ref.float()).abs().max())
    cases.append(
        {
            "case": "fused_add_rmsnorm_bf16 256x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": max(fused_norm_diff, residual_diff),
            "normalized_max_abs_diff": fused_norm_diff,
            "residual_max_abs_diff": residual_diff,
            "correct": bool(
                torch.equal(residual_out, residual_ref)
                and torch.allclose(
                    fused_norm_out.float(),
                    fused_norm_ref,
                    rtol=5e-2,
                    atol=5e-2,
                )
            ),
        }
    )

    gated_x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    gated_gate = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    gated_weight = 1.0 + torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
    gated_out = gated_rmsnorm.gated_rmsnorm(
        gated_x, gated_gate, gated_weight, stream=stream
    )
    stream.synchronize()
    gated_ref = (
        gated_x.float()
        * torch.rsqrt(gated_x.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * gated_weight.float()
        * torch.nn.functional.silu(gated_gate.float())
    )
    cases.append(
        {
            "case": "gated_rmsnorm_bf16 256x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((gated_out.float() - gated_ref).abs().max()),
            "correct": bool(
                torch.allclose(gated_out.float(), gated_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    conv_channels, conv_tokens = 2048, 64
    conv_x = (
        torch.randn(
            conv_channels,
            conv_tokens,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )
    conv_weight = (
        torch.randn(conv_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    )
    conv_out = causal_conv1d.causal_conv1d(conv_x, conv_weight, stream=stream)
    stream.synchronize()
    conv_linear = torch.nn.functional.conv1d(
        conv_x.float().unsqueeze(0),
        conv_weight.float().unsqueeze(1),
        padding=3,
        groups=conv_channels,
    )[..., :conv_tokens].squeeze(0)
    conv_ref = torch.nn.functional.silu(conv_linear)
    cases.append(
        {
            "case": "causal_conv1d_bf16 2048x64 width4",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((conv_out.float() - conv_ref).abs().max()),
            "correct": bool(
                torch.allclose(conv_out.float(), conv_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rows, half = 256, 64
    x_rope = torch.randn(rows, 2 * half, device="cuda", dtype=torch.bfloat16)
    cos_half = torch.rand(rows, half, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.rand(rows, half, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=1).contiguous()
    sin = torch.cat((sin_half, sin_half), dim=1).contiguous()
    rope_out = rope.rope(x_rope, cos, sin, stream=stream)
    stream.synchronize()
    x_low, x_high = x_rope.float().chunk(2, dim=1)
    rope_ref = torch.cat(
        (
            x_low * cos_half.float() - x_high * sin_half.float(),
            x_high * cos_half.float() + x_low * sin_half.float(),
        ),
        dim=1,
    )
    cases.append(
        {
            "case": "rope_bf16 256x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((rope_out.float() - rope_ref).abs().max()),
            "correct": bool(
                torch.allclose(rope_out.float(), rope_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    rows, tokens, head_dim, value_dim = 32, 128, 128, 128
    query_attn = torch.randn(rows, head_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    key_attn = torch.randn(tokens, head_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    value_attn = (
        torch.randn(tokens, value_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    )
    attention_out = attention.attention(query_attn, key_attn, value_attn, stream=stream)
    stream.synchronize()
    score_ref = query_attn.float() @ key_attn.float().T / math.sqrt(head_dim)
    attention_ref = torch.softmax(score_ref, dim=-1) @ value_attn.float()
    cases.append(
        {
            "case": "attention_bf16 32x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((attention_out.float() - attention_ref).abs().max()),
            "correct": bool(
                torch.allclose(
                    attention_out.float(), attention_ref, rtol=1e-1, atol=8e-2
                )
            ),
        }
    )

    def run_paged_decode_case(
        *, q_heads: int, kv_heads: int, valid_count: int, label: str
    ) -> None:
        bucket_tokens = 16
        paged_head_dim = 256
        cache_rows = 1024
        request_rows = 65
        max_context_len = 4096
        request_row = 7
        paged_query = (
            torch.randn(
                q_heads,
                paged_head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        key_cache = (
            torch.randn(
                cache_rows,
                kv_heads * paged_head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        value_cache = torch.randn_like(key_cache) * 0.2
        # Every padded request-table entry remains slot 0. Make that row
        # adversarial so a missing valid-length mask produces an obvious error.
        key_cache[0].zero_()
        value_cache[0].fill_(8.0)
        req_to_token = torch.zeros(
            request_rows,
            max_context_len,
            device="cuda",
            dtype=torch.int32,
        )
        selected = torch.randperm(cache_rows - 1, device="cuda")[:valid_count] + 1
        req_to_token[request_row, :valid_count] = selected.to(torch.int32)
        request_index = torch.tensor(
            [request_row], device="cuda", dtype=torch.int64
        )
        valid_tokens = torch.tensor(
            [valid_count], device="cuda", dtype=torch.int64
        )
        row_width = kv_heads * paged_head_dim
        current_key = (
            torch.randn(1, row_width, device="cuda", dtype=torch.bfloat16) * 0.2
        )
        current_value = (
            torch.randn(1, row_width, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        physical_row = selected[valid_count - 1 : valid_count].contiguous()
        torch.cuda.synchronize()
        write_anchor = attention.paged_cache_write(
            key_cache,
            value_cache,
            physical_row,
            current_key,
            current_value,
            stream=stream,
        )
        stream.synchronize()
        written_key = key_cache[physical_row].view(1, row_width)
        written_value = value_cache[physical_row].view(1, row_width)
        key_write_diff = float((written_key.float() - current_key.float()).abs().max())
        value_write_diff = float(
            (written_value.float() - current_value.float()).abs().max()
        )
        anchor_ref = current_key.float() + current_value.float()
        anchor_diff = float((write_anchor.float() - anchor_ref).abs().max())
        cases.append(
            {
                "case": f"{label} cache_write",
                "implementation": "native-tile-dsl-scatter-rows",
                "launches": 1,
                "key_max_abs_diff": key_write_diff,
                "value_max_abs_diff": value_write_diff,
                "anchor_max_abs_diff": anchor_diff,
                "max_abs_diff": max(key_write_diff, value_write_diff, anchor_diff),
                "correct": bool(
                    torch.equal(written_key, current_key)
                    and torch.equal(written_value, current_value)
                    and torch.allclose(
                        write_anchor.float(), anchor_ref, rtol=5e-2, atol=5e-2
                    )
                ),
            }
        )
        paged_out = attention.paged_attention_decode(
            paged_query,
            key_cache,
            value_cache,
            req_to_token,
            request_index,
            valid_tokens,
            kv_heads=kv_heads,
            bucket_tokens=bucket_tokens,
            stream=stream,
        )
        stream.synchronize()

        key_heads = key_cache.view(cache_rows, kv_heads, paged_head_dim)
        value_heads = value_cache.view(cache_rows, kv_heads, paged_head_dim)
        queries_per_kv = q_heads // kv_heads
        reference_heads = []
        for q_head in range(q_heads):
            kv_head = q_head // queries_per_kv
            selected_keys = key_heads[selected, kv_head].float()
            selected_values = value_heads[selected, kv_head].float()
            scores = (
                paged_query[q_head].float() @ selected_keys.T
            ) / math.sqrt(paged_head_dim)
            reference_heads.append(torch.softmax(scores, dim=-1) @ selected_values)
        paged_ref = torch.stack(reference_heads)
        max_abs_diff = float((paged_out.float() - paged_ref).abs().max())
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-request-table",
                "launches": 1,
                "bucket_tokens": bucket_tokens,
                "valid_tokens": valid_count,
                "max_abs_diff": max_abs_diff,
                "correct": bool(
                    torch.allclose(
                        paged_out.float(), paged_ref, rtol=1e-1, atol=8e-2
                    )
                ),
            }
        )

    run_paged_decode_case(
        q_heads=8,
        kv_heads=2,
        valid_count=13,
        label="attention_paged_decode_0_8b 8q2kv bucket16 valid13 d256",
    )
    run_paged_decode_case(
        q_heads=16,
        kv_heads=4,
        valid_count=16,
        label="attention_paged_decode_9b 16q4kv bucket16 valid16 d256",
    )

    prefill_rows, prefill_width = 13, 512
    prefill_key_cache = torch.randn(
        1024, prefill_width, device="cuda", dtype=torch.bfloat16
    )
    prefill_value_cache = torch.randn_like(prefill_key_cache)
    prefill_physical = torch.randperm(1023, device="cuda")[:prefill_rows] + 1
    prefill_key = torch.randn(
        prefill_rows, prefill_width, device="cuda", dtype=torch.bfloat16
    )
    prefill_value = torch.randn_like(prefill_key)
    torch.cuda.synchronize()
    prefill_anchor = attention.paged_cache_write(
        prefill_key_cache,
        prefill_value_cache,
        prefill_physical,
        prefill_key,
        prefill_value,
        stream=stream,
    )
    stream.synchronize()
    prefill_written_key = prefill_key_cache[prefill_physical]
    prefill_written_value = prefill_value_cache[prefill_physical]
    prefill_anchor_ref = prefill_key.float() + prefill_value.float()
    prefill_key_diff = float(
        (prefill_written_key.float() - prefill_key.float()).abs().max()
    )
    prefill_value_diff = float(
        (prefill_written_value.float() - prefill_value.float()).abs().max()
    )
    prefill_anchor_diff = float(
        (prefill_anchor.float() - prefill_anchor_ref).abs().max()
    )
    cases.append(
        {
            "case": "attention_paged_cache_write_prefill 13x512",
            "implementation": "native-tile-dsl-scatter-rows",
            "launches": 1,
            "key_max_abs_diff": prefill_key_diff,
            "value_max_abs_diff": prefill_value_diff,
            "anchor_max_abs_diff": prefill_anchor_diff,
            "max_abs_diff": max(
                prefill_key_diff, prefill_value_diff, prefill_anchor_diff
            ),
            "correct": bool(
                torch.equal(prefill_written_key, prefill_key)
                and torch.equal(prefill_written_value, prefill_value)
                and torch.allclose(
                    prefill_anchor.float(),
                    prefill_anchor_ref,
                    rtol=5e-2,
                    atol=5e-2,
                )
            ),
        }
    )

    def run_paged_prefill_case(
        *, q_heads: int, kv_heads: int, label: str
    ) -> None:
        query_rows = 13
        prefix_count = 2
        bucket_tokens = 16
        prefill_head_dim = 256
        cache_rows = 1024
        request_rows = 65
        max_context_len = 4096
        request_row = 9
        row_width = kv_heads * prefill_head_dim
        prefill_query = (
            torch.randn(
                query_rows,
                q_heads * prefill_head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        key_cache = (
            torch.randn(
                cache_rows,
                row_width,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        value_cache = torch.randn_like(key_cache) * 0.2
        key_cache[0].zero_()
        value_cache[0].fill_(8.0)
        req_to_token = torch.zeros(
            request_rows,
            max_context_len,
            device="cuda",
            dtype=torch.int32,
        )
        valid_total = prefix_count + query_rows
        selected = torch.randperm(cache_rows - 1, device="cuda")[:valid_total] + 1
        req_to_token[request_row, :valid_total] = selected.to(torch.int32)
        current_key = (
            torch.randn(
                query_rows,
                row_width,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        current_value = torch.randn_like(current_key) * 0.2
        current_rows = selected[prefix_count:valid_total].contiguous()
        request_index = torch.tensor(
            [request_row], device="cuda", dtype=torch.int64
        )
        prefix_tokens = torch.tensor(
            [prefix_count], device="cuda", dtype=torch.int32
        )
        torch.cuda.synchronize()
        attention.paged_cache_write(
            key_cache,
            value_cache,
            current_rows,
            current_key,
            current_value,
            stream=stream,
        )
        stream.synchronize()
        prefill_out = attention.paged_attention_prefill(
            prefill_query,
            key_cache,
            value_cache,
            req_to_token,
            request_index,
            prefix_tokens,
            kv_heads=kv_heads,
            bucket_tokens=bucket_tokens,
            stream=stream,
        )
        stream.synchronize()

        key_heads = key_cache.view(cache_rows, kv_heads, prefill_head_dim)
        value_heads = value_cache.view(cache_rows, kv_heads, prefill_head_dim)
        query_heads = prefill_query.view(query_rows, q_heads, prefill_head_dim)
        queries_per_kv = q_heads // kv_heads
        reference_rows = []
        for query_row in range(query_rows):
            valid_tokens = prefix_count + query_row + 1
            physical = selected[:valid_tokens]
            reference_heads = []
            for q_head in range(q_heads):
                kv_head = q_head // queries_per_kv
                selected_keys = key_heads[physical, kv_head].float()
                selected_values = value_heads[physical, kv_head].float()
                scores = (
                    query_heads[query_row, q_head].float() @ selected_keys.T
                ) / math.sqrt(prefill_head_dim)
                reference_heads.append(
                    torch.softmax(scores, dim=-1) @ selected_values
                )
            reference_rows.append(torch.stack(reference_heads))
        prefill_ref = torch.stack(reference_rows).reshape_as(prefill_out)
        prefill_diff = float((prefill_out.float() - prefill_ref).abs().max())
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-causal-paged-prefill",
                "launches": 2,
                "cache_write_launches": 1,
                "attention_launches": 1,
                "prefix_tokens": prefix_count,
                "query_rows": query_rows,
                "bucket_tokens": bucket_tokens,
                "max_abs_diff": prefill_diff,
                "correct": bool(
                    torch.allclose(
                        prefill_out.float(), prefill_ref, rtol=1e-1, atol=8e-2
                    )
                ),
            }
        )

    run_paged_prefill_case(
        q_heads=8,
        kv_heads=2,
        label="attention_paged_prefill_0_8b prefix2 extend13 bucket16",
    )
    run_paged_prefill_case(
        q_heads=16,
        kv_heads=4,
        label="attention_paged_prefill_9b prefix2 extend13 bucket16",
    )

    linear_input = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    linear_weight = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    linear_out = linear.linear(linear_input, linear_weight, stream=stream)
    stream.synchronize()
    linear_ref = linear_input.float() @ linear_weight.float().T
    cases.append(
        {
            "case": "linear_bf16 32x1024x1024",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((linear_out.float() - linear_ref).abs().max()),
            "correct": bool(
                torch.allclose(linear_out.float(), linear_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )

    heads, dk, dv = 16, 128, 128
    query = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    decay = torch.rand(heads, dk, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    key = torch.randn(heads, dk, device="cuda", dtype=torch.bfloat16) * 0.2
    value = torch.randn(heads, dv, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05

    gdn_out = gdn.gdn_read(query, decay, gate, key, value, state, stream=stream)
    stream.synchronize()

    qd_ref = query.float() * decay.float()
    read_ref = torch.einsum("hd,hdv->hv", qd_ref, state.float())
    compose_ref = (
        query.float() * torch.nn.functional.softplus(gate.float()) * key.float()
    )
    dot_ref = compose_ref.sum(-1, keepdim=True)
    gdn_ref = read_ref + dot_ref * value.float()
    cases.append(
        {
            "case": "gdn_complete_read_bf16 16x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((gdn_out.float() - gdn_ref).abs().max()),
            "correct": bool(
                torch.allclose(gdn_out.float(), gdn_ref, rtol=8e-2, atol=8e-2)
            ),
        }
    )

    # GDN state update is one native tile graph and one launch.
    state = torch.randn(heads, dk, dv, device="cuda", dtype=torch.bfloat16) * 0.05
    state_decay = torch.rand(heads, dk, 1, device="cuda", dtype=torch.bfloat16)
    beta_key = torch.randn(heads, dk, 1, device="cuda", dtype=torch.bfloat16) * 0.05
    update_value = torch.randn(heads, 1, dv, device="cuda", dtype=torch.bfloat16) * 0.1
    updated = gdn.gdn_state_update(
        state,
        state_decay,
        beta_key,
        update_value,
        stream=stream,
    )
    stream.synchronize()
    update_ref = (
        state.float() * state_decay.float() + beta_key.float() * update_value.float()
    )
    cases.append(
        {
            "case": "gdn_state_update_bf16 16x128x128",
            "implementation": "native-tile-dsl",
            "launches": 1,
            "max_abs_diff": float((updated.float() - update_ref).abs().max()),
            "correct": bool(
                torch.allclose(updated.float(), update_ref, rtol=5e-2, atol=5e-2)
            ),
        }
    )
    ok = all(c["correct"] for c in cases)
    dso = pathlib.Path(DSO_PATH)
    result = {
        "schema": 2,
        "kind": "pypto-kernels-exec-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": bootstrap()["compiler"]
        .get_nvidia_backend_build_info()
        .pypto_revision,
        "native_tile_ops": [
            "silu_and_mul",
            "sigmoid_mul",
            "embedding",
            "qk_rmsnorm_rope",
            "rmsnorm",
            "fused_add_rmsnorm",
            "gated_rmsnorm",
            "causal_conv1d",
            "rope",
            "attention",
            "attention_paged_decode",
            "attention_paged_cache_write",
            "attention_paged_prefill",
            "linear",
            "gdn_read",
            "gdn_state_update",
        ],
        "all_correct": ok,
        "cases": cases,
    }
    rendered = json.dumps(result, indent=1)
    pathlib.Path(__file__).with_name("exec_results.json").write_text(
        rendered + "\n", encoding="utf-8"
    )
    print(rendered)
    return 0 if ok else 75


if __name__ == "__main__":
    raise SystemExit(main())
