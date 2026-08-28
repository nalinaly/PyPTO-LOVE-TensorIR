#!/usr/bin/env python3
"""Execution acceptance for the Qwen3.5 PyPTO operators on SM120."""

import argparse
import json
import hashlib
import math
import os
import pathlib
import sys

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from pypto_kernels._boot import bootstrap, loaded_dso_path  # noqa: E402
from pypto_kernels import (  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == PACKAGE_ROOT or PACKAGE_ROOT in output.parents:
        raise ValueError("execution output must be outside the source package")
    seed = 3
    torch.manual_seed(seed)
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
    for integer_dtype in (torch.int32, torch.int64):
        for count in (1, 19):
            integer_table = torch.arange(
                65, device="cuda", dtype=integer_dtype
            ).mul_(7)
            integer_indices = (
                torch.arange(count, device="cuda", dtype=torch.int64) * 11 + 3
            ) % integer_table.numel()
            integer_out = embedding.integer_gather(
                integer_table, integer_indices, stream=stream
            )
            stream.synchronize()
            integer_ref = integer_table.index_select(0, integer_indices)
            cases.append(
                {
                    "case": f"integer_gather_{integer_dtype}_rows{count}",
                    "implementation": "native-tile-dsl-integer-gather",
                    "launches": 1,
                    "table_shape": list(integer_table.shape),
                    "table_stride": list(integer_table.stride()),
                    "indices_shape": list(integer_indices.shape),
                    "indices_stride": list(integer_indices.stride()),
                    "output_shape": list(integer_out.shape),
                    "dtype": str(integer_dtype),
                    "max_abs_diff": int(
                        (integer_out.to(torch.int64) - integer_ref.to(torch.int64))
                        .abs()
                        .max()
                    ),
                    "correct": bool(torch.equal(integer_out, integer_ref)),
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

    def run_stateful_conv_case(
        *, batch_size: int, tokens_per_request: int, index_dtype, label: str
    ) -> None:
        channels = 4096
        rows = batch_size * tokens_per_request
        conv_x = (
            torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        conv_weight = (
            torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        state_slots = 8
        state_stride = channels * 3 + 128
        conv_state = torch.empty_strided(
            (state_slots, 3, channels),
            (state_stride, channels, 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        conv_state.normal_().mul_(0.2)
        state_indices = torch.arange(
            2, 2 + batch_size, device="cuda", dtype=index_dtype
        )
        reference_state = conv_state.clone()
        torch.cuda.synchronize()
        conv_out = causal_conv1d.causal_conv1d(
            conv_x,
            conv_weight,
            conv_state,
            state_indices,
            batch_size=batch_size,
            tokens_per_request=tokens_per_request,
            stream=stream,
        )
        stream.synchronize()
        x_rows = conv_x.view(batch_size, tokens_per_request, channels)
        reference_rows = []
        for batch_row in range(batch_size):
            slot = int(state_indices[batch_row])
            history = reference_state[slot]
            outputs = []
            for token in range(tokens_per_request):
                current = x_rows[batch_row, token]
                linear = (
                    history[0].float() * conv_weight[:, 0].float()
                    + history[1].float() * conv_weight[:, 1].float()
                    + history[2].float() * conv_weight[:, 2].float()
                    + current.float() * conv_weight[:, 3].float()
                )
                outputs.append(torch.nn.functional.silu(linear))
                history = torch.stack((history[1], history[2], current), dim=0)
            reference_state[slot] = history
            reference_rows.append(torch.stack(outputs))
        conv_ref = torch.stack(reference_rows).view(rows, channels)
        output_diff = float((conv_out.float() - conv_ref).abs().max())
        state_diff = float((conv_state.float() - reference_state.float()).abs().max())
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-stateful-conv",
                "launches": 1,
                "batch_size": batch_size,
                "tokens_per_request": tokens_per_request,
                "state_row_stride": state_stride,
                "output_max_abs_diff": output_diff,
                "state_max_abs_diff": state_diff,
                "max_abs_diff": max(output_diff, state_diff),
                "correct": bool(
                    torch.allclose(
                        conv_out.float(), conv_ref, rtol=5e-2, atol=5e-2
                    )
                    and torch.equal(conv_state, reference_state)
                ),
            }
        )

    run_stateful_conv_case(
        batch_size=2,
        tokens_per_request=1,
        index_dtype=torch.int32,
        label="causal_conv1d_stateful_decode_b2_d4096_width4",
    )
    run_stateful_conv_case(
        batch_size=1,
        tokens_per_request=5,
        index_dtype=torch.int64,
        label="causal_conv1d_stateful_prefill_b1_t5_d4096_width4",
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
        *,
        q_heads: int,
        kv_heads: int,
        valid_counts: tuple[int, ...],
        label: str,
        cache_row_stride: int | None = None,
    ) -> None:
        bucket_tokens = 16
        paged_head_dim = 256
        cache_rows = 1024
        request_rows = 65
        max_context_len = 4096
        batch_size = len(valid_counts)
        row_width = kv_heads * paged_head_dim
        if cache_row_stride is None:
            cache_row_stride = row_width
        assert batch_size > 0
        assert all(0 < count <= bucket_tokens for count in valid_counts)
        paged_query = (
            torch.randn(
                batch_size,
                q_heads * paged_head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        key_cache = torch.empty_strided(
            (cache_rows, row_width),
            (cache_row_stride, 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.empty_strided(
            (cache_rows, row_width),
            (cache_row_stride, 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_cache.normal_().mul_(0.2)
        value_cache.normal_().mul_(0.2)
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
        request_index = torch.arange(
            7, 7 + batch_size, device="cuda", dtype=torch.int64
        )
        physical_pool = (
            torch.randperm(cache_rows - 1, device="cuda")[: sum(valid_counts)] + 1
        )
        virtual_pool = torch.arange(
            1, 1 + sum(valid_counts), device="cuda", dtype=torch.int64
        )
        virtual_to_physical = torch.arange(
            cache_rows, device="cuda", dtype=torch.int64
        )
        virtual_to_physical[virtual_pool] = physical_pool
        selected_by_batch = []
        virtual_by_batch = []
        selected_offset = 0
        for batch_row, valid_count in enumerate(valid_counts):
            selected = physical_pool[
                selected_offset : selected_offset + valid_count
            ].contiguous()
            virtual = virtual_pool[
                selected_offset : selected_offset + valid_count
            ].contiguous()
            selected_offset += valid_count
            selected_by_batch.append(selected)
            virtual_by_batch.append(virtual)
            req_to_token[7 + batch_row, :valid_count] = virtual.to(torch.int32)
        valid_tokens = torch.tensor(
            valid_counts, device="cuda", dtype=torch.int64
        )
        current_key = (
            torch.randn(
                batch_size, row_width, device="cuda", dtype=torch.bfloat16
            )
            * 0.2
        )
        current_value = (
            torch.randn(
                batch_size, row_width, device="cuda", dtype=torch.bfloat16
            )
            * 0.2
        )
        virtual_row = torch.stack(
            [selected[-1] for selected in virtual_by_batch]
        ).contiguous()
        physical_row = virtual_to_physical[virtual_row]
        torch.cuda.synchronize()
        write_anchor = attention.paged_cache_write(
            key_cache,
            value_cache,
            virtual_row,
            virtual_to_physical,
            current_key,
            current_value,
            stream=stream,
        )
        stream.synchronize()
        written_key = key_cache[physical_row].view(batch_size, row_width)
        written_value = value_cache[physical_row].view(batch_size, row_width)
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
            virtual_to_physical,
            kv_heads=kv_heads,
            bucket_tokens=bucket_tokens,
            stream=stream,
        )
        stream.synchronize()

        key_heads = key_cache.view(cache_rows, kv_heads, paged_head_dim)
        value_heads = value_cache.view(cache_rows, kv_heads, paged_head_dim)
        queries_per_kv = q_heads // kv_heads
        query_heads = paged_query.view(batch_size, q_heads, paged_head_dim)
        reference_batches = []
        for batch_row, selected in enumerate(selected_by_batch):
            reference_heads = []
            for q_head in range(q_heads):
                kv_head = q_head // queries_per_kv
                selected_keys = key_heads[selected, kv_head].float()
                selected_values = value_heads[selected, kv_head].float()
                scores = (
                    query_heads[batch_row, q_head].float() @ selected_keys.T
                ) / math.sqrt(paged_head_dim)
                reference_heads.append(
                    torch.softmax(scores, dim=-1) @ selected_values
                )
            reference_batches.append(torch.stack(reference_heads))
        paged_ref = torch.stack(reference_batches).view(batch_size, -1)
        max_abs_diff = float((paged_out.float() - paged_ref).abs().max())
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-request-table",
                "launches": 1,
                "batch_size": batch_size,
                "bucket_tokens": bucket_tokens,
                "cache_row_stride": cache_row_stride,
                "valid_tokens": list(valid_counts),
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
        valid_counts=(13,),
        label="attention_paged_decode_0_8b 8q2kv bucket16 valid13 d256",
    )
    run_paged_decode_case(
        q_heads=16,
        kv_heads=4,
        valid_counts=(16,),
        label="attention_paged_decode_9b 16q4kv bucket16 valid16 d256",
    )
    run_paged_decode_case(
        q_heads=8,
        kv_heads=2,
        valid_counts=(13, 7),
        label="attention_paged_decode_batch2_0_8b 8q2kv bucket16 valid13_7 d256",
        cache_row_stride=2048,
    )

    prefill_rows, prefill_width = 13, 512
    prefill_key_cache = torch.randn(
        1024, prefill_width, device="cuda", dtype=torch.bfloat16
    )
    prefill_value_cache = torch.randn_like(prefill_key_cache)
    prefill_physical = torch.randperm(1023, device="cuda")[:prefill_rows] + 1
    prefill_v2p = torch.arange(1024, device="cuda", dtype=torch.int64)
    prefill_key = torch.randn(
        prefill_rows, prefill_width, device="cuda", dtype=torch.bfloat16
    )
    prefill_value = torch.randn_like(prefill_key)
    torch.cuda.synchronize()
    prefill_anchor = attention.paged_cache_write(
        prefill_key_cache,
        prefill_value_cache,
        prefill_physical,
        prefill_v2p,
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
        virtual = torch.arange(
            1, valid_total + 1, device="cuda", dtype=torch.int64
        )
        virtual_to_physical = torch.arange(
            cache_rows, device="cuda", dtype=torch.int64
        )
        virtual_to_physical[virtual] = selected
        req_to_token[request_row, :valid_total] = virtual.to(torch.int32)
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
        current_rows = virtual[prefix_count:valid_total].contiguous()
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
            virtual_to_physical,
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
            virtual_to_physical,
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
        attention_launches = attention._paged_prefill_partition_count(kv_heads)
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-causal-paged-prefill",
                "launches": 1 + attention_launches,
                "launch_count": 1 + attention_launches,
                "cache_write_launches": 1,
                "attention_launches": attention_launches,
                "attention_launch_topology": (
                    "one_fused_all_kv_heads_launch"
                    if attention_launches == 1
                    else "one_single_kv_head_launch_per_kv_head"
                ),
                "kv_heads": kv_heads,
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

    def run_gdn_recurrent_case(
        *, batch_size: int, tokens_per_request: int, index_dtype, label: str
    ) -> None:
        q_heads, value_heads, dk, dv = 8, 16, 128, 128
        rows = batch_size * tokens_per_request
        mixed_width = 2 * q_heads * dk + value_heads * dv
        mixed_qkv = (
            torch.randn(
                rows, mixed_width, device="cuda", dtype=torch.bfloat16
            )
            * 0.2
        )
        a = torch.randn(
            rows, value_heads, device="cuda", dtype=torch.bfloat16
        )
        b = torch.randn_like(a)
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
        dt_bias = (
            torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
            * 0.1
        )
        state_slots = 65
        state_width = value_heads * dv * dk
        state_stride = state_width + 4096
        state = torch.empty_strided(
            (state_slots, value_heads, dv, dk),
            (state_stride, dv * dk, dk, 1),
            device="cuda",
            dtype=torch.float32,
        )
        state.normal_().mul_(0.02)
        state_indices = torch.arange(
            3, 3 + batch_size, device="cuda", dtype=index_dtype
        )
        reference_state = state.clone()
        torch.cuda.synchronize()
        output = gdn.gdn_recurrent(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            state_indices,
            batch_size=batch_size,
            tokens_per_request=tokens_per_request,
            stream=stream,
        )
        stream.synchronize()

        mixed = mixed_qkv.view(batch_size, tokens_per_request, mixed_width)
        a_rows = a.view(batch_size, tokens_per_request, value_heads)
        b_rows = b.view(batch_size, tokens_per_request, value_heads)
        groups = value_heads // q_heads
        reference_outputs = []
        for batch_row in range(batch_size):
            slot = int(state_indices[batch_row])
            current = reference_state[slot]
            token_outputs = []
            for token in range(tokens_per_request):
                packed = mixed[batch_row, token]
                query = packed[: q_heads * dk].view(q_heads, dk).float()
                key = packed[q_heads * dk : 2 * q_heads * dk].view(
                    q_heads, dk
                ).float()
                value = packed[2 * q_heads * dk :].view(value_heads, dv).float()
                query = query / torch.sqrt(
                    torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6
                )
                key = key / torch.sqrt(
                    torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6
                )
                query = query.repeat_interleave(groups, dim=0) / math.sqrt(dk)
                key = key.repeat_interleave(groups, dim=0)
                log_decay = -torch.exp(A_log) * torch.nn.functional.softplus(
                    a_rows[batch_row, token].float() + dt_bias
                )
                current = current * torch.exp(log_decay)[:, None, None]
                residual = value - torch.einsum("hvk,hk->hv", current, key)
                beta = torch.sigmoid(b_rows[batch_row, token].float()).to(
                    torch.bfloat16
                ).float()
                current = current + (
                    residual * beta
                )[:, :, None] * key[:, None, :]
                token_outputs.append(torch.einsum("hk,hvk->hv", query, current))
            reference_state[slot] = current
            reference_outputs.append(torch.stack(token_outputs))
        reference_output = torch.stack(reference_outputs)
        output_diff = float((output.float() - reference_output).abs().max())
        state_diff = float((state.float() - reference_state).abs().max())
        cases.append(
            {
                "case": label,
                "implementation": "native-tile-dsl-gdn-recurrent",
                "launches": 1,
                "batch_size": batch_size,
                "tokens_per_request": tokens_per_request,
                "state_row_stride": state_stride,
                "output_max_abs_diff": output_diff,
                "state_max_abs_diff": state_diff,
                "max_abs_diff": max(output_diff, state_diff),
                "correct": bool(
                    torch.allclose(
                        output.float(), reference_output, rtol=8e-2, atol=8e-2
                    )
                    and torch.allclose(
                        state.float(), reference_state, rtol=2e-3, atol=2e-3
                    )
                ),
            }
        )

    run_gdn_recurrent_case(
        batch_size=2,
        tokens_per_request=1,
        index_dtype=torch.int32,
        label="gdn_recurrent_decode_b2_h8_hv16_k128_v128",
    )
    run_gdn_recurrent_case(
        batch_size=1,
        tokens_per_request=3,
        index_dtype=torch.int64,
        label="gdn_recurrent_prefill_b1_t3_h8_hv16_k128_v128",
    )
    ok = all(c["correct"] for c in cases)
    dso = loaded_dso_path()
    result = {
        "schema": 2,
        "kind": "pypto-kernels-exec-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "seed": seed,
        "thresholds": {
            "pointwise_rtol": 5e-2,
            "pointwise_atol": 5e-2,
            "attention_rtol": 1e-1,
            "attention_atol": 8e-2,
            "gdn_output_rtol": 8e-2,
            "gdn_output_atol": 8e-2,
            "gdn_state_rtol": 2e-3,
            "gdn_state_atol": 2e-3
        },
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": bootstrap()["compiler"]
        .get_nvidia_backend_build_info()
        .pypto_revision,
        "native_tile_ops": [
            "silu_and_mul",
            "sigmoid_mul",
            "embedding",
            "integer_gather",
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
            "gdn_recurrent",
        ],
        "all_correct": ok,
        "cases": cases,
    }
    rendered = json.dumps(result, indent=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if ok else 75


if __name__ == "__main__":
    raise SystemExit(main())
