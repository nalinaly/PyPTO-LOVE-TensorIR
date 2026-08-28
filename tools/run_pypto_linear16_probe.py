#!/usr/bin/env python3
"""Numerically probe the narrow Qwen GDN projection geometry."""

from __future__ import annotations

import json

from run_qwen35_0p8b_pypto_smoke import configure_environment


def main() -> int:
    configure_environment()
    import torch
    from pypto_kernels.embedding import embedding, integer_gather
    from pypto_kernels.gdn_projection import split_projection
    from pypto_kernels.gdn import gdn_recurrent
    from pypto_kernels.linear import linear, linear_to_float
    from pypto_kernels.silu_and_mul import silu_and_mul
    from pypto_plugins.sglang.stream import pypto_stream

    torch.manual_seed(7)
    x = torch.randn((19, 1024), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((32, 1024), device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight)
    reduced_precision = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    expected_full_reduction = torch.nn.functional.linear(x, weight)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
    ground_truth = torch.nn.functional.linear(x.float(), weight.float())
    packed_weight = torch.zeros((128, 1024), device="cuda", dtype=torch.bfloat16)
    packed_weight[:32].copy_(weight)
    with pypto_stream(x.device) as stream:
        actual = linear(x, packed_weight, stream=stream)[:, :32]
    projected_qkvz = torch.randn(
        (19, 6144), device="cuda", dtype=torch.bfloat16
    )
    with pypto_stream(x.device) as stream:
        mixed, z, b, a = split_projection(
            projected_qkvz,
            actual,
            q_heads=8,
            value_heads=16,
            key_dim=128,
            value_dim=128,
            stream=stream,
        )
    torch.cuda.synchronize()
    difference = (actual.float() - expected.float()).abs()
    pypto_ground_difference = (actual.float() - ground_truth).abs()
    vendor_ground_difference = (expected.float() - ground_truth).abs()
    full_reduction_difference = (
        actual.float() - expected_full_reduction.float()
    ).abs()
    projection_difference = max(
        float((mixed - projected_qkvz[:, :4096]).abs().max()),
        float((z.reshape(19, -1) - projected_qkvz[:, 4096:]).abs().max()),
        float((b - actual[:, :16]).abs().max()),
        float((a - actual[:, 16:]).abs().max()),
    )
    packed_gate = torch.randn(
        (19, 7168), device="cuda", dtype=torch.bfloat16
    )
    expected_silu = torch.nn.functional.silu(packed_gate[:, :3584]) * packed_gate[
        :, 3584:
    ]
    with pypto_stream(x.device) as stream:
        actual_silu = silu_and_mul(
            packed_gate[:, :3584], packed_gate[:, 3584:], stream=stream
        )
    torch.cuda.synchronize()
    silu_difference = (actual_silu.float() - expected_silu.float()).abs()
    head_x = x[:1]
    head_weight = weight.new_empty((128, 1024)).normal_()
    expected_head = torch.nn.functional.linear(head_x, head_weight)
    with pypto_stream(x.device) as stream:
        actual_head = linear(head_x, head_weight, stream=stream)
        actual_head_float = linear_to_float(head_x, head_weight, stream=stream)
    torch.cuda.synchronize()
    head_difference = (actual_head.float() - expected_head.float()).abs()
    head_float_difference = (actual_head_float - expected_head.float()).abs()
    wide_weight = torch.randn(
        (8192, 1024), device="cuda", dtype=torch.bfloat16
    )
    reduced_precision = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    expected_wide = torch.nn.functional.linear(x, wide_weight)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
    with pypto_stream(x.device) as stream:
        actual_wide = linear(x, wide_weight, stream=stream)
    torch.cuda.synchronize()
    wide_difference = (actual_wide.float() - expected_wide.float()).abs()
    clear_state = torch.randn(
        (2, 1, 128, 128), device="cuda", dtype=torch.float32
    )
    with pypto_stream(x.device) as stream:
        gdn_recurrent(
            torch.zeros((1, 384), device="cuda", dtype=torch.bfloat16),
            torch.zeros((1, 1), device="cuda", dtype=torch.bfloat16),
            torch.zeros((1, 1), device="cuda", dtype=torch.bfloat16),
            torch.full((1,), float("inf"), device="cuda", dtype=torch.float32),
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16),
            clear_state,
            torch.ones((1,), device="cuda", dtype=torch.int32),
            batch_size=1,
            tokens_per_request=1,
            stream=stream,
        )
    torch.cuda.synchronize()
    recurrent_clear_max = float(clear_state[1].abs().max())
    embedding_weight = torch.randn(
        (256, 1024), device="cuda", dtype=torch.bfloat16
    )
    token_ids = torch.arange(19, device="cuda", dtype=torch.int64)
    with pypto_stream(x.device) as stream:
        actual_embedding = embedding(token_ids, embedding_weight, stream=stream)
    torch.cuda.synchronize()
    embedding_max = float(
        (actual_embedding - embedding_weight[:19]).abs().max()
    )
    integer_table = torch.tensor([3, 17], device="cuda", dtype=torch.int32)
    integer_indices = torch.ones((1,), device="cuda", dtype=torch.int64)
    with pypto_stream(x.device) as stream:
        gathered_integer = integer_gather(
            integer_table, integer_indices, stream=stream
        )
    torch.cuda.synchronize()
    integer_gather_value = int(gathered_integer.cpu()[0])
    result = {
        "finite": bool(torch.isfinite(actual).all()),
        "integer_gather_value": integer_gather_value,
        "embedding_max_abs": embedding_max,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "full_reduction_max_abs": float(full_reduction_difference.max()),
        "full_reduction_mean_abs": float(full_reduction_difference.mean()),
        "pypto_ground_max_abs": float(pypto_ground_difference.max()),
        "pypto_ground_mean_abs": float(pypto_ground_difference.mean()),
        "one_row_max_abs": float(head_difference.max()),
        "one_row_mean_abs": float(head_difference.mean()),
        "one_row_float_max_abs": float(head_float_difference.max()),
        "projection_max_abs": projection_difference,
        "recurrent_clear_max_abs": recurrent_clear_max,
        "shape": list(actual.shape),
        "silu_max_abs": float(silu_difference.max()),
        "silu_mean_abs": float(silu_difference.mean()),
        "vendor_ground_max_abs": float(vendor_ground_difference.max()),
        "vendor_ground_mean_abs": float(vendor_ground_difference.mean()),
        "wide_full_reduction_max_abs": float(wide_difference.max()),
        "wide_full_reduction_mean_abs": float(wide_difference.mean()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return (
        0
        if result["finite"]
        and result["integer_gather_value"] == 17
        and result["embedding_max_abs"] == 0.0
        and result["max_abs"] <= 0.5
        and result["one_row_max_abs"] <= 0.5
        and result["one_row_float_max_abs"] <= 0.5
        and result["projection_max_abs"] == 0.0
        and result["recurrent_clear_max_abs"] == 0.0
        and result["silu_max_abs"] <= 0.5
        and result["wide_full_reduction_max_abs"] <= 0.25
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
