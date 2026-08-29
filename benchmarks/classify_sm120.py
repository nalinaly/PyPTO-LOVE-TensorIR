#!/usr/bin/env python3
"""Compile classification for every Qwen3.5 operator graph on SM120."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import pathlib
import sys

import torch

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from pypto_kernels._boot import bootstrap, classify, loaded_dso_path  # noqa: E402
from pypto_kernels import (  # noqa: E402
    attention,
    causal_conv1d,
    embedding,
    fused_add_rmsnorm,
    gdn,
    gdn_projection,
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
        raise ValueError("classification output must be outside the source package")
    sample = torch.empty((256, 1024), dtype=torch.bfloat16, device="meta")
    cases = (
        (
            "silu_and_mul",
            silu_and_mul.silu_and_mul_kernel.specialize(sample, sample, sample),
            [128],
        ),
        (
            "sigmoid_mul",
            sigmoid_mul.sigmoid_mul_kernel.specialize(sample, sample, sample),
            [128],
        ),
        ("embedding", embedding.build(1, 248320, 1024), [128]),
        (
            "integer_gather",
            embedding.integer_gather_kernel.specialize(
                torch.empty((65, 1), dtype=torch.int32, device="meta"),
                torch.empty((19, 1), dtype=torch.int64, device="meta"),
                torch.empty((19, 1), dtype=torch.int32, device="meta"),
            ),
            [1],
        ),
        (
            "qk_rmsnorm_rope",
            qk_rmsnorm_rope.build(2, 8, 2, 256, 64, 262144),
            [1, 1, 1, 1, 1, 32],
        ),
        ("rmsnorm", rmsnorm.build(1, 1024), [128]),
        (
            "fused_add_rmsnorm",
            fused_add_rmsnorm.build(1, 1024),
            [128],
        ),
        ("gated_rmsnorm", gated_rmsnorm.build(1, 128), [128]),
        (
            "causal_conv1d_stateful_decode",
            causal_conv1d.build(2, 1, 4096),
            [1, 1, 128],
        ),
        (
            "causal_conv1d_stateful_token_primitive",
            causal_conv1d.build(1, 1, 4096),
            [1, 1, 128],
        ),
        ("rope", rope.build(1, 64), [1, 64]),
        ("attention", attention.build(1, 128, 128, 128), [64]),
        (
            "attention_paged_decode_0_8b",
            attention.build_paged_decode(
                1,
                4,
                1,
                16,
                256,
                1024,
                65,
                4096,
                cache_row_stride=512,
                query_row_stride=2048,
                result_row_stride=2048,
            ),
            [1, 64],
        ),
        (
            "attention_paged_decode_9b",
            attention.build_paged_decode(
                1,
                4,
                1,
                16,
                256,
                1024,
                65,
                4096,
                cache_row_stride=1024,
                query_row_stride=4096,
                result_row_stride=4096,
            ),
            [1, 64],
        ),
        (
            "attention_paged_decode_batch2_strided_0_8b",
            attention.build_paged_decode(
                2,
                4,
                1,
                16,
                256,
                1024,
                65,
                4096,
                cache_row_stride=2048,
                query_row_stride=2048,
                result_row_stride=2048,
            ),
            [1, 1, 64],
        ),
        (
            "attention_paged_cache_write_0_8b",
            attention.build_paged_cache_write(1024, 1, 512),
            [128],
        ),
        (
            "attention_paged_cache_write_9b",
            attention.build_paged_cache_write(1024, 1, 1024),
            [128],
        ),
        (
            "attention_paged_cache_write_prefill_strided_0_8b",
            attention.build_paged_cache_write(1024, 13, 512, 2048),
            [1, 128],
        ),
        (
            "attention_paged_prefill_strided_0_8b",
            attention.build_paged_prefill(13, 8, 2, 16, 256, 1024, 65, 4096, 2048),
            [1, 1, 1, 128],
        ),
        (
            "attention_paged_prefill_9b",
            attention.build_paged_prefill(13, 16, 4, 16, 256, 1024, 65, 4096),
            [1, 1, 1, 128],
        ),
        ("linear", linear.build(1, 1024, 1024), [128]),
        (
            "linear_to_float",
            linear.build(19, 4096, 248320, "float32"),
            [1, 128],
        ),
        (
            "gdn_projection_split",
            gdn_projection.build(13, 8, 16, 128, 128),
            [1, 16],
        ),
        (
            "gdn_recurrent_decode",
            gdn.build_recurrent(2, 1, 8, 16, 128, 128, 65),
            [1, 1, 1, 64],
        ),
        (
            "gdn_recurrent_prefill_token_primitive",
            gdn.build_recurrent(1, 1, 8, 16, 128, 128, 65),
            [1, 1, 64],
        ),
    )
    decode_launch_counts = {
        "attention_paged_decode_0_8b": attention._paged_decode_partition_count(2),
        "attention_paged_decode_9b": attention._paged_decode_partition_count(4),
        "attention_paged_decode_batch2_strided_0_8b": (
            attention._paged_decode_partition_count(2)
        ),
    }
    results = []
    for name, program, tiles in cases:
        result = classify(program, tiles)
        record = {"case": name, "tiles": tiles, **result}
        if name in decode_launch_counts:
            launches = decode_launch_counts[name]
            record.update(
                {
                    "compiled_artifacts": 1,
                    "launch_count": launches,
                    "attention_launches": launches,
                    "attention_launch_topology": (
                        "one_reused_single_kv_head_artifact_launch_per_kv_head"
                    ),
                }
            )
        results.append(record)
    all_compiled = all(item["status"] == "compiled" for item in results)
    dso = loaded_dso_path()
    result = {
        "schema": 2,
        "kind": "pypto-kernels-classify-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": bootstrap()["compiler"]
        .get_nvidia_backend_build_info()
        .pypto_revision,
        "all_compiled": all_compiled,
        "cases": results,
    }
    rendered = json.dumps(result, sort_keys=True, indent=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_compiled else 75


if __name__ == "__main__":
    raise SystemExit(main())
