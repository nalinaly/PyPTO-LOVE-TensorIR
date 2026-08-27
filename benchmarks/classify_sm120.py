#!/usr/bin/env python3
"""Compile classification for every Qwen3.5 operator graph on SM120."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import sys

import torch

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src",
)

from pypto_kernels._boot import DSO_PATH, bootstrap, classify  # noqa: E402
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


def main() -> int:
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
        ("embedding", embedding.build(32, 248320, 1024), [8, 128]),
        (
            "qk_rmsnorm_rope",
            qk_rmsnorm_rope.build(2, 8, 2, 256, 64, 262144),
            [1, 1, 1, 1, 1, 32],
        ),
        ("rmsnorm", rmsnorm.build(256, 1024), [1, 128]),
        (
            "fused_add_rmsnorm",
            fused_add_rmsnorm.build(256, 1024),
            [1, 128],
        ),
        ("gated_rmsnorm", gated_rmsnorm.build(256, 128), [1, 128]),
        ("causal_conv1d", causal_conv1d.build(2048, 64), [128, 1]),
        ("rope", rope.build(256, 64), [1, 1, 64]),
        ("attention", attention.build(32, 128, 128, 128), [1, 64]),
        (
            "attention_paged_decode_0_8b",
            attention.build_paged_decode(1, 8, 2, 16, 256, 1024, 65, 4096),
            [1, 1, 64],
        ),
        (
            "attention_paged_decode_9b",
            attention.build_paged_decode(1, 16, 4, 16, 256, 1024, 65, 4096),
            [1, 1, 64],
        ),
        (
            "attention_paged_decode_batch2_0_8b",
            attention.build_paged_decode(2, 8, 2, 16, 256, 1024, 65, 4096),
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
            "attention_paged_cache_write_prefill_0_8b",
            attention.build_paged_cache_write(1024, 13, 512),
            [1, 128],
        ),
        (
            "attention_paged_prefill_0_8b",
            attention.build_paged_prefill(13, 8, 2, 16, 256, 1024, 65, 4096),
            [1, 1, 128],
        ),
        (
            "attention_paged_prefill_9b",
            attention.build_paged_prefill(13, 16, 4, 16, 256, 1024, 65, 4096),
            [1, 1, 128],
        ),
        ("linear", linear.build(32, 1024, 1024), [1, 128]),
        ("gdn_read", gdn.build_read(16, 128, 128), [1, 64]),
        (
            "gdn_state_update",
            gdn.build_state_update(16, 128, 128),
            [1, 32, 32],
        ),
    )
    results = []
    for name, program, tiles in cases:
        result = classify(program, tiles)
        results.append({"case": name, "tiles": tiles, **result})
    all_compiled = all(item["status"] == "compiled" for item in results)
    dso = pathlib.Path(DSO_PATH)
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
    pathlib.Path(__file__).with_name("classify_results.json").write_text(
        rendered + "\n", encoding="utf-8"
    )
    print(rendered)
    return 0 if all_compiled else 75


if __name__ == "__main__":
    raise SystemExit(main())
