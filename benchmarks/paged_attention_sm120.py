#!/usr/bin/env python3
"""Focused SM120 numerical gate for Qwen3.5 paged attention phases."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "PYPTO_KERNEL_DSO_PATH",
    str(
        ROOT
        / "builds/pypto-paged-f34c3f5/product/"
        "pypto_core.cpython-314-x86_64-linux-gnu.so"
    ),
)
os.environ.setdefault(
    "PYPTO_KERNEL_PACKAGE_PATH",
    str(ROOT / "worktrees/pypto-paged-decode/python/pypto"),
)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pypto_kernels import attention  # noqa: E402
from pypto_kernels._boot import DSO_PATH, bootstrap  # noqa: E402

RTOL = 1.0e-1
ATOL = 8.0e-2


def main() -> int:
    torch.manual_seed(11)
    stream = torch.cuda.Stream()
    cases: list[dict[str, object]] = []

    def run_decode(
        *,
        q_heads: int,
        kv_heads: int,
        valid_counts: tuple[int, ...],
        label: str,
        cache_row_stride: int | None = None,
    ) -> None:
        print(f"START {label}", flush=True)
        bucket_tokens, head_dim = 16, 256
        cache_rows, request_rows, max_context_len = 1024, 65, 4096
        batch_size = len(valid_counts)
        row_width = kv_heads * head_dim
        cache_row_stride = cache_row_stride or row_width
        query = (
            torch.randn(
                batch_size,
                q_heads * head_dim,
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
            torch.randperm(cache_rows - 1, device="cuda")[: sum(valid_counts)]
            + 1
        )
        virtual_pool = torch.arange(
            1, 1 + sum(valid_counts), device="cuda", dtype=torch.int64
        )
        virtual_to_physical = torch.arange(
            cache_rows, device="cuda", dtype=torch.int64
        )
        virtual_to_physical[virtual_pool] = physical_pool
        physical_by_batch: list[torch.Tensor] = []
        virtual_by_batch: list[torch.Tensor] = []
        offset = 0
        for batch_row, valid_count in enumerate(valid_counts):
            physical = physical_pool[offset : offset + valid_count].contiguous()
            virtual = virtual_pool[offset : offset + valid_count].contiguous()
            offset += valid_count
            physical_by_batch.append(physical)
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
            [virtual[-1] for virtual in virtual_by_batch]
        ).contiguous()
        physical_row = virtual_to_physical[virtual_row]
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
        anchor_reference = current_key.float() + current_value.float()
        key_diff = float((written_key.float() - current_key.float()).abs().max())
        value_diff = float(
            (written_value.float() - current_value.float()).abs().max()
        )
        anchor_diff = float(
            (write_anchor.float() - anchor_reference).abs().max()
        )
        write_correct = bool(
            torch.equal(written_key, current_key)
            and torch.equal(written_value, current_value)
            and torch.allclose(
                write_anchor.float(), anchor_reference, rtol=5.0e-2, atol=5.0e-2
            )
        )
        cases.append(
            {
                "case": f"{label} cache_write",
                "launches": 1,
                "key_max_abs_diff": key_diff,
                "value_max_abs_diff": value_diff,
                "anchor_max_abs_diff": anchor_diff,
                "correct": write_correct,
            }
        )

        output = attention.paged_attention_decode(
            query,
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
        key_heads = key_cache.view(cache_rows, kv_heads, head_dim)
        value_heads = value_cache.view(cache_rows, kv_heads, head_dim)
        query_heads = query.view(batch_size, q_heads, head_dim)
        queries_per_kv = q_heads // kv_heads
        reference_batches = []
        for batch_row, physical in enumerate(physical_by_batch):
            reference_heads = []
            for q_head in range(q_heads):
                kv_head = q_head // queries_per_kv
                selected_keys = key_heads[physical, kv_head].float()
                selected_values = value_heads[physical, kv_head].float()
                scores = (
                    query_heads[batch_row, q_head].float() @ selected_keys.T
                ) / math.sqrt(head_dim)
                reference_heads.append(
                    torch.softmax(scores, dim=-1) @ selected_values
                )
            reference_batches.append(torch.stack(reference_heads))
        reference = torch.stack(reference_batches).view(batch_size, -1)
        max_abs_diff = float((output.float() - reference).abs().max())
        correct = bool(
            torch.allclose(output.float(), reference, rtol=RTOL, atol=ATOL)
        )
        cases.append(
            {
                "case": label,
                "launches": 1,
                "batch_size": batch_size,
                "bucket_tokens": bucket_tokens,
                "cache_row_stride": cache_row_stride,
                "valid_tokens": list(valid_counts),
                "max_abs_diff": max_abs_diff,
                "correct": correct,
            }
        )
        print(f"DONE {label} correct={correct} max_abs={max_abs_diff}", flush=True)

    def run_prefill(*, q_heads: int, kv_heads: int, label: str) -> None:
        print(f"START {label}", flush=True)
        query_rows, prefix_count, bucket_tokens, head_dim = 13, 2, 16, 256
        cache_rows, request_rows, max_context_len, request_row = 1024, 65, 4096, 9
        row_width = kv_heads * head_dim
        query = (
            torch.randn(
                query_rows,
                q_heads * head_dim,
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
        current_virtual_rows = virtual[prefix_count:valid_total].contiguous()
        request_index = torch.tensor(
            [request_row], device="cuda", dtype=torch.int64
        )
        prefix_tokens = torch.tensor(
            [prefix_count], device="cuda", dtype=torch.int32
        )
        write_anchor = attention.paged_cache_write(
            key_cache,
            value_cache,
            current_virtual_rows,
            virtual_to_physical,
            current_key,
            current_value,
            stream=stream,
        )
        stream.synchronize()
        current_physical_rows = selected[prefix_count:valid_total]
        written_key = key_cache[current_physical_rows]
        written_value = value_cache[current_physical_rows]
        anchor_reference = current_key.float() + current_value.float()
        write_correct = bool(
            torch.equal(written_key, current_key)
            and torch.equal(written_value, current_value)
            and torch.allclose(
                write_anchor.float(), anchor_reference, rtol=5.0e-2, atol=5.0e-2
            )
        )
        cases.append(
            {
                "case": f"{label} cache_write",
                "launches": 1,
                "key_max_abs_diff": float(
                    (written_key.float() - current_key.float()).abs().max()
                ),
                "value_max_abs_diff": float(
                    (written_value.float() - current_value.float()).abs().max()
                ),
                "anchor_max_abs_diff": float(
                    (write_anchor.float() - anchor_reference).abs().max()
                ),
                "correct": write_correct,
            }
        )
        output = attention.paged_attention_prefill(
            query,
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
        key_heads = key_cache.view(cache_rows, kv_heads, head_dim)
        value_heads = value_cache.view(cache_rows, kv_heads, head_dim)
        query_heads = query.view(query_rows, q_heads, head_dim)
        queries_per_kv = q_heads // kv_heads
        reference_rows = []
        for query_row in range(query_rows):
            physical = selected[: prefix_count + query_row + 1]
            reference_heads = []
            for q_head in range(q_heads):
                kv_head = q_head // queries_per_kv
                keys = key_heads[physical, kv_head].float()
                values = value_heads[physical, kv_head].float()
                scores = (query_heads[query_row, q_head].float() @ keys.T) / math.sqrt(
                    head_dim
                )
                reference_heads.append(torch.softmax(scores, dim=-1) @ values)
            reference_rows.append(torch.stack(reference_heads))
        reference = torch.stack(reference_rows).reshape_as(output)
        max_abs_diff = float((output.float() - reference).abs().max())
        correct = bool(
            torch.allclose(output.float(), reference, rtol=RTOL, atol=ATOL)
        )
        cases.append(
            {
                "case": label,
                "launches": 2,
                "cache_write_launches": 1,
                "attention_launches": 1,
                "prefix_tokens": prefix_count,
                "query_rows": query_rows,
                "bucket_tokens": bucket_tokens,
                "max_abs_diff": max_abs_diff,
                "correct": correct,
            }
        )
        print(f"DONE {label} correct={correct} max_abs={max_abs_diff}", flush=True)

    run_decode(
        q_heads=8,
        kv_heads=2,
        valid_counts=(13,),
        label="decode_0_8b_valid13",
    )
    run_decode(
        q_heads=16,
        kv_heads=4,
        valid_counts=(16,),
        label="decode_9b_valid16",
    )
    run_decode(
        q_heads=8,
        kv_heads=2,
        valid_counts=(13, 7),
        label="decode_batch2_0_8b_valid13_7_strided",
        cache_row_stride=2048,
    )
    run_prefill(q_heads=8, kv_heads=2, label="prefill_0_8b_prefix2_extend13")
    run_prefill(q_heads=16, kv_heads=4, label="prefill_9b_prefix2_extend13")

    all_correct = all(bool(case["correct"]) for case in cases)
    dso = pathlib.Path(DSO_PATH)
    result = {
        "schema_version": 1,
        "kind": "pypto-paged-attention-sm120",
        "run_id": os.environ["PYPTO_RUN_ID"],
        "thresholds": {"rtol": RTOL, "atol": ATOL},
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "pypto_commit": (
            bootstrap()["compiler"].get_nvidia_backend_build_info().pypto_revision
        ),
        "all_correct": all_correct,
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    run_dir = ROOT / "runs" / result["run_id"]
    (run_dir / "paged-attention-result.json").write_text(
        rendered, encoding="utf-8"
    )
    print(rendered, end="")
    return 0 if all_correct else 75


if __name__ == "__main__":
    raise SystemExit(main())
