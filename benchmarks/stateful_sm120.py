#!/usr/bin/env python3
"""Focused numerical gate for Qwen3.5 projection and recurrent state kernels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/zhaosiying/pypto-love-tensor-ir")
KERNEL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT / "src"))
os.environ.setdefault(
    "PYPTO_KERNEL_DSO_PATH",
    str(
        ROOT / "builds/pypto-paged-f34c3f5/product/"
        "pypto_core.cpython-314-x86_64-linux-gnu.so"
    ),
)
os.environ.setdefault(
    "PYPTO_KERNEL_PACKAGE_PATH",
    str(ROOT / "worktrees/pypto-paged-decode/python/pypto"),
)

import torch  # noqa: E402

from pypto_kernels import causal_conv1d, gdn, gdn_projection  # noqa: E402
from pypto_kernels._boot import DSO_PATH, bootstrap  # noqa: E402


def projection_case(stream: torch.cuda.Stream) -> dict[str, object]:
    rows, q_heads, value_heads, dk, dv = 13, 8, 16, 128, 128
    mixed_width = 2 * q_heads * dk + value_heads * dv
    z_width = value_heads * dv
    qkvz = torch.randn(rows, mixed_width + z_width, device="cuda", dtype=torch.bfloat16)
    ba = torch.randn(rows, 2 * value_heads, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    actual = gdn_projection.split_projection(
        qkvz,
        ba,
        q_heads=q_heads,
        value_heads=value_heads,
        key_dim=dk,
        value_dim=dv,
        stream=stream,
    )
    stream.synchronize()
    expected = (
        qkvz[:, :mixed_width],
        qkvz[:, mixed_width:].view(rows, value_heads, dv),
        ba[:, :value_heads],
        ba[:, value_heads:],
    )
    diffs = [
        float((observed.float() - reference.float()).abs().max())
        for observed, reference in zip(actual, expected)
    ]
    return {
        "case": "gdn_projection_packed_t13_h8_hv16",
        "launches": 1,
        "outputs_contiguous": [value.is_contiguous() for value in actual],
        "shared_storage": len({value.untyped_storage().data_ptr() for value in actual})
        == 1,
        "max_abs_diff": max(diffs),
        "correct": all(
            torch.equal(observed, reference)
            for observed, reference in zip(actual, expected)
        ),
    }


def conv_case(
    stream: torch.cuda.Stream,
    *,
    batch_size: int,
    tokens: int,
    index_dtype: torch.dtype,
    repetitions: int = 1,
) -> dict[str, object]:
    channels = 4096
    rows = batch_size * tokens
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    slots = 8
    slot_stride = channels * 3 + 128
    state = torch.empty_strided(
        (slots, 3, channels),
        (slot_stride, channels, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state.normal_().mul_(0.2)
    indices = torch.arange(2, 2 + batch_size, device="cuda", dtype=index_dtype)
    initial_state = state.clone()
    reference_state = initial_state.clone()
    observed_outputs = []
    observed_states = []
    for repetition in range(repetitions):
        if repetition:
            state.copy_(initial_state)
        torch.cuda.synchronize()
        observed_outputs.append(
            causal_conv1d.causal_conv1d(
                x,
                weight,
                state,
                indices,
                batch_size=batch_size,
                tokens_per_request=tokens,
                stream=stream,
            ).clone()
        )
        stream.synchronize()
        observed_states.append(state.clone())
    actual = observed_outputs[-1]

    x_rows = x.view(batch_size, tokens, channels)
    reference_outputs = []
    for batch in range(batch_size):
        slot = int(indices[batch])
        history = reference_state[slot]
        token_outputs = []
        for token in range(tokens):
            current = x_rows[batch, token]
            linear = (
                sum(
                    history[index].float() * weight[:, index].float()
                    for index in range(3)
                )
                + current.float() * weight[:, 3].float()
            )
            token_outputs.append(torch.nn.functional.silu(linear))
            history = torch.stack((history[1], history[2], current), dim=0)
        reference_state[slot] = history
        reference_outputs.append(torch.stack(token_outputs))
    reference = torch.stack(reference_outputs).view(rows, channels)
    output_diff = max(
        float((observed.float() - reference).abs().max())
        for observed in observed_outputs
    )
    state_diff = max(
        float((observed.float() - reference_state.float()).abs().max())
        for observed in observed_states
    )
    output_drift = max(
        float((observed.float() - observed_outputs[0].float()).abs().max())
        for observed in observed_outputs
    )
    state_drift = max(
        float((observed.float() - observed_states[0].float()).abs().max())
        for observed in observed_states
    )
    actual_slot_changes = (
        (state.float() - initial_state.float()).abs().flatten(1).amax(dim=1)
    )
    expected_slot_changes = (
        (reference_state.float() - initial_state.float()).abs().flatten(1).amax(dim=1)
    )
    selected_actual = state.index_select(0, indices.long()).float()
    selected_expected = reference_state.index_select(0, indices.long()).float()
    channel_interleaved = (
        selected_expected.transpose(1, 2).contiguous().view(batch_size, 3, channels)
    )
    return {
        "case": f"causal_conv_plane_b{batch_size}_t{tokens}_{index_dtype}",
        "launches": 1,
        "repetitions": repetitions,
        "output_drift": output_drift,
        "state_drift": state_drift,
        "state_slot_stride": slot_stride,
        "output_max_abs_diff": output_diff,
        "state_max_abs_diff": state_diff,
        "actual_changed_slots": torch.nonzero(actual_slot_changes > 0, as_tuple=False)
        .flatten()
        .tolist(),
        "expected_changed_slots": torch.nonzero(
            expected_slot_changes > 0, as_tuple=False
        )
        .flatten()
        .tolist(),
        "actual_change_max": float(actual_slot_changes.max()),
        "expected_change_max": float(expected_slot_changes.max()),
        "selected_plane_diff": float((selected_actual - selected_expected).abs().max()),
        "selected_channel_interleaved_diff": float(
            (selected_actual - channel_interleaved).abs().max()
        ),
        "actual_plane_change_max": (
            selected_actual - initial_state.index_select(0, indices.long()).float()
        )
        .abs()
        .flatten(2)
        .amax(dim=2)
        .amax(dim=0)
        .tolist(),
        "selected_plane_max_diff": (selected_actual - selected_expected)
        .abs()
        .flatten(2)
        .amax(dim=2)
        .amax(dim=0)
        .tolist(),
        "max_abs_diff": max(output_diff, state_diff),
        "correct": bool(
            torch.allclose(actual.float(), reference, rtol=5e-2, atol=5e-2)
            and torch.equal(state, reference_state)
            and output_drift == 0.0
            and state_drift == 0.0
        ),
    }


def gdn_case(
    stream: torch.cuda.Stream,
    *,
    batch_size: int,
    tokens: int,
    index_dtype: torch.dtype,
    controlled_outer: bool = False,
    controlled_decay: bool = False,
) -> dict[str, object]:
    q_heads, value_heads, dk, dv = 8, 16, 128, 128
    rows = batch_size * tokens
    mixed_width = 2 * q_heads * dk + value_heads * dv
    mixed_qkv = (
        torch.randn(rows, mixed_width, device="cuda", dtype=torch.bfloat16) * 0.2
    )
    a = torch.randn(rows, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    slots = 65
    state_width = value_heads * dv * dk
    slot_stride = state_width + 4096
    state = torch.empty_strided(
        (slots, value_heads, dv, dk),
        (slot_stride, dv * dk, dk, 1),
        device="cuda",
        dtype=torch.float32,
    )
    if controlled_outer:
        state.zero_()
        a.zero_()
        b.zero_()
        a_log.zero_()
        dt_bias.zero_()
    else:
        state.normal_().mul_(0.02)
        if controlled_decay:
            mixed_qkv.zero_()
    indices = torch.arange(3, 3 + batch_size, device="cuda", dtype=index_dtype)
    initial_state = state.clone()
    reference_state = initial_state.clone()
    alternate_state = initial_state.clone()
    torch.cuda.synchronize()
    actual = gdn.gdn_recurrent(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        indices,
        batch_size=batch_size,
        tokens_per_request=tokens,
        stream=stream,
    )
    stream.synchronize()

    mixed = mixed_qkv.view(batch_size, tokens, mixed_width)
    a_rows = a.view(batch_size, tokens, value_heads)
    b_rows = b.view(batch_size, tokens, value_heads)
    groups = value_heads // q_heads
    reference_outputs = []
    alternate_outputs = []
    for batch in range(batch_size):
        slot = int(indices[batch])
        current = reference_state[slot]
        alternate_current = alternate_state[slot]
        token_outputs = []
        alternate_token_outputs = []
        for token in range(tokens):
            packed = mixed[batch, token]
            query = packed[: q_heads * dk].view(q_heads, dk).float()
            key = packed[q_heads * dk : 2 * q_heads * dk].view(q_heads, dk).float()
            value = packed[2 * q_heads * dk :].view(value_heads, dv).float()
            query = query / torch.sqrt(
                torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6
            )
            key = key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)
            alternate_query = query.repeat(groups, 1) / math.sqrt(dk)
            alternate_key = key.repeat(groups, 1)
            query = query.repeat_interleave(groups, dim=0) / math.sqrt(dk)
            key = key.repeat_interleave(groups, dim=0)
            log_decay = -torch.exp(a_log) * torch.nn.functional.softplus(
                a_rows[batch, token].float() + dt_bias
            )
            current = current * torch.exp(log_decay)[:, None, None]
            residual = value - torch.einsum("hvk,hk->hv", current, key)
            beta = (
                torch.sigmoid(b_rows[batch, token].float()).to(torch.bfloat16).float()
            )
            current = current + (residual * beta[:, None])[:, :, None] * key[:, None, :]
            alternate_current = alternate_current * torch.exp(log_decay)[:, None, None]
            alternate_residual = value - torch.einsum(
                "hvk,hk->hv", alternate_current, alternate_key
            )
            alternate_current = (
                alternate_current
                + (alternate_residual * beta[:, None])[:, :, None]
                * alternate_key[:, None, :]
            )
            token_outputs.append(torch.einsum("hk,hvk->hv", query, current))
            alternate_token_outputs.append(
                torch.einsum("hk,hvk->hv", alternate_query, alternate_current)
            )
        reference_state[slot] = current
        alternate_state[slot] = alternate_current
        reference_outputs.append(torch.stack(token_outputs))
        alternate_outputs.append(torch.stack(alternate_token_outputs))
    reference = torch.stack(reference_outputs)
    output_diff = float((actual.float() - reference).abs().max())
    state_diff = float((state.float() - reference_state).abs().max())
    actual_slot_changes = (
        (state.float() - initial_state.float()).abs().flatten(1).amax(dim=1)
    )
    expected_slot_changes = (
        (reference_state.float() - initial_state.float()).abs().flatten(1).amax(dim=1)
    )
    selected_actual = state.index_select(0, indices.long()).float()
    selected_expected = reference_state.index_select(0, indices.long()).float()
    selected_alternate = alternate_state.index_select(0, indices.long()).float()
    matrix_transposed = selected_expected.transpose(-1, -2).contiguous()
    group_swapped = (
        selected_expected.view(batch_size, q_heads, groups, dv, dk)
        .transpose(1, 2)
        .contiguous()
        .view(batch_size, value_heads, dv, dk)
    )
    group_and_matrix_swapped = group_swapped.transpose(-1, -2).contiguous()
    value_half_swapped = torch.cat(
        (selected_expected[:, :, dv // 2 :], selected_expected[:, :, : dv // 2]),
        dim=2,
    )
    group_flipped = (
        selected_expected.view(batch_size, q_heads, groups, dv, dk)
        .flip(2)
        .contiguous()
        .view(batch_size, value_heads, dv, dk)
    )
    tile_reinterpreted = (
        selected_expected.view(batch_size, value_heads, 2, 64 * 128)
        .view(batch_size, value_heads, 2, 128, 64)
        .transpose(-1, -2)
        .contiguous()
        .view(batch_size, value_heads, dv, dk)
    )
    actual_element_changes = (
        selected_actual != initial_state.index_select(0, indices.long()).float()
    )
    expected_element_changes = (
        selected_expected != initial_state.index_select(0, indices.long()).float()
    )
    launches = 1 if tokens == 1 else batch_size * tokens
    selected_initial = initial_state.index_select(0, indices.long()).float()
    decay_numerator = (selected_actual * selected_initial).flatten(2).sum(dim=2)
    decay_denominator = (selected_initial * selected_initial).flatten(2).sum(dim=2)
    fitted_decay = (decay_numerator / decay_denominator).mean(dim=0)
    expected_decay = torch.exp(
        -torch.exp(a_log)
        * torch.nn.functional.softplus(a_rows[:, 0].float() + dt_bias).mean(dim=0)
    )
    return {
        "case": (
            f"gdn_outer_b{batch_size}_t{tokens}_{index_dtype}"
            if controlled_outer
            else (
                f"gdn_decay_b{batch_size}_t{tokens}_{index_dtype}"
                if controlled_decay
                else f"gdn_recurrent_b{batch_size}_t{tokens}_{index_dtype}"
            )
        ),
        "launches": launches,
        "state_slot_stride": slot_stride,
        "output_max_abs_diff": output_diff,
        "state_max_abs_diff": state_diff,
        "actual_changed_slots": torch.nonzero(actual_slot_changes > 0, as_tuple=False)
        .flatten()
        .tolist(),
        "expected_changed_slots": torch.nonzero(
            expected_slot_changes > 0, as_tuple=False
        )
        .flatten()
        .tolist(),
        "actual_change_max": float(actual_slot_changes.max()),
        "expected_change_max": float(expected_slot_changes.max()),
        "selected_direct_diff": float(
            (selected_actual - selected_expected).abs().max()
        ),
        "selected_repeat_grouping_diff": float(
            (selected_actual - selected_alternate).abs().max()
        ),
        "output_repeat_grouping_diff": float(
            (actual.float() - torch.stack(alternate_outputs)).abs().max()
        ),
        "selected_matrix_transpose_diff": float(
            (selected_actual - matrix_transposed).abs().max()
        ),
        "selected_group_swap_diff": float(
            (selected_actual - group_swapped).abs().max()
        ),
        "selected_group_matrix_swap_diff": float(
            (selected_actual - group_and_matrix_swapped).abs().max()
        ),
        "selected_value_half_swap_diff": float(
            (selected_actual - value_half_swapped).abs().max()
        ),
        "selected_group_flip_diff": float(
            (selected_actual - group_flipped).abs().max()
        ),
        "selected_tile_reinterpret_diff": float(
            (selected_actual - tile_reinterpreted).abs().max()
        ),
        "actual_head_change_max": (
            selected_actual - initial_state.index_select(0, indices.long()).float()
        )
        .abs()
        .flatten(2)
        .amax(dim=2)
        .amax(dim=0)
        .tolist(),
        "selected_head_max_diff": (selected_actual - selected_expected)
        .abs()
        .flatten(2)
        .amax(dim=2)
        .amax(dim=0)
        .tolist(),
        "actual_head_change_fraction": actual_element_changes.float()
        .flatten(2)
        .mean(dim=2)
        .mean(dim=0)
        .tolist(),
        "expected_head_change_fraction": expected_element_changes.float()
        .flatten(2)
        .mean(dim=2)
        .mean(dim=0)
        .tolist(),
        "fitted_decay": fitted_decay.tolist(),
        "expected_decay": expected_decay.tolist(),
        "max_abs_diff": max(output_diff, state_diff),
        "correct": bool(
            torch.allclose(actual.float(), reference, rtol=8e-2, atol=8e-2)
            and torch.allclose(state.float(), reference_state, rtol=2e-3, atol=2e-3)
        ),
    }


def main() -> int:
    torch.manual_seed(19)
    stream = torch.cuda.Stream()
    requested = set(sys.argv[1:])
    all_cases = {
        "projection": lambda: projection_case(stream),
        "conv_decode": lambda: conv_case(
            stream, batch_size=2, tokens=1, index_dtype=torch.int32
        ),
        "conv_prefill": lambda: conv_case(
            stream,
            batch_size=1,
            tokens=5,
            index_dtype=torch.int64,
            repetitions=10,
        ),
        "gdn_decode": lambda: gdn_case(
            stream, batch_size=2, tokens=1, index_dtype=torch.int32
        ),
        "gdn_outer": lambda: gdn_case(
            stream,
            batch_size=1,
            tokens=1,
            index_dtype=torch.int32,
            controlled_outer=True,
        ),
        "gdn_decay": lambda: gdn_case(
            stream,
            batch_size=1,
            tokens=1,
            index_dtype=torch.int32,
            controlled_decay=True,
        ),
        "gdn_prefill3": lambda: gdn_case(
            stream, batch_size=1, tokens=3, index_dtype=torch.int64
        ),
        "gdn_prefill13": lambda: gdn_case(
            stream, batch_size=1, tokens=13, index_dtype=torch.int32
        ),
    }
    unknown = requested - all_cases.keys()
    if unknown:
        raise ValueError(f"unknown stateful cases: {sorted(unknown)}")
    selected = requested or set(all_cases)
    cases = [factory() for name, factory in all_cases.items() if name in selected]
    dso = pathlib.Path(DSO_PATH)
    result = {
        "schema": 1,
        "kind": "pypto-stateful-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "dso_path": str(dso),
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "bootstrap": {
            name: str(module.__file__) for name, module in bootstrap().items()
        },
        "all_correct": all(bool(case["correct"]) for case in cases),
        "cases": cases,
    }
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    output = ROOT / "runs" / run_id / "stateful-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
