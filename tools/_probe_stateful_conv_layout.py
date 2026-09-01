#!/usr/bin/env python3
"""One-case diagnostic for the typed stateful causal-conv output layout."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/pypto-kernels/src"))

import torch

from pypto_kernels import causal_conv1d


def main() -> None:
    torch.manual_seed(20260830)
    batch = 1
    tokens = 1
    channels = 2048
    slots = 8
    stride = channels * 3 + 128
    x = torch.randn(batch * tokens, channels, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.empty_strided(
        (slots, 3, channels),
        (stride, channels, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state.normal_().mul_(0.2)
    indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    initial = state.clone()
    actual = causal_conv1d.causal_conv1d(
        x,
        weight,
        state,
        indices,
        batch_size=batch,
        tokens_per_request=tokens,
    )
    torch.cuda.synchronize()
    history = initial[2].clone()
    current = x.view(batch, tokens, channels)[0, 0]
    linear = sum(history[index].float() * weight[:, index].float() for index in range(3))
    linear = linear + current.float() * weight[:, 3].float()
    reference = torch.nn.functional.silu(linear).to(torch.bfloat16)
    expected_state = initial.clone()
    expected_state[2] = torch.stack((initial[2, 1], initial[2, 2], current))
    candidates = {
        "silu_history_current": reference,
        "anchor_sum": expected_state[2].sum(dim=0).to(torch.bfloat16),
        "raw_linear": linear.to(torch.bfloat16),
        "silu_current_only": torch.nn.functional.silu(current.float()).to(torch.bfloat16),
    }
    print("actual", actual[0, :16].float().cpu().tolist())
    print("reference", reference[:16].float().cpu().tolist())
    print("diff", float((actual.float().flatten() - reference.float()).abs().max()))
    print("reverse", float((actual.float().flatten() - reference.flip(0).float()).abs().max()))
    print("state_diff", float((state - initial).abs().max()))
    print("state_error", float((state - expected_state).abs().max()))
    for name, candidate in candidates.items():
        print(name, float((actual.float().flatten() - candidate.float()).abs().max()))
    print("actual_shape", tuple(actual.shape), "stride", tuple(actual.stride()))


if __name__ == "__main__":
    main()
