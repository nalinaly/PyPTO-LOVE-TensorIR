#!/usr/bin/env python3
"""Capture and replay the Qwen GDN projection/Conv/GDN PyPTO chain."""

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
from pypto_kernels._boot import (  # noqa: E402
    DSO_PATH,
    acquire_cuda_graph_leases,
    bootstrap,
)


def reference(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_initial: torch.Tensor,
    gdn_initial: torch.Tensor,
    state_index: int,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads, value_heads, key_dim, value_dim = 8, 16, 128, 128
    mixed_width = 2 * q_heads * key_dim + value_heads * value_dim
    mixed = qkvz[:, :mixed_width]
    z = qkvz[:, mixed_width:].view(1, value_heads, value_dim)
    a = ba[:, :value_heads]
    b = ba[:, value_heads:]

    conv_state = conv_initial.clone()
    history = conv_state[state_index]
    current_input = mixed[0]
    linear = (
        sum(
            history[index].float() * conv_weight[:, index].float() for index in range(3)
        )
        + current_input.float() * conv_weight[:, 3].float()
    )
    convolved = torch.nn.functional.silu(linear).to(torch.bfloat16)
    conv_state[state_index] = torch.stack(
        (history[1], history[2], current_input), dim=0
    )

    packed = convolved.float()
    query = packed[: q_heads * key_dim].view(q_heads, key_dim)
    key = packed[q_heads * key_dim : 2 * q_heads * key_dim].view(q_heads, key_dim)
    value = packed[2 * q_heads * key_dim :].view(value_heads, value_dim)
    query = query / torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6)
    key = key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)
    groups = value_heads // q_heads
    query = query.repeat_interleave(groups, dim=0) / math.sqrt(key_dim)
    key = key.repeat_interleave(groups, dim=0)
    log_decay = -torch.exp(a_log) * torch.nn.functional.softplus(a[0].float() + dt_bias)
    gdn_state = gdn_initial.clone()
    current = gdn_state[state_index] * torch.exp(log_decay)[:, None, None]
    residual = value - torch.einsum("hvk,hk->hv", current, key)
    beta = torch.sigmoid(b[0].float()).to(torch.bfloat16).float()
    current = current + (residual * beta[:, None])[:, :, None] * key[:, None, :]
    gdn_state[state_index] = current
    output = torch.einsum("hk,hvk->hv", query, current)
    return z, output, conv_state, gdn_state


def main() -> int:
    torch.manual_seed(29)
    q_heads, value_heads, key_dim, value_dim = 8, 16, 128, 128
    mixed_width = 2 * q_heads * key_dim + value_heads * value_dim
    qkvz = (
        torch.randn(
            (1, mixed_width + value_heads * value_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )
    ba = torch.randn((1, 2 * value_heads), device="cuda", dtype=torch.bfloat16)
    conv_weight = (
        torch.randn((mixed_width, 4), device="cuda", dtype=torch.bfloat16) * 0.2
    )
    slots = 8
    conv_state = (
        torch.randn((slots, 3, mixed_width), device="cuda", dtype=torch.bfloat16) * 0.2
    )
    gdn_state = (
        torch.randn(
            (slots, value_heads, value_dim, key_dim),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    state_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    conv_initial = conv_state.clone()
    gdn_initial = gdn_state.clone()

    stream = torch.cuda.Stream()

    def chain():
        mixed, z, a, b = gdn_projection.split_projection(
            qkvz,
            ba,
            q_heads=q_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            stream=stream,
        )
        convolved = causal_conv1d.causal_conv1d(
            mixed,
            conv_weight,
            conv_state,
            state_indices,
            batch_size=1,
            tokens_per_request=1,
            stream=stream,
        )
        output = gdn.gdn_recurrent(
            convolved,
            a,
            b,
            a_log,
            dt_bias,
            gdn_state,
            state_indices,
            batch_size=1,
            tokens_per_request=1,
            stream=stream,
        )
        return z, output

    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        chain()
    stream.synchronize()
    conv_state.copy_(conv_initial)
    gdn_state.copy_(gdn_initial)
    torch.cuda.synchronize()

    leases = acquire_cuda_graph_leases()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_z, captured_output = chain()

    expected_z, expected_output, expected_conv, expected_gdn = reference(
        qkvz,
        ba,
        conv_weight,
        conv_initial,
        gdn_initial,
        int(state_indices[0]),
        a_log,
        dt_bias,
    )
    observed_outputs = []
    observed_conv = []
    observed_gdn = []
    for _ in range(5):
        conv_state.copy_(conv_initial)
        gdn_state.copy_(gdn_initial)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        observed_outputs.append(captured_output.clone())
        observed_conv.append(conv_state.clone())
        observed_gdn.append(gdn_state.clone())

    output_diff = max(
        float((value.float() - expected_output).abs().max())
        for value in observed_outputs
    )
    conv_diff = max(
        float((value.float() - expected_conv.float()).abs().max())
        for value in observed_conv
    )
    gdn_diff = max(
        float((value.float() - expected_gdn.float()).abs().max())
        for value in observed_gdn
    )
    output_drift = max(
        float((value.float() - observed_outputs[0].float()).abs().max())
        for value in observed_outputs
    )
    conv_drift = max(
        float((value.float() - observed_conv[0].float()).abs().max())
        for value in observed_conv
    )
    gdn_drift = max(
        float((value.float() - observed_gdn[0].float()).abs().max())
        for value in observed_gdn
    )
    z_diff = float((captured_z.float() - expected_z.float()).abs().max())
    dso = pathlib.Path(DSO_PATH)
    correct = bool(
        z_diff == 0.0
        and output_diff <= 0.08
        and conv_diff == 0.0
        and gdn_diff <= 0.002
        and output_drift == 0.0
        and conv_drift == 0.0
        and gdn_drift == 0.0
        and len(leases) == 3
    )
    result = {
        "schema": 1,
        "kind": "pypto-stateful-cuda-graph-sm120",
        "run_id": os.environ.get("PYPTO_RUN_ID"),
        "dso_sha256": hashlib.sha256(dso.read_bytes()).hexdigest(),
        "bootstrap": {
            name: str(module.__file__) for name, module in bootstrap().items()
        },
        "captured_launches": 3,
        "graph_leases": len(leases),
        "replays": len(observed_outputs),
        "z_max_abs_diff": z_diff,
        "output_max_abs_diff": output_diff,
        "conv_state_max_abs_diff": conv_diff,
        "gdn_state_max_abs_diff": gdn_diff,
        "output_drift": output_drift,
        "conv_state_drift": conv_drift,
        "gdn_state_drift": gdn_drift,
        "correct": correct,
    }
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    output = ROOT / "runs" / run_id / "cuda-graph-stateful-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
