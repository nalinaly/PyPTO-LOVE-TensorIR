#!/usr/bin/env python3
"""Compile classification for every broadcast-dependent v2 graph on SM120."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels-v2/src",
)

from pypto_kernels_v2._boot import classify  # noqa: E402
from pypto_kernels_v2.ops import (  # noqa: E402
    attention_design,
    rmsnorm,
    rope,
)
from pypto_kernels_v2._graph import (  # noqa: E402
    gdn_delta_combine_graph,
    gdn_delta_graph,
    gdn_q_decay_graph,
    gdn_state_read_graph,
    gdn_state_update_graph,
    row_sum_graph,
)


def main() -> int:
    cases = (
        ("rmsnorm", rmsnorm.build(256, 1024), [32, 32]),
        ("rope_even", rope.build_even(256, 512), [32, 32]),
        ("rope_odd", rope.build_odd(256, 512), [32, 32]),
        (
            "attention_softmax_scale",
            attention_design.build_softmax_scale(256, 1024),
            [32, 32],
        ),
        ("gdn_delta", gdn_delta_graph(16, 128), [16, 32]),
        (
            "attention_softmax_normalize",
            attention_design.build_softmax_normalize(256, 128),
            [32, 32],
        ),
        (
            "attention_value_mix",
            attention_design.build_value_mix(256, 128, 128),
            [32, 32],
        ),
        ("gdn_q_decay", gdn_q_decay_graph(16, 128), [128]),
        ("gdn_dot", row_sum_graph(16, 128), [16]),
        ("gdn_state_read", gdn_state_read_graph(16, 128, 128), [16, 32]),
        (
            "gdn_delta_combine",
            gdn_delta_combine_graph(16, 128),
            [16, 32],
        ),
        (
            "gdn_state_update",
            gdn_state_update_graph(16, 128, 128),
            [16, 32, 32],
        ),
    )
    results = []
    for name, program, tiles in cases:
        result = classify(program, tiles)
        results.append({"case": name, "tiles": tiles, **result})
    all_compiled = all(item["status"] == "compiled" for item in results)
    print(
        json.dumps(
            {
                "schema": 1,
                "kind": "pypto-kernels-v2-classify-sm120",
                "run_id": os.environ.get("PYPTO_RUN_ID"),
                "all_compiled": all_compiled,
                "cases": results,
            },
            sort_keys=True,
            indent=1,
        )
    )
    return 0 if all_compiled else 75


if __name__ == "__main__":
    raise SystemExit(main())
