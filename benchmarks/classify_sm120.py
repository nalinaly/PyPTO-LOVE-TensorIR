#!/usr/bin/env python3
"""Compile classification for every Qwen3.5 operator graph on SM120."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import sys

sys.path.insert(
    0,
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src",
)

from pypto_kernels._boot import DSO_PATH, bootstrap, classify  # noqa: E402
from pypto_kernels import (  # noqa: E402
    attention,
    rmsnorm,
    rope,
)
from pypto_kernels._graph import (  # noqa: E402
    gdn_delta_combine_graph,
    gdn_delta_graph,
    gdn_q_decay_graph,
    gdn_state_read_graph,
    gdn_state_update_graph,
    row_sum_graph,
)


def main() -> int:
    cases = (
        ("rmsnorm", rmsnorm.build(256, 1024), [1, 128]),
        ("rope", rope.build(256, 64), [1, 1, 64]),
        (
            "attention_softmax_scale",
            attention.build_softmax_scale(256, 1024),
            [32, 32],
        ),
        ("gdn_delta", gdn_delta_graph(16, 128), [16, 32]),
        (
            "attention_softmax_normalize",
            attention.build_softmax_normalize(256, 128),
            [32, 32],
        ),
        (
            "attention_value_mix",
            attention.build_value_mix(256, 128, 128),
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
