#!/usr/bin/env python3
"""Render release charts from frozen evidence JSONs (no test re-runs).

Reads only committed ``state/evidence/*-current.json`` files and writes PNG
charts to ``docs/assets/charts/`` for the README and blog. Data sources:

- three-lane end-to-end  : qwen35-9b-release-results-current.json (performance)
- operator A/B ratios    : qwen35-9b-operator-performance-breakdown-current.json
- SwiGLU fusion ablation : qwen35-9b-inductor-ablation-current.json
- CUPTI phase attribution: qwen35-9b-release-results-current.json (profile)

Labels are English because the plotting environment has no CJK font; captions
in the documents carry the Chinese explanation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "state/evidence"
OUTPUT = ROOT / "docs/assets/charts"

ACCENT = "#2457a7"
MUTED = "#8a94a0"
WARM = "#c05621"
INK = "#17202a"

LANE_COLORS = {
    "pypto": ACCENT,
    "sglang-matched": MUTED,
    "sglang-optimized": WARM,
}
LANE_LABELS = {
    "pypto": "PyPTO\n(100% PyPTO kernels)",
    "sglang-matched": "SGLang\nmatched",
    "sglang-optimized": "SGLang\noptimized",
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def style_axis(ax) -> None:
    ax.grid(axis="x", color="#d9e0e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


def chart_three_lane() -> None:
    release = load("qwen35-9b-release-results-current.json")
    lanes = release["performance"]["lanes"]
    names = ["pypto", "sglang-matched", "sglang-optimized"]
    throughput = [lanes[n]["output_tokens_per_second"]["p50"] for n in names]
    e2e = [lanes[n]["e2e_ms"]["p50"] / 1000.0 for n in names]
    tpot = [lanes[n]["tpot_ms"]["p50"] for n in names]
    ttft = [lanes[n]["ttft_ms"]["p50"] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4))
    metrics = (
        ("Output throughput (tok/s)", throughput, "{:.2f}"),
        ("E2E latency (s)", e2e, "{:.2f}"),
        ("TTFT (ms)", ttft, "{:.0f}"),
        ("TPOT (ms)", tpot, "{:.1f}"),
    )
    for ax, (title, values, fmt) in zip(axes, metrics):
        bars = ax.bar(
            [LANE_LABELS[n] for n in names],
            values,
            color=[LANE_COLORS[n] for n in names],
            width=0.62,
        )
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.grid(axis="y", color="#d9e0e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.6, colors=INK)
        ax.tick_params(axis="y", labelsize=8, colors=INK)
        top = max(values)
        ax.set_ylim(0, top * 1.22)
        for bar, value in zip(bars, values):
            ax.annotate(
                fmt.format(value),
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=INK,
            )
    fig.suptitle(
        "Qwen3.5-9B end-to-end, 31-in/64-out greedy (p50 of 4 fresh starts per lane)",
        fontsize=10.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.78, wspace=0.42)
    save(fig, "three-lane-end-to-end.png")


def chart_operator_ab() -> None:
    breakdown = load("qwen35-9b-operator-performance-breakdown-current.json")
    rows = []
    for name, entry in sorted(breakdown["comparisons"].items()):
        interval = entry["median_ratio_bootstrap_95ci_percent"]
        rows.append(
            (
                name,
                entry["pypto_latency_percent_of_stock"] / 100.0,
                interval["lower"] / 100.0,
                interval["upper"] / 100.0,
            )
        )
    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    lows = [r[1] - r[2] for r in rows]
    highs = [r[3] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    position = range(len(rows))
    ax.barh(
        position,
        ratios,
        xerr=[lows, highs],
        color=ACCENT,
        height=0.62,
        error_kw={"ecolor": WARM, "elinewidth": 1.4, "capsize": 3},
    )
    ax.set_yticks(position, labels, fontsize=8.6, color=INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("stock = 1x", (1.0, len(rows) - 0.4), fontsize=8.6, color=MUTED,
                ha="left", xytext=(6, 0), textcoords="offset points")
    for i, (label, ratio) in enumerate(zip(labels, ratios)):
        ax.annotate(f"{ratio:.1f}x", (ratio, i), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.6, color=INK)
    ax.set_xlabel("PyPTO / stock latency per call (log scale, bar = 95% CI)",
                  fontsize=9.5, color=INK)
    ax.set_title(
        "Operator-level A/B: 7 aligned operators, p50 of 4+4 fresh starts",
        fontsize=10.5,
        color=INK,
    )
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "operator-ab-breakdown.png")


def chart_inductor_ablation() -> None:
    ablation = load("qwen35-9b-inductor-ablation-current.json")
    phases = ("prefill", "decode")
    modes = (
        ("eager", "Eager (6 kernels)", MUTED),
        ("inductor_nv", "Official Inductor CUDA (Triton, 1 kernel)", WARM),
        ("pypto", "PyPTO backend (1 kernel)", ACCENT),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    derived = ablation["phases"]
    for ax, phase in zip(axes, phases):
        values = [
            derived[phase][key]["warm_call_ms"] for key, _label, _color in modes
        ]
        labels = [label for _key, label, _color in modes]
        colors = [color for _key, _label, color in modes]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_yscale("log")
        ax.set_title(f"{phase} shape", fontsize=10.5, color=INK)
        ax.set_ylabel("warm ms/call (log)", fontsize=9.5, color=INK)
        ax.grid(axis="y", color="#d9e0e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.4, colors=INK)
        ax.tick_params(axis="y", labelsize=8, colors=INK)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=INK,
            )
        eager = values[0]
        for bar, value, (key, _label, _color) in zip(bars, values, modes):
            change = (eager / value - 1) * 100
            marker = "+" if change >= 0 else "\u2212"
            ax.annotate(
                f"{marker}{abs(change):.1f}% vs eager",
                (bar.get_x() + bar.get_width() / 2, value * 1.55),
                ha="center",
                fontsize=8.2,
                color=(("#2f855a" if change >= 0 else "#c53030")),
            )
    fig.suptitle(
        "SwiGLU fusion ablation (real Qwen3.5-9B shapes): "
        "both compiled backends fuse 6 kernels into 1",
        fontsize=10.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.76, wspace=0.34)
    save(fig, "inductor-swiglu-ablation.png")


def chart_cupti_attribution() -> None:
    release = load("qwen35-9b-release-results-current.json")
    summaries = release["profile_reconciliation"]["lane_summaries"]
    names = ["pypto", "sglang-matched", "sglang-optimized"]
    phase_order = [
        "attention_core_gate",
        "lm_head",
        "gdn_recurrent_norm",
        "mlp_gate_up",
        "mlp_down",
        "mlp_swiglu",
        "gdn_projection",
        "gdn_conv",
        "attention_projection",
        "embedding_gather",
        "residual_norm",
        "final_norm",
        "kv_cache_write",
    ]
    pretty = {
        "attention_core_gate": "attention core+gate",
        "lm_head": "LM head",
        "gdn_recurrent_norm": "GDN recurrent+norm",
        "mlp_gate_up": "MLP gate/up GEMM",
        "mlp_down": "MLP down GEMM",
        "mlp_swiglu": "MLP SwiGLU",
        "gdn_projection": "GDN projection",
        "gdn_conv": "GDN conv",
        "attention_projection": "attention projection",
        "embedding_gather": "embedding gather",
        "residual_norm": "residual norm",
        "final_norm": "final norm",
        "kv_cache_write": "KV cache write",
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(10.6, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    phases = [
        p
        for p in phase_order
        if any(p in summaries[n]["phase_totals"] for n in names)
    ]
    for ax, lane in zip(axes, names):
        totals = summaries[lane]["phase_totals"]
        values = [
            (
                totals[p]["gpu_time_ns_per_request"]["p50"] / 1e6
                if p in totals
                else 0.0
            )
            for p in phases
        ]
        unattributed = (
            totals["unattributed_compute"]["gpu_time_ns_per_request"]["p50"] / 1e6
        )
        compute = (
            summaries[lane]["compute_gpu_time_ns_per_request"]["p50"] / 1e6
        )
        positions = list(range(len(phases)))
        ax.bar(
            positions,
            values,
            color=[LANE_COLORS[lane]] * len(phases),
            width=0.62,
        )
        ax.bar(
            [len(phases)],
            [unattributed],
            color="#d7dee7",
            edgecolor=LANE_COLORS[lane],
            width=0.62,
        )
        ax.set_yscale("log")
        ax.set_title(
            f"{LANE_LABELS[lane]}  |  forward compute total {compute:,.1f} ms/request",
            fontsize=9.6,
            color=INK,
            loc="left",
        )
        ax.grid(axis="y", color="#d9e0e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors=INK)
        for i, value in enumerate(values + [unattributed]):
            if value > 0:
                ax.annotate(
                    f"{value:,.1f}",
                    (i, value),
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    color=INK,
                )
    axes[-1].set_xticks(
        list(range(len(phases) + 1)),
        [pretty[p] for p in phases] + ["unattributed"],
        rotation=38,
        ha="right",
        fontsize=8,
    )
    axes[-1].tick_params(axis="x", colors=INK)
    fig.suptitle(
        "CUPTI logical-phase attribution (p50 GPU ms/request, log scale)",
        fontsize=10.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.93, hspace=0.55)
    save(fig, "cupti-phase-attribution.png")


def main() -> int:
    chart_three_lane()
    chart_operator_ab()
    chart_inductor_ablation()
    chart_cupti_attribution()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
