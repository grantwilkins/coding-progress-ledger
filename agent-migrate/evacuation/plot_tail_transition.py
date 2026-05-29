"""Tail-weight phase transition: H* cliff + which model breaks first.

X-axis is the token-length spread sigma. Left axis is the worst-class cost H*,
which steps from seconds to the d_miss floor the instant any job goes unmoved.
Right axis stacks per-model unmoved counts, ordered by breaking point. Unlike
the infrastructure sweeps (smooth curves), heavy-tailed workload produces a
step function: the dominant evacuation risk is the workload distribution, not
the infrastructure. Reads `outputs/sweeps.json` only.

Usage:
    cd evacuation && uv run python plot_tail_transition.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
# stack order = order in which models break (earliest first)
BREAK_ORDER = ["Qwen3 235B", "Kimi K2.6", "GLM 5"]


def main() -> None:
    d = json.loads((OUT / "sweeps.json").read_text())
    runs = d["runs"]["sigma_scale"]
    sig = d["diagnostics"]["sigma_scale"]
    x = [r["sweep_value"] for r in runs]
    H = sig["H_star_curve"]
    z = sig["model_z_curve"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax2 = ax.twinx()

    # right axis: stacked unmoved counts
    base = np.zeros(len(x))
    palette = ["#c44536", "#e8943a", "#4a9b54"]
    for name, c in zip(BREAK_ORDER, palette):
        vals = np.array(z[name])
        ax2.fill_between(x, base, base + vals, color=c, alpha=0.55, label=name, zorder=1)
        base = base + vals
    ax2.set_ylabel("unmoved jobs (stacked by model)")
    ax2.set_ylim(0, base.max() * 1.15)

    # left axis: H* cliff
    ax.plot(x, H, color="black", marker="o", ms=5, lw=2.2, zorder=3, label="$H^\\star$")
    ax.set_ylabel("$H^\\star$ worst-class cost (s)")
    ax.set_ylim(0, max(H) * 1.08)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # mark the transition at the steepest H* rise (the d_miss-floor cliff)
    j = int(np.argmax(np.diff(H)))
    cliff = 0.5 * (x[j] + x[j + 1])
    ax.axvline(cliff, color="0.5", ls="--", lw=1.2)
    ax.annotate("first miss\n$\\to H^\\star$ cliff", (cliff, max(H) * 0.5),
                fontsize=8, color="0.3", textcoords="offset points", xytext=(6, 0))

    ax.set_xlabel("token-length spread $\\sigma$ scale")
    ax.set_title("Tail-weight phase transition: workload distribution is the dominant risk")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "tail_transition.pdf")
    fig.savefig(OUT / "tail_transition.png", dpi=150)
    print(f"wrote {OUT / 'tail_transition.pdf'}")


if __name__ == "__main__":
    main()
