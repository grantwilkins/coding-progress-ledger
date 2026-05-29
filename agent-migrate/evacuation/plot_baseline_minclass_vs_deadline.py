"""Worst-class evacuated fraction vs deadline: max-min-optimal vs the 5 baselines.

Single-story figure. The optimizer's max-min objective protects the worst class;
greedy/random heuristics starve it. Reads outputs/baselines.json.

Usage:
    cd evacuation && uv run python plot_baseline_minclass_vs_deadline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {"random": "random", "replay_only": "replay-only",
          "state_only": "state-only", "least_loaded": "least-loaded"}
COLORS = {"random": "#888888", "replay_only": "#c44536",
          "state_only": "#8052a0", "least_loaded": "#d98c00"}


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    data = json.loads((out / "baselines.json").read_text())
    D = data["D"]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(D, data["optimizer"]["maxmin_minclass"], color="black", lw=2.2,
            marker="o", ms=3, label="max-min (optimal)", zorder=3)
    for name, m in data["baselines"].items():
        mean, std = np.array(m["min_class_mean"]), np.array(m["min_class_std"])
        ax.plot(D, mean, color=COLORS[name], lw=1.5, label=LABELS[name])
        if np.any(std > 1e-9):
            ax.fill_between(D, mean - std, mean + std, color=COLORS[name], alpha=0.2)

    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Worst-class evacuated fraction")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_title("Worst-class protection: optimizer vs baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "baseline_minclass_vs_D.pdf")
    fig.savefig(out / "baseline_minclass_vs_D.png", dpi=150)
    print(f"wrote {out / 'baseline_minclass_vs_D.pdf'}")


if __name__ == "__main__":
    main()
