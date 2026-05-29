"""Token-weighted evacuation vs deadline: throughput-optimal vs the 4 baselines.

Single-story figure. Token-weighted evac is order-robust (unlike raw job count):
a greedy that cherry-picks small jobs pumps its count but still moves few tokens,
so the optimizer's advantage shows clearly here. Reads outputs/baselines.json.

Usage:
    cd evacuation && uv run python baselines_compare.py
    cd evacuation && uv run python plot_baseline_token_vs_deadline.py
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
    ax.plot(D, data["optimizer"]["throughput_tokenwtd"], color="black", lw=2.2,
            marker="o", ms=3, label="throughput (optimal)", zorder=3)
    for name, m in data["baselines"].items():
        mean, std = np.array(m["token_wtd_mean"]), np.array(m["token_wtd_std"])
        ax.plot(D, mean, color=COLORS[name], lw=1.5, label=LABELS[name])
        if np.any(std > 1e-9):
            ax.fill_between(D, mean - std, mean + std, color=COLORS[name], alpha=0.2)

    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Token-weighted evacuation")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_title("Token-weighted evacuation: optimizer vs baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "baseline_token_vs_D.pdf")
    fig.savefig(out / "baseline_token_vs_D.png", dpi=150)
    print(f"wrote {out / 'baseline_token_vs_D.pdf'}")


if __name__ == "__main__":
    main()
