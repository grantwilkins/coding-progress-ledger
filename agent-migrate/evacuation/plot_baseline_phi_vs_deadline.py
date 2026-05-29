"""Peak pressure phi vs deadline: pipeline-optimal vs the 5 baselines (move-all).

Single-story figure. In cap-soft mode the baselines insist on moving every job;
phi = max(load/cap), so phi > 1 means the deadline is missed. The pipeline
optimizer keeps phi <= 1 by construction. Reads outputs/baselines.json.

Usage:
    cd evacuation && uv run python plot_baseline_phi_vs_deadline.py
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
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label="deadline miss ($\\phi=1$)")
    ax.plot(D, data["optimizer"]["pipeline_phi"], color="black", lw=2.2,
            marker="o", ms=3, label="pipeline (optimal)", zorder=3)
    for name, m in data["baselines"].items():
        mean, std = np.array(m["phi_mean"]), np.array(m["phi_std"])
        ax.plot(D, mean, color=COLORS[name], lw=1.5, label=LABELS[name])
        if np.any(std > 1e-9):
            ax.fill_between(D, np.maximum(mean - std, 1e-9), mean + std,
                            color=COLORS[name], alpha=0.2)

    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Peak pressure $\\phi$ (max load / capacity)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Move-everything pressure: optimizer vs baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "baseline_phi_vs_D.pdf")
    fig.savefig(out / "baseline_phi_vs_D.png", dpi=150)
    print(f"wrote {out / 'baseline_phi_vs_D.pdf'}")


if __name__ == "__main__":
    main()
