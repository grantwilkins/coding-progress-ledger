"""Sensitivity fan: evacuation fraction vs every swept parameter on one axis.

Each sweep's parameter is normalized to [0, 1] over its own range, so the
slopes are directly comparable. The reader sees the sensitivity ranking at a
glance: load (jobs) is the steepest lever, token-spread sigma bends down,
bandwidth Lambda ramps up, prefill rho and warm-pool W are nearly flat, and
seed is a tight scatter band. Reads `outputs/sweeps.json` only.

Usage:
    cd evacuation && uv run python plot_sensitivity_fan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"

LABELS = {"total_jobs": "jobs (load)", "sigma_scale": "$\\sigma$ (token spread)",
          "lambda_scale": "$\\Lambda$ (bandwidth)", "W_rebalance": "$W$ (warm pool)",
          "rho_scale": "$\\rho$ (prefill)", "seed": "seed"}
# (anchor side, dy in points) for the end-of-line text label
ANCHOR = {"total_jobs": ("r", 0), "sigma_scale": ("r", -6), "W_rebalance": ("r", 7),
          "lambda_scale": ("l", -4), "rho_scale": ("l", 4)}


def _norm(v):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v * 0.0


def main() -> None:
    R = json.loads((OUT / "sweeps.json").read_text())["runs"]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for k in LABELS:
        rs = R[k]
        x = _norm([r["sweep_value"] for r in rs])
        y = np.array([100 * r["evac_fraction"] for r in rs])
        if k == "seed":
            ax.scatter(x, y, s=20, color="0.55", zorder=2)
            ax.annotate("seed (scatter)", (0.5, y.mean()), color="0.4", fontsize=9,
                        textcoords="offset points", xytext=(0, -12), ha="center")
            continue
        line, = ax.plot(x, y, marker="o", ms=4, lw=1.9, zorder=3)
        side, dy = ANCHOR[k]
        i, dx, ha = (-1, 6, "left") if side == "r" else (0, -6, "right")
        ax.annotate(LABELS[k], (x[i], y[i]), color=line.get_color(), fontsize=9,
                    textcoords="offset points", xytext=(dx, dy), va="center", ha=ha)

    ax.set_xlim(-0.18, 1.18)
    ax.set_xlabel("swept parameter, normalized to its own range  $[0, 1]$")
    ax.set_ylabel("evacuated (%)")
    ax.set_title("Sensitivity fan: evacuation fraction vs each swept parameter")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_fan.pdf")
    fig.savefig(OUT / "sensitivity_fan.png", dpi=150)
    print(f"wrote {OUT / 'sensitivity_fan.pdf'}")


if __name__ == "__main__":
    main()
