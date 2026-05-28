"""Per-model action mix: Stage 1 vs Stage 2 (replay/state-transfer split).

For each model m, aggregate over its classes and destinations:
    R_m = sum_{q in m, l} x_R[q, l]
    S_m = sum_{q in m, l} x_S[q, l]

A side-by-side bar (Stage 1 | Stage 2) per model, stacked with R (replay)
on bottom and S (state transfer) on top, expressed as a fraction of
moved jobs R_m + S_m. Identical pairs => Stage 2 only re-routed across
destinations; shifted pairs => Stage 2 traded prefill pressure for
state-ingest by changing the action mix.

Usage:
    cd evacuation && uv run python plot_stage2_action_mix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2


def per_model_split(x_R, x_S, model_idx, M):
    R = np.array([x_R[model_idx == m].sum() for m in range(M)])
    S = np.array([x_S[model_idx == m].sum() for m in range(M)])
    moved = R + S
    R_pct = 100.0 * np.where(moved > 0, R / moved, 0.0)
    S_pct = 100.0 * np.where(moved > 0, S / moved, 0.0)
    return R, S, R_pct, S_pct


def main() -> None:
    inst = build_instance()
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    M = len(inst.M_names)

    R1, S1, R1_pct, S1_pct = per_model_split(s1.x_R, s1.x_S, inst.model_idx, M)
    R2, S2, R2_pct, S2_pct = per_model_split(s2.x_R, s2.x_S, inst.model_idx, M)

    print(f"phi* = {s2.phi_star:.4f}   Z* = {s1.Z_star:.1f}")
    print(f"{'model':>20s}  {'Stage1 R/S':>14s}  {'Stage2 R/S':>14s}   shift")
    for m in range(M):
        shift = R2_pct[m] - R1_pct[m]
        print(f"{inst.M_names[m]:>20s}  {R1_pct[m]:5.1f}/{S1_pct[m]:5.1f}%  "
              f"{R2_pct[m]:5.1f}/{S2_pct[m]:5.1f}%   {shift:+5.1f}pp R")

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(M)
    w = 0.38
    ax.bar(x - w/2, R1_pct, w, color="#3a7ca5", label="Stage 1 — replay")
    ax.bar(x - w/2, S1_pct, w, bottom=R1_pct, color="#a5c8e1", label="Stage 1 — state")
    ax.bar(x + w/2, R2_pct, w, color="#c44536", label="Stage 2 — replay")
    ax.bar(x + w/2, S2_pct, w, bottom=R2_pct, color="#edaaa0", label="Stage 2 — state")

    # Annotate replay% on top of the R segment of each bar
    for i in range(M):
        ax.text(x[i] - w/2, R1_pct[i] + 1, f"{R1_pct[i]:.0f}%",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i] + w/2, R2_pct[i] + 1, f"{R2_pct[i]:.0f}%",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(inst.M_names, rotation=20, ha="right")
    ax.set_ylabel("share of moved jobs (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Per-model action mix: Stage 1 vs Stage 2\n"
                 f"replay (dark) / state-transfer (light);  "
                 f"$\\phi^\\star={s2.phi_star:.4f}$")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "stage2_action_mix.pdf")
    fig.savefig(out / "stage2_action_mix.png", dpi=150)
    print(f"wrote {out / 'stage2_action_mix.pdf'}")


if __name__ == "__main__":
    main()
