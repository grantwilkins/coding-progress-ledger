"""Integrality gap of the Stage 2 LP relaxation (Section 17.1).

Round the fractional Stage 2 optimum to integer job counts two ways:
  - round_plan        : floor + largest-remainder with capacity repair (feasible;
                        overflow spills to z, leaving more jobs behind);
  - round_plan_naive  : floor + largest-remainder, NO repair (can overload).
Over 10 seeds at the default load (total_jobs=10000, D=300, phi*~1, the
saturation boundary) we report the gap as a fraction of N: extra jobs left
behind by the feasible rounding, and the capacity overload incurred by the naive
one. A small gap means the relaxation is tight and practically useful.

Usage:
    cd evacuation && uv run python plot_rounding_gap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from rounding import evaluate_plan, round_plan, round_plan_naive
from stage1 import solve_stage1
from stage2 import solve_stage2

SEEDS = range(10)
TOTAL_JOBS = 10000
D = 300.0


def main() -> None:
    rows = []
    for seed in SEEDS:
        inst = build_instance(D=D, total_jobs=TOTAL_JOBS, seed=seed)
        s1 = solve_stage1(inst)
        s2 = solve_stage2(inst, s1)
        N = float(inst.n.sum())
        rep = evaluate_plan(inst, *round_plan(inst, s2.x_R, s2.x_S, s2.z))
        nai = evaluate_plan(inst, *round_plan_naive(inst, s2.x_R, s2.x_S, s2.z))
        rows.append(dict(
            N=N, phi=s2.phi_star, Zstar=s2.Z_star,
            extra_rep=(rep[2] - s2.Z_star) / N * 100,    # % of N stranded beyond LP
            extra_nai=(nai[2] - s2.Z_star) / N * 100,
            dphi_rep=rep[0] - s2.phi_star,
            viol_rep=rep[1] * 100, viol_nai=nai[1] * 100, # % over capacity
        ))
        print(f"seed {seed}: phi*={s2.phi_star:.4f} Z*={s2.Z_star:.2f}  "
              f"repair: +{rows[-1]['extra_rep']:.3f}%N stranded, viol={rows[-1]['viol_rep']:.2f}%, "
              f"dphi={rows[-1]['dphi_rep']:+.4f}  |  naive: viol={rows[-1]['viol_nai']:.2f}%")

    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    def strip(ax, series, labels, colors, ylabel, title):
        ax.boxplot(series, positions=range(len(series)), widths=0.5,
                   showfliers=False, medianprops=dict(color="0.2"))
        for i, (s, c) in enumerate(zip(series, colors)):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, color=c, s=22, zorder=3, alpha=0.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)

    strip(ax1,
          [[r["extra_rep"] for r in rows], [r["extra_nai"] for r in rows]],
          ["capacity repair", "naive"], ["#3a7ca5", "#c44536"],
          "extra jobs left behind  (% of $N$)",
          "Integrality gap: jobs stranded beyond the LP")
    ax1.axhline(1.0, color="0.5", ls="--", lw=1.0)
    ax1.text(0.02, 1.02, "1% of $N$", transform=ax1.get_yaxis_transform(), color="0.4", fontsize=8)

    strip(ax2,
          [[r["viol_rep"] for r in rows], [r["viol_nai"] for r in rows]],
          ["capacity repair", "naive"], ["#3a7ca5", "#c44536"],
          "max capacity overload  (%)",
          "Capacity feasibility after rounding")

    med_gap = np.median([r["extra_rep"] for r in rows])
    fig.suptitle(f"Stage 2 LP relaxation tightness  —  $N={int(rows[0]['N'])}$, "
                 f"median gap {med_gap:.3f}% of $N$", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "rounding_gap.pdf", bbox_inches="tight")
    fig.savefig(out / "rounding_gap.png", dpi=150, bbox_inches="tight")
    print(f"median repair gap = {med_gap:.4f}% of N;  "
          f"median naive overload = {np.median([r['viol_nai'] for r in rows]):.3f}%")
    print(f"wrote {out / 'rounding_gap.pdf'}")


if __name__ == "__main__":
    main()
