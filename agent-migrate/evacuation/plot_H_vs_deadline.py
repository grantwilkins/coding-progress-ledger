"""Sweep deadline D and plot Stage 3's worst-class reconstruction cost H*.

Usage:
    cd evacuation && uv run python plot_H_vs_deadline.py

Writes `outputs/H_vs_D.{pdf,png}` and `outputs/H_sweep_D.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage3 import solve_stage3

D_SWEEP_S = (10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)
JOB_COUNTS = (10_000,)


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    series_H: dict[int, list[float]] = {}
    series_med: dict[int, list[float]] = {}
    for jobs in JOB_COUNTS:
        H = []
        med = []
        for D in D_SWEEP_S:
            inst = build_instance(D=float(D), total_jobs=jobs)
            s1 = solve_stage1(inst)
            s2 = solve_stage2(inst, s1)
            s3 = solve_stage3(inst, s2)
            r_sorted = sorted(s3.r_q.tolist())
            r_median = r_sorted[len(r_sorted) // 2]
            H.append(s3.H_star)
            med.append(r_median)
            rows.append((jobs, D, s1.Z_star, s2.phi_star, s3.H_star, r_median))
            print(f"jobs={jobs:6d}  D={D:4d}s  Z*={s1.Z_star:8.1f}  "
                  f"phi*={s2.phi_star:.4f}  H*={s3.H_star:8.2f}s  med r_q={r_median:6.2f}s")
        series_H[jobs] = H
        series_med[jobs] = med

    with (out / "H_sweep_D.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_jobs", "D_s", "Z_star", "phi_star", "H_star_s", "r_q_median_s"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for jobs in JOB_COUNTS:
        ax.plot(D_SWEEP_S, series_H[jobs], marker="o", linewidth=1.6,
                label=f"$H^\\star$ (worst class)")
        ax.plot(D_SWEEP_S, series_med[jobs], marker="x", linestyle="--",
                linewidth=1.0, alpha=0.7, label=f"median $r_q$")
    ax.plot(D_SWEEP_S, [2 * D for D in D_SWEEP_S], linestyle=":",
            color="gray", linewidth=1.0, label="$d^{\\mathrm{miss}} = 2D$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Per-class avg. reconstruction cost (s)")
    ax.set_title(f"Stage 3: worst-class $H^\\star$ vs deadline ({JOB_COUNTS[0]:,} jobs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "H_vs_D.pdf")
    fig.savefig(out / "H_vs_D.png", dpi=150)
    print(f"wrote {out / 'H_vs_D.pdf'}")


if __name__ == "__main__":
    main()
