"""Sweep deadline D and plot percent of jobs evacuated.

Usage:
    cd evacuation && uv run python plot_evacuation_vs_deadline.py

Writes `outputs/percent_evacuated_vs_D.pdf` and `outputs/sweep_D.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from instance import build_instance, TOTAL_JOBS_DEFAULT
from stage1 import solve_stage1

D_SWEEP_S = (10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    for D in D_SWEEP_S:
        res = solve_stage1(build_instance(D=float(D)))
        evacuated = TOTAL_JOBS_DEFAULT - res.Z_star
        rows.append((D, res.Z_star, 100.0 * evacuated / TOTAL_JOBS_DEFAULT))
        print(f"D={D:4d}s  Z*={res.Z_star:7.1f}  {rows[-1][2]:5.1f}% evacuated")

    with (out / "sweep_D.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["D_s", "Z_star", "percent_evacuated"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Jobs evacuated (%)")
    ax.set_ylim(0, 102)
    ax.set_title(f"Stage 1: percent evacuated vs deadline ({TOTAL_JOBS_DEFAULT} jobs)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "percent_evacuated_vs_D.pdf")
    fig.savefig(out / "percent_evacuated_vs_D.png", dpi=150)
    print(f"wrote {out / 'percent_evacuated_vs_D.pdf'}")


if __name__ == "__main__":
    main()
