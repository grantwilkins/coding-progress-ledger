"""Sweep deadline D and plot percent of jobs evacuated.

Edit `JOB_COUNTS` to overlay additional workload sizes.

Usage:
    cd evacuation && uv run python plot_evacuation_vs_deadline.py

Writes `outputs/percent_evacuated_vs_D.{pdf,png}` and `outputs/sweep_D.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from instance import build_instance
from stage1 import solve_stage1

D_SWEEP_S = (1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)
JOB_COUNTS = (10_000, 20_000)


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    series: dict[int, list[float]] = {}
    for jobs in JOB_COUNTS:
        pct = []
        for D in D_SWEEP_S:
            res = solve_stage1(build_instance(D=float(D), total_jobs=jobs))
            evacuated_pct = 100.0 * (jobs - res.Z_star) / jobs
            pct.append(evacuated_pct)
            rows.append((jobs, D, res.Z_star, evacuated_pct))
            print(f"jobs={jobs:6d}  D={D:4d}s  Z*={res.Z_star:8.1f}  {evacuated_pct:5.1f}% evacuated")
        series[jobs] = pct

    with (out / "sweep_D.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_jobs", "D_s", "Z_star", "percent_evacuated"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for jobs, pct in series.items():
        ax.plot(D_SWEEP_S, pct, marker="o", label=f"{jobs:,} jobs", linewidth=1.5)
    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("Jobs evacuated (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, max(D_SWEEP_S))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "percent_evacuated_vs_D.pdf")
    fig.savefig(out / "percent_evacuated_vs_D.png", dpi=150)
    print(f"wrote {out / 'percent_evacuated_vs_D.pdf'}")


if __name__ == "__main__":
    main()
