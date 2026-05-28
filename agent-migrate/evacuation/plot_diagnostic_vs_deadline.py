"""Sweep deadline D and plot Section 14's per-resource overload.

Records sigma* (worst-resource overload factor) plus the worst slack within
each resource type (net, pfill, ing). Below the feasibility threshold these
curves identify WHICH resource limits evacuation and BY HOW MUCH; above it
all curves collapse to ~0 (matching Stage 1's Z* = 0 regime).

Usage:
    cd evacuation && uv run python plot_diagnostic_vs_deadline.py

Writes `outputs/diagnostic_vs_D.{pdf,png}` and `outputs/diagnostic_sweep_D.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnostic import solve_diagnostic
from instance import build_instance
from stage1 import solve_stage1

D_SWEEP_S = (10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)
TOTAL_JOBS = 10_000


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    sigma_series: list[float] = []
    snet_series: list[float] = []
    spfill_series: list[float] = []
    sing_series: list[float] = []
    for D in D_SWEEP_S:
        inst = build_instance(D=float(D), total_jobs=TOTAL_JOBS)
        s1 = solve_stage1(inst)
        diag = solve_diagnostic(inst)
        max_snet = float(diag.s_net.max())
        max_spfill = float(diag.s_pfill.max())
        max_sing = float(diag.s_ing.max())
        sigma_series.append(diag.sigma_star)
        snet_series.append(max_snet)
        spfill_series.append(max_spfill)
        sing_series.append(max_sing)
        rows.append((TOTAL_JOBS, D, s1.Z_star, diag.sigma_star,
                     max_snet, max_spfill, max_sing))
        print(f"D={D:4d}s  Z*={s1.Z_star:8.1f}  sigma*={diag.sigma_star:7.3f}  "
              f"max s_net={max_snet:7.3f}  s_pfill={max_spfill:7.3f}  s_ing={max_sing:7.3f}")

    with (out / "diagnostic_sweep_D.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_jobs", "D_s", "Z_star", "sigma_star",
                         "max_s_net", "max_s_pfill", "max_s_ing"])
        writer.writerows(rows)

    # Floor for log axis: replace exact zeros with a small sentinel.
    floor = 1e-3
    def _floor(arr):
        return [max(v, floor) for v in arr]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(D_SWEEP_S, _floor(sigma_series), color="black", marker=None,
            linewidth=2.0, label=r"$\sigma^\star$ (worst overall)")
    ax.plot(D_SWEEP_S, _floor(snet_series), marker="o", linewidth=1.3,
            label=r"max $s^{\mathrm{net}}_\ell$")
    ax.plot(D_SWEEP_S, _floor(spfill_series), marker="s", linewidth=1.3,
            label=r"max $s^{\mathrm{pfill}}_{\ell m}$")
    ax.plot(D_SWEEP_S, _floor(sing_series), marker="^", linewidth=1.3,
            label=r"max $s^{\mathrm{ing}}_{\ell m}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel(r"Overload slack $s_i$ (dim.-less)")
    ax.set_title(f"Section 14 diagnostic: worst-resource overload vs $D$ "
                 f"({TOTAL_JOBS:,} jobs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "diagnostic_vs_D.pdf")
    fig.savefig(out / "diagnostic_vs_D.png", dpi=150)
    print(f"wrote {out / 'diagnostic_vs_D.pdf'}")


if __name__ == "__main__":
    main()
