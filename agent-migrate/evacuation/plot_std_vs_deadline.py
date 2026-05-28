"""Sweep deadline D and plot n_q-weighted std of r_q across Stages 2/3/4.

Stage 4's objective is V = max_q |r_q - r_bar|, but here we report the
n_q-weighted standard deviation, since "spread of per-class costs" is the
quantity readers want to compare across stages. Stage 4 reducing V tends to
also reduce sigma, but they are distinct metrics.

Stage 1 is skipped: its (x, z) witness is solver-arbitrary among Z*-feasible
plans, so its r_q spread has no canonical meaning.

Usage:
    cd evacuation && uv run python plot_std_vs_deadline.py

Writes `outputs/std_vs_D.{pdf,png}` and `outputs/std_sweep_D.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage3 import recon_costs, solve_stage3
from stage4 import solve_stage4

D_SWEEP_S = (10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900)
TOTAL_JOBS = 10_000


def _r_q(inst, x_R, x_S, z):
    c_R, c_S = recon_costs(inst)
    return ((c_R * x_R).sum(axis=1)
            + (c_S * x_S).sum(axis=1)
            + inst.d_miss * z) / inst.n


def _weighted_std(r_q: np.ndarray, n: np.ndarray) -> float:
    N = n.sum()
    r_bar = float((n * r_q).sum() / N)
    var = float((n * (r_q - r_bar) ** 2).sum() / N)
    return float(np.sqrt(max(var, 0.0)))


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    sigmas: dict[str, list[float]] = {"s2": [], "s3": [], "s4": []}
    for D in D_SWEEP_S:
        inst = build_instance(D=float(D), total_jobs=TOTAL_JOBS)
        s1 = solve_stage1(inst)
        s2 = solve_stage2(inst, s1)
        s3 = solve_stage3(inst, s2)
        s4 = solve_stage4(inst, s3)
        sigma_s2 = _weighted_std(_r_q(inst, s2.x_R, s2.x_S, s2.z), inst.n)
        sigma_s3 = _weighted_std(s3.r_q, inst.n)
        sigma_s4 = _weighted_std(s4.r_q, inst.n)
        sigmas["s2"].append(sigma_s2)
        sigmas["s3"].append(sigma_s3)
        sigmas["s4"].append(sigma_s4)
        rows.append((TOTAL_JOBS, D, s1.Z_star, s2.phi_star, s3.H_star,
                     s4.V_star, sigma_s2, sigma_s3, sigma_s4))
        print(f"D={D:4d}s  Z*={s1.Z_star:8.1f}  phi*={s2.phi_star:.4f}  "
              f"H*={s3.H_star:8.2f}s  V*={s4.V_star:8.2f}s  "
              f"sigma s2={sigma_s2:8.2f}  s3={sigma_s3:8.2f}  s4={sigma_s4:8.2f}")

    with (out / "std_sweep_D.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_jobs", "D_s", "Z_star", "phi_star", "H_star_s",
                         "V_star_s", "sigma_s2_s", "sigma_s3_s", "sigma_s4_s"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(D_SWEEP_S, sigmas["s2"], marker="o", linewidth=1.4,
            label="Stage 2 (peak pressure only)")
    ax.plot(D_SWEEP_S, sigmas["s3"], marker="s", linewidth=1.4,
            label="Stage 3 (+ worst-class $H^\\star$)")
    ax.plot(D_SWEEP_S, sigmas["s4"], marker="^", linewidth=1.6,
            label="Stage 4 (+ fairness $V^\\star$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deadline $D$ (s)")
    ax.set_ylabel("$n_q$-weighted std of $r_q$ (s)")
    ax.set_title(f"Spread of per-class reconstruction cost vs $D$ "
                 f"({TOTAL_JOBS:,} jobs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "std_vs_D.pdf")
    fig.savefig(out / "std_vs_D.png", dpi=150)
    print(f"wrote {out / 'std_vs_D.pdf'}")


if __name__ == "__main__":
    main()
