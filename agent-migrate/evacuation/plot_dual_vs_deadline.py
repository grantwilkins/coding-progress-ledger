"""D-parametric prefill prices and the envelope theorem (Section 16).

Prefill capacity is C^pfill = W*D, so as the deadline D grows, prefill pressure
falls and its marginal value decays. The left panel plots each model's prefill
term of the deadline sensitivity, e_m(D) = (1/D) sum_l pi_{lm} r_{lm} — the
model-resolved piece of -dphi*/dD built from the ceiling duals pi and realized
pressures r. Qwen3 235B (worst Sigma_m/Delta_m) carries the largest prefill
term; the normalized ceiling share itself is scale-invariant in the interior, so
the decay lives in this envelope quantity, not the bare simplex weight.

Right panel validates the envelope theorem df*/dC = -pi mapped through C = W*D:
since every capacity is linear in D, dphi*/dD = -(1/D) sum_i pi_i r_i, where pi_i
are the ceiling duals and r_i the realized pressures (two independent LP
outputs). We overlay that dual-predicted slope on the finite-difference slope of
phi*(D). They agree in the interior (phi*<1) and separate once the deadline gets
short enough to strand jobs (phi* pinned at 1, the deadline absorbed by Stage 1
instead of by pressure).

Uses total_jobs=2000 so phi* sweeps the unsaturated regime; the default 10000
saturates at phi*=1 across the whole range.

Usage:
    cd evacuation && uv run python plot_dual_vs_deadline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import MODELS, build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2_duals import ceiling_duals

D_GRID = np.logspace(np.log10(30.0), np.log10(600.0), 20)
TOTAL_JOBS = 2000
MODEL_COLORS = ["#3a7ca5", "#c44536", "#4a9b54", "#8a3122", "#e8943a", "#6a4c93"]


def main() -> None:
    M_names = tuple(m.name for m in MODELS)
    pfill_env = np.zeros((len(D_GRID), len(MODELS)))     # per-model (1/D) sum_l pi_lm r_lm
    phi = np.zeros(len(D_GRID))
    pi_r = np.zeros(len(D_GRID))                          # sum_i pi_i r_i (all resources)

    for d_i, D in enumerate(D_GRID):
        inst = build_instance(D=float(D), total_jobs=TOTAL_JOBS, seed=0)
        s1 = solve_stage1(inst)
        s2 = solve_stage2(inst, s1)
        pi, phi_star = ceiling_duals(inst, s1)
        phi[d_i] = phi_star
        for (kind, *idx), v in pi.items():
            lname = inst.L_names[idx[0]]
            key = (f"net|{lname}" if kind == "net"
                   else f"{kind}|{lname}|{inst.M_names[idx[1]]}")
            r = s2.pressures[key]
            pi_r[d_i] += v * r                            # sum_i pi_i r_i
            if kind == "pfill":
                pfill_env[d_i, idx[1]] += v * r / D       # per-model envelope term

    plt.rcParams.update({
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 4.4))

    for m, (name, c) in enumerate(zip(M_names, MODEL_COLORS)):
        if pfill_env[:, m].max() > 1e-6:
            axl.plot(D_GRID, pfill_env[:, m], "o-", color=c, lw=1.8, ms=4, label=name)
    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xlabel("deadline $D$ (s)")
    axl.set_ylabel("prefill price sensitivity\n"
                   r"$\frac{1}{D}\sum_\ell \pi^{pfill}_{\ell m}\,\hat p_{\ell m}$")
    axl.grid(True, which="both", alpha=0.3)
    axl.legend(loc="upper right", framealpha=0.9)

    # finite-difference slope of phi*(D) vs dual-predicted -(1/D) sum pi_i hat p_i
    dphi_dD = np.gradient(phi, D_GRID)
    pred = -pi_r / D_GRID
    axr.axhline(0, color="0.75", lw=0.8, zorder=0)
    axr.plot(D_GRID, dphi_dD, "o-", color="#3a7ca5", lw=1.8, ms=4, zorder=3,
             label=r"finite difference  $d\phi^\star/dD$")
    axr.plot(D_GRID, pred, "s--", color="#c44536", lw=1.8, ms=4, zorder=2,
             label=r"dual prediction  $-\frac{1}{D}\sum_i \pi_i\,\hat p_i$")
    axr.set_xscale("log")
    axr.set_xlabel("deadline $D$ (s)")
    axr.set_ylabel(r"$d\phi^\star/dD$")
    axr.grid(True, which="both", alpha=0.3)
    axr.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "dual_vs_deadline.pdf", bbox_inches="tight")
    fig.savefig(out / "dual_vs_deadline.png", dpi=150, bbox_inches="tight")

    print("  D     phi*    fd_slope    pred_slope")
    for d_i, D in enumerate(D_GRID):
        print(f"{D:6.1f}  {phi[d_i]:.4f}   {dphi_dD[d_i]:+.2e}  {pred[d_i]:+.2e}")

    for target in (90.0, 273.0):
        j = int(np.argmin(np.abs(D_GRID - target)))
        print(f"\nnearest D={target:.0f}s -> D={D_GRID[j]:.1f}s: "
              f"fd_slope={dphi_dD[j]:+.2e}  dual_pred={pred[j]:+.2e}")
    print(f"\nwrote {out / 'dual_vs_deadline.pdf'}")


if __name__ == "__main__":
    main()
