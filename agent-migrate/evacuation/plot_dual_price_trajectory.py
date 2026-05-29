"""Congestion prices forming over subgradient iterations (Section 16).

The dual decomposition prices the pressure ceilings; each destination publishes
prices and classes route to the cheapest option. We plot the net and prefill
price trajectories pi_i^k from the subgradient solver and overlay the CVXPY
shadow prices as horizontal targets: the prices discover which resources are
scarce and settle onto the LP dual equilibrium.

Uses a loaded instance (total_jobs=2000, D=60s -> phi*~0.97) so both network and
prefill prices are meaningfully nonzero; the underloaded default-size instance
prices almost everything at the network.

Usage:
    cd evacuation && uv run python plot_dual_price_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2_dual import subgradient
from stage2_duals import ceiling_duals

MAX_ITER = 600
NET_COLORS = ["#3a7ca5", "#1f4e6b", "#6fa8c9"]
PFILL_COLORS = ["#c44536", "#e8943a", "#8a3122", "#d4a017"]


def main() -> None:
    inst = build_instance(D=60.0, total_jobs=2000, seed=0)
    s1 = solve_stage1(inst)
    pi_target, phi_star = ceiling_duals(inst, s1)
    traj = subgradient(inst, s1, phi_star=phi_star, max_iter=MAX_ITER)
    P, meta = traj.prices, traj.I_meta
    it = traj.iters

    # Running (ergodic) average smooths subgradient oscillation into the price
    # the market is actually converging toward.
    P_avg = np.cumsum(P, axis=0) / it[:, None]

    net_cols = [(j, inst.L_names[l]) for j, (k, l, m) in enumerate(meta) if k == "net"]
    pfill_cols = [(j, f"{inst.L_names[l]} / {inst.M_names[m]}", pi_target[(k, l, m)])
                  for j, (k, l, m) in enumerate(meta) if k == "pfill"]
    # keep the prefill indices the LP actually prices (binding); drop the rest
    pfill_cols = sorted([c for c in pfill_cols if c[2] > 0.01], key=lambda c: -c[2])[:4]

    fig, (axn, axp) = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)

    for (j, name), c in zip(net_cols, NET_COLORS):
        axn.plot(it, P_avg[:, j], color=c, lw=1.6, label=name)
        tgt = pi_target[("net", inst.L_names.index(name))]
        axn.axhline(tgt, color=c, ls=":", lw=1.2)
    axn.set_title("Network prices  $\\pi^{net}_\\ell$")
    axn.set_ylabel(r"price $\pi_i$  (share of unit simplex)")

    for (j, name, tgt), c in zip(pfill_cols, PFILL_COLORS):
        axp.plot(it, P_avg[:, j], color=c, lw=1.6, label=name)
        axp.axhline(tgt, color=c, ls=":", lw=1.2)
    axp.set_title("Prefill prices  $\\pi^{pfill}_{\\ell m}$")

    for ax in (axn, axp):
        ax.set_xlabel("subgradient iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Congestion prices converging to CVXPY shadow prices "
                 f"(dotted)  —  $\\phi^\\star={phi_star:.3f}$,  $Q={inst.T.size}$ classes", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "dual_price_trajectory.pdf", bbox_inches="tight")
    fig.savefig(out / "dual_price_trajectory.png", dpi=150, bbox_inches="tight")

    print(f"phi*={phi_star:.4f}")
    for j, name in net_cols:
        print(f"  net {name:8s}: final_avg={P_avg[-1, j]:.4f}  target={pi_target[('net', inst.L_names.index(name))]:.4f}")
    for j, name, tgt in pfill_cols:
        print(f"  pfill {name:24s}: final_avg={P_avg[-1, j]:.4f}  target={tgt:.4f}")
    print(f"wrote {out / 'dual_price_trajectory.pdf'}")


if __name__ == "__main__":
    main()
