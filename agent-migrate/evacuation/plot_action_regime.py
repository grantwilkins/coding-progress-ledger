"""Action regime diagram: replay fraction vs bandwidth, at two prefill speeds.

For each (lambda, rho) Stage 2 picks an action mix; we plot the
replay fraction R / (R + S) of moved jobs. More bandwidth makes state transfer
(KV-cache, network-heavy) cheaper and pushes the mix toward state; faster
prefill makes replay cheaper and pushes it back up. The 50/50 crossover is the
endogenous boundary (Section 15): prefill speed slides it along the bandwidth
axis without changing the curve's shape.

The default-rho curve and a pessimistic rho=0.5 curve are computed here over an
extended lambda grid (the saved sweep tops out at lambda=8, before the
crossover) using the (model, log-T) class aggregation for speed.

Degeneracy note: Stage 2 minimizes the *peak* normalized pressure phi. At high
bandwidth the network ceiling goes slack and phi is set by the prefill/ingest
(GPU) resources, which are insensitive to the replay/state split over a wide
range. The R/(R+S) split is then under-determined: `min phi` admits a whole
face of optima and a bare LP solve returns an arbitrary, non-monotone vertex
(the spurious high-lambda kink). We pin the canonical point with a lexicographic
tie-break -- among all phi*-optimal mixes, prefer the most replay -- via a tiny
reward EPS_REG * (replay share) added to the objective. It is small enough to
leave phi* unchanged (relative perturbation < 1e-9) and only selects within the
optimal face, so the curve below the crossover is untouched and the tail is
smooth and monotone.

Usage:
    cd evacuation && uv run python plot_action_regime.py
"""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from loads import inv_cap, loads, norm_cap
from stage1 import solve_stage1

OUT = Path(__file__).resolve().parent / "outputs"
LAM = np.logspace(np.log10(0.25), np.log10(64.0), 18)
EPS_REG = 1e-3  # lexicographic replay-preference weight; does not move phi*


def _replay_frac(rho_scale: float) -> np.ndarray:
    """Replay share R/(R+S) of moved jobs vs LAM, with the degeneracy-breaking
    tie-break folded into a single Stage-2-equivalent solve (throughput link)."""
    out = []
    for lam in LAM:
        inst = build_instance(total_jobs=10_000, n_bins=160,
                              lambda_scale=float(lam), rho_scale=rho_scale)
        s1 = solve_stage1(inst)
        C_net, C_pfill, C_ing, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)

        x_R = cp.Variable(s1.x_R.shape, nonneg=True)
        x_S = cp.Variable(s1.x_S.shape, nonneg=True)
        z = cp.Variable(inst.T.size, nonneg=True)
        phi = cp.Variable(nonneg=True)

        L_net = b_net_R @ x_R + b_net_S @ x_S
        L_pf = S_pfill @ x_R
        L_in = S_ing @ x_S
        (a_net, r_net), (a_pf, r_pf), (a_in, r_in) = map(norm_cap, (C_net, C_pfill, C_ing))

        cons = [
            cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
            cp.sum(z) == s1.Z_star,
            cp.multiply(a_net, L_net) <= r_net,
            cp.multiply(a_pf, L_pf) <= r_pf,
            cp.multiply(a_in, L_in) <= r_in,
            cp.multiply(inv_cap(C_net), L_net) <= phi,
            cp.multiply(inv_cap(C_pfill), L_pf) <= phi,
            cp.multiply(inv_cap(C_ing), L_in) <= phi,
        ]
        reg = cp.sum(x_R) / float(inst.n.sum())
        prob = cp.Problem(cp.Minimize(phi - EPS_REG * reg), cons)
        prob.solve(solver=cp.SCIPY)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"action-regime solve returned {prob.status}")

        r, s = float(x_R.value.sum()), float(x_S.value.sum())
        out.append(r / (r + s) if r + s > 0 else np.nan)
    return np.array(out)


def _crossover(frac, target=0.5):
    """log-interpolated lambda where frac first crosses target."""
    f = frac - target
    for i in range(len(f) - 1):
        if f[i] * f[i + 1] < 0:
            a = f[i] / (f[i] - f[i + 1])
            return float(LAM[i] * (LAM[i + 1] / LAM[i]) ** a)
    return None


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.4), constrained_layout=True)

    curves = [(1.0, "#3a7ca5", r"$\rho$ default"),
              (0.5, "#c44536", r"$\rho \times 0.5$ (slow prefill)")]
    crossovers = []
    for rho_scale, color, label in curves:
        frac = _replay_frac(rho_scale)
        xc = _crossover(frac)
        crossovers.append((rho_scale, xc))
        leg = f"{label} (50/50 at {xc:.1f}$\\times$)" if xc else label
        ax.plot(LAM, 100 * frac, marker="o", ms=5.5, lw=2.2, color=color, label=leg)
        if xc:
            ax.plot([xc], [50], marker="o", ms=9, mfc="white", mec=color,
                    mew=2.0, zorder=5)

    # Single unobtrusive 50/50 reference line.
    ax.axhline(50, color="0.55", lw=1.0, ls="--", zorder=1)
    ax.text(LAM[0], 51.5, "50/50", color="0.4", fontsize=15, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel(r"bandwidth scale $\Lambda$", fontsize=19)
    ax.set_ylabel(r"replay share of moved jobs  $R/(R+S)$  (%)", fontsize=19)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=15, loc="lower left", framealpha=1.0).set_zorder(6)
    fig.savefig(OUT / "action_regime.pdf")
    fig.savefig(OUT / "action_regime.png", dpi=150)
    for rho_scale, xc in crossovers:
        print(f"50/50 crossover  rho_scale={rho_scale}:  Lambda* = {xc:.4f}x")
    print(f"wrote {OUT / 'action_regime.pdf'}")


if __name__ == "__main__":
    main()
