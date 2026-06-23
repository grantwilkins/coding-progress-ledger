"""T8: load↔memory regime boundary diagnostic, walked two ways.

(a) idle/cold × γ at a fixed short context, (b) context E[T] short→long. Both push total
load L across the constant threshold occupancy·N·ρ* (pool-sized populations fix S_held/s_node = occupancy·N).
Plotted vs the regime ratio R = (S_held/s_node)/(L/ρ*): memory still marks a capacity regime,
but it no longer supplies certified watts. The dispatch uses active-work power in every regime;
memory-pressure correlations are reported only as diagnostics.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from dispatch import Event, bind_dp, solve
from impact import compute
from instance import Workload, _mean_T, generate
from power import PoolPower

POOL, WL, EVENT = PoolPower(), replace(Workload(), t_mix=Workload().t_mixes[-1]), Event()
SEEDS, N_NODES = range(8), 32
ET_A = 13_000.0  # fixed context for (a): crossover lands mid active-sweep


def shift_tmix(target):
    """Center mixture shape, log-shifted so analytic E[T] = target."""
    c = np.log(target / _mean_T(WL))
    return tuple((w, mu + c, s) for w, mu, s in WL.t_mix)


def point(pool, wl):
    """Mean,std over seeds of (R, corr_cert, corr_mem, rho_div, feas)."""
    rows = []
    for sd in SEEDS:
        pop = generate(pool, wl, n_nodes=N_NODES, seed=sd)
        imp = compute(pop, pool)
        dp = bind_dp(imp)
        mv = dp > 0
        plan = solve(pop, pool, imp, 0.3 * dp.sum(), EVENT)
        real = plan.y > 0.5
        rows.append((
            (len(pop) / pool.s_node) / (pop.ell.sum() / pool.rho_star),
            spearmanr(real[mv], dp[mv]).correlation,
            spearmanr(real[mv], imp.dp_memory[mv]).correlation,
            spearmanr(dp[mv], imp.dp_memory[mv]).correlation,
            float(plan.feasible),
        ))
    a = np.array(rows)
    return a.mean(0), a.std(0)


def walk(configs):
    M, S = zip(*(point(pool, wl) for pool, wl in configs))
    return np.array(M), np.array(S)  # (n,5): R, corr_cert, corr_mem, rho_div, feas


def walk_a(gamma):
    pool = replace(POOL, gamma=gamma, mean_context_tokens=ET_A)
    cfg = []
    for act in np.linspace(0.05, 0.6, 11):
        rest = 1 - act
        wl = replace(WL, state_mix=(act, 0.25 / 0.70 * rest, 0.45 / 0.70 * rest), t_mix=shift_tmix(ET_A))
        cfg.append((pool, wl))
    return walk(cfg)


def walk_b():
    cfg = []
    for et in np.geomspace(5e3, 4e4, 11):
        wl = replace(WL, t_mix=shift_tmix(et))
        cfg.append((replace(POOL, mean_context_tokens=_mean_T(wl)), wl))
    return walk(cfg)


SERIES = [("(a) γ=0.5", walk_a(0.5)), ("(a) γ=1.0", walk_a(1.0)), ("(b) context", walk_b())]

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
for (lbl, (M, S)), c in zip(SERIES, colors):
    o = np.argsort(M[:, 0])
    R = M[o, 0]
    for ax, col in ((ax0, 1), (ax2, 3)):
        ax.plot(R, M[o, col], "-o", ms=3, color=c, label=lbl)
        ax.fill_between(R, M[o, col] - S[o, col], M[o, col] + S[o, col], color=c, alpha=0.2)
    ax1.plot(R, M[o, 2], "--s", ms=3, color=c, label=lbl)

for ax in (ax0, ax1, ax2):
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
ax0.set_ylabel("Spearman\n(shed set, certificate)")
ax1.set_ylabel("Spearman\n(shed set, memory)")
ax2.set_ylabel("Spearman\n(certificate, memory)")
ax2.set_xlabel("regime ratio  R = (S_held/s_node)/(L/ρ*)")
ax0.legend(fontsize=8)
ax0.set_title("T8 — memory regime is diagnostic, not certified watts")
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/regime_boundary.{ext}", dpi=150)

for lbl, (M, _) in SERIES:
    R, near = M[:, 0], M[np.argmin(np.abs(M[:, 0] - 1))]
    print(f"{lbl:12s}  R∈[{R.min():.2f},{R.max():.2f}] brackets 1: {R.min() < 1 < R.max()}"
          f"  corr_cert@R≈1={near[1]:+.2f}  corr_mem@R≈1={near[2]:+.2f}  feasible frac={M[:, 4].mean():.2f}")
rho = np.concatenate([M[:, 3] for _, (M, _) in SERIES])
print(f"Spearman(certificate, dp_memory): mean={rho.mean():+.3f} std={rho.std():.3f}  (measured, not assumed)")
print("memory pressure remains a constraint/diagnostic; it no longer changes bind_dp")
