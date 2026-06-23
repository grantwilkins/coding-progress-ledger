"""Certify low, report high (T6; formulation.md §"Certify low, report high").

Sweep the grid's requested power cut and read two prices off each dispatch plan:
the *guaranteed* floor (realized even if no node ever shuts off) and the *expected*
upside (realized once removed load lets idle nodes shut off). Two pools, plain English
on every axis — no bare symbols.

Left (compute-bound pool): the guaranteed floor is a small fraction of the current
single-price expected proxy. The gap is exactly the bracket ratio (30×). There is no
extra prefill/decode power adjustment in this plot; token-energy work power is not calibrated yet.

Right (memory-bound pool, mirror image): the bottleneck is KV cache, so the certified
price is the memory price (power freed by draining a full node). The load-only future
proxy is not the certificate in this regime; the 30× load bracket does not transfer.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, solve
from impact import Movement, compute
from instance import Workload, generate
from power import PoolPower

# Slack movement budgets so no link/headroom constraint binds: the shed ceiling is set
# only by the population's certifiable watts, isolating the price story (T5 owns links).
SLACK_E = Event(D=1e9, W=10**7, dest_nodes=10**7)
SLACK_M = replace(Movement(), lambda_src=1e18, mu_in=1e18)
kW = 1e3
BR = PoolPower().bracket_ratio  # single-price bracket p̄/s_plat = 30×


def build(pool, wl, n_nodes, regime):
    pop = generate(pool, wl, n_nodes=n_nodes, seed=42)
    imp = compute(pop, pool)
    assert imp.regime == regime, f"expected {regime}-bound population, got {imp.regime}"
    ceil = solve(pop, pool, imp, 2 * bind_dp(imp).sum(), SLACK_E, SLACK_M).shed_guaranteed
    S = np.linspace(0.04, 1.15, 20) * ceil  # past the ceiling so the frontier shows
    return S, [solve(pop, pool, imp, s, SLACK_E, SLACK_M) for s in S]


def ratios(plans):
    return np.array([p.shed_expected / p.shed_guaranteed
                     for p in plans if p.feasible and p.shed_guaranteed > 0])


# Compute-bound: short context (~3.4k tok) keeps KV from binding. Memory-bound: defaults.
# Both are controlled price-story fixtures, not live traffic mixes.
LOAD_POOL = replace(PoolPower(), mean_context_tokens=3378)
LOAD_WL = replace(Workload(), t_mix=((1.0, 8.0, 0.5),))
Sl, pl = build(LOAD_POOL, LOAD_WL, 4, "load")
Sm, pm = build(PoolPower(), Workload(), 32, "memory")
rl, rm = ratios(pl), ratios(pm)

fig, (axL, axM) = plt.subplots(1, 2, figsize=(12, 4.8))

# --- compute-bound pool: expected ≫ guaranteed ---
g, e, x = (np.array([p.shed_guaranteed for p in pl]) / kW,
           np.array([p.shed_expected for p in pl]) / kW, Sl / kW)
axL.fill_between(x, g, e, color="tab:blue", alpha=0.10)
axL.plot(x, g, color="tab:blue", lw=2.4, label="power freed while nodes keep running (guaranteed)")
axL.plot(x, e, color="tab:red", lw=2.4, label="power freed once idle nodes shut off (expected)")
axL.plot(x, BR * g, color="0.4", lw=1.3, ls="--", label=f"single-price bracket ({BR:.0f}×)")
axL.annotate(f"expected ≈ {rl.mean():.0f}× guaranteed\n(single-price bracket only)",
             xy=(0.50, 0.62), xycoords="axes fraction", fontsize=9, ha="center", color="0.2")
axL.set(xlabel="requested power cut (kW)", ylabel="power freed (kW)")
axL.set_title("Compute-bound pool: certified floor far below expected power freed", fontsize=10)
axL.legend(loc="upper left", fontsize=8)

# --- memory-bound pool: the memory floor, not the load proxy, is the certificate ---
gm, em, xm = (np.array([p.shed_guaranteed for p in pm]) / kW,
              np.array([p.shed_expected for p in pm]) / kW, Sm / kW)
axM.fill_between(xm, np.minimum(gm, em), np.maximum(gm, em), color="tab:gray", alpha=0.15)
axM.plot(xm, gm, color="tab:blue", lw=2.4, label="power freed by draining full nodes (guaranteed)")
axM.plot(xm, em, color="tab:red", lw=2.4, label="load-only future proxy (not the memory floor)")
axM.annotate(f"30× load bracket does not transfer\nload proxy = {rm.min():.1f}–{rm.max():.1f}× memory floor",
             xy=(0.50, 0.40), xycoords="axes fraction", fontsize=9, ha="center", color="0.2")
axM.set(xlabel="requested power cut (kW)", ylabel="power freed (kW)")
axM.set_title("Memory-bound pool: memory floor is the certificate", fontsize=10)
axM.legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/certify_report_validation.{ext}", dpi=150)

contain = all(p.shed_expected >= p.shed_guaranteed - 1e-6 for p in pl if p.feasible)
print(f"LOAD  pool: single-price gap = {BR:.0f}× (bracket ratio); "
      f"expected/guaranteed = {rl.mean():.1f}×")
print(f"MEM   pool: expected/guaranteed = {rm.mean():.1f}× "
      f"(range {rm.min():.1f}–{rm.max():.1f}); 30× bracket does NOT transfer")
print(f"containment (every feasible S* certified low is met high): {'PASS' if contain else 'FAIL'}")
