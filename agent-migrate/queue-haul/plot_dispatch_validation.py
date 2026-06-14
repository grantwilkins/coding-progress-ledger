"""Validation plot for dispatch.py (formulation.md §Dispatch program).

Both solvers respect the same movement budgets the deadline window allows (source
egress bytes, rebuild capacity, destination headroom). Left: achieved shed vs
requested S*. Each tracks y=x while feasible, then flattens at its ceiling. The
decentralized greedy (each job self-selects its cheaper action, best-deal jobs
move first, shared budgets drawn down in one myopic pass) hits a *lower* ceiling
than the coordinated LP — which repacks transfers into replays to ship fewer bytes
per watt, fitting far more shed under the same egress link. That vertical gap is
the value of central coordination. Right: total downtime vs *achieved* shed (the
fair, per-watt comparison). On this cost frontier greedy and the LP/MILP coincide
exactly — greedy is optimal where it can operate; it simply stops sooner.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, greedy, solve
from impact import Movement, compute
from instance import generate
from power import PoolPower

POOL = PoolPower()
# Destination headroom slack; the 1 GB/s source egress link is the binding budget.
POP = generate(POOL, n_nodes=16)
IMP = compute(POP, POOL)
EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()

DP = bind_dp(IMP)
lp_ceil = solve(POP, POOL, IMP, 2 * DP.sum(), EVENT, MOVE).shed_guaranteed  # finite unreachable → max shed
S = np.linspace(0.04, 1.15, 20) * lp_ceil

lp = [solve(POP, POOL, IMP, s, EVENT, MOVE) for s in S]
mi = [solve(POP, POOL, IMP, s, EVENT, MOVE, integer=True) for s in S]
gr = [greedy(POP, POOL, IMP, s, EVENT, MOVE) for s in S]
g_ceil = max(p.shed_guaranteed for p in gr)  # greedy's reachable shed

kW = 1e3
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# --- Panel A: achieved shed vs requested S* (the two ceilings) ---
axA.plot(S / kW, S / kW, "k--", lw=1, label="requested (y=x)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in gr], color="tab:orange", lw=2, label="greedy (decentralized)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in lp], color="tab:blue", lw=2, label="LP (coordinated)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in mi], color="tab:red", lw=1.2, ls="--", label="MILP")
axA.axhline(g_ceil / kW, color="tab:orange", ls=":", lw=0.8)
axA.axhline(lp_ceil / kW, color="tab:blue", ls=":", lw=0.8)
xg = 0.92 * S.max() / kW
axA.annotate("", xy=(xg, lp_ceil / kW), xytext=(xg, g_ceil / kW),
             arrowprops=dict(arrowstyle="<->", color="0.3"))
axA.text(xg - 2, (g_ceil + lp_ceil) / 2 / kW, "coordination\ngap", fontsize=8, ha="right", va="center")
axA.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="achieved shed (kW)",
        title="Same links, higher ceiling: coordination ~2× the shed")
axA.legend(loc="upper left", fontsize=8)

# --- Panel B: total downtime vs ACHIEVED shed (fair per-watt frontier) ---
def frontier(plans):
    pts = sorted((p.shed_guaranteed / kW, p.cost) for p in plans if p.feasible)
    return np.array(pts).T

axB.plot(*frontier(lp), color="tab:blue", lw=2, label="LP (coordinated)")
axB.plot(*frontier(mi), color="tab:red", lw=1.2, ls="--", label="MILP")
axB.plot(*frontier(gr), color="tab:orange", lw=4, alpha=0.5, label="greedy (stops at its ceiling)")
axB.set(xlabel="achieved shed (kW)", ylabel="total downtime $\\Sigma\\,c$ (s)",
        title="Same per-watt cost where feasible; greedy stops sooner")
axB.legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation.{ext}", dpi=150)

print(f"regime={IMP.regime}  jobs={len(POP)}  greedy_ceiling={g_ceil/kW:.1f}kW  "
      f"LP_ceiling={lp_ceil/kW:.1f}kW  coordination={lp_ceil/g_ceil:.2f}x")
k = np.searchsorted(S, 0.5 * g_ceil)
print(f"@S*={S[k]/kW:.1f}kW (both feasible)  greedy cost={gr[k].cost:.0f}  LP cost={lp[k].cost:.0f}  "
      f"coincide={'yes' if abs(gr[k].cost-lp[k].cost)<1e-3*lp[k].cost else 'no'}")
print(f"greedy infeasible beyond its ceiling: shed caps at {g_ceil/kW:.1f}kW while LP reaches {lp_ceil/kW:.1f}kW")
