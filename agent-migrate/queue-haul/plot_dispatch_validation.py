"""Validation plot for dispatch.py (formulation.md §Dispatch program).

Three policies, all respecting the same movement budgets the deadline window
allows (source egress bytes, rebuild capacity, destination headroom): a random
baseline (shuffle the movable jobs, pick each action by coin flip), the
decentralized greedy (each job self-selects its cheaper action, best-deal jobs
move first), and the coordinated LP/MILP. Left: achieved shed vs requested S*.
Each tracks y=x while feasible, then flattens at its ceiling — random < greedy <
LP, because selection (greedy over random) and then global repacking
(transfers→replays, LP over greedy) each fit more shed under the same egress link.
Right: total downtime vs *achieved* shed (the fair per-watt view). Greedy lies
exactly on the LP/MILP cost frontier (optimal where it can operate, just stops
sooner); random sits well above it — it pays more downtime per watt shed.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, greedy, random_dispatch, solve
from impact import Movement, compute
from instance import generate
from power import PoolPower

POOL = PoolPower()
# Destination headroom slack; the 1 GB/s source egress link is the binding budget.
POP = generate(POOL, n_nodes=16)
IMP = compute(POP, POOL)
EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
SEEDS = range(8)

DP = bind_dp(IMP)
lp_ceil = solve(POP, POOL, IMP, 2 * DP.sum(), EVENT, MOVE).shed_guaranteed  # finite unreachable → max shed
S = np.linspace(0.04, 1.15, 20) * lp_ceil

lp = [solve(POP, POOL, IMP, s, EVENT, MOVE) for s in S]
mi = [solve(POP, POOL, IMP, s, EVENT, MOVE, integer=True) for s in S]
gr = [greedy(POP, POOL, IMP, s, EVENT, MOVE) for s in S]
rd = [[random_dispatch(POP, POOL, IMP, s, EVENT, MOVE, seed=sd) for sd in SEEDS] for s in S]
g_ceil = max(p.shed_guaranteed for p in gr)
r_shed = np.array([[p.shed_guaranteed for p in row] for row in rd]) / 1e3  # (len(S), seeds)

kW = 1e3
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# --- Panel A: achieved shed vs requested S* (the three ceilings) ---
axA.plot(S / kW, S / kW, "k--", lw=1, label="requested (y=x)")
axA.fill_between(S / kW, r_shed.min(1), r_shed.max(1), color="tab:gray", alpha=0.2)
axA.plot(S / kW, r_shed.mean(1), color="tab:gray", lw=2, label="random (shuffled)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in gr], color="tab:orange", lw=2, label="greedy (decentralized)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in lp], color="tab:blue", lw=2, label="LP (coordinated)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in mi], color="tab:red", lw=1.2, ls="--", label="MILP")
xg = 0.92 * S.max() / kW
axA.annotate("", xy=(xg, lp_ceil / kW), xytext=(xg, g_ceil / kW), arrowprops=dict(arrowstyle="<->", color="0.3"))
axA.text(xg - 2, (g_ceil + lp_ceil) / 2 / kW, "coordination\ngap", fontsize=8, ha="right", va="center")
axA.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="achieved shed (kW)",
        title="random < greedy < LP: selection then coordination")
axA.legend(loc="upper left", fontsize=8)

# --- Panel B: total downtime vs ACHIEVED shed (fair per-watt view) ---
def frontier(plans):
    return np.array(sorted((p.shed_guaranteed / kW, p.cost) for p in plans if p.feasible)).T

r_pts = [(p.shed_guaranteed / kW, p.cost) for row in rd for p in row if p.feasible]
axB.scatter(*zip(*r_pts), s=14, color="tab:gray", alpha=0.4, label="random (feasible runs)")
axB.plot(*frontier(lp), color="tab:blue", lw=2, label="LP (coordinated)")
axB.plot(*frontier(mi), color="tab:red", lw=1.2, ls="--", label="MILP")
axB.plot(*frontier(gr), color="tab:orange", lw=4, alpha=0.5, label="greedy (on the optimum)")
axB.set(xlabel="achieved shed (kW)", ylabel="total downtime $\\Sigma\\,c$ (s)",
        title="random pays more per watt; greedy = LP optimum where feasible")
axB.legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation.{ext}", dpi=150)

print(f"regime={IMP.regime}  jobs={len(POP)}  ceilings kW: "
      f"random={r_shed.max(1).mean():.1f}  greedy={g_ceil/kW:.1f}  LP={lp_ceil/kW:.1f}")
k = np.searchsorted(S, 0.5 * g_ceil)
print(f"@S*={S[k]/kW:.1f}kW: random cost={np.mean([rd[k][s].cost for s in SEEDS]):.0f} "
      f"(shed {np.mean([rd[k][s].shed_guaranteed for s in SEEDS])/kW:.1f}kW, feas "
      f"{np.mean([rd[k][s].feasible for s in SEEDS]):.0%})  greedy={gr[k].cost:.0f}  LP={lp[k].cost:.0f}")
