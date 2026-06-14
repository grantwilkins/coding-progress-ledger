"""Validation plot for dispatch.py (formulation.md §Dispatch program).

Left: achieved shed vs requested S*. The guaranteed shed (the committed floor the
≥S* constraint binds on) tracks y=x exactly while feasible, then flattens at the
movement-limited ceiling (source egress) where a shortfall opens; the expected
shed (reported, not bound) rides above it — the full guaranteed/expected bracket
spread is the load-regime result T6 measures. Right: total downtime vs S* for the
resource-blind greedy, the fractional LP, and the integer MILP. Greedy and LP
coincide where no constraint binds (their only difference is resource-blindness,
both split the last job); the MILP rides just above (integrality can't split, so
it overshoots) — that gap is the relaxation/granularity cost.
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
# Destination headroom slack at low S*, so greedy and LP coincide there; the
# source egress link (Λ_src·(D−τ)) is the binding ceiling at high S*.
POP = generate(POOL, n_nodes=16)
IMP = compute(POP, POOL)
EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()

DP = bind_dp(IMP)
ceiling = solve(POP, POOL, IMP, 2 * DP.sum(), EVENT, MOVE).shed_guaranteed  # finite unreachable → max shed
S = np.linspace(0.04, 1.15, 20) * ceiling

lp = [solve(POP, POOL, IMP, s, EVENT, MOVE) for s in S]
mi = [solve(POP, POOL, IMP, s, EVENT, MOVE, integer=True) for s in S]
gr = [greedy(POP, POOL, IMP, s, EVENT, MOVE) for s in S]

kW = 1e3
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# --- Panel A: achieved shed vs requested S* ---
axA.plot(S / kW, S / kW, "k--", lw=1, label="requested (y=x)")
axA.plot(S / kW, [p.shed_guaranteed / kW for p in lp], color="tab:blue", lw=2,
         label="guaranteed (binds ≥S*)")
axA.plot(S / kW, [p.shed_expected / kW for p in lp], color="tab:green", lw=2,
         label="expected (reported)")
axA.axvline(ceiling / kW, color="gray", lw=0.8, alpha=0.7)
axA.text(ceiling / kW, ceiling / kW * 0.25, " ceiling →\n shortfall", fontsize=8, color="gray")
axA.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="achieved shed (kW)",
        title="Sheds exactly $S^\\star$ until the movement ceiling")
axA.legend(loc="upper left", fontsize=8)

# --- Panel B: total downtime vs S*, three solvers ---
gcost, lcost = np.array([p.cost for p in gr]), np.array([p.cost for p in lp])
coincide = np.abs(gcost - lcost) < 1e-3 * np.maximum(lcost, 1)
bound = S[coincide].max() if coincide.any() else S[0]  # last S* where greedy=LP
axB.axvspan(0, bound / kW, color="tab:green", alpha=0.08)
axB.text(bound / kW / 2, lcost.max() * 0.5, "greedy = LP\n(no limit binds)", fontsize=8,
         color="green", ha="center", va="center")
axB.plot(S / kW, gcost, color="tab:orange", lw=2, label="greedy (resource-blind)")
axB.plot(S / kW, lcost, color="tab:blue", lw=2, ls="-", label="LP (fractional)")
axB.plot(S / kW, [p.cost for p in mi], color="tab:red", lw=1.6, ls="--", label="MILP (integer)")
axB.axvline(ceiling / kW, color="gray", lw=0.8, alpha=0.7)
axB.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="total downtime $\\Sigma\\,c$ (s)",
        title="greedy = LP off-boundary; MILP overshoot = relaxation gap")
axB.legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation.{ext}", dpi=150)

# mid-sweep sanity line (a feasible point) and the ceiling
k = np.searchsorted(S, 0.5 * ceiling)
print(f"regime={IMP.regime}  jobs={len(POP)}  ceiling={ceiling/kW:.1f} kW  "
      f"held_cap={EVENT.s_dest(POOL):.0f}")
print(f"@S*={S[k]/kW:.1f}kW  LP shed={lp[k].shed_guaranteed/kW:.2f}  exp={lp[k].shed_expected/kW:.2f}  "
      f"feasible={lp[k].feasible}  greedy cost={gr[k].cost:.1f}  LP cost={lp[k].cost:.1f}  MILP cost={mi[k].cost:.1f}")
print(f"infeasible tail: LP feasible={lp[-1].feasible}  shortfall={lp[-1].shortfall/kW:.1f}kW")
