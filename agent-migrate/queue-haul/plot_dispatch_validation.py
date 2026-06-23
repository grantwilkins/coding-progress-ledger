"""Dispatch validation by workload class.

Each panel isolates one active, cache-resident session class. The policies see
the same movement budgets: random, decentralized greedy, and coordinated LP.
Ceilings separate because selection and action repacking fit different amounts
of shed under the same shared link.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, greedy, random_dispatch, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
SEEDS = range(4)
N_NODES = 4
CASES = (
    ("ordinary chat", "ordinary_chat"),
    ("long chat / code", "long_chat_code"),
    ("reasoning chat", "reasoning_chat"),
    ("agentic tool loop", "agentic_tool_loop"),
)


def case(label, cls):
    wl = class_workload(cls, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    imp = compute(pop, pool)
    dp = bind_dp(imp)
    lp_ceil = solve(pop, pool, imp, 2 * dp.sum(), EVENT, MOVE).shed_guaranteed
    S = np.linspace(0.04, 1.15, 18) * lp_ceil
    lp = [solve(pop, pool, imp, s, EVENT, MOVE) for s in S]
    gr = [greedy(pop, pool, imp, s, EVENT, MOVE) for s in S]
    rd = [[random_dispatch(pop, pool, imp, s, EVENT, MOVE, seed=sd) for sd in SEEDS] for s in S]
    return label, pop, imp, S, lp, gr, np.array([[p.shed_guaranteed for p in row] for row in rd]) / 1e3


results = [case(*c) for c in CASES]

kW = 1e3
fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), sharex=False, sharey=False)
for ax, (label, pop, imp, S, lp, gr, r_shed) in zip(axs.ravel(), results):
    lp_ceil = max(p.shed_guaranteed for p in lp)
    g_ceil = max(p.shed_guaranteed for p in gr)
    ax.plot(S / kW, S / kW, "k--", lw=1, label="requested")
    ax.fill_between(S / kW, r_shed.min(1), r_shed.max(1), color="tab:gray", alpha=0.2)
    ax.plot(S / kW, r_shed.mean(1), color="tab:gray", lw=2, label="random")
    ax.plot(S / kW, [p.shed_guaranteed / kW for p in gr], color="tab:orange", lw=2, label="greedy")
    ax.plot(S / kW, [p.shed_guaranteed / kW for p in lp], color="tab:blue", lw=2, label="LP")
    xg = 0.92 * S.max() / kW
    if lp_ceil - g_ceil > 0.5 * kW:
        ax.annotate("", xy=(xg, lp_ceil / kW), xytext=(xg, g_ceil / kW),
                    arrowprops=dict(arrowstyle="<->", color="0.3"))
        ax.text(xg, (g_ceil + lp_ceil) / (2 * kW), "gap", fontsize=8, ha="center", va="center")
    else:
        ax.text(xg, lp_ceil / kW, "greedy=LP", fontsize=8, ha="right", va="bottom", color="0.3")
    ax.set(title=f"{label}: {imp.regime}, {len(pop)} jobs",
           xlabel="requested shed $S^\\star$ (kW)", ylabel="achieved shed (kW)")
axs[0, 0].legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation.{ext}", dpi=150)

for label, pop, imp, _, lp, gr, r_shed in results:
    print(f"{label:16s} regime={imp.regime:6s} jobs={len(pop):4d} ceilings kW: "
          f"random={r_shed.max():5.1f} greedy={max(p.shed_guaranteed for p in gr)/kW:5.1f} "
          f"LP={max(p.shed_guaranteed for p in lp)/kW:5.1f}")
