"""Agentic dispatch scale check.

Scale source nodes and report the best LP reduction in disruption intensity
(seconds per certified kW) over a common target sweep. Fixed destination capacity
tests the current small-fixture limit; scaled destination capacity preserves the
4-node fixture ratio as the source grows.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, greedy, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

NODES = np.array([4, 8, 16, 32, 64, 128])
FRACS = np.linspace(0.30, 0.95, 12)
MOVE = Movement()
WL = class_workload("agentic_tool_loop", state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
POOL = replace(PoolPower(), mean_context_tokens=_mean_T(WL))


def scan(n_nodes, event):
    pop = generate(POOL, WL, n_nodes=int(n_nodes))
    imp = compute(pop, POOL)
    target = 2 * bind_dp(imp).sum()
    ceil = solve(pop, POOL, imp, target, event, MOVE).shed_guaranteed
    common = min(ceil, greedy(pop, POOL, imp, target, event, MOVE).shed_guaranteed)
    best = None
    for frac in FRACS:
        s_star = frac * common
        lp = solve(pop, POOL, imp, s_star, event, MOVE)
        gr = greedy(pop, POOL, imp, s_star, event, MOVE)
        if not (lp.feasible and gr.feasible):
            continue
        lp_norm, gr_norm = lp.cost / (s_star / 1e3), gr.cost / (s_star / 1e3)
        cut = (gr_norm - lp_norm) / gr_norm if gr_norm else 0.0
        if best is None or cut > best[0]:
            best = (cut, s_star / 1e3, gr_norm, lp_norm)
    return len(pop), ceil / 1e3, *best


def fixed_event(n_nodes):
    return Event(dest_nodes=48, W=16)


def scaled_event(n_nodes):
    return Event(dest_nodes=12 * int(n_nodes), W=4 * int(n_nodes))


series = {
    "fixed destination": [scan(n, fixed_event(n)) for n in NODES],
    "scaled destination": [scan(n, scaled_event(n)) for n in NODES],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for label, rows in series.items():
    rows = np.array(rows)
    ax1.plot(NODES, 100 * rows[:, 2], marker="o", lw=2, label=label)
    ax2.plot(NODES, rows[:, 1], marker="o", lw=2, label=label)

for ax in (ax1, ax2):
    ax.set_xscale("log", base=2)
    ax.set_xticks(NODES)
    ax.set_xticklabels([str(n) for n in NODES])
    ax.set_xlabel("source nodes")
    ax.grid(True, alpha=0.25)

ax1.set(ylabel="best LP disruption cut (%)", title="A. Coordination value over target sweep")
ax2.set(ylabel="LP shed ceiling (kW)", title="B. Event size implied by destination capacity")
ax1.legend(fontsize=8)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88, wspace=0.28)

os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_scale.{ext}", dpi=150)

for label, rows in series.items():
    print(label)
    for n, row in zip(NODES, rows):
        jobs, ceil, cut, target_kw, gr_norm, lp_norm = row
        print(f"  source_nodes={n:3d} jobs={jobs:4.0f} ceiling={ceil:6.1f} kW "
              f"best_cut={100 * cut:5.1f}% at S*={target_kw:6.1f} kW "
              f"({gr_norm:5.1f}->{lp_norm:5.1f} s/kW)")
