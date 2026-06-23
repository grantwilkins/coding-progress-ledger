"""128-node agentic dispatch validation.

Uses the 128-source-node agentic fixture from plot_dispatch_scale.py and draws
plot_dispatch_validation.py-style disruption-intensity curves for greedy, MILP,
and LP under both scale events.
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

N_NODES = 128
FRACS = np.linspace(0.04, 0.98, 20)
MOVE = Movement()


def population():
    wl = class_workload("agentic_tool_loop", state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    return pool, pop, compute(pop, pool)


def fixed_event():
    return Event(dest_nodes=48, W=16)


def scaled_event():
    return Event(dest_nodes=12 * N_NODES, W=4 * N_NODES)


def run_case(pool, pop, imp, event, fracs=FRACS):
    target = 2 * bind_dp(imp).sum()
    ceilings = {
        "integer greedy": greedy(pop, pool, imp, target, event, MOVE).shed_guaranteed,
        "MILP": solve(pop, pool, imp, target, event, MOVE, integer=True).shed_guaranteed,
        "LP": solve(pop, pool, imp, target, event, MOVE).shed_guaranteed,
    }
    S = np.asarray(fracs) * min(ceilings.values())
    plans = {
        "integer greedy": [greedy(pop, pool, imp, s, event, MOVE) for s in S],
        "MILP": [solve(pop, pool, imp, s, event, MOVE, integer=True) for s in S],
        "LP": [solve(pop, pool, imp, s, event, MOVE) for s in S],
    }
    return S, plans, ceilings


def norm_cost(plans, S):
    return {k: np.array([p.cost if p.feasible else np.nan for p in v]) / (S / 1e3) for k, v in plans.items()}


def main():
    pool, pop, imp = population()
    cases = (("fixed destination", fixed_event()), ("scaled destination", scaled_event()))
    results = [(label, *run_case(pool, pop, imp, event)) for label, event in cases]

    styles = {
        "integer greedy": dict(color="tab:orange", ls="-", marker="o"),
        "MILP": dict(color="tab:green", ls="--", marker="s"),
        "LP": dict(color="tab:blue", ls="-", marker=None),
    }
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (label, S, plans, ceilings) in zip(axs, results):
        costs = norm_cost(plans, S)
        for name in ("integer greedy", "MILP", "LP"):
            ax.plot(S / 1e3, costs[name], lw=2, ms=3, markevery=4, label=name, **styles[name])
        ax.set(
            title=f"{label}: {len(pop)} jobs",
            xlabel="requested shed $S^\\star$ (kW)",
            ylabel="disruption intensity (s/kW)",
        )
        ax.grid(True, alpha=0.25)
        print(f"{label}: common ceiling={min(ceilings.values())/1e3:.1f} kW")
        for name in ("integer greedy", "MILP", "LP"):
            vals = costs[name]
            print(f"  {name:15s} ceiling={ceilings[name]/1e3:6.1f} kW best={np.nanmin(vals):5.2f} s/kW worst={np.nanmax(vals):5.2f} s/kW")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88, wspace=0.28)

    os.makedirs("outputs", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"outputs/dispatch_128_validation.{ext}", dpi=150)


if __name__ == "__main__":
    main()
