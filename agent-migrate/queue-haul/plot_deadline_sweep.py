"""Deadline sweep by workload class.

For each class, keep the population fixed and vary only the event deadline.
The reported value is the max shed ceiling under that deadline for decentralized
greedy, integer coordination, and fractional LP coordination.
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

BASE_EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
N_NODES, kW = 4, 1e3
STARTUP = max(BASE_EVENT.tau_src, BASE_EVENT.tau_pre, BASE_EVENT.tau_in)
DEADLINES = np.unique(np.concatenate(([1.0, STARTUP], np.linspace(5.5, 30, 22), np.linspace(40, 300, 14))))
CASES = (
    ("ordinary chat", "ordinary_chat"),
    ("long chat / code", "long_chat_code"),
    ("reasoning chat", "reasoning_chat"),
    ("agentic tool loop", "agentic_tool_loop"),
)


def ceilings(cls):
    wl = class_workload(cls, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    imp = compute(pop, pool)
    target = 2 * bind_dp(imp).sum()
    gr, mi, lp = [], [], []
    for D in DEADLINES:
        if D <= STARTUP:
            gr.append(0.0)
            mi.append(0.0)
            lp.append(0.0)
        else:
            ev = replace(BASE_EVENT, D=float(D))
            gr.append(greedy(pop, pool, imp, target, ev, MOVE).shed_guaranteed / kW)
            mi.append(solve(pop, pool, imp, target, ev, MOVE, integer=True).shed_guaranteed / kW)
            lp.append(solve(pop, pool, imp, target, ev, MOVE).shed_guaranteed / kW)
    return pop, imp, np.array(gr), np.array(mi), np.array(lp)


results = [(label, *ceilings(cls)) for label, cls in CASES]

fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=False)
for ax, (label, pop, imp, gr, mi, lp) in zip(axs.ravel(), results):
    gap = lp - gr
    ax.plot(DEADLINES, lp, color="tab:blue", lw=2.2, label="LP")
    ax.plot(DEADLINES, mi, color="tab:green", lw=1.8, ls="-.", label="MILP")
    ax.plot(DEADLINES, gr, color="tab:orange", lw=2.0, ls="--", label="greedy")
    ax.fill_between(DEADLINES, gr, lp, where=gap > 0.05, color="tab:blue", alpha=0.14)
    ax.axvline(STARTUP, color="0.5", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set(title=f"{label}: {imp.regime}, {len(pop)} jobs",
           xlabel="deadline (seconds)", ylabel="max shed (kW)")
    if gap.max() > 0.5:
        D = DEADLINES[gap.argmax()]
        ax.text(D, lp.max() * 0.55, f"max gap {gap.max():.1f} kW", fontsize=8, ha="center", color="0.25")
    else:
        ax.text(20, lp.max() * 0.55, "greedy=LP", fontsize=8, ha="center", color="0.25")
axs[0, 0].legend(loc="lower right", fontsize=8)
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/deadline_sweep.{ext}", dpi=150)

for label, _, _, gr, mi, lp in results:
    plateau = lp.max()
    knee = DEADLINES[lp >= 0.99 * plateau][0] if plateau > 0 else np.nan
    gap = lp - gr
    igap = mi - gr
    print(f"{label:16s} plateau={plateau:5.1f} kW by D≈{knee:5.1f}s; "
          f"max LP-greedy gap={gap.max():4.1f} kW at D={DEADLINES[gap.argmax()]:.1f}s; "
          f"max MILP-greedy gap={igap.max():4.1f} kW")
