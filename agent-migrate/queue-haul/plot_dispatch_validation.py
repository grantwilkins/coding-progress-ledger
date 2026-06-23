"""Dispatch validation by workload class.

Each panel isolates one active, cache-resident session class. Integer greedy and
LP see the same movement budgets. The y-axis is aggregate downtime normalized by
the requested certified shed, i.e. seconds of disruption per kW. The console
diagnostics distinguish resource-bound coordination gaps from fractional LP
granularity gaps.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dispatch import Event, bind_dp, dispatch_diagnostics, greedy, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
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
    S = np.linspace(0.04, 0.98, 20) * lp_ceil
    lp = [solve(pop, pool, imp, s, EVENT, MOVE) for s in S]
    gr = [greedy(pop, pool, imp, s, EVENT, MOVE) for s in S]
    return label, pool, pop, imp, S, lp, gr


results = [case(*c) for c in CASES]

kW = 1e3
fig, axs = plt.subplots(2, 2, figsize=(11, 7.8), sharex=False, sharey=False)
for ax, (label, pool, pop, imp, S, lp, gr) in zip(axs.ravel(), results):
    lp_cost = np.array([p.cost if p.feasible else np.nan for p in lp])
    gr_cost = np.array([p.cost if p.feasible else np.nan for p in gr])
    lp_norm, gr_norm = lp_cost / (S / kW), gr_cost / (S / kW)
    save = np.divide(
        gr_norm - lp_norm, gr_norm, out=np.zeros_like(gr_norm), where=gr_norm > 0
    )
    ax.plot(S / kW, gr_norm, color="tab:orange", lw=2, label="integer greedy")
    ax.plot(S / kW, lp_norm, color="tab:blue", lw=2, label="LP")
    if np.nanmax(save) > 0.01:
        i = int(np.nanargmax(save))
        ax.annotate(
            f"LP cuts disruption {100 * save[i]:.0f}%",
            xy=(S[i] / kW, lp_norm[i]),
            xytext=(S[i] / kW, 0.72 * gr_norm[i]),
            arrowprops=dict(arrowstyle="->", color="0.3"),
            fontsize=8,
            ha="center",
        )
    else:
        ax.text(
            0.60 * S.max() / kW,
            0.55 * np.nanmax(gr_norm),
            "same sorted plan",
            fontsize=8,
            ha="center",
            color="0.3",
        )
    ax.set(
        title=f"{label}: {len(pop)} jobs",
        xlabel="requested shed $S^\\star$ (kW)",
        ylabel="disruption intensity (s/kW)",
    )
axs[0, 0].legend(loc="upper left", fontsize=8)

fig.subplots_adjust(
    left=0.08, right=0.98, bottom=0.08, top=0.94, wspace=0.28, hspace=0.38
)
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation.{ext}", dpi=150)


def fmt(d):
    return " ".join(f"{k}={v:.2g}" for k, v in d.items())


for label, pool, pop, imp, S, lp, gr in results:
    lp_cost = np.array([p.cost for p in lp])
    gr_cost = np.array([p.cost for p in gr])
    lp_norm, gr_norm = lp_cost / (S / kW), gr_cost / (S / kW)
    save = np.divide(
        gr_norm - lp_norm, gr_norm, out=np.zeros_like(gr_norm), where=gr_norm > 0
    )
    i = int(np.nanargmax(save))
    diag = dispatch_diagnostics(pop, pool, imp, lp[i], S[i], EVENT, MOVE)
    print(
        f"{label:16s} regime={imp.regime:6s} jobs={len(pop):4d} "
        f"max disruption cut={100 * save[i]:4.1f}% at S*={S[i]/kW:4.1f} kW "
        f"({gr_norm[i]:.1f}->{lp_norm[i]:.1f} s/kW)"
    )
    print(
        f"  active={','.join(diag['active_constraints']) or 'none'} "
        f"duals[{fmt(diag['duals'])}] frac_vars={diag['fractional_variables']} "
        f"max_dp/S*={diag['max_dp_over_s']:.3g}"
    )
    print(f"  max_draw/budget[{fmt(diag['max_resource_draw_over_budget'])}]")
    print(f"  spearman(dp,*)[{fmt(diag['spearman'])}]")
