"""Deadline sweep for node-knee expected power.

Fixed active-agentic source population, explicit memory-first source placement, and
a fixed modeled node-expected shed target. The old additive LP is shown as the
baseline: it solves the old active-floor target, then we evaluate its node-expected
power on the same source placement.
"""

from __future__ import annotations

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from node_knee import (
    _result,
    place_source_nodes,
    solve_active_knee_lp,
    solve_active_knee_milp,
    solve_live_greedy,
    solve_random_jobs,
    with_source_nodes,
)
from power import PoolPower

BASE_EVENT = Event(dest_nodes=48)
MOVE = Movement()
N_NODES, kW = 4, 1e3
STARTUP = max(BASE_EVENT.tau_src, BASE_EVENT.tau_pre, BASE_EVENT.tau_in)
DEADLINES = np.unique(np.concatenate(([1.0, STARTUP], np.linspace(6, 30, 13), np.linspace(45, 300, 12))))


def population(policy: str = "memory"):
    wl = class_workload("agentic_tool_loop", state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES, seed=3)
    return pool, with_source_nodes(pop, place_source_nodes(pop, pool, N_NODES, policy))


def additive_result(pop, pool, imp, target, event):
    plan = solve(pop, pool, imp, target, event, MOVE)
    return _result(pop, pool, imp, plan.y_R, plan.y_S, plan.cost, "additive LP", target,
                   event, MOVE, plan.feasible)


def run_sweep(deadlines=DEADLINES):
    pool, pop = population()
    imp = compute(pop, pool)
    target = 0.45 * imp.dp_certified.sum()
    methods = (
        ("additive LP", lambda ev: additive_result(pop, pool, imp, target, ev), "0.45"),
        ("active-knee LP relaxation", lambda ev: solve_active_knee_lp(pop, pool, imp, target, ev, MOVE), "tab:green"),
        ("active-knee MILP", lambda ev: solve_active_knee_milp(pop, pool, imp, target, ev, MOVE), "tab:blue"),
        ("live greedy", lambda ev: solve_live_greedy(pop, pool, imp, target, ev, MOVE), "tab:orange"),
        ("random jobs", lambda ev: solve_random_jobs(pop, pool, imp, target, ev, MOVE, seed=0), "tab:purple"),
    )
    rows = []
    for D in deadlines:
        for name, fn, color in methods:
            if D <= STARTUP:
                rows.append({"deadline": D, "method": name, "color": color, "node_kw": 0.0,
                             "active_kw": 0.0, "cost_s": np.nan, "hit": False})
                continue
            r = fn(replace(BASE_EVENT, D=float(D)))
            rows.append({"deadline": D, "method": name, "color": color,
                         "node_kw": r.node_expected_w / kW, "active_kw": r.active_floor_w / kW,
                         "cost_s": r.cost if r.true_expected_feasible else np.nan,
                         "hit": r.true_expected_feasible})
    return pop, target / kW, rows


def main():
    pop, target_kw, rows = run_sweep()
    methods = list(dict.fromkeys(r["method"] for r in rows))
    colors = {r["method"]: r["color"] for r in rows}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        D = np.array([r["deadline"] for r in rs])
        node = np.array([r["node_kw"] for r in rs])
        cost = np.array([r["cost_s"] for r in rs], float)
        ax1.plot(D, node, lw=2, label=method, color=colors[method])
        ax2.plot(D, cost / np.maximum(node, 1e-12), lw=2, label=method, color=colors[method])
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(STARTUP, color="0.5", ls=":", lw=1)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("deadline (seconds)")
    ax1.axhline(target_kw, color="0.2", ls="--", lw=1, label=f"target {target_kw:.1f} kW")
    ax1.set(ylabel="modeled node-expected shed (kW)", title=f"A. Power outcome, {len(pop)} jobs")
    ax2.set(ylabel="disruption intensity (s/kW node-expected)", title="B. Cost normalized by modeled shed")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"outputs/node_knee_deadline_sweep.{ext}", dpi=150)

    print(f"target={target_kw:.1f} kW node-expected, jobs={len(pop)}")
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        hit = [r for r in rs if r["hit"]]
        first = min((r["deadline"] for r in hit), default=np.nan)
        best = max(r["node_kw"] for r in rs)
        min_intensity = min((r["cost_s"] / r["node_kw"] for r in hit), default=np.nan)
        print(f"{method:17s} first_hit_D={first:6.1f}s  max_node={best:6.1f} kW  min_intensity={min_intensity:7.1f}s/kW")


if __name__ == "__main__":
    main()
