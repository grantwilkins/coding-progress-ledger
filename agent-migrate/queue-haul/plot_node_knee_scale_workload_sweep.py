"""Node-knee sweep over source size, workload class, deadline, and target."""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, solve
from impact import Movement, compute
from instance import SESSION_CLASSES, _mean_T, class_workload, generate
from node_knee import (
    _result,
    evaluate_node_expected_w,
    place_source_nodes,
    solve_active_knee_lp,
    solve_active_knee_milp,
    solve_live_greedy,
    solve_random_jobs,
    with_source_nodes,
)
from power import PoolPower

kW = 1e3
MOVE = Movement()
NODE_COUNTS = (1, 2, 4)
DEADLINES = (10.0, 30.0, 120.0)
TARGET_FRACS = (0.25, 0.45, 0.65)
BASE_EVENT = Event()
STARTUP = max(BASE_EVENT.tau_src, BASE_EVENT.tau_pre, BASE_EVENT.tau_in)
COLORS = {
    "additive LP": "0.45",
    "active-knee LP relaxation": "tab:green",
    "active-knee MILP": "tab:blue",
    "live greedy": "tab:orange",
    "random jobs": "tab:purple",
}


def scaled_event(n_nodes: int, deadline: float) -> Event:
    return replace(BASE_EVENT, D=float(deadline), dest_nodes=12 * int(n_nodes), W=4 * int(n_nodes))


def population(session_class: str, n_nodes: int, seed: int = 3):
    wl = class_workload(session_class, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=int(n_nodes), seed=seed)
    return pool, with_source_nodes(pop, place_source_nodes(pop, pool, int(n_nodes), "memory"))


def additive_result(pop, pool, imp, target, event):
    plan = solve(pop, pool, imp, target, event, MOVE)
    return _result(pop, pool, imp, plan.y_R, plan.y_S, plan.cost, "additive LP", target,
                   event, MOVE, plan.feasible)


def method_specs(pop, pool, imp, target):
    return (
        ("additive LP", lambda ev: additive_result(pop, pool, imp, target, ev)),
        ("active-knee LP relaxation", lambda ev: solve_active_knee_lp(pop, pool, imp, target, ev, MOVE)),
        ("active-knee MILP", lambda ev: solve_active_knee_milp(pop, pool, imp, target, ev, MOVE)),
        ("live greedy", lambda ev: solve_live_greedy(pop, pool, imp, target, ev, MOVE)),
        ("random jobs", lambda ev: solve_random_jobs(pop, pool, imp, target, ev, MOVE, seed=0)),
    )


def _row(session_class, n_nodes, jobs, full_kw, deadline, frac, target_kw, event, method, result=None):
    hit = bool(result and result.true_expected_feasible)
    node_kw = 0.0 if result is None else result.node_expected_w / kW
    cost = np.nan if not hit else result.cost
    return {
        "session_class": session_class,
        "source_nodes": int(n_nodes),
        "jobs": int(jobs),
        "deadline_s": float(deadline),
        "target_frac": float(frac),
        "target_kw": float(target_kw),
        "full_node_kw": float(full_kw),
        "dest_nodes": int(event.dest_nodes),
        "workers": int(event.W),
        "method": method,
        "hit": hit,
        "node_kw": float(node_kw),
        "active_kw": 0.0 if result is None else result.active_floor_w / kW,
        "cost_s": float(cost),
        "intensity_s_per_kw": float(cost / node_kw) if hit and node_kw > 0 else np.nan,
    }


def run_sweep(workloads=SESSION_CLASSES, nodes=NODE_COUNTS, deadlines=DEADLINES, target_fracs=TARGET_FRACS):
    rows = []
    for session_class in workloads:
        for n_nodes in nodes:
            pool, pop = population(session_class, int(n_nodes))
            imp = compute(pop, pool)
            full_w = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
            for frac in target_fracs:
                target = float(frac * full_w)
                for deadline in deadlines:
                    event = scaled_event(int(n_nodes), float(deadline))
                    for method, fn in method_specs(pop, pool, imp, target):
                        result = None if deadline <= STARTUP else fn(event)
                        rows.append(_row(session_class, n_nodes, len(pop), full_w / kW, deadline,
                                         frac, target / kW, event, method, result))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _median(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def plot(rows, path_base="outputs/node_knee_scale_workload_sweep"):
    workloads = list(dict.fromkeys(r["session_class"] for r in rows))
    methods = list(COLORS)
    nodes = sorted({r["source_nodes"] for r in rows})
    fig, axs = plt.subplots(2, len(workloads), figsize=(3.9 * len(workloads), 6.2), sharex=True, squeeze=False)
    for col, cls in enumerate(workloads):
        for method in methods:
            hit, intensity = [], []
            for n in nodes:
                rs = [r for r in rows if r["session_class"] == cls and r["method"] == method and r["source_nodes"] == n]
                hit.append(np.mean([r["hit"] for r in rs]))
                intensity.append(_median([r["intensity_s_per_kw"] for r in rs]))
            axs[0, col].plot(nodes, hit, marker="o", lw=1.8, ms=4, color=COLORS[method], label=method)
            axs[1, col].plot(nodes, intensity, marker="o", lw=1.8, ms=4, color=COLORS[method])
        axs[0, col].set(title=cls.replace("_", " "), ylim=(-0.03, 1.03))
        axs[1, col].set_yscale("log")
        axs[1, col].set_xlabel("source nodes")
        for ax in axs[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(nodes)
            ax.set_xticklabels([str(n) for n in nodes])
            ax.grid(True, alpha=0.25)
    axs[0, 0].set_ylabel("target hit rate")
    axs[1, 0].set_ylabel("median disruption intensity (s/kW)")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Node-knee dispatch over deadlines and modeled power targets", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    rows = run_sweep()
    os.makedirs("outputs", exist_ok=True)
    write_csv(rows, "outputs/node_knee_scale_workload_sweep.csv")
    plot(rows)
    print(f"rows={len(rows)} configs={len(rows) // len(COLORS)}")
    for cls in SESSION_CLASSES:
        print(cls)
        for method in COLORS:
            rs = [r for r in rows if r["session_class"] == cls and r["method"] == method]
            print(f"  {method:17s} hit={np.mean([r['hit'] for r in rs]):.2f} "
                  f"median_intensity={_median([r['intensity_s_per_kw'] for r in rs]):7.1f}s/kW")


if __name__ == "__main__":
    main()
