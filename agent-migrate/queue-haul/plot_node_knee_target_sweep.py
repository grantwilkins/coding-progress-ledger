"""Fixed-deadline 4-node node-knee target sweep."""

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
    solve_live_greedy,
    solve_random_jobs,
    with_source_nodes,
)
from power import PoolPower

kW = 1e3
N_NODES = 4
EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
TARGET_FRACS = np.linspace(0.05, 0.95, 19)
COLORS = {
    "additive LP": "0.45",
    "active-knee LP": "tab:green",
    "live greedy": "tab:orange",
    "random jobs": "tab:purple",
}


def population(session_class: str, seed: int = 42):
    wl = class_workload(session_class, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES, seed=seed)
    return pool, with_source_nodes(pop, place_source_nodes(pop, pool, N_NODES, "memory"))


def additive_result(pop, pool, imp, target):
    plan = solve(pop, pool, imp, target, EVENT, MOVE)
    return _result(pop, pool, imp, plan.y_R, plan.y_S, plan.cost, "additive LP", target)


def method_specs(pop, pool, imp, target):
    return (
        ("additive LP", lambda: additive_result(pop, pool, imp, target)),
        ("active-knee LP", lambda: solve_active_knee_lp(pop, pool, imp, target, EVENT, MOVE)),
        ("live greedy", lambda: solve_live_greedy(pop, pool, imp, target, EVENT, MOVE)),
        ("random jobs", lambda: solve_random_jobs(pop, pool, imp, target, EVENT, MOVE, seed=0)),
    )


def _row(session_class, jobs, full_kw, frac, target_kw, method, result):
    hit = result.true_expected_feasible
    node_kw = result.node_expected_w / kW
    cost = result.cost if hit else np.nan
    return {
        "session_class": session_class,
        "source_nodes": N_NODES,
        "jobs": int(jobs),
        "deadline_s": float(EVENT.D),
        "target_frac": float(frac),
        "target_kw": float(target_kw),
        "full_node_kw": float(full_kw),
        "method": method,
        "hit": bool(hit),
        "node_kw": float(node_kw),
        "achieved_over_target": float(node_kw / target_kw),
        "active_kw": float(result.active_floor_w / kW),
        "cost_s": float(cost),
        "intensity_s_per_kw": float(cost / node_kw) if hit and node_kw > 0 else np.nan,
        "requested_intensity_s_per_kw": float(cost / target_kw) if hit else np.nan,
    }


def run_sweep(workloads=SESSION_CLASSES, target_fracs=TARGET_FRACS):
    rows = []
    for session_class in workloads:
        pool, pop = population(session_class)
        imp = compute(pop, pool)
        full_w = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
        for frac in target_fracs:
            target = float(frac * full_w)
            for method, fn in method_specs(pop, pool, imp, target):
                rows.append(_row(session_class, len(pop), full_w / kW, frac, target / kW, method, fn()))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _median(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def plot(rows, path_base="outputs/node_knee_target_sweep"):
    workloads = list(dict.fromkeys(r["session_class"] for r in rows))
    fig, axs = plt.subplots(3, len(workloads), figsize=(3.9 * len(workloads), 8.3), sharex=False, squeeze=False)
    for col, cls in enumerate(workloads):
        for method, color in COLORS.items():
            rs = [r for r in rows if r["session_class"] == cls and r["method"] == method]
            x = np.array([r["target_kw"] for r in rs])
            ratio = np.array([r["achieved_over_target"] for r in rs])
            intensity = np.array([r["intensity_s_per_kw"] for r in rs], float)
            requested = np.array([r["requested_intensity_s_per_kw"] for r in rs], float)
            axs[0, col].plot(x, ratio, marker="o", lw=1.8, ms=3.5, color=color, label=method)
            axs[1, col].plot(x, intensity, marker="o", lw=1.8, ms=3.5, color=color)
            axs[2, col].plot(x, requested, marker="o", lw=1.8, ms=3.5, color=color)
        axs[0, col].axhline(1.0, color="0.2", ls="--", lw=1)
        axs[0, col].set(title=cls.replace("_", " "), ylim=(0, None))
        axs[1, col].set_yscale("log")
        axs[2, col].set_yscale("log")
        axs[2, col].set_xlabel("requested modeled shed (kW)")
        for ax in axs[:, col]:
            ax.grid(True, alpha=0.25)
    axs[0, 0].set_ylabel("modeled shed / request")
    axs[1, 0].set_ylabel("cost / achieved kW")
    axs[2, 0].set_ylabel("cost / requested kW")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"4-node fixed-deadline node-knee target sweep (D={EVENT.D:.0f}s)", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    rows = run_sweep()
    os.makedirs("outputs", exist_ok=True)
    write_csv(rows, "outputs/node_knee_target_sweep.csv")
    plot(rows)
    print(f"rows={len(rows)} configs={len(rows) // len(COLORS)} deadline={EVENT.D:.0f}s source_nodes={N_NODES}")
    for cls in SESSION_CLASSES:
        print(cls)
        for method in COLORS:
            rs = [r for r in rows if r["session_class"] == cls and r["method"] == method]
            hit = [r for r in rs if r["hit"]]
            max_hit = max((r["target_kw"] for r in hit), default=np.nan)
            print(f"  {method:14s} hit={np.mean([r['hit'] for r in rs]):.2f} "
                  f"max_hit={max_hit:6.1f} kW median_intensity={_median([r['intensity_s_per_kw'] for r in rs]):7.1f}s/kW")


if __name__ == "__main__":
    main()
