"""Agentic requested-shed disruption sweep with deterministic replay."""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Plan, solve
from node_knee import (
    _result,
    evaluate_node_expected_w,
    execution_realization_metrics,
    solve_active_knee_lp,
    solve_active_knee_milp,
    solve_live_greedy,
    solve_random_jobs,
)
from plot_node_knee_target_sweep import EVENT, MOVE, N_NODES, TARGET_FRACS, population
from impact import compute
from simulate import simulate

kW = 1e3
SESSION_CLASS = "agentic_tool_loop"
MODE = "sf"
ORDERING = "node_marginal_pd"
STAGES = {
    "selected": "selected plan",
    "egress_realized": "DES egress by D",
    "rebuild_realized": "DES rebuild by D",
}
COLORS = {
    "additive LP": "0.45",
    "active-knee LP relaxation": "tab:green",
    "active-knee MILP": "tab:blue",
    "live greedy": "tab:orange",
    "random jobs": "tab:purple",
}


def _plan(result, method):
    return Plan(result.y_R, result.y_S, result.active_floor_w, result.node_expected_w,
                result.cost, result.movement_feasible, result.expected_shortfall_w, "load", method)


def _additive(pop, pool, imp, target):
    plan = solve(pop, pool, imp, target, EVENT, MOVE)
    res = _result(pop, pool, imp, plan.y_R, plan.y_S, plan.cost, "additive LP", target,
                  EVENT, MOVE, plan.feasible)
    return plan, res


def _wrapped(result, method):
    return _plan(result, method), result


def method_specs(pop, pool, imp, target):
    return (
        ("additive LP", lambda: _additive(pop, pool, imp, target)),
        ("active-knee LP relaxation", lambda: _wrapped(
            solve_active_knee_lp(pop, pool, imp, target, EVENT, MOVE), "active-knee LP relaxation"
        )),
        ("active-knee MILP", lambda: _wrapped(
            solve_active_knee_milp(pop, pool, imp, target, EVENT, MOVE), "active-knee MILP"
        )),
        ("live greedy", lambda: _wrapped(
            solve_live_greedy(pop, pool, imp, target, EVENT, MOVE), "live greedy"
        )),
        ("random jobs", lambda: _wrapped(
            solve_random_jobs(pop, pool, imp, target, EVENT, MOVE, seed=0), "random jobs"
        )),
    )


def _hit(w, target):
    return bool(w >= target - 1e-6 * max(target, 1.0))


def _row(frac, target, full_node, method, plan, metrics):
    row = {
        "session_class": SESSION_CLASS,
        "source_nodes": N_NODES,
        "jobs": len(plan.y_R),
        "deadline_s": EVENT.D,
        "mode": MODE,
        "ordering": ORDERING,
        "target_basis": "full_node_expected",
        "target_frac": float(frac),
        "full_node_kw": full_node / kW,
        "target_kw": target / kW,
        "method": method,
        "cost_s": float(plan.cost),
    }
    for stage in STAGES:
        node = metrics[f"{stage}_node_expected_w"]
        active = metrics[f"{stage}_active_floor_w"]
        hit = _hit(node, target)
        row[f"{stage}_node_kw"] = node / kW
        row[f"{stage}_active_kw"] = active / kW
        row[f"{stage}_over_target"] = node / target
        row[f"{stage}_hit"] = hit
        row[f"{stage}_requested_s_per_kw"] = plan.cost / (target / kW) if hit else np.nan
        row[f"{stage}_delivered_s_per_kw"] = plan.cost / (node / kW) if node > 0 else np.nan
    return row


def run_sweep(target_fracs=TARGET_FRACS):
    pool, pop = population(SESSION_CLASS)
    imp = compute(pop, pool)
    full_node = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
    rows = []
    for frac in target_fracs:
        target = float(frac * full_node)
        for method, fn in method_specs(pop, pool, imp, target):
            plan, _ = fn()
            sim = simulate(pop, pool, imp, plan, EVENT, MOVE, MODE, ORDERING)
            metrics = execution_realization_metrics(pop, pool, imp, plan, sim, EVENT.D)
            rows.append(_row(frac, target, full_node, method, plan, metrics))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def plot(rows, path_base="outputs/node_knee_agentic_des_sweep"):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (stage, title) in zip(axs, STAGES.items()):
        for method, color in COLORS.items():
            rs = [r for r in rows if r["method"] == method]
            ax.plot([r["target_kw"] for r in rs], [r[f"{stage}_requested_s_per_kw"] for r in rs],
                    marker="o", lw=2, ms=3.5, color=color, label=method)
        ax.set(title=title, xlabel="requested node-expected shed (kW)", yscale="log")
        ax.grid(True, alpha=0.25)
    axs[0].set_ylabel("disruption / requested kW (s/kW)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Agentic requested-shed disruption with DES replay (D={EVENT.D:.0f}s)", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    rows = run_sweep()
    os.makedirs("outputs", exist_ok=True)
    write_csv(rows, "outputs/node_knee_agentic_des_sweep.csv")
    plot(rows)
    print(f"rows={len(rows)} target_points={len(set(r['target_kw'] for r in rows))} "
          f"deadline={EVENT.D:.0f}s ordering={ORDERING}")


if __name__ == "__main__":
    main()
