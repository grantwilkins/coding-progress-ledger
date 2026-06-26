"""Execution validation for active-knee node-expected power."""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Plan
from impact import compute
from node_knee import (
    evaluate_node_expected_w,
    execution_realization_metrics,
    solve_active_knee_lp,
    solve_active_knee_milp,
)
from plot_node_knee_deadline_sweep import BASE_EVENT, DEADLINES, MOVE, N_NODES, STARTUP, population
from simulate import simulate

kW = 1e3
MODE = "sf"
TARGET_FRAC = 0.45
FIXED_PLAN_DEADLINE = BASE_EVENT.D
ORDERS = {
    "fifo": ("FIFO", "tab:orange"),
    "certified_pd": ("certified PD", "tab:blue"),
    "node_marginal_pd": ("node-marginal PD", "tab:green"),
}
VARIANTS = {
    "active-knee MILP": solve_active_knee_milp,
    "active-knee LP relaxation": solve_active_knee_lp,
}


def _plan(result, method):
    return Plan(result.y_R, result.y_S, result.active_floor_w, result.node_expected_w,
                result.cost, result.movement_feasible, result.expected_shortfall_w, "load", method)


def _hit(w, target):
    return bool(w >= target - 1e-6 * max(target, 1.0))


def _row(sweep, deadline, plan_deadline, variant, order, target, full_node, jobs, cost, metrics):
    row = {
        "sweep": sweep,
        "deadline_s": float(deadline),
        "plan_deadline_s": float(plan_deadline),
        "source_nodes": N_NODES,
        "jobs": int(jobs),
        "mode": MODE,
        "variant": variant,
        "ordering": order,
        "target_basis": "full_node_expected",
        "target_frac": TARGET_FRAC,
        "full_node_kw": full_node / kW,
        "target_kw": target / kW,
        "cost_s": cost,
    }
    for stage in ("selected", "egress_realized", "rebuild_realized"):
        node = metrics[f"{stage}_node_expected_w"]
        active = metrics[f"{stage}_active_floor_w"]
        row[f"{stage}_node_kw"] = node / kW
        row[f"{stage}_active_kw"] = active / kW
        row[f"{stage}_over_target"] = node / target
        row[f"{stage}_hit"] = _hit(node, target)
        row[f"{stage}_node_s_per_kw"] = metrics[f"{stage}_node_s_per_kw"]
        row[f"{stage}_active_s_per_kw"] = metrics[f"{stage}_active_s_per_kw"]
    return row


def _zero_metrics():
    return {f"{stage}_{metric}": 0.0 for stage in ("selected", "egress_realized", "rebuild_realized")
            for metric in ("node_expected_w", "active_floor_w")} | {
        f"{stage}_{metric}": np.nan for stage in ("selected", "egress_realized", "rebuild_realized")
        for metric in ("node_s_per_kw", "active_s_per_kw")
    }


def _target(pop, pool):
    full_node = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
    return TARGET_FRAC * full_node, full_node


def _rows_for_plan(sweep, pop, pool, imp, target, full_node, event, variant, result, plan_deadline):
    plan = _plan(result, variant)
    rows = []
    for order in ORDERS:
        sim = simulate(pop, pool, imp, plan, event, MOVE, MODE, order)
        metrics = execution_realization_metrics(pop, pool, imp, plan, sim, event.D)
        rows.append(_row(sweep, event.D, plan_deadline, variant, order, target, full_node,
                         len(pop), result.cost, metrics))
    return rows


def run_sweep(deadlines=DEADLINES):
    pool, pop = population()
    imp = compute(pop, pool)
    target, full_node = _target(pop, pool)
    rows = []
    for D in deadlines:
        event = replace(BASE_EVENT, D=float(D))
        for variant, solve_fn in VARIANTS.items():
            if D <= STARTUP:
                for order in ORDERS:
                    rows.append(_row("operational_resolve", D, D, variant, order, target, full_node,
                                     len(pop), np.nan, _zero_metrics()))
                continue
            result = solve_fn(pop, pool, imp, target, event, MOVE)
            rows.extend(_rows_for_plan("operational_resolve", pop, pool, imp, target, full_node,
                                       event, variant, result, event.D))
    return pop, target / kW, rows


def run_fixed_plan_sweep(deadlines=DEADLINES, plan_deadline=FIXED_PLAN_DEADLINE):
    pool, pop = population()
    imp = compute(pop, pool)
    target, full_node = _target(pop, pool)
    plan_event = replace(BASE_EVENT, D=float(plan_deadline))
    solved = {
        variant: solve_fn(pop, pool, imp, target, plan_event, MOVE)
        for variant, solve_fn in VARIANTS.items()
    }
    rows = []
    for D in deadlines:
        event = replace(BASE_EVENT, D=float(D))
        for variant, result in solved.items():
            rows.extend(_rows_for_plan("fixed_plan_replay", pop, pool, imp, target, full_node,
                                       event, variant, result, plan_event.D))
    return pop, target / kW, rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def plot(rows, path_base="outputs/node_knee_execution_validation",
         title="Operational active-knee execution validation"):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, variant in zip(axs, VARIANTS):
        base = [r for r in rows if r["variant"] == variant and r["ordering"] == "fifo"]
        D = np.array([r["deadline_s"] for r in base])
        ax.plot(D, [r["selected_over_target"] for r in base], color="0.35", ls="--", lw=2.2,
                label="selected")
        for order, (label, color) in ORDERS.items():
            rs = [r for r in rows if r["variant"] == variant and r["ordering"] == order]
            D = np.array([r["deadline_s"] for r in rs])
            ax.plot(D, [r["egress_realized_over_target"] for r in rs], color=color, lw=2,
                    label=f"{label} egress")
            ax.plot(D, [r["rebuild_realized_over_target"] for r in rs], color=color, lw=2, ls=":",
                    label=f"{label} rebuild")
        ax.axhline(1.0, color="0.2", ls=":", lw=1)
        ax.axvline(STARTUP, color="0.55", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set(title=variant, xlabel="deadline (seconds)")
        ax.grid(True, alpha=0.25)
    axs[0].set_ylabel("node-expected power / target")
    axs[0].legend(fontsize=7)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    pop, target_kw, rows = run_sweep()
    _, _, fixed_rows = run_fixed_plan_sweep()
    os.makedirs("outputs", exist_ok=True)
    write_csv(rows, "outputs/node_knee_execution_validation.csv")
    write_csv(fixed_rows, "outputs/node_knee_fixed_plan_replay.csv")
    plot(rows)
    plot(fixed_rows, "outputs/node_knee_fixed_plan_replay",
         f"Fixed-plan replay: plans solved at D={FIXED_PLAN_DEADLINE:.0f}s")
    print(f"target={target_kw:.1f} kW node-expected ({TARGET_FRAC:.0%} of full node model), jobs={len(pop)}")
    for label, table in (("operational_resolve", rows), ("fixed_plan_replay", fixed_rows)):
        print(label)
        for variant in VARIANTS:
            print(f"  {variant}")
            for order in ORDERS:
                rs = [r for r in table if r["variant"] == variant and r["ordering"] == order]
                sel = min((r["deadline_s"] for r in rs if r["selected_hit"]), default=np.nan)
                eg = min((r["deadline_s"] for r in rs if r["egress_realized_hit"]), default=np.nan)
                rb = min((r["deadline_s"] for r in rs if r["rebuild_realized_hit"]), default=np.nan)
                print(f"    {order:16s} selected={sel:6.1f}s egress={eg:6.1f}s rebuild={rb:6.1f}s")


if __name__ == "__main__":
    main()
