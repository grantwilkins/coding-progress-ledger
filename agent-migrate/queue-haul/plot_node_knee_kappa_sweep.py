"""Planner rebuild-cushion sensitivity: solve with kappa-derated prefill/ingest rows,
replay each plan in the DES (interference alpha_in at its default), and report realized
relief and per-session rebuild misses. This figure is the justification for any non-1.0
mainline kappa."""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from impact import compute
from node_knee import evaluate_node_expected_w, execution_realization_metrics, solve_active_knee_milp
from plot_node_knee_deadline_sweep import BASE_EVENT, MOVE, population
from plot_node_knee_execution_validation import _plan
from simulate import simulate

kW = 1e3
MODE, ORDERING = "sf", "certified_pd"
TARGET_FRAC = 0.45  # of full node-expected removable power (the shared basis)
KAPPAS = (0.7, 0.8, 0.9, 1.0)
DEADLINES = (10.0, 15.0, 30.0, 300.0)  # tight -> slack; kappa only matters when rebuild binds


def run_sweep(kappas=KAPPAS, deadlines=DEADLINES):
    pool, pop = population()
    imp = compute(pop, pool)
    target = TARGET_FRAC * evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
    rows = []
    for D in deadlines:
        event = replace(BASE_EVENT, D=float(D))
        for kappa in kappas:
            res = solve_active_knee_milp(pop, pool, imp, target, event, MOVE, kappa=kappa)
            plan = _plan(res, "active-knee MILP")
            sim = simulate(pop, pool, imp, plan, event, MOVE, MODE, ORDERING)
            m = execution_realization_metrics(pop, pool, imp, plan, sim, event.D)
            movers = int((plan.y > 1e-9).sum())
            selected = m["selected_node_expected_w"]
            rebuild = m["rebuild_realized_node_expected_w"]
            misses = movers - sim.reconstruction_success_count
            rows.append({
                "deadline_s": float(D), "kappa": float(kappa), "mode": MODE, "ordering": ORDERING,
                "target_basis": "full_node_expected", "target_kw": target / kW,
                "jobs": len(pop), "movers": movers, "cost_s": res.cost,
                "selected_hit": selected >= target - 1e-6 * max(target, 1.0),
                "rebuild_hit": rebuild >= target - 1e-6 * max(target, 1.0),
                "planner_shortfall_w": max(0.0, target - selected),
                "deadline_miss_count": misses,
                "selected_node_kw": selected / kW,
                "rebuild_node_kw": rebuild / kW,
                "rebuild_over_target": rebuild / target,
                "miss_frac": misses / movers if movers else 0.0,
            })
    return target / kW, rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def plot(rows, target_kw, path_base="outputs/node_knee_kappa_sweep"):
    deadlines = sorted({r["deadline_s"] for r in rows})
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(deadlines)))  # magnitude -> one-hue ramp
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)
    for D, c in zip(deadlines, shades):
        rs = sorted((r for r in rows if r["deadline_s"] == D), key=lambda r: r["kappa"])
        k = [r["kappa"] for r in rs]
        ax1.plot(k, [r["rebuild_over_target"] for r in rs], marker="o", ms=5, lw=2, color=c,
                 label=f"D={D:.0f}s")
        ax2.plot(k, [r["miss_frac"] for r in rs], marker="o", ms=5, lw=2, color=c)
    ax1.axhline(1.0, color="0.2", ls=":", lw=1)
    ax1.set(xlabel="planner rebuild cushion kappa", ylabel="rebuild-realized relief / target",
            title="A. Delivered relief")
    ax2.set(xlabel="planner rebuild cushion kappa", ylabel="moved sessions missing D (fraction)",
            title="B. Per-session rebuild misses", ylim=(0, None))  # zero-anchored, zoomed to data
    for ax in (ax1, ax2):
        ax.set_xticks(list(KAPPAS))
        ax.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, title=f"target {target_kw:.1f} kW", title_fontsize=8)
    fig.suptitle("Rebuild-cushion sensitivity (active-knee MILP plans replayed in the DES)", y=0.99)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    target_kw, rows = run_sweep()
    os.makedirs("outputs", exist_ok=True)
    write_csv(rows, "outputs/node_knee_kappa_sweep.csv")
    plot(rows, target_kw)
    print(f"target={target_kw:.1f} kW node-expected (basis=full_node_expected), mode={MODE}")
    for r in rows:
        print(f"D={r['deadline_s']:6.1f}s kappa={r['kappa']:.1f} movers={r['movers']:3d} "
              f"rebuild/target={r['rebuild_over_target']:.3f} miss_frac={r['miss_frac']:.3f} "
              f"cost={r['cost_s']:.1f}s")


if __name__ == "__main__":
    main()
