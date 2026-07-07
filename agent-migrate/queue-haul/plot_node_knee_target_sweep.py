"""Fixed-deadline 4-node node-knee target sweep."""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event
from impact import Movement, compute
from instance import SESSION_CLASSES, _mean_T, class_workload, generate
from node_knee import (
    evaluate_node_expected_w,
    place_source_nodes,
    solve_active_knee_lp,
    solve_active_knee_milp,
    solve_power_function_lp,
    solve_live_greedy,
    solve_random_jobs,
    with_source_nodes,
)
from power import PoolPower

kW = 1e3
N_NODES = 4
EVENT = Event(dest_nodes=48)
MOVE = Movement()
POWER_CURVE = os.getenv("QUEUE_HAUL_POWER_CURVE", "knee")
OUT_BASE = os.getenv("QUEUE_HAUL_TARGET_SWEEP_OUT", "outputs/node_knee_target_sweep")
TARGET_FRACS = np.linspace(0.05, 0.95, 19)
COLORS = {"LP relaxation": "tab:blue", "MILP": "tab:green", "greedy": "tab:orange", "random": "tab:purple"}
METHOD_LABELS = {
    "active-knee LP relaxation": "LP relaxation",
    "active-knee MILP": "MILP",
    "power-function LP relaxation": "LP relaxation",
    "live greedy": "greedy",
    "random jobs": "random",
    **{k: k for k in COLORS},
}
NUMERIC_FIELDS = (
    "source_nodes", "jobs", "deadline_s", "target_frac", "target_kw", "full_node_kw",
    "node_kw", "achieved_over_target", "active_kw", "cost_s",
    "intensity_s_per_kw", "requested_intensity_s_per_kw",
)


def population(session_class: str, seed: int = 42):
    wl = class_workload(session_class, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl), power_curve=POWER_CURVE)
    pop = generate(pool, wl, n_nodes=N_NODES, seed=seed)
    return pool, with_source_nodes(pop, place_source_nodes(pop, pool, N_NODES, "memory"))


def method_specs(pop, pool, imp, target):
    if pool.power_curve == "log":
        return (
            ("LP relaxation", lambda: solve_power_function_lp(pop, pool, imp, target, EVENT, MOVE)),
            ("greedy", lambda: solve_live_greedy(pop, pool, imp, target, EVENT, MOVE)),
            ("random", lambda: solve_random_jobs(pop, pool, imp, target, EVENT, MOVE, seed=0)),
        )
    return (
        ("LP relaxation", lambda: solve_active_knee_lp(pop, pool, imp, target, EVENT, MOVE)),
        ("MILP", lambda: solve_active_knee_milp(pop, pool, imp, target, EVENT, MOVE)),
        ("greedy", lambda: solve_live_greedy(pop, pool, imp, target, EVENT, MOVE)),
        ("random", lambda: solve_random_jobs(pop, pool, imp, target, EVENT, MOVE, seed=0)),
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


def env_workloads():
    return tuple(x for x in os.getenv("QUEUE_HAUL_WORKLOADS", "").split(",") if x) or SESSION_CLASSES


def env_target_fracs():
    text = os.getenv("QUEUE_HAUL_TARGET_FRACS", "")
    return np.array([float(x) for x in text.split(",") if x]) if text else TARGET_FRACS


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_rows(rows):
    out = []
    for r in rows:
        label = METHOD_LABELS.get(r["method"])
        if label is None:
            continue
        row = {**r, "method": label, "hit": r["hit"] in (True, "True", "true", "1")}
        row.update({k: float(row[k]) for k in NUMERIC_FIELDS})
        row["source_nodes"], row["jobs"] = int(row["source_nodes"]), int(row["jobs"])
        out.append(row)
    return out


def _median(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def plot(rows, path_base=OUT_BASE):
    workloads = list(dict.fromkeys(r["session_class"] for r in rows))
    fig, axs = plt.subplots(1, len(workloads), figsize=(3.8 * len(workloads), 3.2), sharex=False, squeeze=False)
    for col, cls in enumerate(workloads):
        for method in dict.fromkeys(r["method"] for r in rows):
            color = COLORS[method]
            rs = [r for r in rows if r["session_class"] == cls and r["method"] == method]
            x = np.array([r["target_kw"] for r in rs])
            intensity = np.array([r["intensity_s_per_kw"] for r in rs], float)
            axs[0, col].plot(x, intensity, marker="o", lw=1.8, ms=3.5, color=color, label=method)
        axs[0, col].set(title=cls.replace("_", " "), yscale="log", xlabel="requested modeled shed (kW)")
        axs[0, col].grid(True, alpha=0.25)
    axs[0, 0].set_ylabel("disruption (s/kW)")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    title = "node power-function" if POWER_CURVE == "log" else "node-knee"
    fig.suptitle(f"4-node fixed-deadline {title} target sweep (D={EVENT.D:.0f}s)", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)


def main():
    path = f"{OUT_BASE}.csv"
    rows = plot_rows(read_csv(path)) if os.path.exists(path) and not os.getenv("QUEUE_HAUL_FORCE_RUN") else run_sweep(env_workloads(), env_target_fracs())
    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    write_csv(rows, path)
    plot(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"rows={len(rows)} configs={len(rows) // len(methods)} deadline={EVENT.D:.0f}s source_nodes={N_NODES} power_curve={POWER_CURVE}")
    for cls in dict.fromkeys(r["session_class"] for r in rows):
        print(cls)
        for method in methods:
            rs = [r for r in rows if r["session_class"] == cls and r["method"] == method]
            hit = [r for r in rs if r["hit"]]
            max_hit = max((r["target_kw"] for r in hit), default=np.nan)
            print(f"  {method:14s} hit={np.mean([r['hit'] for r in rs]):.2f} "
                  f"max_hit={max_hit:6.1f} kW median_intensity={_median([r['intensity_s_per_kw'] for r in rs]):7.1f}s/kW")


if __name__ == "__main__":
    main()
