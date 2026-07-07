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
STARTUP = max(EVENT.tau_src, EVENT.tau_pre, EVENT.tau_in)
POWER_CURVE = os.getenv("QUEUE_HAUL_POWER_CURVE", "knee")
OUT_BASE = os.getenv("QUEUE_HAUL_TARGET_SWEEP_OUT", "outputs/node_knee_target_sweep")
DEADLINE_OUT_BASE = os.getenv("QUEUE_HAUL_DEADLINE_SWEEP_OUT", f"{OUT_BASE}_max_power_deadline_sweep")
TARGET_FRACS = np.linspace(0.05, 0.95, 19)
DEADLINES = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0])
RANDOM_SEEDS = tuple(range(int(os.getenv("QUEUE_HAUL_RANDOM_RUNS", "3"))))
if not RANDOM_SEEDS:
    raise ValueError("QUEUE_HAUL_RANDOM_RUNS must be positive")
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


def method_specs(pop, pool, imp, target, event=EVENT):
    if pool.power_curve == "log":
        fixed = (
            ("LP relaxation", 0, lambda: solve_power_function_lp(pop, pool, imp, target, event, MOVE)),
            ("greedy", 0, lambda: solve_live_greedy(pop, pool, imp, target, event, MOVE)),
        )
    else:
        fixed = (
            ("LP relaxation", 0, lambda: solve_active_knee_lp(pop, pool, imp, target, event, MOVE)),
            ("MILP", 0, lambda: solve_active_knee_milp(pop, pool, imp, target, event, MOVE)),
            ("greedy", 0, lambda: solve_live_greedy(pop, pool, imp, target, event, MOVE)),
        )
    return fixed + tuple(
        ("random", seed, lambda seed=seed: solve_random_jobs(pop, pool, imp, target, event, MOVE, seed=seed))
        for seed in RANDOM_SEEDS
    )


def _row(session_class, jobs, full_kw, deadline, frac, target_kw, method, replicate, result):
    hit = bool(result and result.true_expected_feasible)
    node_kw = 0.0 if result is None else result.node_expected_w / kW
    cost = result.cost if hit else np.nan
    return {
        "session_class": session_class,
        "source_nodes": N_NODES,
        "jobs": int(jobs),
        "deadline_s": float(deadline),
        "target_frac": float(frac),
        "target_kw": float(target_kw),
        "full_node_kw": float(full_kw),
        "method": method,
        "replicate": int(replicate),
        "hit": bool(hit),
        "node_kw": float(node_kw),
        "achieved_over_target": float(node_kw / target_kw),
        "active_kw": 0.0 if result is None else float(result.active_floor_w / kW),
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
            for method, replicate, fn in method_specs(pop, pool, imp, target):
                rows.append(_row(session_class, len(pop), full_w / kW, EVENT.D, frac,
                                 target / kW, method, replicate, fn()))
    return rows


def run_deadline_sweep(workloads=SESSION_CLASSES, deadlines=DEADLINES):
    rows = []
    for session_class in workloads:
        pool, pop = population(session_class)
        imp = compute(pop, pool)
        full_w = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
        for deadline in deadlines:
            event = replace(EVENT, D=float(deadline))
            for method, replicate, fn in method_specs(pop, pool, imp, full_w, event):
                result = None if deadline <= STARTUP else fn()
                rows.append(_row(session_class, len(pop), full_w / kW, deadline, 1.0,
                                 full_w / kW, method, replicate, result))
    return rows


def env_workloads():
    return tuple(x for x in os.getenv("QUEUE_HAUL_WORKLOADS", "").split(",") if x) or SESSION_CLASSES


def env_target_fracs():
    text = os.getenv("QUEUE_HAUL_TARGET_FRACS", "")
    return np.array([float(x) for x in text.split(",") if x]) if text else TARGET_FRACS


def env_deadlines():
    text = os.getenv("QUEUE_HAUL_DEADLINES", "")
    return np.array([float(x) for x in text.split(",") if x]) if text else DEADLINES


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
        row.setdefault("replicate", 0)
        row.update({k: float(row[k]) for k in NUMERIC_FIELDS})
        row["source_nodes"], row["jobs"], row["replicate"] = int(row["source_nodes"]), int(row["jobs"]), int(row["replicate"])
        out.append(row)
    return out


def _median(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def _methods(rows):
    return [m for m in COLORS if any(r["method"] == m for r in rows)]


def _series(rows, x_key, y_key, group_key=None):
    groups = sorted({r[group_key or x_key] for r in rows})
    x, y, lo, hi, n = [], [], [], [], []
    for group in groups:
        rs = [r for r in rows if r[group_key or x_key] == group]
        xs = [r[x_key] for r in rs if np.isfinite(r[x_key])]
        ys = [r[y_key] for r in rs if np.isfinite(r[y_key])]
        x.append(float(np.mean(xs)) if xs else np.nan)
        y.append(float(np.mean(ys)) if ys else np.nan)
        lo.append(float(np.min(ys)) if ys else np.nan)
        hi.append(float(np.max(ys)) if ys else np.nan)
        n.append(len(ys))
    return np.array(x), np.array(y), np.array(lo), np.array(hi), n


def _plot_xy(rows, path_base, x_key, y_key, xlabel, ylabel, title,
             yscale=None, xscale=None, xlim=None, xticks=None, diagonal=False, hline_key=None,
             group_key=None):
    workloads = list(dict.fromkeys(r["session_class"] for r in rows))
    fig, axs = plt.subplots(1, len(workloads), figsize=(3.8 * len(workloads), 3.2), sharex=False, squeeze=False)
    for col, cls in enumerate(workloads):
        cls_rows = [r for r in rows if r["session_class"] == cls]
        for method in _methods(rows):
            color = COLORS[method]
            rs = [r for r in cls_rows if r["method"] == method]
            x, y, lo, hi, n = _series(rs, x_key, y_key, group_key)
            axs[0, col].plot(x, y, marker="o", lw=1.8, ms=3.5, color=color, label=method)
            if method == "random" and max(n) > 1:
                axs[0, col].fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
        if diagonal:
            hi = max(r[x_key] for r in cls_rows)
            axs[0, col].plot([0.0, hi], [0.0, hi], color="0.25", ls="--", lw=1)
        if hline_key:
            axs[0, col].axhline(cls_rows[0][hline_key], color="0.25", ls="--", lw=1)
        axs[0, col].set(title=cls.replace("_", " "), xlabel=xlabel)
        if xscale:
            axs[0, col].set_xscale(xscale)
        if yscale:
            axs[0, col].set_yscale(yscale)
        if xlim:
            axs[0, col].set_xlim(*xlim)
        if xticks is not None:
            axs[0, col].set_xticks(xticks)
            axs[0, col].set_xticklabels([f"{x:g}" for x in xticks], rotation=45, ha="right", fontsize=8)
        axs[0, col].grid(True, alpha=0.25)
    axs[0, 0].set_ylabel(ylabel)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150)
    plt.close(fig)


def plot(rows, path_base=OUT_BASE):
    title = "node power-function" if POWER_CURVE == "log" else "node-knee"
    _plot_xy(rows, path_base, "target_kw", "intensity_s_per_kw", "requested modeled shed (kW)",
             "disruption (s/kW)", f"4-node fixed-deadline {title} target sweep (D={EVENT.D:.0f}s)",
             yscale="log", group_key="target_kw")


def plot_power_shed_vs_requested(rows, path_base=f"{OUT_BASE}_power_shed_vs_requested"):
    title = "node power-function" if POWER_CURVE == "log" else "node-knee"
    _plot_xy(rows, path_base, "target_kw", "node_kw", "requested modeled shed (kW)",
             "achieved modeled shed (kW)", f"4-node fixed-deadline {title}: achieved vs requested shed",
             diagonal=True, group_key="target_kw")


def plot_deadline_power_shed(rows, path_base=DEADLINE_OUT_BASE):
    title = "node power-function" if POWER_CURVE == "log" else "node-knee"
    _plot_xy(rows, path_base, "deadline_s", "node_kw", "deadline (s)", "achieved modeled shed (kW)",
             f"4-node max-request {title}: achieved shed by deadline", xscale="log",
             xlim=(1.0, EVENT.D), xticks=DEADLINES, hline_key="target_kw", group_key="deadline_s")


def plot_disruption_by_power_shed(rows, path_base=f"{OUT_BASE}_disruption_by_power_shed"):
    title = "node power-function" if POWER_CURVE == "log" else "node-knee"
    _plot_xy(rows, path_base, "node_kw", "cost_s", "achieved modeled shed (kW)",
             "disruption (s)", f"4-node fixed-deadline {title}: disruption by achieved shed (D={EVENT.D:.0f}s)",
             yscale="log", group_key="target_kw")


def main():
    path = f"{OUT_BASE}.csv"
    rows = plot_rows(read_csv(path)) if os.path.exists(path) and not os.getenv("QUEUE_HAUL_FORCE_RUN") else run_sweep(env_workloads(), env_target_fracs())
    deadline_path = f"{DEADLINE_OUT_BASE}.csv"
    deadline_rows = plot_rows(read_csv(deadline_path)) if os.path.exists(deadline_path) and not os.getenv("QUEUE_HAUL_FORCE_RUN") else run_deadline_sweep(env_workloads(), env_deadlines())
    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    write_csv(rows, path)
    write_csv(deadline_rows, deadline_path)
    plot(rows)
    plot_power_shed_vs_requested(rows)
    plot_deadline_power_shed(deadline_rows)
    plot_disruption_by_power_shed(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"rows={len(rows)} deadline_rows={len(deadline_rows)} configs={len(rows) // len(methods)} deadline={EVENT.D:.0f}s source_nodes={N_NODES} power_curve={POWER_CURVE}")
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
