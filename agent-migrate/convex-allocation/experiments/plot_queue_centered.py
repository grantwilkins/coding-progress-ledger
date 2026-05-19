from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from baselines import (
    solve_crossover_greedy,
    solve_mixed_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import get_model
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation
from metrics import shed_achieved
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import evaluate_rounded_queue_trace, round_allocation

MAIN_POLICIES = (
    "CVXPY-rounded",
    "mirror-descent-rounded",
    "crossover-greedy",
    "mixed-greedy",
    "replay-only",
    "state-only",
)
REFERENCE_POLICIES = (
    "local-repair-oracle",
    "deadline-aware-m0.8-rounded",
    "deadline-aware-m1.0-rounded",
)
PLOT_POLICIES = MAIN_POLICIES + REFERENCE_POLICIES
OUTPUT_FILES = (
    "safe_shed_frontier_lines.pdf",
    "miss_rate_frontier_lines.pdf",
    "delay_cdf_hard_case.pdf",
    "queue_depth_hard_case.pdf",
    "resource_pressure_scatter.pdf",
)
POLICY_COLORS = {
    "CVXPY-rounded": "#1f77b4",
    "mirror-descent-rounded": "#ff7f0e",
    "crossover-greedy": "#2ca02c",
    "mixed-greedy": "#9467bd",
    "replay-only": "#8c564b",
    "state-only": "#7f7f7f",
    "local-repair-oracle": "#17becf",
    "deadline-aware-m0.8-rounded": "#bcbd22",
    "deadline-aware-m1.0-rounded": "#e377c2",
}


def plot_queue_centered(
    workload_config: WorkloadConfig = WorkloadConfig(),
    hard_shed_fraction: float = 0.5,
    hard_slack_multiplier: float = 0.5,
) -> None:
    out = workload_config.output_dir(ROOT)
    rows = _read_rows(out / "shed_slack_sweep.csv")
    _plot_frontier(
        rows,
        "p95_normalized_delay",
        1.0,
        out / "safe_shed_frontier_lines.pdf",
        "p95 reconstruction delay / slack",
    )
    _plot_frontier(
        rows,
        "miss_rate",
        0.01,
        out / "miss_rate_frontier_lines.pdf",
        "deadline miss rate",
    )
    _plot_resource_pressure(rows, out / "resource_pressure_scatter.pdf")

    traces = _hard_case_traces(workload_config, hard_shed_fraction, hard_slack_multiplier)
    _plot_delay_cdf(traces, out / "delay_cdf_hard_case.pdf")
    _plot_queue_depth(traces, out / "queue_depth_hard_case.pdf")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _plot_frontier(rows, y_key, threshold, path, ylabel):
    slack_values = sorted({_as_float(row["slack_multiplier"]) for row in rows})
    fig, axes = plt.subplots(1, len(slack_values), figsize=(3.4 * len(slack_values), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, slack in zip(axes, slack_values):
        for policy in PLOT_POLICIES:
            points = _policy_points(
                rows,
                policy,
                "shed_fraction",
                y_key,
                {"slack_multiplier": slack},
            )
            if points:
                x, y = np.asarray(points).T
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    color=POLICY_COLORS[policy],
                    linestyle=_linestyle(policy),
                    label=policy,
                )
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"slack {slack:g}x")
        ax.set_xlabel("shed fraction")
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel(ylabel)
    _legend(fig, axes)
    fig.savefig(path)
    plt.close(fig)


def _plot_resource_pressure(rows, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    for policy in PLOT_POLICIES:
        points = [
            row
            for row in rows
            if row["policy"] == policy
            and _finite(row, "max_net_busy")
            and _finite(row, "max_prefill_busy")
            and _finite(row, "p95_normalized_delay")
        ]
        if not points:
            continue
        size = _scatter_sizes([_as_float(row["p95_normalized_delay"]) for row in points])
        ax.scatter(
            [_as_float(row["max_net_busy"]) for row in points],
            [_as_float(row["max_prefill_busy"]) for row in points],
            s=size,
            color=POLICY_COLORS[policy],
            alpha=0.72,
            label=policy,
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("max network busy window")
    ax.set_ylabel("max prefill busy window")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def _hard_case_traces(workload_config, shed_fraction, slack_multiplier):
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        shed_fraction=shed_fraction,
        slack_multiplier=slack_multiplier,
        **workload_config.problem_kwargs(),
    )
    traces = {}
    for policy in PLOT_POLICIES:
        try:
            y = _hard_case_allocation(policy, problem)
            if shed_achieved(problem, y) < problem.B_shed - 1e-5:
                continue
            rounded = round_allocation(problem, y)
            _, trace = evaluate_rounded_queue_trace(problem, rounded.y)
        except (RuntimeError, ValueError):
            continue
        traces[policy] = trace
    return traces


def _hard_case_allocation(policy, problem):
    if policy == "CVXPY-rounded":
        return solve_cvxpy(problem).y
    if policy == "mirror-descent-rounded":
        return solve_mirror_descent(problem, eta_x0=500.0).y
    if policy == "crossover-greedy":
        return solve_crossover_greedy(problem).allocation
    if policy == "mixed-greedy":
        return solve_mixed_greedy(problem).allocation
    if policy == "replay-only":
        return solve_replay_only(problem).allocation
    if policy == "state-only":
        return solve_state_only(problem).allocation
    if policy == "deadline-aware-m0.8-rounded":
        return solve_deadline_aware_cvxpy(problem, 0.8, shed_cap=problem.B_shed).y
    if policy == "deadline-aware-m1.0-rounded":
        return solve_deadline_aware_cvxpy(problem, 1.0, shed_cap=problem.B_shed).y
    if policy == "local-repair-oracle":
        rounded = round_allocation(problem, solve_cvxpy(problem).y)
        return repair_rounded_allocation(problem, rounded.y).y
    raise ValueError(policy)


def _plot_delay_cdf(traces, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    for policy, trace in traces.items():
        values = [record.reconstruction_delay / record.slack for record in trace]
        points = _cdf_points(values)
        if points:
            x, y = np.asarray(points).T
            ax.plot(
                x,
                y,
                color=POLICY_COLORS[policy],
                linestyle=_linestyle(policy),
                linewidth=1.5,
                label=policy,
            )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("reconstruction delay / slack")
    ax.set_ylabel("empirical CDF")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def _plot_queue_depth(traces, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, constrained_layout=True)
    for ax, resource in zip(axes, ("network", "prefill")):
        for policy, trace in traces.items():
            points = _max_waiting_depth_points(trace, resource)
            if points:
                x, y = np.asarray(points).T
                ax.step(
                    x,
                    y,
                    where="post",
                    color=POLICY_COLORS[policy],
                    linestyle=_linestyle(policy),
                    linewidth=1.4,
                    label=policy,
                )
        ax.set_title(f"{resource} queue")
        ax.set_xlabel("time since shed event (s)")
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("max waiting queue depth")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def _policy_points(rows, policy, x_key, y_key, filters=None):
    filters = filters or {}
    points = []
    for row in rows:
        if row["policy"] != policy or not _finite(row, x_key) or not _finite(row, y_key):
            continue
        if any(_as_float(row[key]) != value for key, value in filters.items()):
            continue
        points.append((_as_float(row[x_key]), _as_float(row[y_key])))
    return sorted(points)


def _cdf_points(values):
    values = np.sort(np.asarray(list(values), dtype=float))
    return list(zip(values, np.arange(1, values.size + 1) / values.size)) if values.size else []


def _max_waiting_depth_points(trace, resource):
    events: dict[float, dict[int, int]] = {}
    for record in trace:
        if resource == "network":
            arrival = 0.0
            start = record.network_queue_wait
        elif resource == "prefill":
            if record.prefill_service_time == 0.0:
                continue
            arrival = record.network_queue_wait + record.network_service_time
            start = arrival + record.prefill_queue_wait
        else:
            raise ValueError(resource)
        if start <= arrival + 1e-12:
            continue
        events.setdefault(arrival, {}).setdefault(record.k, 0)
        events[arrival][record.k] += 1
        events.setdefault(start, {}).setdefault(record.k, 0)
        events[start][record.k] -= 1
    if not events:
        return [(0.0, 0.0)]
    depth: dict[int, int] = {}
    points = [(0.0, 0.0)]
    for time in sorted(events):
        for dest, delta in events[time].items():
            depth[dest] = depth.get(dest, 0) + delta
        points.append((float(time), float(max(depth.values(), default=0))))
    return points


def _finite(row, key):
    try:
        return np.isfinite(_as_float(row[key]))
    except (KeyError, TypeError, ValueError):
        return False


def _as_float(value):
    return float(value)


def _linestyle(policy):
    return "--" if policy in REFERENCE_POLICIES else "-"


def _scatter_sizes(values):
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0]):
        return np.full(values.size, 60.0)
    scaled = (values - np.min(values)) / (np.max(values) - np.min(values))
    return 30.0 + 110.0 * scaled


def _legend(fig, axes):
    handles_by_label = {}
    for ax in np.atleast_1d(axes):
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))
    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label),
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=7,
        )


if __name__ == "__main__":
    plot_queue_centered(parse_workload_config("Plot queue-centered analysis outputs."))
