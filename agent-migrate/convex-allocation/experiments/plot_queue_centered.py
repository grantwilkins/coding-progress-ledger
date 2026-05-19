from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

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
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation
from metrics import source_load_moved_s
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import evaluate_rounded_queue_trace, round_allocation

MAIN_POLICIES = (
    "CVXPY-rounded",
    "deadline-penalty-rounded",
    "mirror-descent-rounded",
    "crossover-greedy",
    "replay-only",
    "state-only",
)
REFERENCE_POLICIES = (
    "local-repair-oracle",
    "deadline-aware-m0.8-rounded",
    "deadline-aware-m1.0-rounded",
)
PLOT_DRAIN_WINDOW_S = 1800.0
PLOT_POLICIES = MAIN_POLICIES
OUTPUT_FILES = (
    "source_load_frontier.pdf",
    "deadline_miss_frontier.pdf",
    "deadline_delay_cdf.pdf",
    "queue_depth_example.pdf",
    "network_prefill_busy_scatter.pdf",
)
POLICY_COLORS = {
    "CVXPY-rounded": "#0072B2",
    "deadline-penalty-rounded": "#000000",
    "mirror-descent-rounded": "#E69F00",
    "crossover-greedy": "#009E73",
    "mixed-greedy": "#CC79A7",
    "replay-only": "#D55E00",
    "state-only": "#666666",
    "local-repair-oracle": "#56B4E9",
    "deadline-aware-m0.8-rounded": "#8C8C8C",
    "deadline-aware-m1.0-rounded": "#4D4D4D",
}
POLICY_LABELS = {
    "CVXPY-rounded": "CVXPY",
    "deadline-penalty-rounded": "Deadline penalty",
    "mirror-descent-rounded": "Mirror descent",
    "crossover-greedy": "Crossover greedy",
    "mixed-greedy": "Mixed greedy",
    "replay-only": "Replay only",
    "state-only": "State transfer",
    "local-repair-oracle": "Repair oracle",
    "deadline-aware-m0.8-rounded": "Hard cap 0.8x",
    "deadline-aware-m1.0-rounded": "Hard cap 1.0x",
}


def plot_queue_centered(
    workload_config: WorkloadConfig = WorkloadConfig(),
    example_source_prefill_fraction: float = 0.5,
    example_deadline_scale: float = 0.5,
) -> None:
    out = workload_config.output_dir(ROOT)
    rows = [
        row
        for row in _read_rows(out / "source_load_deadline_sweep.csv")
        if _as_float(row["drain_window_s"]) == PLOT_DRAIN_WINDOW_S
    ]
    _plot_frontier(
        rows,
        "p95_delay_over_deadline",
        1.0,
        out / "source_load_frontier.pdf",
        "p95 delay / deadline",
    )
    _plot_frontier(
        rows,
        "deadline_miss_rate",
        0.01,
        out / "deadline_miss_frontier.pdf",
        "deadline miss rate",
    )
    _plot_busy_scatter(rows, out / "network_prefill_busy_scatter.pdf")

    traces = _example_traces(workload_config, example_source_prefill_fraction, example_deadline_scale)
    _plot_delay_cdf(traces, out / "deadline_delay_cdf.pdf")
    _plot_queue_depth(traces, out / "queue_depth_example.pdf")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _plot_frontier(rows, y_key, threshold, path, ylabel):
    deadline_scales = sorted({_as_float(row["deadline_scale"]) for row in rows})
    cols = min(2, len(deadline_scales))
    rows_n = int(np.ceil(len(deadline_scales) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.35 * cols, 2.55 * rows_n), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, deadline_scale in zip(axes, deadline_scales):
        for policy in PLOT_POLICIES:
            points = _policy_points(
                rows,
                policy,
                "source_prefill_fraction",
                y_key,
                {"deadline_scale": deadline_scale},
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
                    label=POLICY_LABELS[policy],
                )
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0)
        ax.text(0.98, threshold, f"{threshold:g}", ha="right", va="bottom", fontsize=7)
        ax.set_title(f"deadline scale = {deadline_scale:g}x")
        ax.set_xlabel("source prefill fraction moved")
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(deadline_scales) :]:
        ax.set_visible(False)
    axes[0].set_ylabel(ylabel)
    _legend(fig, axes[: len(deadline_scales)])
    fig.subplots_adjust(top=0.84, bottom=0.11, wspace=0.12, hspace=0.38)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_busy_scatter(rows, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    for policy in PLOT_POLICIES:
        points = [
            row
            for row in rows
            if row["policy"] == policy
            and _finite(row, "network_capacity_pressure")
            and _finite(row, "prefill_capacity_pressure")
            and _finite(row, "p95_delay_over_deadline")
        ]
        if not points:
            continue
        size = _scatter_sizes([_as_float(row["p95_delay_over_deadline"]) for row in points])
        ax.scatter(
            [_as_float(row["network_capacity_pressure"]) for row in points],
            [_as_float(row["prefill_capacity_pressure"]) for row in points],
            s=size,
            color=POLICY_COLORS[policy],
            alpha=0.72,
            label=POLICY_LABELS[policy],
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("network capacity pressure")
    ax.set_ylabel("GPU prefill capacity pressure")
    ax.text(0.03, 0.97, "marker size = p95 delay / deadline", transform=ax.transAxes, va="top")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _example_traces(workload_config, source_prefill_fraction, deadline_scale):
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        source_load_fraction=source_prefill_fraction,
        deadline_scale=deadline_scale,
        **workload_config.problem_kwargs(),
    )
    traces = {}
    for policy in PLOT_POLICIES:
        try:
            y = _example_allocation(policy, problem)
            if source_load_moved_s(problem, y) < problem.source_load_target_s - 1e-5:
                continue
            rounded = round_allocation(problem, y)
            _, trace = evaluate_rounded_queue_trace(problem, rounded.y, drain_window_s=PLOT_DRAIN_WINDOW_S)
        except (RuntimeError, ValueError):
            continue
        traces[policy] = trace
    return traces


def _example_allocation(policy, problem):
    if policy == "CVXPY-rounded":
        return solve_cvxpy(problem).y
    if policy == "deadline-penalty-rounded":
        return solve_soft_deadline_cvxpy(problem).y
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
        return solve_deadline_aware_cvxpy(problem, 0.8, source_load_cap=problem.source_load_target_s).y
    if policy == "deadline-aware-m1.0-rounded":
        return solve_deadline_aware_cvxpy(problem, 1.0, source_load_cap=problem.source_load_target_s).y
    if policy == "local-repair-oracle":
        rounded = round_allocation(problem, solve_cvxpy(problem).y)
        return repair_rounded_allocation(problem, rounded.y).y
    raise ValueError(policy)


def _plot_delay_cdf(traces, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    for policy, trace in traces.items():
        values = [record.reconstruction_delay / record.deadline_s for record in trace]
        points = _cdf_points(values)
        if points:
            x, y = np.asarray(points).T
            ax.plot(
                x,
                y,
                color=POLICY_COLORS[policy],
                linestyle=_linestyle(policy),
                linewidth=1.5,
                label=POLICY_LABELS[policy],
            )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.text(1.02, 0.04, "deadline", rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("reconstruction delay / deadline")
    ax.set_ylabel("empirical CDF")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_queue_depth(traces, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
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
                    label=POLICY_LABELS[policy],
                )
        ax.set_title(f"{resource} queue")
        ax.set_xlabel("time since drain start (s)")
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("max waiting queue depth")
    _legend(fig, axes)
    fig.subplots_adjust(top=0.75, bottom=0.18, wspace=0.12)
    fig.savefig(path, bbox_inches="tight")
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
        release = getattr(record, "release_time_s", 0.0)
        if resource == "network":
            arrival = release
            start = arrival + record.network_queue_wait
        elif resource == "prefill":
            if record.prefill_service_time == 0.0:
                continue
            arrival = release + record.network_queue_wait + record.network_service_time
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
