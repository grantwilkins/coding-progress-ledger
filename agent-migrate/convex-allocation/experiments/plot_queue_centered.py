from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from baselines import (
    solve_crossover_greedy,
    solve_least_loaded_destination,
    solve_online_queue_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import catalog_models, get_model
from coefficients import ACTIONS, compute_coefficients
from cvxpy_solver import solve_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import retained_prefill_action_mix, retained_prefill_moved_s
from problem import ProblemData, make_problem
from queueing import RELEASE_POLICIES, evaluate_rounded_queue_trace, round_allocation

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

REPORT_DEADLINE_SCALE = 1.0
REPORT_RETAINED_PREFILL_FRACTION = 0.5
PLOT_DRAIN_WINDOW_S = 1200.0
H1_DRAIN_WINDOW_S = 20.0
H1_RETAINED_PREFILL_FRACTION = 0.25
H1_RELEASE_POLICY = "edf"
NETWORK_SCALES = (0.6, 1.0, 2.0)
REPORT_POLICIES = (
    "deadline-penalty-rounded",
    "online-queue-greedy",
    "least-loaded-destination",
    "replay-only",
    "state-only",
)
FRONTIER_POLICY = "deadline-penalty-rounded"
FRONTIER_RELEASE_POLICIES = RELEASE_POLICIES
H1_POLICIES = (
    "deadline-penalty-rounded",
    "online-queue-greedy",
    "least-loaded-destination",
    "replay-only",
    "state-only",
)
POLICY_LABELS = {
    "deadline-penalty-rounded": "Deadline-aware",
    "CVXPY-rounded": "CVXPY rounded",
    "online-queue-greedy": "Online queue",
    "least-loaded-destination": "Least loaded",
    "crossover-greedy": "Crossover greedy",
    "replay-only": "Replay only",
    "state-only": "State only",
}
POLICY_COLORS = {
    "deadline-penalty-rounded": "#1b1b1b",
    "CVXPY-rounded": "#4c78a8",
    "online-queue-greedy": "#72b7b2",
    "least-loaded-destination": "#54a24b",
    "crossover-greedy": "#54a24b",
    "replay-only": "#e45756",
    "state-only": "#b279a2",
}
RELEASE_POLICY_LABELS = {
    "edf": "EDF",
    "shortest-context-first": "Shortest context",
    "random": "Random",
}
RELEASE_POLICY_COLORS = {
    "edf": "#1b1b1b",
    "shortest-context-first": "#4c78a8",
    "random": "#e45756",
}
OUTPUT_FILES = (
    "h1_fixed_target_stress.csv",
    "h2_safe_frontier.pdf",
    "h2_delay_cdf.pdf",
    "h3_action_mix_by_model.pdf",
    "h4_state_manifest_heatmap.pdf",
    "integer_benchmark_summary.csv",
)
H1_STRESS_COLUMNS = (
    "policy",
    "drain_window_s",
    "retained_prefill_fraction",
    "release_policy",
    "target_moved_fraction",
    "network_capacity_pressure",
    "prefill_capacity_pressure",
    "absolute_p95_delay_over_deadline",
    "absolute_deadline_miss_rate",
    "verdict",
)
INTEGER_TABLE_POLICIES = (
    "best-integer-objective",
    "best-integer-queue",
    "CVXPY-rounded",
    "repaired-deadline-aware-CVXPY-rounded",
)


def plot_queue_centered(
    workload_config: WorkloadConfig = WorkloadConfig(source="fixed"),
    report_deadline_scale: float = REPORT_DEADLINE_SCALE,
    report_retained_prefill_fraction: float = REPORT_RETAINED_PREFILL_FRACTION,
) -> None:
    out = workload_config.output_dir(ROOT)
    sweep = _read_rows(out / "retained_state_drain_sweep.csv")
    _write_h1_stress_table(sweep, out / "h1_fixed_target_stress.csv", report_deadline_scale)
    _plot_safe_frontier(sweep, out / "h2_safe_frontier.pdf")
    _plot_delay_cdf(workload_config, out / "h2_delay_cdf.pdf", report_retained_prefill_fraction, report_deadline_scale)
    _plot_action_mix_by_model(out / "h3_action_mix_by_model.pdf")
    _plot_state_manifest_heatmap(out / "h4_state_manifest_heatmap.pdf", report_retained_prefill_fraction, report_deadline_scale)
    _write_integer_summary(ROOT / "outputs" / "sweep" / "integer_optimality_cases.csv", out / "integer_benchmark_summary.csv")


def _write_h1_stress_table(rows: list[dict[str, str]], path: Path, deadline_scale: float) -> None:
    table = _h1_stress_rows(rows, deadline_scale)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, H1_STRESS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def _h1_stress_rows(rows: list[dict[str, str]], deadline_scale: float) -> list[dict[str, object]]:
    df = _frame(rows)
    df = df[
        (df["deadline_scale"] == deadline_scale)
        & (df["release_policy"] == H1_RELEASE_POLICY)
        & np.isclose(df["drain_window_s"], H1_DRAIN_WINDOW_S)
        & np.isclose(df["retained_prefill_fraction"], H1_RETAINED_PREFILL_FRACTION)
        & df["policy"].isin(H1_POLICIES)
    ].copy()
    _require_policies(df, H1_POLICIES, "H1 fixed-target stress table")
    rows_by_policy = {row.policy: row for row in df.itertuples()}
    table = []
    for policy in H1_POLICIES:
        row = rows_by_policy[policy]
        target_moved_fraction = row.retained_prefill_moved_s / row.retained_prefill_target_s
        table.append(
            {
                "policy": POLICY_LABELS[policy],
                "drain_window_s": row.drain_window_s,
                "retained_prefill_fraction": row.retained_prefill_fraction,
                "release_policy": row.release_policy,
                "target_moved_fraction": target_moved_fraction,
                "network_capacity_pressure": row.network_capacity_pressure,
                "prefill_capacity_pressure": row.prefill_capacity_pressure,
                "absolute_p95_delay_over_deadline": row.absolute_p95_delay_over_deadline,
                "absolute_deadline_miss_rate": row.absolute_deadline_miss_rate,
                "verdict": _h1_verdict(row, target_moved_fraction),
            }
        )
    return table


def _h1_verdict(row, target_moved_fraction: float) -> str:
    if target_moved_fraction < 1.0 - 1e-9:
        return "Target shortfall"
    if row.network_capacity_pressure > 1.0:
        return "Network overload"
    if row.prefill_capacity_pressure > 1.0:
        return "Prefill overload"
    if row.absolute_p95_delay_over_deadline > 1.0:
        return "P95 deadline miss"
    if row.absolute_deadline_miss_rate > 0.01:
        return "Deadline miss rate"
    return "Pass"


def _plot_safe_frontier(rows: list[dict[str, str]], path: Path) -> None:
    frontier = _safe_frontier(rows)
    _require_keys(set(frontier["release_policy"]), FRONTIER_RELEASE_POLICIES, "safe-frontier sweep")
    if not frontier.empty:
        frontier["Release policy"] = frontier["release_policy"].map(RELEASE_POLICY_LABELS)
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    sns.lineplot(
        frontier,
        x="drain_window_s",
        y="max_safe_retained_prefill_fraction",
        hue="Release policy",
        marker="o",
        palette={RELEASE_POLICY_LABELS[k]: v for k, v in RELEASE_POLICY_COLORS.items()},
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Drain window (s)")
    ax.set_ylabel("Max safe retained-prefill fraction evacuated")
    ax.set_title("H2: same allocation policy; only release order changes")
    ax.set_ylim(0.0, 1.0)
    _simple_legend(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_cdf(
    workload_config: WorkloadConfig,
    path: Path,
    retained_prefill_fraction: float,
    deadline_scale: float,
) -> None:
    traces = _example_traces(workload_config, retained_prefill_fraction, deadline_scale)
    _require_keys(traces, REPORT_POLICIES, "delay CDF traces")
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    for policy in REPORT_POLICIES:
        trace = traces.get(policy)
        if not trace:
            continue
        x, y = np.asarray(_cdf_points(record.reconstruction_delay / record.deadline_s for record in trace)).T
        ax.plot(x, y, color=POLICY_COLORS[policy], linewidth=1.7, label=POLICY_LABELS[policy])
    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Reconstruction delay / deadline")
    ax.set_ylabel("Fraction of moved requests")
    ax.set_title("H2 detail: normalized delay tail")
    ax.set_xlim(left=0.0)
    _simple_legend(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_action_mix_by_model(path: Path) -> None:
    df = _architecture_action_mix()
    models = [model.name for model in catalog_models()]
    fig, axes = plt.subplots(1, len(NETWORK_SCALES), figsize=(8.6, 3.8), sharey=True, constrained_layout=True)
    for ax, scale in zip(np.atleast_1d(axes), NETWORK_SCALES):
        sub = df[df["network_bandwidth_scale"] == scale].set_index("model").reindex(models)
        y = np.arange(len(models))
        replay = sub["replay_retained_prefill_fraction"].to_numpy()
        state = sub["state_transfer_retained_prefill_fraction"].to_numpy()
        ax.barh(y, replay, color="#4c78a8", label="Replay")
        ax.barh(y, state, left=replay, color="#f58518", label="State transfer")
        ax.set_title(f"{scale:g}x baseline network")
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Action fraction")
    axes[0].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H3: model architecture changes replay/state mix")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_state_manifest_heatmap(path: Path, retained_prefill_fraction: float, deadline_scale: float) -> None:
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        retained_prefill_fraction=retained_prefill_fraction,
        deadline_scale=deadline_scale,
        workload_source="fixed",
    )
    allocation = solve_soft_deadline_cvxpy(problem).y
    heatmap, row_labels, col_labels = _allocation_heatmap(problem, allocation, max_rows=6)
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    sns.heatmap(
        heatmap,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Fraction of class"},
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
    )
    ax.set_xlabel("Destination / action")
    ax.set_ylabel("Session class: context, deadline, locality")
    ax.set_title("H4: routing depends on per-session state")
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _safe_frontier(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = _frame(rows)
    df = df[df["policy"] == FRONTIER_POLICY].copy()
    df["queue_safe"] = _safe_series(df)
    safe = df[df["queue_safe"]]
    grouped = safe.groupby(["policy", "release_policy", "drain_window_s"], as_index=False)[
        "retained_prefill_fraction"
    ].max()
    grouped = grouped.sort_values(["release_policy", "drain_window_s"])
    return grouped.rename(columns={"retained_prefill_fraction": "max_safe_retained_prefill_fraction"})


def _architecture_action_mix() -> pd.DataFrame:
    rows = []
    for scale in NETWORK_SCALES:
        for model in catalog_models():
            problem = _scaled_network_problem(
                make_problem(model, "bandwidth-spread", retained_prefill_fraction=0.5, workload_source="fixed"),
                scale,
            )
            mix = retained_prefill_action_mix(problem, solve_cvxpy(problem).y)
            rows.append({"model": model.name, "network_bandwidth_scale": scale, **mix})
    return pd.DataFrame(rows)


def _allocation_heatmap(
    problem: ProblemData, allocation: np.ndarray, max_rows: int | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    coeffs = compute_coefficients(problem)
    y = allocation / problem.d[:, None]
    order = np.lexsort((problem.deadline_s, -problem.T))
    if max_rows is not None:
        order = order[:max_rows]
    col_labels = [
        f"site {int(k)}\n{ACTIONS[int(action)]}"
        for k, action in zip(coeffs.option_dest, coeffs.option_action)
    ] + ["stay"]
    row_labels = [
        f"class {g}: T={problem.T[g] / 1000:.1f}k, ddl={problem.deadline_s[g]:.0f}s, "
        f"ctx={np.max(problem.h_ctx[g]):.2f}, kv={np.max(problem.h_kv[g]):.2f}"
        for g in order
    ]
    return y[order], row_labels, col_labels


def _example_traces(workload_config: WorkloadConfig, retained_prefill_fraction: float, deadline_scale: float):
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        retained_prefill_fraction=retained_prefill_fraction,
        deadline_scale=deadline_scale,
        **workload_config.problem_kwargs(),
    )
    traces = {}
    for policy in REPORT_POLICIES:
        try:
            y = _example_allocation(policy, problem)
            if retained_prefill_moved_s(problem, y) < problem.retained_prefill_target_s - 1e-5:
                raise RuntimeError(f"{policy} missed the retained-prefill target")
            integer_y = y if np.allclose(y, np.rint(y)) else round_allocation(problem, y).y
            _, trace = evaluate_rounded_queue_trace(problem, integer_y, drain_window_s=PLOT_DRAIN_WINDOW_S)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"cannot build delay CDF for {policy}") from exc
        traces[policy] = trace
    return traces


def _example_allocation(policy: str, problem: ProblemData) -> np.ndarray:
    if policy == "deadline-penalty-rounded":
        return solve_soft_deadline_cvxpy(problem).y
    if policy == "CVXPY-rounded":
        return solve_cvxpy(problem).y
    if policy == "online-queue-greedy":
        return solve_online_queue_greedy(problem).allocation
    if policy == "least-loaded-destination":
        return solve_least_loaded_destination(problem).allocation
    if policy == "crossover-greedy":
        return solve_crossover_greedy(problem).allocation
    if policy == "replay-only":
        return solve_replay_only(problem).allocation
    if policy == "state-only":
        return solve_state_only(problem).allocation
    raise ValueError(policy)


def _write_integer_summary(source: Path, target: Path) -> None:
    rows = _integer_summary_rows(_read_rows(source))
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            ("case", "policy", "integer_objective_gap_to_best", "p95_delay", "miss_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _integer_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    present = {row["policy"] for row in rows}
    missing = [policy for policy in INTEGER_TABLE_POLICIES if policy not in present]
    if missing:
        raise ValueError(f"integer benchmark missing policies: {missing}")
    keep = []
    for row in rows:
        if row["policy"] in INTEGER_TABLE_POLICIES:
            keep.append(
                {
                    "case": row["case"],
                    "policy": row["policy"],
                    "integer_objective_gap_to_best": row["integer_objective_gap_to_best"],
                    "p95_delay": row["p95_delay"],
                    "miss_rate": row["miss_rate"],
                }
            )
    return keep


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


def _scaled_network_problem(problem: ProblemData, scale: float) -> ProblemData:
    return ProblemData(
        model=problem.model,
        regime=problem.regime,
        T=problem.T,
        d=problem.d,
        deadline_s=problem.deadline_s,
        lambda_Bps=problem.lambda_Bps * scale,
        rho_prefill=problem.rho_prefill,
        C_net=problem.C_net * scale,
        C_prefill=problem.C_prefill,
        ell_net=problem.ell_net * scale,
        ell_prefill=problem.ell_prefill,
        h_ctx=problem.h_ctx,
        h_kv=problem.h_kv,
        retained_prefill_target_s=problem.retained_prefill_target_s,
        w=problem.w,
    )


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = (
        "retained_prefill_fraction",
        "deadline_scale",
        "drain_window_s",
        "network_capacity_pressure",
        "prefill_capacity_pressure",
        "deadline_miss_rate",
        "p95_delay_over_deadline",
        "absolute_deadline_miss_rate",
        "absolute_p95_delay_over_deadline",
        "retained_prefill_target_s",
        "retained_prefill_moved_s",
        "max_safe_retained_prefill_fraction",
        "network_capacity_pressure_at_frontier",
        "prefill_capacity_pressure_at_frontier",
    )
    for column in numeric:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_series(df: pd.DataFrame) -> pd.Series:
    return (df["absolute_deadline_miss_rate"] <= 0.01) & (df["absolute_p95_delay_over_deadline"] <= 1.0)


def _cdf_points(values) -> list[tuple[float, float]]:
    values = np.sort(np.asarray(list(values), dtype=float))
    return list(zip(values, np.arange(1, values.size + 1) / values.size)) if values.size else []


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _require_policies(df: pd.DataFrame, policies: tuple[str, ...], context: str) -> None:
    present = set(df["policy"])
    missing = [policy for policy in policies if policy not in present]
    if missing:
        raise ValueError(f"{context} missing policies: {missing}")


def _require_keys(values: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{context} missing policies: {missing}")


def _simple_legend(ax) -> None:
    legend = ax.legend(frameon=False, title=None, loc="best")
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(8)


if __name__ == "__main__":
    plot_queue_centered(parse_workload_config("Plot report figures."))
