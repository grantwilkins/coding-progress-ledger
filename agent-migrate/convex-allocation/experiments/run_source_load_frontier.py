from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

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
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import source_load_moved_s
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import fractional_queue_load_proxy, queue_metrics, round_allocation
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation

SOURCE_PREFILL_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
DEADLINE_SCALES = (0.25, 0.50, 1.00, 2.00)
DRAIN_WINDOWS_S = (0.0, 900.0, 1800.0, 3600.0)
PLOT_DRAIN_WINDOW_S = 1800.0
DEADLINE_MARGINS = (0.8, 1.0)
REPAIR_ORACLE_POLICY = "local-repair-oracle"
POLICIES = (
    ("CVXPY-rounded", solve_cvxpy),
    ("deadline-penalty-rounded", solve_soft_deadline_cvxpy),
    (
        "deadline-aware-m0.8-rounded",
        lambda problem: solve_deadline_aware_cvxpy(problem, 0.8, source_load_cap=problem.source_load_target_s),
    ),
    (
        "deadline-aware-m1.0-rounded",
        lambda problem: solve_deadline_aware_cvxpy(problem, 1.0, source_load_cap=problem.source_load_target_s),
    ),
    ("mirror-descent-rounded", lambda problem: solve_mirror_descent(problem, eta_x0=500.0)),
    ("crossover-greedy", solve_crossover_greedy),
    ("mixed-greedy", solve_mixed_greedy),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
FRONTIER_POLICIES = tuple(policy for policy, _ in POLICIES) + (REPAIR_ORACLE_POLICY,)
SWEEP_COLUMNS = (
    "policy",
    "source_prefill_fraction",
    "deadline_scale",
    "drain_window_s",
    "status",
    "safe",
    "failure_mode",
    "source_prefill_target_s",
    "source_prefill_moved_s",
    "source_prefill_removal_rate_s_per_s",
    "request_migration_fraction",
    "deadline_overrun_mean",
    "deadline_overrun_p95",
    "deadline_overrun_max",
    "deadline_load_max",
    "mean_delay_s",
    "p50_delay_s",
    "p95_delay_s",
    "p99_delay_s",
    "p95_delay_over_deadline",
    "deadline_miss_rate",
    "network_capacity_pressure",
    "prefill_capacity_pressure",
    "replay_source_prefill_fraction",
    "state_transfer_source_prefill_fraction",
    "fractional_network_capacity_pressure",
    "fractional_prefill_capacity_pressure",
    "drain_completion_s",
    "objective",
)
FRONTIER_COLUMNS = (
    "policy",
    "deadline_scale",
    "drain_window_s",
    "max_safe_source_prefill_fraction",
    "p95_delay_at_frontier",
    "p95_delay_over_deadline_at_frontier",
    "deadline_miss_rate_at_frontier",
    "network_capacity_pressure_at_frontier",
    "prefill_capacity_pressure_at_frontier",
    "replay_source_prefill_fraction_at_frontier",
    "state_transfer_source_prefill_fraction_at_frontier",
    "drain_completion_s_at_frontier",
)


def run_source_load_frontier(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    model = get_model("GLM-5")
    for deadline_scale in DEADLINE_SCALES:
        for source_prefill_fraction in SOURCE_PREFILL_FRACTIONS:
            problem = make_problem(
                model,
                "transition-coupled",
                source_load_fraction=source_prefill_fraction,
                deadline_scale=deadline_scale,
                **workload_config.problem_kwargs(),
            )
            for policy, solver in POLICIES:
                rows.extend(_policy_rows(problem, policy, solver, source_prefill_fraction, deadline_scale))
            rows.extend(_repair_oracle_rows(problem, source_prefill_fraction, deadline_scale))

    frontier = _frontier_rows(rows)
    _write_rows(out / "source_load_deadline_sweep.csv", rows, SWEEP_COLUMNS)
    _write_rows(out / "source_load_frontier.csv", frontier, FRONTIER_COLUMNS)
    _print_latex_frontier(frontier)
    _print_diagnostics(frontier, rows)
    return rows, frontier


def _policy_rows(problem, policy, solver, source_prefill_fraction, deadline_scale):
    try:
        result = solver(problem)
    except RuntimeError:
        return [
            _empty_policy_row(policy, source_prefill_fraction, deadline_scale, drain_window_s, problem.source_load_target_s)
            for drain_window_s in DRAIN_WINDOWS_S
        ]

    y = result.allocation if hasattr(result, "allocation") else result.y
    solver_metrics = _semantic_solver_metrics(problem, y, getattr(result, "diagnostics", None))
    feasible = (
        getattr(result, "feasible", True)
        and source_load_moved_s(problem, y) >= problem.source_load_target_s - 1e-5
    )
    rows = []
    for drain_window_s in DRAIN_WINDOWS_S:
        base = _empty_policy_row(policy, source_prefill_fraction, deadline_scale, drain_window_s, problem.source_load_target_s)
        base["objective"] = getattr(result, "objective", math.nan)
        base.update(solver_metrics)
        if not feasible:
            rows.append(base)
            continue
        proxy = fractional_queue_load_proxy(problem, y)
        base.update(
            {
                "fractional_network_capacity_pressure": proxy["fractional_network_capacity_pressure"],
                "fractional_prefill_capacity_pressure": proxy["fractional_prefill_capacity_pressure"],
            }
        )
        try:
            metrics = queue_metrics(problem, y, drain_window_s=drain_window_s)
        except ValueError:
            base.update({"status": "ROUNDING_FAILED", "failure_mode": "rounding artifact"})
            rows.append(base)
            continue
        rows.append(_complete_policy_row(base, metrics))
    return rows


def _repair_oracle_rows(problem, source_prefill_fraction, deadline_scale):
    try:
        rounded = round_allocation(problem, solve_cvxpy(problem).y)
    except (RuntimeError, ValueError):
        return [
            _empty_policy_row(
                REPAIR_ORACLE_POLICY, source_prefill_fraction, deadline_scale, drain_window_s, problem.source_load_target_s
            )
            for drain_window_s in DRAIN_WINDOWS_S
        ]
    rows = []
    for drain_window_s in DRAIN_WINDOWS_S:
        base = _empty_policy_row(
            REPAIR_ORACLE_POLICY, source_prefill_fraction, deadline_scale, drain_window_s, problem.source_load_target_s
        )
        try:
            repair = repair_rounded_allocation(problem, rounded.y, drain_window_s=drain_window_s)
        except (RuntimeError, ValueError):
            rows.append(base)
            continue
        proxy = fractional_queue_load_proxy(problem, repair.y)
        base.update(
            {
                "fractional_network_capacity_pressure": proxy["fractional_network_capacity_pressure"],
                "fractional_prefill_capacity_pressure": proxy["fractional_prefill_capacity_pressure"],
            }
        )
        rows.append(_complete_policy_row(base, repair.metrics))
    return rows


def _empty_policy_row(policy, source_prefill_fraction, deadline_scale, drain_window_s, source_prefill_target_s):
    row = {
        "policy": policy,
        "source_prefill_fraction": source_prefill_fraction,
        "deadline_scale": deadline_scale,
        "drain_window_s": drain_window_s,
        "status": "INFEASIBLE",
        "safe": False,
        "failure_mode": "infeasible",
        **{column: math.nan for column in SWEEP_COLUMNS[7:]},
    }
    row["source_prefill_target_s"] = source_prefill_target_s
    return row


def _complete_policy_row(base, metrics):
    base.update(
        {
            "source_prefill_moved_s": metrics["source_prefill_moved_s"],
            "source_prefill_removal_rate_s_per_s": metrics["source_prefill_removal_rate_s_per_s"],
            "mean_delay_s": metrics["mean_reconstruction_delay"],
            "p50_delay_s": metrics["p50_reconstruction_delay"],
            "p95_delay_s": metrics["p95_reconstruction_delay"],
            "p99_delay_s": metrics["p99_reconstruction_delay"],
            "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
            "deadline_miss_rate": metrics["deadline_miss_rate"],
            "network_capacity_pressure": metrics["network_capacity_pressure"],
            "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
            "replay_source_prefill_fraction": metrics["replay_source_prefill_fraction"],
            "state_transfer_source_prefill_fraction": metrics["state_transfer_source_prefill_fraction"],
            "drain_completion_s": metrics["drain_completion_s"],
        }
    )
    base["safe"] = _is_safe(metrics)
    base["status"] = "SAFE" if base["safe"] else "UNSAFE"
    base["failure_mode"] = "" if base["safe"] else _failure_mode(base)
    return base


def _is_safe(metrics):
    return (
        metrics["source_prefill_moved_s"] >= metrics["source_prefill_target_s"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_reconstruction_delay_ratio"] <= 1.0
    )


def _failure_mode(row):
    if row["source_prefill_moved_s"] < row["source_prefill_target_s"] - 1e-9:
        return "rounding artifact"
    if row["deadline_miss_rate"] > 0.01 or row["p95_delay_over_deadline"] > 1.0:
        if max(row["network_capacity_pressure"], row["prefill_capacity_pressure"]) < 0.95:
            return "deadline misses"
        return (
            "network bottleneck"
            if row["network_capacity_pressure"] >= row["prefill_capacity_pressure"]
            else "prefill bottleneck"
        )
    return "unsafe"


def _semantic_solver_metrics(problem, y, diagnostics):
    diagnostics = diagnostics or {}
    source_prefill_moved = source_load_moved_s(problem, y)
    request_migration_fraction = float(np.sum(y[:, : y.shape[1] - 1]) / np.sum(problem.d))
    return {
        "source_prefill_target_s": problem.source_load_target_s,
        "source_prefill_moved_s": source_prefill_moved,
        "request_migration_fraction": request_migration_fraction,
        "deadline_overrun_mean": diagnostics.get("deadline_overrun_mean", math.nan),
        "deadline_overrun_p95": diagnostics.get("deadline_overrun_p95", math.nan),
        "deadline_overrun_max": diagnostics.get("deadline_overrun_max", math.nan),
        "deadline_load_max": diagnostics.get("deadline_load_max", math.nan),
    }


def _frontier_rows(
    rows,
    policies=FRONTIER_POLICIES,
    deadline_scales=DEADLINE_SCALES,
    drain_window_s=PLOT_DRAIN_WINDOW_S,
):
    frontier = []
    for policy in policies:
        for deadline_scale in deadline_scales:
            safe = [
                row
                for row in rows
                if row["policy"] == policy
                and row["deadline_scale"] == deadline_scale
                and row["drain_window_s"] == drain_window_s
                and row["safe"]
            ]
            if not safe:
                frontier.append(
                    {
                        "policy": policy,
                        "deadline_scale": deadline_scale,
                        "drain_window_s": drain_window_s,
                        **{column: "UNSAFE" for column in FRONTIER_COLUMNS[3:]},
                    }
                )
                continue
            row = max(safe, key=lambda item: item["source_prefill_fraction"])
            frontier.append(
                {
                    "policy": policy,
                    "deadline_scale": deadline_scale,
                    "drain_window_s": drain_window_s,
                    "max_safe_source_prefill_fraction": row["source_prefill_fraction"],
                    "p95_delay_at_frontier": row["p95_delay_s"],
                    "p95_delay_over_deadline_at_frontier": row["p95_delay_over_deadline"],
                    "deadline_miss_rate_at_frontier": row["deadline_miss_rate"],
                    "network_capacity_pressure_at_frontier": row["network_capacity_pressure"],
                    "prefill_capacity_pressure_at_frontier": row["prefill_capacity_pressure"],
                    "replay_source_prefill_fraction_at_frontier": row["replay_source_prefill_fraction"],
                    "state_transfer_source_prefill_fraction_at_frontier": row[
                        "state_transfer_source_prefill_fraction"
                    ],
                    "drain_completion_s_at_frontier": row["drain_completion_s"],
                }
            )
    return frontier


def _write_rows(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _frontier_value(frontier, policy, deadline_scale):
    value = next(
        row["max_safe_source_prefill_fraction"]
        for row in frontier
        if row["policy"] == policy and row["deadline_scale"] == deadline_scale
    )
    return math.nan if value == "UNSAFE" else float(value)


def _print_latex_frontier(frontier):
    print("\nsource-prefill frontier (LaTeX)")
    print("\\begin{tabular}{lrrrr}")
    print("policy & 0.25x & 0.5x & 1x & 2x \\\\")
    print("\\hline")
    for policy in FRONTIER_POLICIES:
        cells = []
        for deadline_scale in DEADLINE_SCALES:
            value = _frontier_value(frontier, policy, deadline_scale)
            cells.append("--" if math.isnan(value) else f"{value:.2f}")
        print(f"{policy} & {' & '.join(cells)} \\\\")
    print("\\end{tabular}")


def _print_diagnostics(frontier, rows):
    md_rounding = []
    cvx_losses = []
    support = []
    for deadline_scale in DEADLINE_SCALES:
        cvx = _frontier_value(frontier, "CVXPY-rounded", deadline_scale)
        md = _frontier_value(frontier, "mirror-descent-rounded", deadline_scale)
        if not math.isnan(md) and (math.isnan(cvx) or md > cvx + 1e-12):
            md_rounding.append(deadline_scale)

        values = [_frontier_value(frontier, policy, deadline_scale) for policy, _ in POLICIES]
        best = max((value for value in values if not math.isnan(value)), default=math.nan)
        if not math.isnan(best) and (math.isnan(cvx) or cvx < best - 1e-12):
            cvx_losses.append((deadline_scale, _cvx_failure_after_frontier(rows, cvx, deadline_scale)))

        rivals = (
            _frontier_value(frontier, "crossover-greedy", deadline_scale),
            _frontier_value(frontier, "replay-only", deadline_scale),
            _frontier_value(frontier, "state-only", deadline_scale),
        )
        if not math.isnan(cvx) and any(math.isnan(value) or cvx > value + 1e-12 for value in rivals):
            support.append(deadline_scale)

    if md_rounding:
        print(
            "\nMirror-descent-rounded exceeds CVXPY-rounded at "
            f"{md_rounding}; this is a rounding effect, not an algorithmic win."
        )
    else:
        print("\nMirror-descent-rounded does not beat CVXPY-rounded on the frontier.")
    if cvx_losses:
        print(f"CVXPY-rounded does not win at {cvx_losses}.")
    else:
        print("CVXPY-rounded wins or ties the frontier at every deadline scale.")
    if support:
        print(
            "CVXPY-rounded supports a larger safe source-prefill fraction than crossover-greedy "
            f"or a single-action policy at deadline scales {support}."
        )
    else:
        print("CVXPY-rounded does not exceed crossover-greedy or either single-action frontier.")
    _print_deadline_diagnostics(frontier, rows)


def _print_deadline_diagnostics(frontier, rows):
    rows = [row for row in rows if row["drain_window_s"] == PLOT_DRAIN_WINDOW_S]
    for margin in DEADLINE_MARGINS:
        policy = f"deadline-aware-m{margin:.1f}-rounded"
        wins = 0
        losses = 0
        ties = 0
        for row in rows:
            if row["policy"] != "CVXPY-rounded" or row["deadline_scale"] not in (0.25, 0.5):
                continue
            other = next(
                candidate
                for candidate in rows
                if candidate["policy"] == policy
                and candidate["deadline_scale"] == row["deadline_scale"]
                and candidate["source_prefill_fraction"] == row["source_prefill_fraction"]
            )
            if math.isnan(row["deadline_miss_rate"]) or math.isnan(other["deadline_miss_rate"]):
                continue
            delta = other["deadline_miss_rate"] - row["deadline_miss_rate"]
            wins += delta < -1e-12
            losses += delta > 1e-12
            ties += abs(delta) <= 1e-12
        frontier_values = [
            _frontier_value(frontier, policy, deadline_scale)
            for deadline_scale in DEADLINE_SCALES
        ]
        print(
            f"{policy} tight-deadline miss-rate comparison vs CVXPY-rounded: "
            f"{wins} lower, {ties} tied, {losses} higher; frontier={frontier_values}."
        )


def _cvx_failure_after_frontier(rows, cvx_frontier, deadline_scale):
    for row in sorted(
        (
            row
            for row in rows
            if row["policy"] == "CVXPY-rounded"
            and row["deadline_scale"] == deadline_scale
            and row["drain_window_s"] == PLOT_DRAIN_WINDOW_S
            and (math.isnan(cvx_frontier) or row["source_prefill_fraction"] > cvx_frontier)
        ),
        key=lambda item: item["source_prefill_fraction"],
    ):
        if not row["safe"]:
            return row["failure_mode"]
    return "unsafe"


if __name__ == "__main__":
    run_source_load_frontier(parse_workload_config("Run source-prefill frontier sweep."))
