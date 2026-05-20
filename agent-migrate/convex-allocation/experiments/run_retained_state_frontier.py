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
    solve_replay_only,
    solve_state_only,
)
from catalog import get_model
from cvxpy_solver import solve_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import (
    nvl72_hbm_fraction,
    resident_state_bytes,
    resident_state_moved_bytes,
    average_equivalent_state_target_bytes,
    retained_prefill_moved_s,
    state_tb,
    total_retained_prefill_s,
)
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import fractional_queue_load_proxy, queue_metrics

RETAINED_PREFILL_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
DEADLINE_SCALES = (0.25, 0.50, 1.00, 2.00)
DRAIN_WINDOWS_S = (0.0, 900.0, 1800.0, 3600.0)
PLOT_DRAIN_WINDOW_S = 1800.0
POLICIES = (
    ("CVXPY-rounded", solve_cvxpy),
    ("deadline-penalty-rounded", solve_soft_deadline_cvxpy),
    ("mirror-descent-rounded", lambda problem: solve_mirror_descent(problem, eta_x0=500.0)),
    ("crossover-greedy", solve_crossover_greedy),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
FRONTIER_POLICIES = tuple(policy for policy, _ in POLICIES)
SWEEP_COLUMNS = (
    "policy",
    "retained_prefill_fraction",
    "deadline_scale",
    "drain_window_s",
    "status",
    "safe",
    "failure_mode",
    "retained_prefill_target_s",
    "retained_prefill_moved_s",
    "resident_state_tb",
    "average_equivalent_state_target_tb",
    "actual_evacuated_state_tb",
    "retained_prefill_moved_fraction",
    "actual_evacuated_nvl72_hbm_fraction",
    "retained_prefill_removal_rate_s_per_s",
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
    "replay_retained_prefill_fraction",
    "state_transfer_retained_prefill_fraction",
    "fractional_network_capacity_pressure",
    "fractional_prefill_capacity_pressure",
    "drain_completion_s",
    "objective",
)
FRONTIER_COLUMNS = (
    "policy",
    "deadline_scale",
    "drain_window_s",
    "max_safe_retained_prefill_fraction",
    "average_equivalent_state_target_tb_at_frontier",
    "actual_evacuated_state_tb_at_frontier",
    "actual_evacuated_nvl72_hbm_fraction_at_frontier",
    "p95_delay_at_frontier",
    "p95_delay_over_deadline_at_frontier",
    "deadline_miss_rate_at_frontier",
    "network_capacity_pressure_at_frontier",
    "prefill_capacity_pressure_at_frontier",
    "replay_retained_prefill_fraction_at_frontier",
    "state_transfer_retained_prefill_fraction_at_frontier",
    "drain_completion_s_at_frontier",
)


def run_retained_state_frontier(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    model = get_model("GLM-5")
    for deadline_scale in DEADLINE_SCALES:
        for retained_prefill_fraction in RETAINED_PREFILL_FRACTIONS:
            problem = make_problem(
                model,
                "transition-coupled",
                retained_prefill_fraction=retained_prefill_fraction,
                deadline_scale=deadline_scale,
                **workload_config.problem_kwargs(),
            )
            for policy, solver in POLICIES:
                rows.extend(_policy_rows(problem, policy, solver, retained_prefill_fraction, deadline_scale))

    frontier = _frontier_rows(rows)
    _write_rows(out / "retained_state_deadline_sweep.csv", rows, SWEEP_COLUMNS)
    _write_rows(out / "retained_state_frontier.csv", frontier, FRONTIER_COLUMNS)
    _print_latex_frontier(frontier)
    _print_diagnostics(frontier, rows)
    return rows, frontier


def _policy_rows(problem, policy, solver, retained_prefill_fraction, deadline_scale):
    try:
        result = solver(problem)
    except RuntimeError:
        return [
            _empty_policy_row(policy, retained_prefill_fraction, deadline_scale, drain_window_s, problem.retained_prefill_target_s)
            for drain_window_s in DRAIN_WINDOWS_S
        ]

    y = result.allocation if hasattr(result, "allocation") else result.y
    solver_metrics = _semantic_solver_metrics(problem, y, getattr(result, "diagnostics", None))
    feasible = (
        getattr(result, "feasible", True)
        and retained_prefill_moved_s(problem, y) >= problem.retained_prefill_target_s - 1e-5
    )
    rows = []
    for drain_window_s in DRAIN_WINDOWS_S:
        base = _empty_policy_row(policy, retained_prefill_fraction, deadline_scale, drain_window_s, problem.retained_prefill_target_s)
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


def _empty_policy_row(policy, retained_prefill_fraction, deadline_scale, drain_window_s, retained_prefill_target_s):
    row = {
        "policy": policy,
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
        "drain_window_s": drain_window_s,
        "status": "INFEASIBLE",
        "safe": False,
        "failure_mode": "infeasible",
        **{column: math.nan for column in SWEEP_COLUMNS[7:]},
    }
    row["retained_prefill_target_s"] = retained_prefill_target_s
    return row


def _complete_policy_row(base, metrics):
    base.update(
        {
            "retained_prefill_moved_s": metrics["retained_prefill_moved_s"],
            "retained_prefill_removal_rate_s_per_s": metrics["retained_prefill_removal_rate_s_per_s"],
            "mean_delay_s": metrics["mean_reconstruction_delay"],
            "p50_delay_s": metrics["p50_reconstruction_delay"],
            "p95_delay_s": metrics["p95_reconstruction_delay"],
            "p99_delay_s": metrics["p99_reconstruction_delay"],
            "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
            "deadline_miss_rate": metrics["deadline_miss_rate"],
            "network_capacity_pressure": metrics["network_capacity_pressure"],
            "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
            "resident_state_tb": metrics["resident_state_tb"],
            "average_equivalent_state_target_tb": metrics["average_equivalent_state_target_tb"],
            "actual_evacuated_state_tb": metrics["actual_evacuated_state_tb"],
            "retained_prefill_moved_fraction": metrics["retained_prefill_moved_fraction"],
            "actual_evacuated_nvl72_hbm_fraction": metrics["actual_evacuated_nvl72_hbm_fraction"],
            "replay_retained_prefill_fraction": metrics["replay_retained_prefill_fraction"],
            "state_transfer_retained_prefill_fraction": metrics["state_transfer_retained_prefill_fraction"],
            "drain_completion_s": metrics["drain_completion_s"],
        }
    )
    base["safe"] = _is_safe(metrics)
    base["status"] = "SAFE" if base["safe"] else "UNSAFE"
    base["failure_mode"] = "" if base["safe"] else _failure_mode(base)
    return base


def _is_safe(metrics):
    return (
        metrics["retained_prefill_moved_s"] >= metrics["retained_prefill_target_s"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_reconstruction_delay_ratio"] <= 1.0
    )


def _failure_mode(row):
    if row["retained_prefill_moved_s"] < row["retained_prefill_target_s"] - 1e-9:
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
    retained_prefill_moved = retained_prefill_moved_s(problem, y)
    total = total_retained_prefill_s(problem)
    moved_bytes = resident_state_moved_bytes(problem, y)
    average_equivalent_target_bytes = average_equivalent_state_target_bytes(problem)
    total_bytes = resident_state_bytes(problem)
    request_migration_fraction = float(np.sum(y[:, : y.shape[1] - 1]) / np.sum(problem.d))
    return {
        "retained_prefill_target_s": problem.retained_prefill_target_s,
        "retained_prefill_moved_s": retained_prefill_moved,
        "resident_state_tb": state_tb(total_bytes),
        "average_equivalent_state_target_tb": state_tb(average_equivalent_target_bytes),
        "actual_evacuated_state_tb": state_tb(moved_bytes),
        "retained_prefill_moved_fraction": retained_prefill_moved / total,
        "actual_evacuated_nvl72_hbm_fraction": nvl72_hbm_fraction(moved_bytes),
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
            row = max(safe, key=lambda item: item["retained_prefill_fraction"])
            frontier.append(
                {
                    "policy": policy,
                    "deadline_scale": deadline_scale,
                    "drain_window_s": drain_window_s,
                    "max_safe_retained_prefill_fraction": row["retained_prefill_fraction"],
                    "average_equivalent_state_target_tb_at_frontier": row["average_equivalent_state_target_tb"],
                    "actual_evacuated_state_tb_at_frontier": row["actual_evacuated_state_tb"],
                    "actual_evacuated_nvl72_hbm_fraction_at_frontier": row["actual_evacuated_nvl72_hbm_fraction"],
                    "p95_delay_at_frontier": row["p95_delay_s"],
                    "p95_delay_over_deadline_at_frontier": row["p95_delay_over_deadline"],
                    "deadline_miss_rate_at_frontier": row["deadline_miss_rate"],
                    "network_capacity_pressure_at_frontier": row["network_capacity_pressure"],
                    "prefill_capacity_pressure_at_frontier": row["prefill_capacity_pressure"],
                    "replay_retained_prefill_fraction_at_frontier": row["replay_retained_prefill_fraction"],
                    "state_transfer_retained_prefill_fraction_at_frontier": row[
                        "state_transfer_retained_prefill_fraction"
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
        row["max_safe_retained_prefill_fraction"]
        for row in frontier
        if row["policy"] == policy and row["deadline_scale"] == deadline_scale
    )
    return math.nan if value == "UNSAFE" else float(value)


def _print_latex_frontier(frontier):
    print("\nretained-state frontier (LaTeX)")
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
            "CVXPY-rounded supports a larger safe retained-prefill fraction than crossover-greedy "
            f"or a single-action policy at deadline scales {support}."
        )
    else:
        print("CVXPY-rounded does not exceed crossover-greedy or either single-action frontier.")


def _cvx_failure_after_frontier(rows, cvx_frontier, deadline_scale):
    for row in sorted(
        (
            row
            for row in rows
            if row["policy"] == "CVXPY-rounded"
            and row["deadline_scale"] == deadline_scale
            and row["drain_window_s"] == PLOT_DRAIN_WINDOW_S
            and (math.isnan(cvx_frontier) or row["retained_prefill_fraction"] > cvx_frontier)
        ),
        key=lambda item: item["retained_prefill_fraction"],
    ):
        if not row["safe"]:
            return row["failure_mode"]
    return "unsafe"


if __name__ == "__main__":
    run_retained_state_frontier(parse_workload_config("Run retained-state frontier sweep."))
