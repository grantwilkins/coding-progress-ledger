from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines import (
    solve_crossover_greedy,
    solve_mixed_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import get_model
from cvxpy_solver import solve_cvxpy
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import fractional_queue_load_proxy, queue_metrics

SHED_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
SLACK_MULTIPLIERS = (0.25, 0.50, 1.00, 2.00)
POLICIES = (
    ("CVXPY-rounded", solve_cvxpy),
    ("mirror-descent-rounded", lambda problem: solve_mirror_descent(problem, eta_x0=500.0)),
    ("crossover-greedy", solve_crossover_greedy),
    ("mixed-greedy", solve_mixed_greedy),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
SWEEP_COLUMNS = (
    "policy",
    "shed_fraction",
    "slack_multiplier",
    "status",
    "safe",
    "failure_mode",
    "rounded_shed_achieved",
    "rounded_shed_target",
    "rounded_shed_ratio",
    "mean_delay",
    "p50_delay",
    "p95_delay",
    "p99_delay",
    "p95_normalized_delay",
    "miss_rate",
    "max_net_busy",
    "max_prefill_busy",
    "replay_shed_frac",
    "state_shed_frac",
    "fractional_max_net_busy_proxy",
    "fractional_max_prefill_busy_proxy",
    "objective",
)
FRONTIER_COLUMNS = (
    "policy",
    "slack_multiplier",
    "max_safe_shed_fraction",
    "p95_delay_at_frontier",
    "p95_normalized_delay_at_frontier",
    "miss_rate_at_frontier",
    "max_net_busy_at_frontier",
    "max_prefill_busy_at_frontier",
)


def run_safe_shed_frontier():
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    model = get_model("GLM-5")
    for slack_multiplier in SLACK_MULTIPLIERS:
        for shed_fraction in SHED_FRACTIONS:
            problem = make_problem(
                model,
                "transition-coupled",
                shed_fraction=shed_fraction,
                slack_multiplier=slack_multiplier,
            )
            for policy, solver in POLICIES:
                rows.append(_policy_row(problem, policy, solver, shed_fraction, slack_multiplier))

    frontier = _frontier_rows(rows)
    _write_rows(out / "shed_slack_sweep.csv", rows, SWEEP_COLUMNS)
    _write_rows(out / "safe_shed_frontier.csv", frontier, FRONTIER_COLUMNS)
    _print_latex_frontier(frontier)
    _print_diagnostics(frontier, rows)
    return rows, frontier


def _policy_row(problem, policy, solver, shed_fraction, slack_multiplier):
    base = {
        "policy": policy,
        "shed_fraction": shed_fraction,
        "slack_multiplier": slack_multiplier,
        "status": "INFEASIBLE",
        "safe": False,
        "failure_mode": "infeasible",
        **{column: math.nan for column in SWEEP_COLUMNS[6:]},
    }
    base["rounded_shed_target"] = problem.B_shed
    try:
        result = solver(problem)
    except RuntimeError:
        return base

    y = result.allocation if hasattr(result, "allocation") else result.y
    base["objective"] = getattr(result, "objective", math.nan)
    if not getattr(result, "feasible", True):
        return base

    proxy = fractional_queue_load_proxy(problem, y)
    base.update(
        {
            "fractional_max_net_busy_proxy": proxy["fractional_max_network_busy_window"],
            "fractional_max_prefill_busy_proxy": proxy["fractional_max_prefill_busy_window"],
        }
    )
    try:
        metrics = queue_metrics(problem, y)
    except ValueError:
        base.update({"status": "ROUNDING_FAILED", "failure_mode": "rounding artifact"})
        return base

    base.update(
        {
            "rounded_shed_achieved": metrics["rounded_shed_achieved"],
            "rounded_shed_target": metrics["rounded_shed_target"],
            "rounded_shed_ratio": metrics["rounded_shed_ratio"],
            "mean_delay": metrics["mean_reconstruction_delay"],
            "p50_delay": metrics["p50_reconstruction_delay"],
            "p95_delay": metrics["p95_reconstruction_delay"],
            "p99_delay": metrics["p99_reconstruction_delay"],
            "p95_normalized_delay": metrics["p95_normalized_reconstruction_delay"],
            "miss_rate": metrics["deadline_miss_rate"],
            "max_net_busy": metrics["max_network_busy_window"],
            "max_prefill_busy": metrics["max_prefill_busy_window"],
            "replay_shed_frac": metrics["replay_shed_frac"],
            "state_shed_frac": metrics["state_shed_frac"],
        }
    )
    base["safe"] = _is_safe(metrics)
    base["status"] = "SAFE" if base["safe"] else "UNSAFE"
    base["failure_mode"] = "" if base["safe"] else _failure_mode(base)
    return base


def _is_safe(metrics):
    return (
        metrics["rounded_shed_achieved"] >= metrics["rounded_shed_target"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_normalized_reconstruction_delay"] <= 1.0
    )


def _failure_mode(row):
    if row["rounded_shed_achieved"] < row["rounded_shed_target"] - 1e-9:
        return "rounding artifact"
    if row["miss_rate"] > 0.01 or row["p95_normalized_delay"] > 1.0:
        if max(row["max_net_busy"], row["max_prefill_busy"]) < 0.95:
            return "slack misses"
        return (
            "network bottleneck"
            if row["max_net_busy"] >= row["max_prefill_busy"]
            else "prefill bottleneck"
        )
    return "unsafe"


def _frontier_rows(rows, policies=tuple(policy for policy, _ in POLICIES), slack_multipliers=SLACK_MULTIPLIERS):
    frontier = []
    for policy in policies:
        for slack_multiplier in slack_multipliers:
            safe = [
                row
                for row in rows
                if row["policy"] == policy and row["slack_multiplier"] == slack_multiplier and row["safe"]
            ]
            if not safe:
                frontier.append(
                    {
                        "policy": policy,
                        "slack_multiplier": slack_multiplier,
                        **{column: "UNSAFE" for column in FRONTIER_COLUMNS[2:]},
                    }
                )
                continue
            row = max(safe, key=lambda item: item["shed_fraction"])
            frontier.append(
                {
                    "policy": policy,
                    "slack_multiplier": slack_multiplier,
                    "max_safe_shed_fraction": row["shed_fraction"],
                    "p95_delay_at_frontier": row["p95_delay"],
                    "p95_normalized_delay_at_frontier": row["p95_normalized_delay"],
                    "miss_rate_at_frontier": row["miss_rate"],
                    "max_net_busy_at_frontier": row["max_net_busy"],
                    "max_prefill_busy_at_frontier": row["max_prefill_busy"],
                }
            )
    return frontier


def _write_rows(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _frontier_value(frontier, policy, slack_multiplier):
    value = next(
        row["max_safe_shed_fraction"]
        for row in frontier
        if row["policy"] == policy and row["slack_multiplier"] == slack_multiplier
    )
    return math.nan if value == "UNSAFE" else float(value)


def _print_latex_frontier(frontier):
    print("\nsafe-shed frontier (LaTeX)")
    print("\\begin{tabular}{lrrrr}")
    print("policy & 0.25x & 0.5x & 1x & 2x \\\\")
    print("\\hline")
    for policy, _ in POLICIES:
        cells = []
        for slack_multiplier in SLACK_MULTIPLIERS:
            value = _frontier_value(frontier, policy, slack_multiplier)
            cells.append("--" if math.isnan(value) else f"{value:.2f}")
        print(f"{policy} & {' & '.join(cells)} \\\\")
    print("\\end{tabular}")


def _print_diagnostics(frontier, rows):
    md_rounding = []
    cvx_losses = []
    support = []
    for slack_multiplier in SLACK_MULTIPLIERS:
        cvx = _frontier_value(frontier, "CVXPY-rounded", slack_multiplier)
        md = _frontier_value(frontier, "mirror-descent-rounded", slack_multiplier)
        if not math.isnan(md) and (math.isnan(cvx) or md > cvx + 1e-12):
            md_rounding.append(slack_multiplier)

        values = [_frontier_value(frontier, policy, slack_multiplier) for policy, _ in POLICIES]
        best = max((value for value in values if not math.isnan(value)), default=math.nan)
        if not math.isnan(best) and (math.isnan(cvx) or cvx < best - 1e-12):
            cvx_losses.append((slack_multiplier, _cvx_failure_after_frontier(rows, cvx, slack_multiplier)))

        rivals = (
            _frontier_value(frontier, "crossover-greedy", slack_multiplier),
            _frontier_value(frontier, "replay-only", slack_multiplier),
            _frontier_value(frontier, "state-only", slack_multiplier),
        )
        if not math.isnan(cvx) and any(math.isnan(value) or cvx > value + 1e-12 for value in rivals):
            support.append(slack_multiplier)

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
        print("CVXPY-rounded wins or ties the frontier at every slack multiplier.")
    if support:
        print(
            "CVXPY-rounded supports a larger safe shed fraction than crossover-greedy "
            f"or a single-action policy at slack multipliers {support}."
        )
    else:
        print("CVXPY-rounded does not exceed crossover-greedy or either single-action frontier.")


def _cvx_failure_after_frontier(rows, cvx_frontier, slack_multiplier):
    for row in sorted(
        (
            row
            for row in rows
            if row["policy"] == "CVXPY-rounded"
            and row["slack_multiplier"] == slack_multiplier
            and (math.isnan(cvx_frontier) or row["shed_fraction"] > cvx_frontier)
        ),
        key=lambda item: item["shed_fraction"],
    ):
        if not row["safe"]:
            return row["failure_mode"]
    return "unsafe"


if __name__ == "__main__":
    run_safe_shed_frontier()
