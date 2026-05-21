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
    solve_least_loaded_destination,
    solve_online_queue_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import get_model
from cvxpy_solver import solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config, run_jobs as _run_jobs
from metrics import (
    average_equivalent_state_target_bytes,
    nvl72_hbm_fraction,
    resident_state_bytes,
    resident_state_moved_bytes,
    retained_prefill_moved_s,
    state_tb,
    total_retained_prefill_s,
)
from problem import make_problem, with_retained_prefill_fraction
from queueing import DEFAULT_RELEASE_SEED, RELEASE_POLICIES, fractional_queue_load_proxy, queue_metrics

DRAIN_WINDOWS_S = (10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1200.0, 2400.0, 3600.0)
BINARY_TOLERANCE = 0.01
VALIDATION_OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)
STRESS_FRACTIONS = (0.25,)
DEADLINE_SCALE = 1.0
MAIN_POLICY = "deadline-penalty-rounded"
POLICIES = (
    (MAIN_POLICY, solve_soft_deadline_cvxpy),
    ("online-queue-greedy", solve_online_queue_greedy),
    ("least-loaded-destination", solve_least_loaded_destination),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
FRONTIER_POLICIES = tuple(policy for policy, _ in POLICIES)
SWEEP_COLUMNS = (
    "policy",
    "release_policy",
    "release_seed",
    "drain_window_s",
    "deadline_scale",
    "retained_prefill_fraction",
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
    "mean_delay_s",
    "p50_delay_s",
    "p95_delay_s",
    "p99_delay_s",
    "p95_delay_over_deadline",
    "deadline_miss_rate",
    "absolute_p95_delay_over_deadline",
    "absolute_deadline_miss_rate",
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
    "release_policy",
    "release_seed",
    "drain_window_s",
    "deadline_scale",
    "max_safe_retained_prefill_fraction",
    "frontier_censored_by_search",
    "first_unsafe_retained_prefill_fraction",
    "first_unsafe_failure_mode",
    "absolute_p95_delay_over_deadline_at_frontier",
    "absolute_deadline_miss_rate_at_frontier",
    "p95_delay_over_deadline_at_frontier",
    "deadline_miss_rate_at_frontier",
    "network_capacity_pressure_at_frontier",
    "prefill_capacity_pressure_at_frontier",
    "average_equivalent_state_target_tb_at_frontier",
    "actual_evacuated_state_tb_at_frontier",
    "actual_evacuated_nvl72_hbm_fraction_at_frontier",
    "replay_retained_prefill_fraction_at_frontier",
    "state_transfer_retained_prefill_fraction_at_frontier",
    "drain_completion_s_at_frontier",
)


def run_retained_state_frontier(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    model = get_model("GLM-5")
    base_by_window = {
        drain_window_s: make_problem(
            model,
            "transition-coupled",
            retained_prefill_fraction=1.0,
            deadline_scale=DEADLINE_SCALE,
            window_s=drain_window_s,
            **workload_config.problem_kwargs(),
        )
        for drain_window_s in DRAIN_WINDOWS_S
    }
    jobs = [
        (
            policy,
            solver,
            release_policy,
            DEFAULT_RELEASE_SEED,
            drain_window_s,
            base_by_window[drain_window_s],
        )
        for drain_window_s in DRAIN_WINDOWS_S
        for policy, solver in POLICIES
        for release_policy in RELEASE_POLICIES
    ]
    batches = _run_jobs("retained-state drain frontier", jobs, _frontier_job)
    rows = [row for sweep_rows, _ in batches for row in sweep_rows]
    frontier = _monotone_frontier([frontier_row for _, frontier_row in batches])
    _write_rows(out / "retained_state_drain_sweep.csv", rows, SWEEP_COLUMNS)
    _write_rows(out / "retained_state_drain_frontier.csv", frontier, FRONTIER_COLUMNS)
    _print_latex_frontier(frontier)
    _print_diagnostics(frontier)
    return rows, frontier


def _frontier_job(job):
    policy, solver, release_policy, release_seed, drain_window_s, base = job
    cache = {}

    def row_at(fraction):
        fraction = _fraction(fraction)
        if fraction not in cache:
            cache[fraction] = _policy_row(
                base, policy, solver, release_policy, release_seed, fraction, drain_window_s
            )
        return cache[fraction]

    low, high = 0.0, 1.0
    while high - low > BINARY_TOLERANCE:
        mid = (low + high) / 2.0
        if row_at(mid)["safe"]:
            low = mid
        else:
            high = mid

    center = round(low, 2)
    for offset in VALIDATION_OFFSETS:
        row_at(center + offset)
    for fraction in STRESS_FRACTIONS:
        row_at(fraction)
    safe = [row for row in cache.values() if row["safe"]]
    frontier = max(safe, key=lambda row: row["retained_prefill_fraction"]) if safe else None
    unsafe_fraction = 0.0 if frontier is None else frontier["retained_prefill_fraction"] + BINARY_TOLERANCE
    if unsafe_fraction <= 1.0:
        row_at(unsafe_fraction)
    return list(cache.values()), _frontier_row(
        policy, release_policy, release_seed, drain_window_s, safe, cache.values()
    )


def _policy_row(base, policy, solver, release_policy, release_seed, retained_prefill_fraction, drain_window_s):
    problem = with_retained_prefill_fraction(base, retained_prefill_fraction)
    row = _empty_row(
        policy, release_policy, release_seed, retained_prefill_fraction, drain_window_s, problem.retained_prefill_target_s
    )
    try:
        result = solver(problem)
    except RuntimeError:
        row["failure_mode"] = "solver_infeasible"
        return row

    y = result.allocation if hasattr(result, "allocation") else result.y
    row["objective"] = getattr(result, "objective", math.nan)
    row.update(_solver_metrics(problem, y))
    row.update(fractional_queue_load_proxy(problem, y))
    feasible = (
        getattr(result, "feasible", True)
        and row["retained_prefill_moved_s"] >= problem.retained_prefill_target_s - 1e-5
    )
    try:
        metrics = queue_metrics(
            problem,
            y,
            drain_window_s=drain_window_s,
            release_policy=release_policy,
            release_seed=release_seed,
        )
        row.update(_queue_fields(metrics) if feasible else _queue_diagnostic_fields(metrics))
    except ValueError:
        pass
    if not feasible:
        row["failure_mode"] = "target_not_met"
        return row

    if math.isnan(row["network_capacity_pressure"]):
        row["failure_mode"] = "target_not_met"
        return row
    row["safe"] = _is_safe(row)
    row["status"] = "SAFE" if row["safe"] else "UNSAFE"
    row["failure_mode"] = "" if row["safe"] else _failure_mode(row)
    return row


def _queue_diagnostic_fields(metrics):
    return {
        key: value
        for key, value in _queue_fields(metrics).items()
        if key
        not in {
            "retained_prefill_moved_s",
            "resident_state_tb",
            "average_equivalent_state_target_tb",
            "actual_evacuated_state_tb",
            "retained_prefill_moved_fraction",
            "actual_evacuated_nvl72_hbm_fraction",
        }
    }


def _empty_row(policy, release_policy, release_seed, retained_prefill_fraction, drain_window_s, retained_prefill_target_s):
    row = {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": _release_seed_value(release_policy, release_seed),
        "drain_window_s": drain_window_s,
        "deadline_scale": DEADLINE_SCALE,
        "retained_prefill_fraction": retained_prefill_fraction,
        "status": "INFEASIBLE",
        "safe": False,
        "failure_mode": "solver_infeasible",
        **{column: math.nan for column in SWEEP_COLUMNS[9:]},
    }
    row["retained_prefill_target_s"] = retained_prefill_target_s
    return row


def _solver_metrics(problem, y):
    retained = retained_prefill_moved_s(problem, y)
    total = total_retained_prefill_s(problem)
    moved_bytes = resident_state_moved_bytes(problem, y)
    return {
        "retained_prefill_moved_s": retained,
        "resident_state_tb": state_tb(resident_state_bytes(problem)),
        "average_equivalent_state_target_tb": state_tb(average_equivalent_state_target_bytes(problem)),
        "actual_evacuated_state_tb": state_tb(moved_bytes),
        "retained_prefill_moved_fraction": retained / total,
        "actual_evacuated_nvl72_hbm_fraction": nvl72_hbm_fraction(moved_bytes),
        "request_migration_fraction": float(np.sum(y[:, : y.shape[1] - 1]) / np.sum(problem.d)),
    }


def _queue_fields(metrics):
    return {
        "retained_prefill_moved_s": metrics["retained_prefill_moved_s"],
        "retained_prefill_removal_rate_s_per_s": metrics["retained_prefill_removal_rate_s_per_s"],
        "mean_delay_s": metrics["mean_reconstruction_delay"],
        "p50_delay_s": metrics["p50_reconstruction_delay"],
        "p95_delay_s": metrics["p95_reconstruction_delay"],
        "p99_delay_s": metrics["p99_reconstruction_delay"],
        "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
        "deadline_miss_rate": metrics["deadline_miss_rate"],
        "absolute_p95_delay_over_deadline": metrics["absolute_p95_delay_over_deadline"],
        "absolute_deadline_miss_rate": metrics["absolute_deadline_miss_rate"],
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


def _is_safe(metrics):
    return (
        metrics["retained_prefill_moved_s"] >= metrics["retained_prefill_target_s"] - 1e-9
        and metrics["absolute_deadline_miss_rate"] <= 0.01
        and metrics["absolute_p95_delay_over_deadline"] <= 1.0
    )


def _failure_mode(row):
    if row["retained_prefill_moved_s"] < row["retained_prefill_target_s"] - 1e-9:
        return "target_not_met"
    if row["absolute_deadline_miss_rate"] > 0.01:
        return "absolute_deadline_miss"
    if row["absolute_p95_delay_over_deadline"] > 1.0:
        return "absolute_p95_delay"
    if row["network_capacity_pressure"] >= row["prefill_capacity_pressure"]:
        return "network_pressure"
    return "prefill_pressure"


def _frontier_row(policy, release_policy, release_seed, drain_window_s, safe, rows):
    rows = list(rows)
    if not safe:
        unsafe = min(rows, key=lambda row: row["retained_prefill_fraction"])
        return {
            "policy": policy,
            "release_policy": release_policy,
            "release_seed": _release_seed_value(release_policy, release_seed),
            "drain_window_s": drain_window_s,
            "deadline_scale": DEADLINE_SCALE,
            "max_safe_retained_prefill_fraction": "UNSAFE",
            "frontier_censored_by_search": False,
            "first_unsafe_retained_prefill_fraction": unsafe["retained_prefill_fraction"],
            "first_unsafe_failure_mode": unsafe["failure_mode"],
            **{column: "UNSAFE" for column in FRONTIER_COLUMNS[9:]},
        }
    best = max(safe, key=lambda row: row["retained_prefill_fraction"])
    first_unsafe = min(
        (row for row in rows if row["retained_prefill_fraction"] > best["retained_prefill_fraction"] and not row["safe"]),
        key=lambda row: row["retained_prefill_fraction"],
        default=None,
    )
    return {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": _release_seed_value(release_policy, release_seed),
        "drain_window_s": drain_window_s,
        "deadline_scale": DEADLINE_SCALE,
        "max_safe_retained_prefill_fraction": best["retained_prefill_fraction"],
        "frontier_censored_by_search": best["retained_prefill_fraction"] >= 1.0 and first_unsafe is None,
        "first_unsafe_retained_prefill_fraction": "" if first_unsafe is None else first_unsafe["retained_prefill_fraction"],
        "first_unsafe_failure_mode": "" if first_unsafe is None else first_unsafe["failure_mode"],
        "absolute_p95_delay_over_deadline_at_frontier": best["absolute_p95_delay_over_deadline"],
        "absolute_deadline_miss_rate_at_frontier": best["absolute_deadline_miss_rate"],
        "p95_delay_over_deadline_at_frontier": best["p95_delay_over_deadline"],
        "deadline_miss_rate_at_frontier": best["deadline_miss_rate"],
        "network_capacity_pressure_at_frontier": best["network_capacity_pressure"],
        "prefill_capacity_pressure_at_frontier": best["prefill_capacity_pressure"],
        "average_equivalent_state_target_tb_at_frontier": best["average_equivalent_state_target_tb"],
        "actual_evacuated_state_tb_at_frontier": best["actual_evacuated_state_tb"],
        "actual_evacuated_nvl72_hbm_fraction_at_frontier": best["actual_evacuated_nvl72_hbm_fraction"],
        "replay_retained_prefill_fraction_at_frontier": best["replay_retained_prefill_fraction"],
        "state_transfer_retained_prefill_fraction_at_frontier": best["state_transfer_retained_prefill_fraction"],
        "drain_completion_s_at_frontier": best["drain_completion_s"],
    }


def _monotone_frontier(rows):
    by_key = {(row["policy"], row["release_policy"], row["drain_window_s"]): row for row in rows}
    out = []
    for policy in FRONTIER_POLICIES:
        for release_policy in RELEASE_POLICIES:
            for drain_window_s in DRAIN_WINDOWS_S:
                out.append(by_key[(policy, release_policy, drain_window_s)])
    return out


def _fraction(value):
    return round(min(1.0, max(0.0, float(value))), 4)


def _write_rows(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _frontier_value(frontier, policy, drain_window_s):
    value = next(
        row["max_safe_retained_prefill_fraction"]
        for row in frontier
        if row["policy"] == policy and row["release_policy"] == "edf" and row["drain_window_s"] == drain_window_s
    )
    return math.nan if value == "UNSAFE" else float(value)


def _print_latex_frontier(frontier):
    print("\nretained-state drain frontier (LaTeX)")
    print("\\begin{tabular}{lrrrrrrrrrr}")
    print("policy & 10s & 20s & 40s & 80s & 160s & 300s & 600s & 1200s & 2400s & 3600s \\\\")
    print("\\hline")
    for policy in FRONTIER_POLICIES:
        cells = []
        for drain_window_s in DRAIN_WINDOWS_S:
            value = _frontier_value(frontier, policy, drain_window_s)
            cells.append("--" if math.isnan(value) else f"{value:.2f}")
        print(f"{policy} & {' & '.join(cells)} \\\\")
    print("\\end{tabular}")


def _print_diagnostics(frontier):
    support = []
    for drain_window_s in DRAIN_WINDOWS_S:
        main = _frontier_value(frontier, MAIN_POLICY, drain_window_s)
        rivals = [
            _frontier_value(frontier, policy, drain_window_s)
            for policy in FRONTIER_POLICIES
            if policy != MAIN_POLICY
        ]
        if not math.isnan(main) and all(math.isnan(value) or main > value + 1e-12 for value in rivals):
            support.append(drain_window_s)
    if support:
        print(f"\n{MAIN_POLICY} supports the largest frontier at drain windows {support}.")
    else:
        print(f"\n{MAIN_POLICY} does not exceed every baseline frontier.")


def _release_seed_value(release_policy, release_seed):
    return release_seed if release_policy == "random" else ""


if __name__ == "__main__":
    run_retained_state_frontier(parse_workload_config("Run retained-state drain frontier sweep."))
