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

from baselines import solve_online_queue_greedy, solve_replay_only
from catalog import ModelParams, catalog_models, get_model
from coefficients import compute_coefficients
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from experiments.run_integer_optimality_cases import (
    CASES,
    exact_integer_objective_optimum,
    exact_integer_queue_optimum,
    make_case_problem,
)
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation
from metrics import retained_prefill_action_mix, retained_prefill_moved_s
from objective import objective
from problem import ProblemData, make_problem
from queueing import evaluate_rounded_queue, round_allocation

RETAINED_PREFILL_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
DEADLINE_HEADROOMS = (0.60, 0.75, 0.85, 1.00)
LINEAR_OVERRUN_WEIGHTS = (5.0, 25.0, 100.0)
QUADRATIC_OVERRUN_WEIGHTS = (25.0, 100.0, 500.0)
DRAIN_WINDOWS_S = (0.0, 900.0, 1800.0, 3600.0)
REPORT_DRAIN_WINDOW_S = 1800.0


def run_report_experiments(workload_config: WorkloadConfig = WorkloadConfig()) -> None:
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rounding_rows, rounding_summary = rounding_gap_study()
    sensitivity_rows = deadline_weight_sensitivity(workload_config)
    architecture_rows = model_architecture_sweep(workload_config)
    adversarial_rows = adversarial_queue_case()
    claim_rows = claim_table(workload_config, architecture_rows)
    _write_rows(out / "claim_table.csv", claim_rows)
    _write_rows(out / "rounding_gap_study.csv", rounding_rows)
    _write_rows(out / "rounding_gap_summary.csv", rounding_summary)
    _write_rows(out / "deadline_weight_sensitivity.csv", sensitivity_rows)
    _write_rows(out / "model_architecture_sweep.csv", architecture_rows)
    _write_rows(out / "adversarial_queue_case.csv", adversarial_rows)


def claim_table(workload_config: WorkloadConfig, architecture_rows: list[dict[str, object]] | None = None):
    transition = make_problem(get_model("GLM-5"), "transition-coupled", **workload_config.problem_kwargs())
    main = _rounded_metrics(transition, solve_soft_deadline_cvxpy, REPORT_DRAIN_WINDOW_S)[1]
    online = _rounded_metrics(transition, solve_online_queue_greedy, REPORT_DRAIN_WINDOW_S)[1]
    main_frontier = _max_safe_fraction(get_model("GLM-5"), "transition-coupled", solve_soft_deadline_cvxpy, workload_config)
    replay_frontier = _max_safe_fraction(get_model("GLM-5"), "transition-coupled", solve_replay_only, workload_config)
    architecture_rows = architecture_rows or model_architecture_sweep(workload_config)
    replay_values = [float(row["replay_fraction"]) for row in architecture_rows if row["status"] == "SAFE"]
    action_range = max(replay_values) - min(replay_values) if replay_values else math.nan
    return [
        _claim_row(
            "deadline-aware reduces miss rate",
            "miss rate",
            "deadline-aware-rounded",
            "online-queue-greedy",
            main["deadline_miss_rate"],
            online["deadline_miss_rate"],
            main["deadline_miss_rate"] < online["deadline_miss_rate"],
        ),
        _claim_row(
            "mixed action beats replay-only",
            "largest tested safe retained-prefill fraction",
            "deadline-aware-rounded",
            "replay-only",
            main_frontier["largest_tested_safe_retained_prefill_fraction"],
            replay_frontier["largest_tested_safe_retained_prefill_fraction"],
            main_frontier["largest_tested_safe_retained_prefill_fraction"]
            > replay_frontier["largest_tested_safe_retained_prefill_fraction"],
        ),
        _claim_row(
            "model architecture changes action mix",
            "replay/state fraction range",
            "varies by model",
            "fixed policy",
            action_range,
            0.0,
            bool(replay_values) and action_range > 0.05,
        ),
    ]


def rounding_gap_study():
    rows = []
    changed = 0
    for case in CASES:
        problem = make_case_problem(case)
        coeffs = compute_coefficients(problem)
        exact_objective = exact_integer_objective_optimum(problem)
        exact_queue = exact_integer_queue_optimum(problem)
        best_queue_metrics = evaluate_rounded_queue(problem, exact_queue.y, drain_window_s=0.0)
        relaxed = solve_cvxpy(problem).y
        rounded = round_allocation(problem, relaxed).y
        repaired = repair_rounded_allocation(problem, rounded, drain_window_s=0.0).y
        rounded_metrics = evaluate_rounded_queue(problem, rounded, drain_window_s=0.0)
        changed += _queue_key(rounded_metrics, objective(problem, coeffs, rounded)) != _queue_key(
            best_queue_metrics, objective(problem, coeffs, exact_queue.y)
        )
        variants = (
            ("relaxed-CVXPY", relaxed, None),
            ("exact-integer-optimum", exact_objective.y, evaluate_rounded_queue(problem, exact_objective.y, drain_window_s=0.0)),
            ("current-rounding", rounded, rounded_metrics),
            ("current-rounding-plus-repair", repaired, evaluate_rounded_queue(problem, repaired, drain_window_s=0.0)),
        )
        for policy, y, metrics in variants:
            rows.append(_rounding_row(case.name, policy, problem, coeffs, y, metrics, exact_objective.objective, best_queue_metrics))
    summary = [{"metric": "fraction_cases_where_current_rounding_changes_queue_winner", "value": changed / len(CASES)}]
    return rows, summary


def deadline_weight_sensitivity(workload_config: WorkloadConfig):
    rows = []
    model = get_model("GLM-5")
    for headroom in DEADLINE_HEADROOMS:
        for linear in LINEAR_OVERRUN_WEIGHTS:
            for quadratic in QUADRATIC_OVERRUN_WEIGHTS:
                problem = make_problem(
                    model,
                    "transition-coupled",
                    retained_prefill_fraction=0.90,
                    deadline_scale=1.0,
                    **workload_config.problem_kwargs(),
                )
                try:
                    result = solve_soft_deadline_cvxpy(problem, headroom, linear, quadratic)
                    y = round_allocation(problem, result.y).y
                except (RuntimeError, ValueError):
                    for drain in DRAIN_WINDOWS_S:
                        rows.append(_empty_sensitivity_row(headroom, linear, quadratic, drain))
                    continue
                for drain in DRAIN_WINDOWS_S:
                    metrics = evaluate_rounded_queue(problem, y, drain_window_s=drain)
                    mix = retained_prefill_action_mix(problem, y)
                    rows.append(
                        {
                            "deadline_headroom": headroom,
                            "linear_overrun_weight": linear,
                            "quadratic_overrun_weight": quadratic,
                            "drain_window_s": drain,
                            "status": "OK",
                            "safe": _safe(metrics),
                            "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
                            "deadline_miss_rate": metrics["deadline_miss_rate"],
                            "network_capacity_pressure": metrics["network_capacity_pressure"],
                            "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
                            **mix,
                        }
                    )
    return rows


def model_architecture_sweep(workload_config: WorkloadConfig):
    rows = []
    for model in catalog_models():
        frontier = _max_safe_fraction(model, "bandwidth-spread", solve_soft_deadline_cvxpy, workload_config)
        rows.append({"model": model.name, **frontier})
    return rows


def adversarial_queue_case():
    problem, relaxed = adversarial_problem_and_relaxed_allocation()
    rounded = round_allocation(problem, relaxed).y
    repaired = repair_rounded_allocation(problem, rounded, drain_window_s=0.0).y
    try:
        deadline_aware = round_allocation(
            problem,
            solve_deadline_aware_cvxpy(problem, 1.0, retained_prefill_cap=problem.retained_prefill_target_s).y,
        ).y
    except RuntimeError:
        deadline_aware = None
    rows = [
        _adversarial_row("current-rounded", problem, rounded),
        _adversarial_row("repaired-rounded", problem, repaired),
    ]
    if deadline_aware is not None:
        rows.append(_adversarial_row("deadline-aware-rounded", problem, deadline_aware))
    return rows


def adversarial_problem_and_relaxed_allocation():
    model = ModelParams("adversarial", 0.0, 1e12, 1.0, 0.0)
    c_prefill = np.array([150.53583109, 149.87117605, 146.77163791])
    window_s = 100.0
    lambda_bps = np.full(3, 1e9)
    problem = ProblemData(
        model=model,
        regime="adversarial-queue",
        T=np.array([10.0]),
        d=np.array([3.0]),
        deadline_s=np.array([15.0]),
        lambda_Bps=lambda_bps,
        rho_prefill=c_prefill / window_s,
        C_net=lambda_bps * window_s,
        C_prefill=c_prefill,
        ell_net=np.zeros(3),
        ell_prefill=np.array([27.54993171, 8.09898146, 97.24003298]),
        h_ctx=np.zeros((1, 3)),
        h_kv=np.zeros((1, 3)),
        retained_prefill_target_s=30.0,
    )
    return problem, solve_cvxpy(problem).y


def _max_safe_fraction(model, regime, solver, workload_config):
    best = None
    for fraction in RETAINED_PREFILL_FRACTIONS:
        problem = make_problem(model, regime, retained_prefill_fraction=fraction, **workload_config.problem_kwargs())
        try:
            y, metrics = _rounded_metrics(problem, solver, REPORT_DRAIN_WINDOW_S)
        except (RuntimeError, ValueError):
            continue
        if _safe(metrics):
            mix = retained_prefill_action_mix(problem, y)
            bottleneck = "network" if metrics["network_capacity_pressure"] >= metrics["prefill_capacity_pressure"] else "prefill"
            best = {
                "status": "SAFE",
                "largest_tested_safe_retained_prefill_fraction": fraction,
                "frontier_censored_by_grid": fraction == RETAINED_PREFILL_FRACTIONS[-1],
                "bottleneck_type": bottleneck,
                "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
                "deadline_miss_rate": metrics["deadline_miss_rate"],
                "network_capacity_pressure": metrics["network_capacity_pressure"],
                "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
                "replay_fraction": mix["replay_retained_prefill_fraction"],
                "state_transfer_fraction": mix["state_transfer_retained_prefill_fraction"],
            }
    return best or {
        "status": "UNSAFE",
        "largest_tested_safe_retained_prefill_fraction": math.nan,
        "frontier_censored_by_grid": False,
        "bottleneck_type": "unsafe",
        "p95_delay_over_deadline": math.nan,
        "deadline_miss_rate": math.nan,
        "network_capacity_pressure": math.nan,
        "prefill_capacity_pressure": math.nan,
        "replay_fraction": math.nan,
        "state_transfer_fraction": math.nan,
    }


def _rounded_metrics(problem, solver, drain_window_s):
    result = solver(problem)
    y = result.allocation if hasattr(result, "allocation") else result.y
    rounded = y if np.allclose(y, np.rint(y)) else round_allocation(problem, y).y
    return rounded, evaluate_rounded_queue(problem, rounded, drain_window_s=drain_window_s)


def _claim_row(claim, metric, winner, baseline, winner_value, baseline_value, passed):
    return {
        "claim": claim,
        "metric": metric,
        "winner": winner,
        "best_baseline": baseline,
        "winner_value": winner_value,
        "best_baseline_value": baseline_value,
        "pass": "yes" if passed else "no",
    }


def _rounding_row(case, policy, problem, coeffs, y, metrics, best_objective, best_queue_metrics):
    moved = retained_prefill_moved_s(problem, y)
    row = {
        "case": case,
        "policy": policy,
        "objective_gap": objective(problem, coeffs, y) - best_objective,
        "movement_shortfall": max(0.0, problem.retained_prefill_target_s - moved),
    }
    if metrics is None:
        row.update({"miss_rate_gap": "NA", "p95_delay_gap": "NA"})
    else:
        row.update(
            {
                "miss_rate_gap": metrics["deadline_miss_rate"] - best_queue_metrics["deadline_miss_rate"],
                "p95_delay_gap": metrics["p95_reconstruction_delay"] - best_queue_metrics["p95_reconstruction_delay"],
            }
        )
    return row


def _queue_key(metrics, value):
    return (
        metrics["deadline_miss_rate"],
        metrics["p95_reconstruction_delay"],
        metrics["mean_reconstruction_delay"],
        value,
    )


def _empty_sensitivity_row(headroom, linear, quadratic, drain):
    return {
        "deadline_headroom": headroom,
        "linear_overrun_weight": linear,
        "quadratic_overrun_weight": quadratic,
        "drain_window_s": drain,
        "status": "INFEASIBLE",
        "safe": False,
        "p95_delay_over_deadline": math.nan,
        "deadline_miss_rate": math.nan,
        "network_capacity_pressure": math.nan,
        "prefill_capacity_pressure": math.nan,
        "replay_retained_prefill_fraction": math.nan,
        "state_transfer_retained_prefill_fraction": math.nan,
    }


def _adversarial_row(policy, problem, y):
    metrics = evaluate_rounded_queue(problem, y, drain_window_s=0.0)
    moved = y[0, :6].reshape(3, 2)
    return {
        "policy": policy,
        "destination_counts": " ".join(str(int(n)) for n in np.sum(moved, axis=1)),
        "replay_requests": int(np.sum(moved[:, 0])),
        "state_transfer_requests": int(np.sum(moved[:, 1])),
        "deadline_miss_rate": metrics["deadline_miss_rate"],
        "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
        "safe": _safe(metrics),
    }


def _safe(metrics):
    return (
        metrics["retained_prefill_moved_s"] >= metrics["retained_prefill_target_s"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_reconstruction_delay_ratio"] <= 1.0
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_report_experiments(parse_workload_config("Run report-facing claim, sensitivity, architecture, and gap tables."))
