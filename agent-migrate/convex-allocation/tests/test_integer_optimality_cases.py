"""
Claim:
Tiny retained-prefill cases can be exhaustively solved over integer class
assignments, separating the best integer convex-objective allocation from the
best integer queue schedule.

Plausible wrong implementations:
- Optimize the relaxed allocation and mistake it for the best integer objective allocation.
- Present the best integer convex-objective allocation as the best queue schedule.
- Compare queue metrics from fractional allocations instead of integer ones.
- Let a rounded policy miss the retained-prefill movement target.
- Omit the deadline-aware or repaired-CVXPY comparison row.
"""

from __future__ import annotations

import math

from experiments.run_integer_optimality_cases import (
    CASES,
    POLICIES,
    _case_rows,
    exact_integer_optimum,
    exact_integer_queue_optimum,
    make_case_problem,
)
from coefficients import compute_coefficients
from objective import objective
from queueing import evaluate_rounded_queue


def test_exact_integer_optimum_is_target_feasible_and_no_worse_than_policy_roundings():
    problem = make_case_problem(CASES[0])
    coeffs = compute_coefficients(problem)
    exact = exact_integer_optimum(problem)
    exact_metrics = evaluate_rounded_queue(problem, exact.y, drain_window_s=0.0)

    assert exact_metrics["retained_prefill_moved_s"] >= exact_metrics["retained_prefill_target_s"]
    for _, _, solver in POLICIES:
        _, integer = solver(problem)
        assert objective(problem, coeffs, exact.y) <= objective(problem, coeffs, integer) + 1e-9


def test_exact_queue_optimum_is_lexicographically_best_queue_schedule():
    problem = make_case_problem(CASES[1])
    coeffs = compute_coefficients(problem)
    queue_best = exact_integer_queue_optimum(problem)
    best_metrics = evaluate_rounded_queue(problem, queue_best.y, drain_window_s=0.0)
    best_key = (
        best_metrics["deadline_miss_rate"],
        best_metrics["p95_reconstruction_delay"],
        best_metrics["mean_reconstruction_delay"],
        objective(problem, coeffs, queue_best.y),
    )

    for _, _, solver in POLICIES:
        _, integer = solver(problem)
        metrics = evaluate_rounded_queue(problem, integer, drain_window_s=0.0)
        key = (
            metrics["deadline_miss_rate"],
            metrics["p95_reconstruction_delay"],
            metrics["mean_reconstruction_delay"],
            objective(problem, coeffs, integer),
        )
        assert best_key <= key


def test_integer_optimality_rows_report_all_policy_metrics():
    rows = _case_rows(CASES[0])
    by_policy = {row["policy"]: row for row in rows}

    assert set(by_policy) == {
        "best-integer-objective",
        "best-integer-queue",
        "CVXPY-rounded",
        "deadline-aware-rounded",
        "repaired-CVXPY-rounded",
        "crossover-greedy",
        "mixed-greedy",
        "replay-only",
        "state-only",
    }
    assert by_policy["best-integer-objective"]["fractional_objective"] == "NA"
    assert by_policy["crossover-greedy"]["fractional_objective"] == "NA"
    assert by_policy["CVXPY-rounded"]["fractional_objective"] != "NA"
    assert by_policy["best-integer-objective"]["integer_objective_gap_to_best"] == "0"
    assert by_policy["best-integer-queue"]["p95_gap_to_best_queue"] == "0"
    assert by_policy["best-integer-queue"]["miss_rate_gap_to_best_queue"] == "0"
    assert all(row["movement_target_met"] == "True" for row in rows)
    assert all(math.isfinite(float(row["integer_objective"])) for row in rows)
    assert all(math.isfinite(float(row["p95_delay"])) for row in rows)


def test_same_convex_objective_can_have_different_queue_behavior():
    rows = _case_rows(CASES[0])
    best = next(row for row in rows if row["policy"] == "best-integer-objective")
    crossover = next(row for row in rows if row["policy"] == "crossover-greedy")

    assert crossover["integer_objective"] == best["integer_objective"]
    assert crossover["miss_rate"] != best["miss_rate"]
