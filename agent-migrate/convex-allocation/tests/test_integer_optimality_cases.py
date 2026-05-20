"""
Claim:
Tiny retained-prefill cases can be exhaustively solved over integer class
assignments, giving a true integer optimum for comparison against rounded
relaxations and greedy baselines.

Plausible wrong implementations:
- Optimize the relaxed allocation and mistake it for the true integer optimum.
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
    for _, solver in POLICIES:
        _, integer = solver(problem)
        assert objective(problem, coeffs, exact.y) <= objective(problem, coeffs, integer) + 1e-9


def test_integer_optimality_rows_report_all_policy_metrics():
    rows = _case_rows(CASES[0])
    by_policy = {row["policy"]: row for row in rows}

    assert set(by_policy) == {
        "true-best-integer",
        "CVXPY-rounded",
        "deadline-aware-rounded",
        "repaired-CVXPY-rounded",
        "crossover-greedy",
        "mixed-greedy",
        "replay-only",
        "state-only",
    }
    assert all(row["movement_target_met"] == "True" for row in rows)
    assert all(math.isfinite(float(row["integer_objective"])) for row in rows)
    assert all(math.isfinite(float(row["p95_delay"])) for row in rows)
