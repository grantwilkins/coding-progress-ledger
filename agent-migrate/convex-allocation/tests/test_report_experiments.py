"""
Claim:
The report-facing experiment tables compare claims to the named baselines,
measure rounding gaps on integer queue schedules, and include a small
adversarial case where rounding creates a deadline miss that local repair fixes.

Plausible wrong implementations:
- Mark a claim as passing without comparing winner and baseline metric values.
- Evaluate rounding-gap queue metrics on fractional allocations.
- Build an adversarial case where the rounded allocation is already safe.
- Let repair change the retained-prefill target while improving queue metrics.
"""

from __future__ import annotations

from experiments.run_report_experiments import (
    _claim_row,
    _rounding_row,
    adversarial_problem_and_relaxed_allocation,
    rounding_gap_study,
)
from experiments.run_integer_optimality_cases import CASES, exact_integer_objective_optimum, exact_integer_queue_optimum, make_case_problem
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation
from coefficients import compute_coefficients
from metrics import retained_prefill_moved_s
from queueing import evaluate_rounded_queue, round_allocation


def test_claim_row_pass_is_machine_checkable_yes_no():
    assert _claim_row("c", "m", "w", "b", 0.1, 0.2, 0.1 < 0.2)["pass"] == "yes"
    assert _claim_row("c", "m", "w", "b", 0.3, 0.2, 0.3 < 0.2)["pass"] == "no"


def test_adversarial_rounding_misses_and_repair_fixes_without_shortfall():
    problem, relaxed = adversarial_problem_and_relaxed_allocation()
    rounded = round_allocation(problem, relaxed).y
    repaired = repair_rounded_allocation(problem, rounded, drain_window_s=0.0).y
    rounded_metrics = evaluate_rounded_queue(problem, rounded, drain_window_s=0.0)
    repaired_metrics = evaluate_rounded_queue(problem, repaired, drain_window_s=0.0)

    assert (relaxed[0, 0:6:2] > 0.25).sum() == 2
    assert rounded[0, 0:6:2].tolist() == [3, 0, 0]
    assert repaired[0, 0:6:2].tolist() == [1, 2, 0]
    assert rounded_metrics["deadline_miss_rate"] > 0.0
    assert repaired_metrics["deadline_miss_rate"] == 0.0
    assert retained_prefill_moved_s(problem, repaired) >= problem.retained_prefill_target_s


def test_rounding_gap_row_uses_integer_queue_reference_metrics():
    problem = make_case_problem(CASES[0])
    coeffs = compute_coefficients(problem)
    exact_objective = exact_integer_objective_optimum(problem)
    exact_queue = exact_integer_queue_optimum(problem)
    exact_queue_metrics = evaluate_rounded_queue(problem, exact_queue.y, drain_window_s=0.0)
    rounded = round_allocation(problem, exact_objective.y).y
    metrics = evaluate_rounded_queue(problem, rounded, drain_window_s=0.0)

    row = _rounding_row(
        CASES[0].name,
        "integer-check",
        problem,
        coeffs,
        rounded,
        metrics,
        exact_objective.objective,
        exact_queue_metrics,
    )

    assert row["miss_rate_gap"] == metrics["deadline_miss_rate"] - exact_queue_metrics["deadline_miss_rate"]
    assert row["p95_delay_gap"] == metrics["p95_reconstruction_delay"] - exact_queue_metrics["p95_reconstruction_delay"]


def test_rounding_gap_study_has_requested_variants_and_summary_fraction():
    rows, summary = rounding_gap_study()
    policies = {row["policy"] for row in rows}

    assert {
        "relaxed-CVXPY",
        "exact-integer-optimum",
        "current-rounding",
        "current-rounding-plus-repair",
    } <= policies
    assert summary[0]["metric"] == "fraction_cases_where_current_rounding_changes_queue_winner"
    assert 0.0 <= summary[0]["value"] <= 1.0
