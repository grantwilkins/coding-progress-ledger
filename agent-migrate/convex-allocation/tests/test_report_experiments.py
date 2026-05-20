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
- Rebuild the generated workload once per weight or frontier grid point.
- Recompute claim frontiers even when the caller already passed them in.
"""

from __future__ import annotations

import numpy as np

from experiments.run_report_experiments import (
    _claim_row,
    _rounding_row,
    claim_table,
    adversarial_problem_and_relaxed_allocation,
    deadline_weight_sensitivity,
    model_architecture_sweep,
    rounding_gap_study,
)
import experiments.run_report_experiments as report
from experiments.run_integer_optimality_cases import CASES, exact_integer_objective_optimum, exact_integer_queue_optimum, make_case_problem
from experiments.run_queue_failure_diagnostics import repair_rounded_allocation
from coefficients import compute_coefficients
from catalog import get_model
from evaluation import WorkloadConfig
from metrics import retained_prefill_moved_s
from problem import ProblemData
from queueing import evaluate_rounded_queue, round_allocation


def tiny_problem(retained_prefill_fraction=0.4):
    model = get_model("GLM-5")
    return ProblemData(
        model=model,
        regime="transition-coupled",
        T=np.array([1.0]),
        d=np.array([1.0]),
        deadline_s=np.array([1.0]),
        lambda_Bps=np.ones(3),
        rho_prefill=np.ones(3),
        C_net=np.ones(3),
        C_prefill=np.ones(3),
        ell_net=np.zeros(3),
        ell_prefill=np.zeros(3),
        h_ctx=np.zeros((1, 3)),
        h_kv=np.zeros((1, 3)),
        retained_prefill_target_s=retained_prefill_fraction / model.prefill_tok_s,
    )


def test_claim_row_pass_is_machine_checkable_yes_no():
    assert _claim_row("c", "m", "w", "b", 0.1, 0.2, 0.1 < 0.2)["pass"] == "yes"
    assert _claim_row("c", "m", "w", "b", 0.3, 0.2, 0.3 < 0.2)["pass"] == "no"


def test_deadline_weight_sensitivity_builds_one_problem_for_weight_grid(monkeypatch):
    built = []

    def make_problem(*args, **kwargs):
        built.append(kwargs["retained_prefill_fraction"])
        return tiny_problem(kwargs["retained_prefill_fraction"])

    def run_jobs(label, jobs, fn):
        assert label == "deadline weight sensitivity"
        assert len({id(job[0]) for job in jobs}) == 1
        return [[{"ok": True}] for _ in jobs]

    monkeypatch.setattr(report, "make_problem", make_problem)
    monkeypatch.setattr(report, "_run_jobs", run_jobs)

    rows = deadline_weight_sensitivity(WorkloadConfig())

    assert built == [0.90]
    assert len(rows) == len(report.DEADLINE_HEADROOMS) * len(report.LINEAR_OVERRUN_WEIGHTS) * len(report.QUADRATIC_OVERRUN_WEIGHTS)


def test_model_architecture_sweep_reuses_one_base_problem_per_model(monkeypatch):
    built = []

    def make_problem(model, regime, **kwargs):
        built.append((model.name, regime, kwargs["retained_prefill_fraction"]))
        return tiny_problem(kwargs["retained_prefill_fraction"])

    def run_jobs(label, jobs, fn):
        assert label == "model architecture frontier"
        by_model = {}
        for key, _, _, problem in jobs:
            by_model.setdefault(key, set()).add(id(problem.T))
        assert all(len(ids) == 1 for ids in by_model.values())
        return [(key, fraction, None, None) for key, fraction, *_ in jobs]

    monkeypatch.setattr(report, "make_problem", make_problem)
    monkeypatch.setattr(report, "_run_jobs", run_jobs)

    rows = model_architecture_sweep(WorkloadConfig())

    assert [row["status"] for row in rows] == ["UNSAFE", "UNSAFE", "UNSAFE"]
    assert built == [(model.name, "bandwidth-spread", 1.0) for model in report.catalog_models()]


def test_claim_table_uses_precomputed_frontiers(monkeypatch):
    frontiers = {
        "main": {"largest_tested_safe_retained_prefill_fraction": 0.9},
        "replay": {"largest_tested_safe_retained_prefill_fraction": 0.4},
    }
    architecture_rows = [
        {"status": "SAFE", "replay_fraction": 0.2},
        {"status": "SAFE", "replay_fraction": 0.4},
    ]

    monkeypatch.setattr(report, "_claim_frontiers", lambda _: (_ for _ in ()).throw(AssertionError("recomputed")))
    monkeypatch.setattr(report, "make_problem", lambda *args, **kwargs: tiny_problem())
    monkeypatch.setattr(
        report,
        "_rounded_metrics",
        lambda *args, **kwargs: (None, {"deadline_miss_rate": 0.0}),
    )

    rows = claim_table(WorkloadConfig(), architecture_rows, frontiers)

    assert rows[1]["pass"] == "yes"


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
