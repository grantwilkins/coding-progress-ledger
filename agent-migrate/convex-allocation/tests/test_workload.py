"""
Claim:
The generated retained-session workload is a deterministic static active-session
batch that preserves long-context variance, weak deadline correlation,
admissible cache locality, compact aggregation, and non-degenerate
transition-coupled allocation behavior.

Plausible wrong implementations:
- Use unseeded randomness or omit seed flow through make_problem.
- Collapse high-variance jobs into a smooth average class and erase the tail.
- Make deadlines a deterministic function of context length.
- Generate resident KV fractions larger than reusable context fractions.
- Aggregate only by length and hide deadline or locality variation.
- Accidentally make the retained-session default look like the fixed six-row smoke workload.
- Produce a generated transition case that collapses to one destination.
- Crash queue-table evaluation when a generated-workload baseline is infeasible.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baselines import solve_crossover_greedy
from catalog import get_model
from coefficients import REPLAY, compute_coefficients
from cvxpy_solver import solve_cvxpy
from evaluation import WorkloadConfig
from experiments.run_catalog_sweep import _infeasible_result, _transition_queue_rows, run_transition_coupled
from metrics import available_rates
from problem import make_problem
from queueing import evaluate_rounded_queue
from workload import assert_workload_quality, generate_workload


def _weighted_log_correlation(x, y, weights):
    x = np.repeat(np.log(x), weights.astype(int))
    y = np.repeat(np.log(y), weights.astype(int))
    return float(np.corrcoef(x, y)[0, 1])


def test_generated_workload_is_seed_reproducible():
    a = generate_workload(3, seed=7, jobs=1000, classes=12)
    b = generate_workload(3, seed=7, jobs=1000, classes=12)

    for name in ("T", "d", "deadline_s", "h_ctx", "h_kv"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name))


def test_different_generated_seeds_change_arrays_while_preserving_invariants():
    a = generate_workload(3, seed=7, jobs=1000, classes=12)
    b = generate_workload(3, seed=11, jobs=1000, classes=12)

    assert not np.allclose(a.T, b.T)
    assert not np.allclose(a.deadline_s, b.deadline_s)
    for workload in (a, b):
        assert workload.T.size <= 12
        assert workload.d.sum() == 1000
        assert np.all(workload.T > 0)
        assert np.all(workload.deadline_s > 0)
        assert np.all((0.0 <= workload.h_kv) & (workload.h_kv <= workload.h_ctx))
        assert np.all(workload.h_ctx <= 1.0)


def test_generated_lengths_have_high_variance_tail_and_weak_deadline_correlation():
    workload = generate_workload(3, seed=7, jobs=2000, classes=40)
    corr = _weighted_log_correlation(workload.T, workload.deadline_s, workload.d)

    assert workload.T.max() >= 180_000
    assert workload.T.max() / workload.T.min() > 15
    assert 0.15 < corr < 0.75
    assert np.any((workload.T >= 100_000) & (workload.deadline_s < np.median(workload.deadline_s)))


def test_joint_aggregation_preserves_length_deadline_and_locality_variation():
    workload = generate_workload(3, seed=7, jobs=1000, classes=12)

    assert workload.T.size == 12
    assert workload.T.max() / workload.T.min() > 15
    assert workload.deadline_s.max() / workload.deadline_s.min() > 5
    assert np.ptp(np.max(workload.h_ctx, axis=1)) > 0.4
    assert np.ptp(np.max(workload.h_kv, axis=1)) > 0.4
    assert np.unique(np.argmax(workload.h_ctx, axis=1)).size > 1


def test_generated_retained_sessions_are_default_and_fixed_is_explicit():
    model = get_model("GLM-5")
    default = make_problem(model, "transition-coupled")
    repeated = make_problem(model, "transition-coupled")
    fixed = make_problem(
        model,
        "transition-coupled",
        workload_source="fixed",
        workload_seed=7,
        workload_jobs=123,
        workload_classes=4,
    )

    assert default.G == 48
    assert default.d.sum() == 10_000
    np.testing.assert_allclose(default.T, repeated.T)
    np.testing.assert_allclose(default.d, repeated.d)
    np.testing.assert_allclose(default.h_ctx, repeated.h_ctx)
    assert fixed.G == 6
    assert not np.allclose(default.T[: fixed.G], fixed.T)


def test_problem_derived_values_recompute_after_array_mutation():
    problem = make_problem(get_model("GLM-5"), "transition-coupled", workload_source="fixed")
    coeffs = compute_coefficients(problem)
    _, lambda_avail, _ = available_rates(problem)

    problem.h_ctx[0, 0] = 0.5
    problem.ell_net[0] *= 0.5

    changed_coeffs = compute_coefficients(problem)
    _, changed_lambda_avail, _ = available_rates(problem)

    assert changed_coeffs.b_net[0, 0, REPLAY] < coeffs.b_net[0, 0, REPLAY]
    assert changed_lambda_avail[0] > lambda_avail[0]


def test_generated_evaluation_config_is_reproducible_and_separate_from_fixed_outputs():
    root = Path("/analysis")
    generated = WorkloadConfig()
    fixed = WorkloadConfig(source="fixed")

    assert fixed.output_dir(root) == root / "outputs" / "sweep"
    assert generated.output_dir(root) == root / "outputs" / "sweep" / generated.label
    assert generated.problem_kwargs()["workload_source"] == "generated"
    assert generated.problem_kwargs()["workload_seed"] == 7
    assert generated.problem_kwargs()["workload_jobs"] == 10_000
    assert generated.problem_kwargs()["workload_classes"] == 48
    assert "sessions10000" in generated.label


def test_generated_transition_workload_is_not_degenerate_after_solving():
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        workload_source="generated",
        workload_seed=7,
        workload_jobs=2000,
        workload_classes=32,
        gpu_count=np.full(3, 0.3),
        window_s=900.0,
    )
    cvx = solve_cvxpy(problem)
    crossover = solve_crossover_greedy(problem)

    assert_workload_quality(
        problem,
        cvx.y,
        crossover.allocation,
        cvx.objective,
        crossover.objective,
        crossover.feasible,
    )


def test_generated_queue_metrics_handle_small_integer_retained_fixture():
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        workload_source="generated",
        workload_seed=7,
        workload_jobs=48,
        workload_classes=12,
    )
    coeffs = compute_coefficients(problem)
    replay_to_site_0 = int(np.flatnonzero((coeffs.option_dest == 0) & (coeffs.option_action == REPLAY))[0])
    y = np.zeros((problem.G, coeffs.M + 1), dtype=int)
    y[:, -1] = problem.d.astype(int)
    for g in range(problem.G):
        if y[g, -1]:
            y[g, replay_to_site_0] = 1
            y[g, -1] -= 1

    metrics = evaluate_rounded_queue(problem, y)

    assert metrics["retained_prefill_moved_s"] > 0.0
    assert np.isfinite(metrics["p95_reconstruction_delay_ratio"])
    assert metrics["resident_state_tb"] > 0.0


def test_transition_queue_rows_mark_infeasible_policies_instead_of_rounding_them():
    problem = make_problem(get_model("GLM-5"), "transition-coupled")
    y = np.zeros((problem.G, 2 * problem.K + 1))
    y[:, -1] = problem.d
    result = SimpleNamespace(allocation=y, feasible=False, objective=None)

    rows = _transition_queue_rows(
        problem,
        {
            "deadline-penalty": result,
            "CVXPY": result,
            "mirror-descent-best": result,
            "crossover-greedy": result,
            "mixed-greedy": result,
            "replay-only": result,
            "state-only": result,
        },
    )

    assert {row["status"] for row in rows} == {"INFEASIBLE"}
    assert [row["policy"] for row in rows[:2]] == ["deadline-penalty-rounded", "CVXPY-rounded"]
    assert all(row["retained_prefill_ratio"] == "INFEASIBLE" for row in rows)


def test_catalog_infeasible_result_preserves_generated_policy_table_shape():
    problem = make_problem(get_model("GLM-5"), "transition-coupled", workload_jobs=48, workload_classes=12)
    result = _infeasible_result(problem)

    rows = _transition_queue_rows(problem, {"mirror-descent-best": result})

    assert len(rows) == 1
    assert rows[0]["policy"] == "mirror-descent-rounded"
    assert rows[0]["status"] == "INFEASIBLE"


def test_fixed_transition_sweep_skips_generated_quality_gate(tmp_path):
    run_transition_coupled(tmp_path, WorkloadConfig(source="fixed"))

    header = (tmp_path / "transition_coupled_policy_table.csv").read_text().splitlines()[0]
    assert "retained_prefill_moved_s" in header
