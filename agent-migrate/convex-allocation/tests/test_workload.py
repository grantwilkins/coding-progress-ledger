"""
Claim:
The generated shed-event workload is a deterministic, opt-in static batch
source that preserves long-context variance, weak slack correlation, admissible
cache locality, compact aggregation, and non-degenerate transition-coupled
allocation behavior.

Plausible wrong implementations:
- Use unseeded randomness or omit seed flow through make_problem.
- Collapse high-variance jobs into a smooth average class and erase the tail.
- Make slack a deterministic function of context length.
- Generate resident KV fractions larger than reusable context fractions.
- Aggregate only by length and hide slack or locality variation.
- Accidentally replace the fixed six-row regression workload.
- Produce a generated transition case that uses one destination or one action.
- Crash queue-table evaluation when a generated-workload baseline is infeasible.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baselines import solve_crossover_greedy
from catalog import get_model
from cvxpy_solver import solve_cvxpy
from evaluation import WorkloadConfig
from experiments.run_catalog_sweep import _transition_queue_rows
from problem import make_problem
from queueing import queue_metrics
from workload import assert_workload_quality, generate_workload


def _weighted_log_correlation(x, y, weights):
    x = np.repeat(np.log(x), weights.astype(int))
    y = np.repeat(np.log(y), weights.astype(int))
    return float(np.corrcoef(x, y)[0, 1])


def test_generated_workload_is_seed_reproducible():
    a = generate_workload(3, seed=7, jobs=1000, classes=12)
    b = generate_workload(3, seed=7, jobs=1000, classes=12)

    for name in ("T", "d", "slack", "h_ctx", "h_kv"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name))


def test_different_generated_seeds_change_arrays_while_preserving_invariants():
    a = generate_workload(3, seed=7, jobs=1000, classes=12)
    b = generate_workload(3, seed=11, jobs=1000, classes=12)

    assert not np.allclose(a.T, b.T)
    assert not np.allclose(a.slack, b.slack)
    for workload in (a, b):
        assert workload.T.size <= 12
        assert workload.d.sum() == 1000
        assert np.all(workload.T > 0)
        assert np.all(workload.slack > 0)
        assert np.all((0.0 <= workload.h_kv) & (workload.h_kv <= workload.h_ctx))
        assert np.all(workload.h_ctx <= 1.0)


def test_generated_lengths_have_high_variance_tail_and_weak_slack_correlation():
    workload = generate_workload(3, seed=7, jobs=2000, classes=40)
    corr = _weighted_log_correlation(workload.T, workload.slack, workload.d)

    assert workload.T.max() >= 100_000
    assert workload.T.max() / workload.T.min() > 50
    assert 0.15 < corr < 0.75
    assert np.any((workload.T >= 100_000) & (workload.slack < np.median(workload.slack)))


def test_joint_aggregation_preserves_length_slack_and_locality_variation():
    workload = generate_workload(3, seed=7, jobs=1000, classes=12)

    assert workload.T.size == 12
    assert workload.T.max() / workload.T.min() > 50
    assert workload.slack.max() / workload.slack.min() > 20
    assert np.ptp(np.max(workload.h_ctx, axis=1)) > 0.4
    assert np.ptp(np.max(workload.h_kv, axis=1)) > 0.4
    assert np.unique(np.argmax(workload.h_ctx, axis=1)).size > 1


def test_fixed_workload_source_preserves_default_problem_behavior():
    model = get_model("GLM-5")
    default = make_problem(model, "transition-coupled")
    explicit = make_problem(
        model,
        "transition-coupled",
        workload_source="fixed",
        workload_seed=7,
        workload_jobs=123,
        workload_classes=4,
    )

    for name in ("T", "d", "slack", "h_ctx", "h_kv", "B_shed"):
        np.testing.assert_allclose(getattr(default, name), getattr(explicit, name))


def test_generated_evaluation_config_is_reproducible_and_separate_from_fixed_outputs():
    root = Path("/analysis")
    fixed = WorkloadConfig()
    generated = WorkloadConfig(source="generated", seed=7)

    assert fixed.output_dir(root) == root / "outputs" / "sweep"
    assert generated.output_dir(root) == root / "outputs" / "sweep" / generated.label
    assert generated.problem_kwargs()["workload_source"] == "generated"
    assert generated.problem_kwargs()["workload_seed"] == 7
    assert generated.problem_kwargs()["workload_jobs"] == 1000


def test_generated_transition_workload_is_not_degenerate_after_solving_and_rounding():
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        workload_source="generated",
        workload_seed=7,
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
    metrics = queue_metrics(problem, cvx.y)
    assert metrics["rounded_shed_achieved"] >= metrics["rounded_shed_target"]
    assert min(metrics["replay_shed_frac"], metrics["state_shed_frac"]) > 0.05


def test_transition_queue_rows_mark_infeasible_policies_instead_of_rounding_them():
    problem = make_problem(get_model("GLM-5"), "transition-coupled")
    y = np.zeros((problem.G, 2 * problem.K + 1))
    y[:, -1] = problem.d
    result = SimpleNamespace(allocation=y, feasible=False, objective=None)

    rows = _transition_queue_rows(
        problem,
        {
            "CVXPY": result,
            "mirror-descent-best": result,
            "crossover-greedy": result,
            "mixed-greedy": result,
            "replay-only": result,
            "state-only": result,
        },
    )

    assert {row["status"] for row in rows} == {"INFEASIBLE"}
    assert all(row["rounded_shed_ratio"] == "INFEASIBLE" for row in rows)
