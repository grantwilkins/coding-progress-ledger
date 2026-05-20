"""
Claim:
The CVXPY oracle and mirror descent with scalar bisection solve the same convex
relaxation up to objective and feasibility tolerances. The deadline-aware CVXPY
planner maximizes retained prefill under cumulative per-destination deadline capacity.

Plausible wrong implementations:
- Give mirror descent a penalty objective that differs from the CVXPY problem.
- Use the wrong load-gradient sign, so larger alpha rewards staying.
- Update the wrong bisection bound and return an infeasible allocation.
- Compare allocations entrywise even when equivalent optima can differ.
- Accept capacity or retained-prefill violations as solver success.
- Mark a partial greedy baseline as feasible after it hits capacity.
- Claim catalog action-mix conclusions from diagnostics instead of the CVXPY oracle.
- Let the mixed greedy baseline ignore current destination load.
- Let the crossover baseline quietly become another coupled-load greedy solver.
- Let the online queue baseline ignore prefill backlog when replay queues build.
- Let the online queue baseline ignore network backlog when choosing destinations.
- Accept a transition-coupled scenario that is single-destination or single-action.
- Use full window capacity instead of available deadline-rate capacity.
- Enforce deadline buckets as disjoint bins instead of cumulative thresholds.
- Ignore the retained-prefill cap when scanning a frontier.
"""

from __future__ import annotations

import numpy as np

from baselines import (
    solve_crossover_greedy,
    solve_mixed_greedy,
    solve_online_queue_greedy,
    solve_replay_only,
)
from catalog import ModelParams, catalog_models, get_model
from coefficients import REPLAY, STATE, compute_coefficients
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy, solve_soft_deadline_cvxpy
from metrics import allocation_diagnostics, assert_feasible, retained_prefill_action_mix, retained_prefill_moved_s
from mirror_descent import solve_mirror_descent
from problem import ProblemData, make_problem


def small_problem() -> ProblemData:
    model = ModelParams("small", 4.0, 120.0, 1_000.0, 0.0)
    T = np.array([80.0, 300.0])
    d = np.array([4.0, 3.0])
    total_retained_prefill_s = float(np.dot(T / model.prefill_tok_s, d))
    return ProblemData(
        model=model,
        regime="small",
        T=T,
        d=d,
        deadline_s=np.array([2.0, 9.0]),
        lambda_Bps=np.array([25_000.0, 80_000.0]),
        rho_prefill=np.array([1_700.0, 2_300.0]),
        C_net=np.array([40_000.0, 55_000.0]),
        C_prefill=np.array([6_000.0, 5_000.0]),
        ell_net=np.array([4_000.0, 8_000.0]),
        ell_prefill=np.array([600.0, 1_200.0]),
        h_ctx=np.array([[0.0, 0.2], [0.1, 0.0]]),
        h_kv=np.array([[0.0, 0.1], [0.3, 0.0]]),
        retained_prefill_target_s=0.35 * total_retained_prefill_s,
    )


def deadline_rate_problem() -> ProblemData:
    model = ModelParams("deadline", 1.0, 1.0, 10.0, 0.0)
    return ProblemData(
        model=model,
        regime="deadline",
        T=np.array([10.0, 10.0]),
        d=np.array([1.0, 1.0]),
        deadline_s=np.array([1.0, 100.0]),
        lambda_Bps=np.array([10.0]),
        rho_prefill=np.array([1_000_000.0]),
        C_net=np.array([100.0]),
        C_prefill=np.array([10_000_000.0]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((2, 1)),
        h_kv=np.zeros((2, 1)),
        retained_prefill_target_s=0.0,
    )


def test_cvxpy_and_mirror_descent_agree_on_small_instance():
    problem = small_problem()
    coeffs = compute_coefficients(problem)
    cvx = solve_cvxpy(problem)
    md = solve_mirror_descent(problem, iterations=700, bisection_iterations=12)
    rel_gap = abs(md.objective - cvx.objective) / max(1.0, abs(cvx.objective))
    assert rel_gap < 1e-3
    assert_feasible(problem, coeffs, md.y, target_tol=1e-3)
    infeasible = md.history["retained_prefill_shortfall_s"] > 1e-5
    assert np.all(np.isnan(md.history["feasible_objective"][infeasible]))
    assert np.all(np.diff(md.history["best_feasible_objective"]) <= 1e-12)


def test_larger_alpha_moves_more_work_on_small_instance():
    problem = small_problem()
    low = solve_mirror_descent(problem, iterations=250, bisection_iterations=1)
    high_alpha_load = low.history["retained_prefill_moved_s"][low.history["alpha"] == 1.0][-1]
    zero_alpha_load = low.history["retained_prefill_moved_s"][low.history["alpha"] == 0.0][-1]
    assert high_alpha_load > zero_alpha_load


def test_deadline_aware_solver_uses_available_rate_by_cumulative_deadline_threshold():
    problem = deadline_rate_problem()

    tight = solve_deadline_aware_cvxpy(problem, deadline_margin=0.5)
    loose = solve_deadline_aware_cvxpy(problem, deadline_margin=1.0)
    moved = np.sum(tight.y[:, : compute_coefficients(problem).M], axis=1)

    np.testing.assert_allclose(moved, [0.5, 1.0], atol=1e-4)
    np.testing.assert_allclose(tight.objective, 1.5, atol=1e-4)
    np.testing.assert_allclose(loose.objective, 2.0, atol=1e-4)


def test_deadline_aware_solver_respects_retained_prefill_cap_for_frontier_scans():
    problem = deadline_rate_problem()
    result = solve_deadline_aware_cvxpy(problem, deadline_margin=1.0, retained_prefill_cap=1.25)

    np.testing.assert_allclose(retained_prefill_moved_s(problem, result.y), 1.25, atol=1e-4)


def test_mirror_descent_preserves_glm_transition_mix():
    model = next(model for model in catalog_models() if model.name == "GLM-5")
    problem = make_problem(model, "bandwidth-spread", workload_source="fixed")
    cvx = solve_cvxpy(problem)
    md = solve_mirror_descent(problem, iterations=1000, bisection_iterations=12)
    gap = max(0.0, (md.objective - cvx.objective) / max(1.0, abs(cvx.objective)))
    cvx_mix = retained_prefill_action_mix(problem, cvx.y)
    md_mix = retained_prefill_action_mix(problem, md.y)
    assert gap < 2e-3
    assert retained_prefill_moved_s(problem, md.y) >= problem.retained_prefill_target_s - 1e-5
    assert abs(md_mix["replay_retained_prefill_fraction"] - cvx_mix["replay_retained_prefill_fraction"]) < 0.08


def test_cvxpy_catalog_action_mix_matches_final_claim():
    regimes = ("bandwidth-spread", "prefill-spread", "background-load-spread")
    shares = {}
    for model in catalog_models():
        shares[model.name] = []
        for regime in regimes:
            problem = make_problem(model, regime, workload_source="fixed")
            shares[model.name].append(
                retained_prefill_action_mix(problem, solve_cvxpy(problem).y)["replay_retained_prefill_fraction"]
            )
    assert max(shares["DeepSeek-V4-Pro"]) < 0.05
    assert min(shares["Qwen3-Next-80B-A3B"]) > 0.95
    assert shares["GLM-5"][0] < 0.30
    assert max(shares["GLM-5"]) - min(shares["GLM-5"]) > 0.80


def test_replay_baseline_reports_infeasible_when_capacity_blocks_target():
    model = ModelParams("capacity", 4.0, 100.0, 10.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="capacity",
        T=np.array([10.0]),
        d=np.array([1.0]),
        deadline_s=np.array([1.0]),
        lambda_Bps=np.array([1e9]),
        rho_prefill=np.array([1.0]),
        C_net=np.array([1e12]),
        C_prefill=np.array([5.0]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        retained_prefill_target_s=1.0,
    )
    result = solve_replay_only(problem)
    assert not result.feasible
    assert result.objective is None
    assert result.retained_prefill_moved_s < problem.retained_prefill_target_s


def loaded_two_dest_problem() -> ProblemData:
    model = ModelParams("loaded", 4.0, 1e12, 10.0, 0.0)
    return ProblemData(
        model=model,
        regime="loaded",
        T=np.array([10.0]),
        d=np.array([1.0]),
        deadline_s=np.array([1.0]),
        lambda_Bps=np.array([1e12, 1e12]),
        rho_prefill=np.array([10.0, 10.0]),
        C_net=np.array([1e15, 1e15]),
        C_prefill=np.array([100.0, 100.0]),
        ell_net=np.array([0.0, 0.0]),
        ell_prefill=np.array([99.0, 0.0]),
        h_ctx=np.zeros((1, 2)),
        h_kv=np.zeros((1, 2)),
        retained_prefill_target_s=1.0,
    )


def test_online_queue_greedy_uses_prefill_backlog_for_action_choice():
    model = ModelParams("online-prefill", 0.0, 15.0, 1.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="online-prefill",
        T=np.array([10.0]),
        d=np.array([2.0]),
        deadline_s=np.array([1.0]),
        lambda_Bps=np.array([100.0]),
        rho_prefill=np.array([10.0]),
        C_net=np.array([10_000.0]),
        C_prefill=np.array([1_000.0]),
        ell_net=np.zeros(1),
        ell_prefill=np.zeros(1),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        retained_prefill_target_s=20.0,
    )

    result = solve_online_queue_greedy(problem)

    np.testing.assert_array_equal(result.allocation[0], [1, 1, 0])


def test_online_queue_greedy_uses_network_backlog_for_destination_choice():
    model = ModelParams("online-network", 20.0, 1.0, 1.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="online-network",
        T=np.array([10.0]),
        d=np.array([2.0]),
        deadline_s=np.array([1.0]),
        lambda_Bps=np.array([10.0, 10.0]),
        rho_prefill=np.array([1_000.0, 1_000.0]),
        C_net=np.array([10_000.0, 10_000.0]),
        C_prefill=np.array([1_000_000.0, 1_000_000.0]),
        ell_net=np.zeros(2),
        ell_prefill=np.zeros(2),
        h_ctx=np.zeros((1, 2)),
        h_kv=np.zeros((1, 2)),
        retained_prefill_target_s=20.0,
    )

    result = solve_online_queue_greedy(problem)

    np.testing.assert_array_equal(result.allocation[0], [0, 1, 0, 1, 0])


def test_mixed_greedy_uses_current_load_marginal_cost():
    result = solve_mixed_greedy(loaded_two_dest_problem())
    assert result.allocation[0, 2] > 0.99


def test_crossover_greedy_ignores_current_load_until_capacity():
    result = solve_crossover_greedy(loaded_two_dest_problem())
    assert 0.09 < result.allocation[0, 0] < 0.11
    assert result.allocation[0, 2] > 0.89


def test_transition_coupled_scenario_quality_gate():
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        workload_source="fixed",
        window_s=60.0,
    )
    coeffs = compute_coefficients(problem)
    soft = solve_soft_deadline_cvxpy(problem)
    crossover = solve_crossover_greedy(problem)
    diag = allocation_diagnostics(problem, coeffs, soft.y)

    assert np.any(problem.h_ctx != problem.h_kv)
    assert np.max(problem.h_ctx[1]) > 0.5
    assert np.max(problem.h_kv[2]) > 0.5
    assert coeffs.R0[4, 1, REPLAY] < coeffs.R0[4, 1, STATE]
    assert coeffs.R0[2, 2, STATE] < coeffs.R0[2, 2, REPLAY]
    assert np.any(np.all((problem.h_ctx == 0.0) & (problem.h_kv == 0.0), axis=1))
    assert diag["active_destinations_used"] >= 2
    assert min(diag["replay_retained_prefill_fraction"], diag["state_transfer_retained_prefill_fraction"]) >= 0.05
    assert max(diag["max_net_util"], diag["max_prefill_util"]) > 0.7
    assert crossover.feasible
    assert retained_prefill_moved_s(problem, soft.y) >= problem.retained_prefill_target_s - 1e-5
    assert soft.diagnostics["deadline_overrun_max"] == 0.0
    assert soft.diagnostics["deadline_load_max"] <= soft.diagnostics["deadline_headroom"] + 1e-5
