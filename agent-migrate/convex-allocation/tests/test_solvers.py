"""
Claim:
The CVXPY oracle and mirror descent with scalar bisection solve the same convex
relaxation up to objective and feasibility tolerances.

Plausible wrong implementations:
- Give mirror descent a penalty objective that differs from the CVXPY problem.
- Use the wrong shed-gradient sign, so larger alpha rewards staying.
- Update the wrong bisection bound and return an infeasible allocation.
- Compare allocations entrywise even when equivalent optima can differ.
- Accept capacity or shed violations as solver success.
- Mark a partial greedy baseline as feasible after it hits capacity.
- Claim catalog action-mix conclusions from diagnostics instead of the CVXPY oracle.
"""

from __future__ import annotations

import numpy as np

from baselines import solve_replay_only
from catalog import ModelParams, catalog_models
from coefficients import compute_coefficients
from cvxpy_solver import solve_cvxpy
from metrics import assert_feasible, shed_achieved, shed_action_mix
from mirror_descent import solve_mirror_descent
from problem import ProblemData, make_problem


def small_problem() -> ProblemData:
    model = ModelParams("small", 4.0, 120.0, 1_000.0, 0.0)
    T = np.array([80.0, 300.0])
    d = np.array([4.0, 3.0])
    total_shed = float(np.dot(T / model.prefill_tok_s, d))
    return ProblemData(
        model=model,
        regime="small",
        T=T,
        d=d,
        slack=np.array([2.0, 9.0]),
        lambda_Bps=np.array([25_000.0, 80_000.0]),
        rho_prefill=np.array([1_700.0, 2_300.0]),
        C_net=np.array([40_000.0, 55_000.0]),
        C_prefill=np.array([6_000.0, 5_000.0]),
        ell_net=np.array([4_000.0, 8_000.0]),
        ell_prefill=np.array([600.0, 1_200.0]),
        h_ctx=np.array([[0.0, 0.2], [0.1, 0.0]]),
        h_kv=np.array([[0.0, 0.1], [0.3, 0.0]]),
        B_shed=0.35 * total_shed,
    )


def test_cvxpy_and_mirror_descent_agree_on_small_instance():
    problem = small_problem()
    coeffs = compute_coefficients(problem)
    cvx = solve_cvxpy(problem)
    md = solve_mirror_descent(problem, iterations=700, bisection_iterations=12)
    rel_gap = abs(md.objective - cvx.objective) / max(1.0, abs(cvx.objective))
    assert rel_gap < 1e-3
    assert_feasible(problem, coeffs, md.y, shed_tol=1e-3)
    infeasible = md.history["shed_violation"] > 1e-5
    assert np.all(np.isnan(md.history["feasible_objective"][infeasible]))
    assert np.all(np.diff(md.history["best_feasible_objective"]) <= 1e-12)


def test_larger_alpha_moves_more_work_on_small_instance():
    problem = small_problem()
    low = solve_mirror_descent(problem, iterations=250, bisection_iterations=1)
    high_alpha_shed = low.history["shed"][low.history["alpha"] == 1.0][-1]
    zero_alpha_shed = low.history["shed"][low.history["alpha"] == 0.0][-1]
    assert high_alpha_shed > zero_alpha_shed


def test_mirror_descent_preserves_glm_transition_mix():
    model = next(model for model in catalog_models() if model.name == "GLM-5")
    problem = make_problem(model, "bandwidth-spread")
    cvx = solve_cvxpy(problem)
    md = solve_mirror_descent(problem, iterations=1000, bisection_iterations=12)
    gap = max(0.0, (md.objective - cvx.objective) / max(1.0, abs(cvx.objective)))
    cvx_mix = shed_action_mix(problem, cvx.y)
    md_mix = shed_action_mix(problem, md.y)
    assert gap < 2e-3
    assert shed_achieved(problem, md.y) >= problem.B_shed - 1e-5
    assert abs(md_mix["replay_shed_frac"] - cvx_mix["replay_shed_frac"]) < 0.08


def test_cvxpy_catalog_action_mix_matches_final_claim():
    regimes = ("bandwidth-spread", "prefill-spread", "background-load-spread")
    shares = {}
    for model in catalog_models():
        shares[model.name] = []
        for regime in regimes:
            problem = make_problem(model, regime)
            shares[model.name].append(
                shed_action_mix(problem, solve_cvxpy(problem).y)["replay_shed_frac"]
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
        slack=np.array([1.0]),
        lambda_Bps=np.array([1e9]),
        rho_prefill=np.array([1.0]),
        C_net=np.array([1e12]),
        C_prefill=np.array([5.0]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        B_shed=1.0,
    )
    result = solve_replay_only(problem)
    assert not result.feasible
    assert result.objective is None
    assert result.shed_achieved < problem.B_shed
