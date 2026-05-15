"""
Claim:
The convex allocation objective implements the stated replay/state resource costs,
barrier domain, Lagrangian shed sign, and per-request crossover units.

Plausible wrong implementations:
- Use bits where bytes are required, or omit the beta correction in the crossover.
- Collapse context-prefix locality and KV-state locality into one coefficient.
- Clip the log-barrier domain and return finite objectives for infeasible loads.
- Put the shed multiplier on the wrong sign in the Lagrangian gradient.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from catalog import ModelParams, catalog_models
from coefficients import REPLAY, STATE, compute_coefficients
from cvxpy_solver import solve_cvxpy
from metrics import shed_action_mix
from objective import lagrangian_gradient, lagrangian_value, objective, objective_gradient
from problem import ProblemData


def one_dest_problem(model: ModelParams, lambda_Bps: float) -> ProblemData:
    T = np.array([100_000.0])
    d = np.array([1.0])
    return ProblemData(
        model=model,
        regime="isolated",
        T=T,
        d=d,
        slack=np.array([1.0]),
        lambda_Bps=np.array([lambda_Bps]),
        rho_prefill=np.array([model.prefill_tok_s]),
        C_net=np.array([1e30]),
        C_prefill=np.array([1e30]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        B_shed=float(T[0] / model.prefill_tok_s),
    )


def test_catalog_crossovers_round_trip_from_units():
    for model in catalog_models():
        rel_err = abs(model.crossover_gbps - model.published_crossover_gbps) / model.published_crossover_gbps
        assert rel_err < 0.03


def test_coefficients_keep_context_and_kv_locality_separate():
    model = ModelParams("toy", 4.0, 100.0, 20.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="toy",
        T=np.array([10.0]),
        d=np.array([1.0]),
        slack=np.array([2.0]),
        lambda_Bps=np.array([50.0]),
        rho_prefill=np.array([25.0]),
        C_net=np.array([1e6]),
        C_prefill=np.array([1e6]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.array([[0.25]]),
        h_kv=np.array([[0.50]]),
        B_shed=0.0,
    )
    coeffs = compute_coefficients(problem)
    assert coeffs.b_net[0, 0, REPLAY] == 4.0 * 10.0 * 0.75
    assert coeffs.b_prefill[0, 0, REPLAY] == 10.0 * 0.75
    assert coeffs.b_net[0, 0, STATE] == 100.0 * 10.0 * 0.50
    assert coeffs.b_prefill[0, 0, STATE] == 0.0
    assert coeffs.R0[0, 0, REPLAY] == coeffs.b_net[0, 0, REPLAY] / 50.0 + coeffs.b_prefill[0, 0, REPLAY] / 25.0
    assert coeffs.R0[0, 0, STATE] == coeffs.b_net[0, 0, STATE] / 50.0


def test_objective_returns_inf_at_capacity_boundary():
    model = ModelParams("toy", 4.0, 100.0, 20.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="toy",
        T=np.array([10.0]),
        d=np.array([1.0]),
        slack=np.array([1.0]),
        lambda_Bps=np.array([100.0]),
        rho_prefill=np.array([100.0]),
        C_net=np.array([40.0]),
        C_prefill=np.array([1000.0]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        B_shed=0.0,
    )
    coeffs = compute_coefficients(problem)
    assert np.isfinite(objective(problem, coeffs, np.array([[0.5, 0.0, 0.5]])))
    assert objective(problem, coeffs, np.array([[1.0, 0.0, 0.0]])) == float("inf")


def test_shed_action_mix_is_tau_weighted_not_request_weighted():
    model = ModelParams("toy", 4.0, 100.0, 10.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="toy",
        T=np.array([10.0, 100.0]),
        d=np.array([1.0, 1.0]),
        slack=np.ones(2),
        lambda_Bps=np.array([1_000.0]),
        rho_prefill=np.array([1_000.0]),
        C_net=np.array([1e9]),
        C_prefill=np.array([1e9]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((2, 1)),
        h_kv=np.zeros((2, 1)),
        B_shed=0.0,
    )
    mix = shed_action_mix(problem, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert_allclose([mix["replay_shed_frac"], mix["state_shed_frac"]], [1 / 11, 10 / 11])


def test_objective_and_lagrangian_gradients_match_finite_difference():
    rng = np.random.default_rng(3)
    model = ModelParams("toy", 4.0, 80.0, 100.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="grad",
        T=np.array([30.0, 70.0]),
        d=np.array([3.0, 2.0]),
        slack=np.array([2.0, 5.0]),
        lambda_Bps=np.array([2_000.0, 5_000.0]),
        rho_prefill=np.array([400.0, 600.0]),
        C_net=np.array([1e7, 1e7]),
        C_prefill=np.array([1e7, 1e7]),
        ell_net=np.array([10.0, 20.0]),
        ell_prefill=np.array([30.0, 40.0]),
        h_ctx=np.array([[0.0, 0.2], [0.3, 0.1]]),
        h_kv=np.array([[0.4, 0.1], [0.0, 0.2]]),
        B_shed=1.0,
    )
    coeffs = compute_coefficients(problem)
    y = rng.uniform(0.05, 0.2, size=(problem.G, coeffs.M + 1))
    y *= problem.d[:, None] / y.sum(axis=1, keepdims=True)
    eps = 1e-6

    for analytic, fn in (
        (objective_gradient(problem, coeffs, y), lambda z: objective(problem, coeffs, z)),
        (lagrangian_gradient(problem, coeffs, y, 0.7), lambda z: lagrangian_value(problem, coeffs, z, 0.7)),
    ):
        numeric = np.zeros_like(analytic)
        for g in range(problem.G):
            for m in range(coeffs.M):
                step = np.zeros_like(y)
                step[g, m] = eps
                numeric[g, m] = (fn(y + step) - fn(y - step)) / (2 * eps)
        assert_allclose(analytic[:, : coeffs.M], numeric[:, : coeffs.M], rtol=1e-5, atol=1e-5)


def test_cvxpy_recovers_replay_state_crossover_for_each_model():
    for model in catalog_models():
        threshold_Bps = model.prefill_tok_s * (model.eta_bytes_per_tok - model.beta_bytes_per_tok)
        low = solve_cvxpy(one_dest_problem(model, 0.5 * threshold_Bps)).y[0]
        high = solve_cvxpy(one_dest_problem(model, 2.0 * threshold_Bps)).y[0]
        assert low[REPLAY] > 0.99
        assert high[STATE] > 0.99
