from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from coefficients import REPLAY, Coefficients, compute_coefficients
from metrics import (
    assert_feasible,
    available_rates,
    deadline_debt_summary,
    deadline_load_ratios,
    relief_achieved_s,
    shed_achieved,
)
from objective import objective
from problem import ProblemData


@dataclass(frozen=True)
class SolverResult:
    y: np.ndarray
    objective: float
    status: str
    diagnostics: dict[str, float] | None = None


def solve_cvxpy(problem: ProblemData, eps: float = 1e-6) -> SolverResult:
    coeffs = compute_coefficients(problem)
    M = coeffs.M
    y = cp.Variable((problem.G, M + 1), nonneg=True)
    x = y[:, :M]
    constraints = [
        cp.sum(y, axis=1) == problem.d,
        problem.tau @ cp.sum(x, axis=1) >= problem.relief_target_s,
    ]

    terms = [cp.sum(cp.multiply(coeffs.q_flat, x))]
    for k in range(problem.K):
        mask = coeffs.option_dest == k
        u_net = problem.ell_net[k] / problem.C_net[k] + cp.sum(
            cp.multiply(coeffs.b_net_flat[:, mask] / problem.C_net[k], x[:, mask])
        )
        u_prefill = problem.ell_prefill[k] / problem.C_prefill[k] + cp.sum(
            cp.multiply(coeffs.b_prefill_flat[:, mask] / problem.C_prefill[k], x[:, mask])
        )
        constraints += [
            u_net <= 1.0 - eps,
            u_prefill <= 1.0 - eps,
        ]
        terms += [
            -problem.w * cp.log(1.0 - u_net),
            -problem.w * cp.log(1.0 - u_prefill),
        ]

    prob = cp.Problem(cp.Minimize(sum(terms)), constraints)
    last_error: Exception | None = None
    for solver, kwargs in (
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-6, "max_iters": 100_000}),
    ):
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or y.value is None:
                raise RuntimeError(f"{solver} returned {prob.status}")
            y_value = np.maximum(np.asarray(y.value, dtype=float), 0.0)
            y_value *= (problem.d / np.sum(y_value, axis=1))[:, None]
            obj = objective(problem, coeffs, y_value)
            assert_feasible(problem, coeffs, y_value, shed_tol=1e-5)
            return SolverResult(y_value, obj, prob.status)
        except (cp.SolverError, AssertionError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"CVXPY solve failed: {last_error}")


def solve_deadline_aware_cvxpy(
    problem: ProblemData,
    deadline_margin: float = 1.0,
    shed_cap: float | None = None,
) -> SolverResult:
    if deadline_margin <= 0.0:
        raise ValueError("deadline_margin must be positive")
    coeffs = compute_coefficients(problem)
    _, lambda_avail, rho_avail = available_rates(problem)
    M = coeffs.M
    y = cp.Variable((problem.G, M + 1), nonneg=True)
    x = y[:, :M]
    shed = problem.tau @ cp.sum(x, axis=1)
    constraints = [cp.sum(y, axis=1) == problem.d]
    if shed_cap is not None:
        constraints.append(shed <= shed_cap)

    deadline_thresholds = np.unique(problem.deadline_s)
    for k in range(problem.K):
        dest = coeffs.option_dest == k
        replay_dest = dest & (coeffs.option_action == REPLAY)
        constraints += [
            cp.sum(cp.multiply(coeffs.b_net_flat * dest[None, :], x))
            <= problem.C_net[k] - problem.ell_net[k],
            cp.sum(cp.multiply(coeffs.b_prefill_flat * replay_dest[None, :], x))
            <= problem.C_prefill[k] - problem.ell_prefill[k],
        ]
        for deadline_s in deadline_thresholds:
            classes = problem.deadline_s <= deadline_s
            constraints += [
                cp.sum(cp.multiply(coeffs.b_net_flat * classes[:, None] * dest[None, :], x))
                <= deadline_margin * lambda_avail[k] * deadline_s,
                cp.sum(
                    cp.multiply(
                        coeffs.b_prefill_flat * classes[:, None] * replay_dest[None, :],
                        x,
                    )
                )
                <= deadline_margin * rho_avail[k] * deadline_s,
            ]

    prob = cp.Problem(cp.Maximize(shed), constraints)
    last_error: Exception | None = None
    for solver, kwargs in (
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-6, "max_iters": 100_000}),
    ):
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or y.value is None:
                raise RuntimeError(f"{solver} returned {prob.status}")
            y_value = np.maximum(np.asarray(y.value, dtype=float), 0.0)
            y_value *= (problem.d / np.sum(y_value, axis=1))[:, None]
            _assert_deadline_feasible(problem, coeffs, y_value, deadline_margin, shed_cap)
            return SolverResult(y_value, shed_achieved(problem, y_value), prob.status)
        except (cp.SolverError, AssertionError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"deadline-aware CVXPY solve failed: {last_error}")


def solve_soft_deadline_cvxpy(
    problem: ProblemData,
    deadline_headroom: float = 0.85,
    debt_linear_weight: float = 25.0,
    debt_quadratic_weight: float = 100.0,
    eps: float = 1e-6,
) -> SolverResult:
    if deadline_headroom <= 0.0:
        raise ValueError("deadline_headroom must be positive")
    coeffs = compute_coefficients(problem)
    _, lambda_avail, rho_avail = available_rates(problem)
    deadlines = np.unique(problem.deadline_s)
    M = coeffs.M
    y = cp.Variable((problem.G, M + 1), nonneg=True)
    x = y[:, :M]
    debt_net = cp.Variable((problem.K, deadlines.size), nonneg=True)
    debt_prefill = cp.Variable((problem.K, deadlines.size), nonneg=True)
    constraints = [
        cp.sum(y, axis=1) == problem.d,
        problem.tau @ cp.sum(x, axis=1) >= problem.relief_target_s,
    ]
    terms = [cp.sum(cp.multiply(coeffs.q_flat, x))]

    for k in range(problem.K):
        dest = coeffs.option_dest == k
        replay_dest = dest & (coeffs.option_action == REPLAY)
        u_net = problem.ell_net[k] / problem.C_net[k] + cp.sum(
            cp.multiply(coeffs.b_net_flat[:, dest] / problem.C_net[k], x[:, dest])
        )
        u_prefill = problem.ell_prefill[k] / problem.C_prefill[k] + cp.sum(
            cp.multiply(coeffs.b_prefill_flat[:, replay_dest] / problem.C_prefill[k], x[:, replay_dest])
        )
        constraints += [u_net <= 1.0 - eps, u_prefill <= 1.0 - eps]
        terms += [-problem.w * cp.log(1.0 - u_net), -problem.w * cp.log(1.0 - u_prefill)]
        for j, deadline_s in enumerate(deadlines):
            classes = problem.deadline_s <= deadline_s
            net_ratio = cp.sum(
                cp.multiply(
                    coeffs.b_net_flat * classes[:, None] * dest[None, :],
                    x,
                )
            ) / (lambda_avail[k] * deadline_s)
            prefill_ratio = cp.sum(
                cp.multiply(
                    coeffs.b_prefill_flat * classes[:, None] * replay_dest[None, :],
                    x,
                )
            ) / (rho_avail[k] * deadline_s)
            constraints += [
                net_ratio <= deadline_headroom + debt_net[k, j],
                prefill_ratio <= deadline_headroom + debt_prefill[k, j],
            ]

    n_debt = 2 * problem.K * deadlines.size
    terms += [
        debt_linear_weight * (cp.sum(debt_net) + cp.sum(debt_prefill)) / n_debt,
        debt_quadratic_weight
        * (cp.sum_squares(debt_net) + cp.sum_squares(debt_prefill))
        / n_debt,
    ]
    prob = cp.Problem(cp.Minimize(sum(terms)), constraints)
    last_error: Exception | None = None
    for solver, kwargs in (
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-6, "max_iters": 100_000}),
    ):
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or y.value is None:
                raise RuntimeError(f"{solver} returned {prob.status}")
            y_value = np.maximum(np.asarray(y.value, dtype=float), 0.0)
            y_value *= (problem.d / np.sum(y_value, axis=1))[:, None]
            assert_feasible(problem, coeffs, y_value, shed_tol=1e-5)
            diagnostics = deadline_debt_summary(problem, coeffs, y_value, deadline_headroom)
            _, net_load, prefill_load = deadline_load_ratios(problem, coeffs, y_value)
            diagnostics.update(
                {
                    "deadline_load_max": float(max(np.max(net_load), np.max(prefill_load))),
                    "deadline_headroom": deadline_headroom,
                    "deadline_debt_linear_weight": debt_linear_weight,
                    "deadline_debt_quadratic_weight": debt_quadratic_weight,
                    "relief_achieved_s": relief_achieved_s(problem, y_value),
                    "relief_target_s": problem.relief_target_s,
                }
            )
            return SolverResult(y_value, float(prob.value), prob.status, diagnostics)
        except (cp.SolverError, AssertionError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"soft-deadline CVXPY solve failed: {last_error}")


def _assert_deadline_feasible(
    problem: ProblemData,
    coeffs: Coefficients,
    y: np.ndarray,
    deadline_margin: float,
    shed_cap: float | None,
) -> None:
    if not np.all(y >= -1e-8):
        raise AssertionError("allocation has negative entries")
    if not np.allclose(np.sum(y, axis=1), problem.d, atol=1e-5):
        raise AssertionError("allocation rows do not sum to class demand")
    if shed_cap is not None and shed_achieved(problem, y) > shed_cap + 1e-5:
        raise AssertionError("shed cap exceeded")
    x = y[:, : coeffs.M]
    _, lambda_avail, rho_avail = available_rates(problem)
    for k in range(problem.K):
        dest = coeffs.option_dest == k
        replay_dest = dest & (coeffs.option_action == REPLAY)
        _assert_leq(
            np.sum(coeffs.b_net_flat[:, dest] * x[:, dest]),
            problem.C_net[k] - problem.ell_net[k],
        )
        _assert_leq(
            np.sum(coeffs.b_prefill_flat[:, replay_dest] * x[:, replay_dest]),
            problem.C_prefill[k] - problem.ell_prefill[k],
        )
        for deadline_s in np.unique(problem.deadline_s):
            classes = problem.deadline_s <= deadline_s
            _assert_leq(
                np.sum(coeffs.b_net_flat[np.ix_(classes, dest)] * x[np.ix_(classes, dest)]),
                deadline_margin * lambda_avail[k] * deadline_s,
            )
            _assert_leq(
                np.sum(
                    coeffs.b_prefill_flat[np.ix_(classes, replay_dest)]
                    * x[np.ix_(classes, replay_dest)]
                ),
                deadline_margin * rho_avail[k] * deadline_s,
            )


def _assert_leq(value: float, limit: float) -> None:
    if value > limit + 1e-5 * max(1.0, abs(limit)):
        raise AssertionError("deadline-aware allocation violates a capacity constraint")
