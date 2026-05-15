from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from coefficients import Coefficients, compute_coefficients
from metrics import assert_feasible
from objective import objective
from problem import ProblemData


@dataclass(frozen=True)
class SolverResult:
    y: np.ndarray
    objective: float
    status: str


def solve_cvxpy(problem: ProblemData, eps: float = 1e-6) -> SolverResult:
    coeffs = compute_coefficients(problem)
    M = coeffs.M
    y = cp.Variable((problem.G, M + 1), nonneg=True)
    x = y[:, :M]
    constraints = [
        cp.sum(y, axis=1) == problem.d,
        problem.tau @ cp.sum(x, axis=1) >= problem.B_shed,
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
