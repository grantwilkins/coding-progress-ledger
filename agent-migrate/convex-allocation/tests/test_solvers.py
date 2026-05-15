"""
Claim:
The CVXPY oracle and primal-dual mirror descent solve the same convex relaxation
up to objective and feasibility tolerances.

Plausible wrong implementations:
- Give mirror descent a penalty objective that differs from the CVXPY problem.
- Use the wrong shed-gradient sign, so the dual update rewards staying.
- Compare allocations entrywise even when equivalent optima can differ.
- Accept capacity or shed violations as solver success.
"""

from __future__ import annotations

import numpy as np

from catalog import ModelParams
from coefficients import compute_coefficients
from cvxpy_solver import solve_cvxpy
from metrics import assert_feasible
from mirror_descent import solve_mirror_descent
from problem import ProblemData


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
    md = solve_mirror_descent(problem, iterations=5000, eta_x0=2.0, eta_l0=0.2, max_backtracks=20)
    rel_gap = abs(md.objective - cvx.objective) / max(1.0, abs(cvx.objective))
    assert rel_gap < 1e-3
    assert_feasible(problem, coeffs, md.y, shed_tol=1e-3)
