"""
Claim:
Source relief is measured in source-prefill seconds, deadline slack is an
explicit reconstruction deadline, transition cost is independent of the
deadline, and deadline violations are reported as soft debt ratios.

Plausible wrong implementations:
- Keep hiding deadline pressure in R0/deadline instead of explicit debt.
- Compute deadline load from bytes or tokens without converting to service time.
- Treat deadline buckets as disjoint rather than cumulative due-by thresholds.
- Penalize debt with the wrong sign or let the soft solver under-relieve.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from catalog import ModelParams
from coefficients import compute_coefficients
from cvxpy_solver import solve_soft_deadline_cvxpy
from metrics import deadline_debt_ratios, relief_achieved_s
from problem import ProblemData


def deadline_problem(deadline_s=1.0, relief_target_s=1.0) -> ProblemData:
    model = ModelParams("deadline-test", 1.0, 1.0, 10.0, 0.0)
    return ProblemData(
        model=model,
        regime="deadline-test",
        T=np.array([10.0]),
        d=np.array([1.0]),
        deadline_s=np.array([deadline_s]),
        lambda_Bps=np.array([5.0]),
        rho_prefill=np.array([1_000.0]),
        C_net=np.array([50.0]),
        C_prefill=np.array([10_000.0]),
        ell_net=np.array([0.0]),
        ell_prefill=np.array([0.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        relief_target_s=relief_target_s,
    )


def test_transition_cost_is_not_deadline_normalized():
    tight = deadline_problem(deadline_s=1.0, relief_target_s=0.0)
    loose = deadline_problem(deadline_s=100.0, relief_target_s=0.0)

    tight_coeffs = compute_coefficients(tight)
    loose_coeffs = compute_coefficients(loose)

    assert_allclose(tight_coeffs.q, tight_coeffs.R0)
    assert_allclose(tight_coeffs.q, loose_coeffs.q)


def test_deadline_debt_is_service_time_due_by_deadline_ratio():
    problem = deadline_problem(deadline_s=1.0)
    coeffs = compute_coefficients(problem)
    y = np.array([[0.0, 1.0, 0.0]])

    debt = deadline_debt_ratios(problem, coeffs, y, deadline_headroom=0.75)

    assert_allclose(np.sort(debt), [0.0, 1.25])


def test_soft_deadline_solver_relieves_target_and_reports_violable_debt():
    problem = deadline_problem(deadline_s=1.0, relief_target_s=1.0)
    result = solve_soft_deadline_cvxpy(problem, deadline_headroom=0.75)

    assert relief_achieved_s(problem, result.y) >= problem.relief_target_s - 1e-5
    assert result.diagnostics["deadline_debt_max"] > 1.0


def test_soft_deadline_solver_has_zero_debt_when_headroom_is_sufficient():
    problem = deadline_problem(deadline_s=1.0, relief_target_s=1.0)
    result = solve_soft_deadline_cvxpy(problem, deadline_headroom=2.5)

    assert relief_achieved_s(problem, result.y) >= problem.relief_target_s - 1e-5
    assert result.diagnostics["deadline_debt_max"] <= 1e-7
