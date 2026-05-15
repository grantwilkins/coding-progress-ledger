from __future__ import annotations

import numpy as np

from coefficients import Coefficients, move_view
from metrics import resource_loads, shed_achieved
from problem import ProblemData


def objective(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> float:
    x = move_view(y, problem)
    L_net, L_prefill = resource_loads(problem, coeffs, y)
    if np.any(L_net / problem.C_net >= 1.0) or np.any(L_prefill / problem.C_prefill >= 1.0):
        return float("inf")
    risk = float(np.sum(coeffs.q * x))
    net_barrier = -np.log(1.0 - L_net / problem.C_net)
    prefill_barrier = -np.log(1.0 - L_prefill / problem.C_prefill)
    return risk + problem.w * float(np.sum(net_barrier) + np.sum(prefill_barrier))


def objective_gradient(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> np.ndarray:
    L_net, L_prefill = resource_loads(problem, coeffs, y)
    if np.any(L_net >= problem.C_net) or np.any(L_prefill >= problem.C_prefill):
        raise ValueError("gradient requested outside the barrier domain")
    grad_move = (
        coeffs.q
        + problem.w * coeffs.b_net / (problem.C_net - L_net)[None, :, None]
        + problem.w * coeffs.b_prefill / (problem.C_prefill - L_prefill)[None, :, None]
    )
    grad = np.zeros((problem.G, coeffs.M + 1))
    grad[:, : coeffs.M] = grad_move.reshape(problem.G, coeffs.M)
    return grad


def lagrangian_value(problem: ProblemData, coeffs: Coefficients, y: np.ndarray, dual: float) -> float:
    return objective(problem, coeffs, y) + dual * (problem.B_shed - shed_achieved(problem, y))


def lagrangian_gradient(problem: ProblemData, coeffs: Coefficients, y: np.ndarray, dual: float) -> np.ndarray:
    grad = objective_gradient(problem, coeffs, y)
    grad[:, : coeffs.M] -= dual * problem.tau[:, None]
    return grad
