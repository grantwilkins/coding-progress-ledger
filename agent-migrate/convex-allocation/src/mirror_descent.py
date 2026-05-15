from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coefficients import Coefficients, compute_coefficients
from metrics import assert_feasible, shed_achieved, utilization
from objective import lagrangian_gradient, objective
from problem import ProblemData


@dataclass(frozen=True)
class MirrorDescentResult:
    y: np.ndarray
    objective: float
    dual: float
    history: dict[str, np.ndarray]


def _softmax_rows(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return d[:, None] * p / np.sum(p, axis=1, keepdims=True)


def _initial_y(problem: ProblemData, M: int) -> np.ndarray:
    move_mass = 1e-3
    y = np.empty((problem.G, M + 1))
    y[:, :M] = problem.d[:, None] * move_mass / M
    y[:, -1] = problem.d * (1.0 - move_mass)
    return y


def solve_mirror_descent(
    problem: ProblemData,
    iterations: int = 2000,
    eta_x0: float = 0.05,
    eta_l0: float = 0.1,
    max_backtracks: int = 10,
) -> MirrorDescentResult:
    coeffs = compute_coefficients(problem)
    y = _initial_y(problem, coeffs.M)
    dual = 0.0
    best_y: np.ndarray | None = None
    best_obj = float("inf")
    obj_hist = []
    best_hist = []
    viol_hist = []
    shed_hist = []
    net_util_hist = []
    prefill_util_hist = []
    dual_hist = []

    for t in range(1, iterations + 1):
        grad = lagrangian_gradient(problem, coeffs, y, dual)
        base_step = eta_x0 / np.sqrt(t)
        log_y = np.log(np.maximum(y, 1e-300))
        for bt in range(max_backtracks + 1):
            step = base_step / (2**bt)
            y_new = _softmax_rows(log_y - step * grad, problem.d)
            if np.isfinite(objective(problem, coeffs, y_new)):
                break
        else:
            raise RuntimeError("mirror descent backtracking could not find a feasible step")

        y = y_new
        shed = shed_achieved(problem, y)
        violation = max(0.0, problem.B_shed - shed)
        dual = max(0.0, dual + eta_l0 / np.sqrt(t) * (problem.B_shed - shed))
        obj = objective(problem, coeffs, y)
        if violation <= 1e-3 and obj < best_obj:
            best_obj = obj
            best_y = y.copy()

        obj_hist.append(obj)
        best_hist.append(best_obj)
        viol_hist.append(violation)
        shed_hist.append(shed)
        u_net, u_prefill = utilization(problem, coeffs, y)
        net_util_hist.append(float(np.max(u_net)))
        prefill_util_hist.append(float(np.max(u_prefill)))
        dual_hist.append(dual)

    if best_y is None:
        raise RuntimeError("mirror descent found no shed-feasible iterate")
    assert_feasible(problem, coeffs, best_y, shed_tol=1e-3)
    return MirrorDescentResult(
        best_y,
        best_obj,
        dual,
        {
            "objective": np.asarray(obj_hist),
            "best_objective": np.asarray(best_hist),
            "shed_violation": np.asarray(viol_hist),
            "shed": np.asarray(shed_hist),
            "max_net_util": np.asarray(net_util_hist),
            "max_prefill_util": np.asarray(prefill_util_hist),
            "dual": np.asarray(dual_hist),
        },
    )
