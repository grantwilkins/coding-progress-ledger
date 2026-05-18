from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from coefficients import Coefficients, compute_coefficients
from metrics import assert_feasible, resource_loads, shed_achieved, utilization
from objective import lagrangian_gradient, objective
from problem import ProblemData


@dataclass(frozen=True)
class MirrorDescentResult:
    y: np.ndarray
    objective: float
    dual: float
    history: dict[str, np.ndarray]
    current_y: np.ndarray
    best_feasible_y: np.ndarray
    average_y: np.ndarray
    average_objective: float
    average_feasible: bool
    eta_x0: float
    eta_l0: float
    feasible: bool = True


def _softmax_rows(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return d[:, None] * p / np.sum(p, axis=1, keepdims=True)


def _initial_y(problem: ProblemData, coeffs: Coefficients) -> np.ndarray:
    y = np.zeros((problem.G, coeffs.M + 1))
    y[:, -1] = problem.d
    remaining_shed = problem.B_shed

    while remaining_shed > 1e-9:
        L_net, L_prefill = resource_loads(problem, coeffs, y)
        choices = []
        for g in range(problem.G):
            if y[g, -1] <= 1e-12:
                continue
            for m, (k, action) in enumerate(
                zip(coeffs.option_dest, coeffs.option_action)
            ):
                b_net = coeffs.b_net[g, k, action]
                b_prefill = coeffs.b_prefill[g, k, action]
                room_net = (1.0 - 1e-9) * problem.C_net[k] - L_net[k]
                room_prefill = (1.0 - 1e-9) * problem.C_prefill[k] - L_prefill[k]
                cap_net = np.inf if b_net == 0.0 else room_net / b_net
                cap_prefill = np.inf if b_prefill == 0.0 else room_prefill / b_prefill
                amount = min(
                    y[g, -1], remaining_shed / problem.tau[g], cap_net, cap_prefill
                )
                if amount > 1e-12:
                    marginal = (
                        coeffs.q_flat[g, m]
                        + problem.w * b_net / (problem.C_net[k] - L_net[k])
                        + problem.w * b_prefill / (problem.C_prefill[k] - L_prefill[k])
                    )
                    choices.append((marginal / problem.tau[g], g, m, amount))
        if not choices:
            break
        _, g, m, amount = min(choices, key=lambda item: item[0])
        y[g, m] += amount
        y[g, -1] -= amount
        remaining_shed -= amount * problem.tau[g]

    if remaining_shed > 1e-7:
        raise RuntimeError("mirror descent initializer could not meet shed target")
    assert_feasible(problem, coeffs, y, shed_tol=1e-7)
    return y


def _stats(
    problem: ProblemData, coeffs: Coefficients, y: np.ndarray, shed_tol: float
) -> tuple[float, float, float, float, float, bool]:
    shed = shed_achieved(problem, y)
    violation = max(0.0, problem.B_shed - shed)
    excess = max(0.0, shed - problem.B_shed)
    u_net, u_prefill = utilization(problem, coeffs, y)
    max_net = float(np.max(u_net))
    max_prefill = float(np.max(u_prefill))
    return shed, violation, excess, max_net, max_prefill, violation <= shed_tol


def solve_mirror_descent(
    problem: ProblemData,
    iterations: int = 10000,
    eta_x0: float = 0.05,
    eta_l0: float = 0.1,
    max_backtracks: int = 10,
    shed_tol: float = 1e-5,
) -> MirrorDescentResult:
    coeffs = compute_coefficients(problem)
    y = _initial_y(problem, coeffs)
    dual = 0.0
    best_y = y.copy()
    best_obj = objective(problem, coeffs, best_y)
    avg_y = y.copy()
    avg_obj = best_obj
    avg_shed = shed_achieved(problem, avg_y)
    obj_hist = []
    feasible_obj_hist = []
    best_feasible_hist = []
    avg_obj_hist = []
    avg_feasible_obj_hist = []
    viol_hist = []
    excess_hist = []
    shed_hist = []
    net_util_hist = []
    prefill_util_hist = []
    dual_hist = []
    avg_viol_hist = []
    avg_excess_hist = []

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
            raise RuntimeError(
                "mirror descent backtracking could not find a feasible step"
            )

        y = y_new
        avg_y += (y - avg_y) / (t + 1)
        shed, violation, excess, max_net, max_prefill, feasible = _stats(
            problem, coeffs, y, shed_tol
        )
        dual = max(
            0.0, dual + eta_l0 / np.sqrt(t) * (problem.B_shed - shed) / problem.B_shed
        )
        obj = objective(problem, coeffs, y)
        avg_obj = objective(problem, coeffs, avg_y)
        avg_shed, avg_violation, avg_excess, _, _, avg_feasible = _stats(
            problem, coeffs, avg_y, shed_tol
        )
        if feasible and obj < best_obj:
            best_obj = obj
            best_y = y.copy()

        obj_hist.append(obj)
        feasible_obj_hist.append(obj if feasible else np.nan)
        best_feasible_hist.append(best_obj)
        avg_obj_hist.append(avg_obj)
        avg_feasible_obj_hist.append(avg_obj if avg_feasible else np.nan)
        viol_hist.append(violation)
        excess_hist.append(excess)
        shed_hist.append(shed)
        net_util_hist.append(max_net)
        prefill_util_hist.append(max_prefill)
        dual_hist.append(dual)
        avg_viol_hist.append(avg_violation)
        avg_excess_hist.append(avg_excess)

    avg_feasible = avg_shed >= problem.B_shed - shed_tol and np.isfinite(avg_obj)
    assert_feasible(problem, coeffs, best_y, shed_tol=shed_tol)
    return MirrorDescentResult(
        best_y,
        best_obj,
        dual,
        {
            "objective": np.asarray(obj_hist),
            "feasible_objective": np.asarray(feasible_obj_hist),
            "best_feasible_objective": np.asarray(best_feasible_hist),
            "average_objective": np.asarray(avg_obj_hist),
            "average_feasible_objective": np.asarray(avg_feasible_obj_hist),
            "shed_violation": np.asarray(viol_hist),
            "excess_shed": np.asarray(excess_hist),
            "shed": np.asarray(shed_hist),
            "max_net_util": np.asarray(net_util_hist),
            "max_prefill_util": np.asarray(prefill_util_hist),
            "dual": np.asarray(dual_hist),
            "average_shed_violation": np.asarray(avg_viol_hist),
            "average_excess_shed": np.asarray(avg_excess_hist),
        },
        y,
        best_y,
        avg_y,
        avg_obj,
        avg_feasible,
        eta_x0,
        eta_l0,
    )
