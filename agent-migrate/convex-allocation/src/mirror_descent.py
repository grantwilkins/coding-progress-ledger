from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from coefficients import Coefficients, compute_coefficients
from metrics import assert_feasible, resource_loads, shed_achieved, utilization
from objective import objective, penalized_gradient
from problem import ProblemData


@dataclass(frozen=True)
class MirrorDescentResult:
    y: np.ndarray
    objective: float
    alpha: float
    history: dict[str, np.ndarray]
    eta_x0: float
    bisection_iterations: int
    feasible: bool = True


def _softmax_rows(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return d[:, None] * p / np.sum(p, axis=1, keepdims=True)


def _interior_y(problem: ProblemData, coeffs: Coefficients) -> np.ndarray:
    move_frac = 1e-3
    for _ in range(12):
        y = np.zeros((problem.G, coeffs.M + 1))
        y[:, : coeffs.M] = problem.d[:, None] * move_frac / coeffs.M
        y[:, -1] = problem.d * (1.0 - move_frac)
        if np.isfinite(objective(problem, coeffs, y)):
            return y
        move_frac *= 0.1
    raise RuntimeError("could not build an interior mirror-descent start")


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
    return (
        shed,
        violation,
        excess,
        max_net,
        max_prefill,
        violation <= shed_tol and max_net < 1.0 and max_prefill < 1.0,
    )


def solve_mirror_descent(
    problem: ProblemData,
    iterations: int = 1500,
    eta_x0: float = 10.0,
    max_backtracks: int = 30,
    shed_tol: float = 1e-5,
    bisection_iterations: int = 16,
) -> MirrorDescentResult:
    coeffs = compute_coefficients(problem)
    start_y = _interior_y(problem, coeffs)
    best_y = _initial_y(problem, coeffs)
    best_obj = objective(problem, coeffs, best_y)
    best_alpha = np.nan
    hist: dict[str, list[float]] = {
        "objective": [],
        "feasible_objective": [],
        "best_feasible_objective": [],
        "shed_violation": [],
        "excess_shed": [],
        "shed": [],
        "max_net_util": [],
        "max_prefill_util": [],
        "alpha": [],
    }

    def run_alpha(alpha: float) -> float:
        nonlocal best_alpha, best_obj, best_y
        y = start_y.copy()
        for t in range(1, iterations + 1):
            grad = penalized_gradient(problem, coeffs, y, alpha)
            log_y = np.log(np.maximum(y, 1e-300))
            base_step = eta_x0 / np.sqrt(t)
            for bt in range(max_backtracks + 1):
                step = base_step / (2**bt)
                y_new = _softmax_rows(log_y - step * grad, problem.d)
                if np.isfinite(objective(problem, coeffs, y_new)):
                    break
            else:
                raise RuntimeError("mirror descent could not find a feasible step")

            y = y_new
            obj = objective(problem, coeffs, y)
            shed, violation, excess, max_net, max_prefill, feasible = _stats(
                problem, coeffs, y, shed_tol
            )
            if feasible and obj < best_obj:
                best_y = y.copy()
                best_obj = obj
                best_alpha = alpha

            hist["objective"].append(obj)
            hist["feasible_objective"].append(obj if feasible else np.nan)
            hist["best_feasible_objective"].append(best_obj)
            hist["shed_violation"].append(violation)
            hist["excess_shed"].append(excess)
            hist["shed"].append(shed)
            hist["max_net_util"].append(max_net)
            hist["max_prefill_util"].append(max_prefill)
            hist["alpha"].append(alpha)
        return shed_achieved(problem, y)

    lo = 0.0
    shed_lo = run_alpha(lo)
    hi = 1.0
    shed_hi = shed_lo
    while shed_hi < problem.B_shed - shed_tol:
        shed_hi = run_alpha(hi)
        if shed_hi < problem.B_shed - shed_tol:
            lo = hi
            hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("could not bracket shed target with scalar multiplier")

    for _ in range(bisection_iterations):
        alpha = 0.5 * (lo + hi)
        shed = run_alpha(alpha)
        if shed < problem.B_shed - shed_tol:
            lo = alpha
        else:
            hi = alpha

    if not np.isfinite(best_alpha):
        best_alpha = hi
    assert_feasible(problem, coeffs, best_y, shed_tol=shed_tol)
    return MirrorDescentResult(
        best_y,
        best_obj,
        best_alpha,
        {key: np.asarray(value) for key, value in hist.items()},
        eta_x0,
        bisection_iterations,
    )
