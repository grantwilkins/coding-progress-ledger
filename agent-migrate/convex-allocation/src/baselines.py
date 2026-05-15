from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coefficients import REPLAY, STATE, Coefficients, compute_coefficients
from metrics import resource_loads, shed_achieved
from objective import objective
from problem import ProblemData


@dataclass(frozen=True)
class BaselineResult:
    feasible: bool
    objective: float | None
    shed_achieved: float
    allocation: np.ndarray


def solve_replay_only(problem: ProblemData) -> BaselineResult:
    return _solve_single_action(problem, REPLAY)


def solve_state_only(problem: ProblemData) -> BaselineResult:
    return _solve_single_action(problem, STATE)


def _solve_single_action(problem: ProblemData, action: int) -> BaselineResult:
    coeffs = compute_coefficients(problem)
    y = np.zeros((problem.G, coeffs.M + 1))
    y[:, -1] = problem.d
    order = np.argsort(-(problem.tau / problem.slack))
    remaining_shed = problem.B_shed

    for g in order:
        remaining_class = problem.d[g]
        while remaining_class > 1e-12 and remaining_shed > 1e-12:
            L_net, L_prefill = resource_loads(problem, coeffs, y)
            choices = []
            for k in range(problem.K):
                m = k * 2 + action
                b_net = coeffs.b_net[g, k, action]
                b_prefill = coeffs.b_prefill[g, k, action]
                room_net = (1.0 - 1e-9) * problem.C_net[k] - L_net[k]
                room_prefill = (1.0 - 1e-9) * problem.C_prefill[k] - L_prefill[k]
                cap_net = np.inf if b_net == 0 else room_net / b_net
                cap_prefill = np.inf if b_prefill == 0 else room_prefill / b_prefill
                cap = min(remaining_class, remaining_shed / problem.tau[g], cap_net, cap_prefill)
                if cap > 1e-12:
                    risk = (
                        coeffs.q[g, k, action]
                        + problem.w * b_net / (problem.C_net[k] - L_net[k])
                        + problem.w * b_prefill / (problem.C_prefill[k] - L_prefill[k])
                    )
                    choices.append((risk, k, m, cap))
            if not choices:
                achieved = shed_achieved(problem, y)
                return BaselineResult(False, None, achieved, y)
            _, _, m, amount = min(choices, key=lambda item: item[0])
            y[g, m] += amount
            y[g, -1] -= amount
            remaining_class -= amount
            remaining_shed -= amount * problem.tau[g]

    achieved = shed_achieved(problem, y)
    obj = objective(problem, coeffs, y)
    feasible = achieved >= problem.B_shed - 1e-8 and np.isfinite(obj)
    return BaselineResult(feasible, obj if feasible else None, achieved, y)
