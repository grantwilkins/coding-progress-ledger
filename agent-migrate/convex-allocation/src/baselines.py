from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coefficients import REPLAY, STATE, Coefficients, compute_coefficients
from metrics import available_rates, capacity_loads, retained_prefill_moved_s
from objective import objective
from problem import ProblemData


@dataclass(frozen=True)
class BaselineResult:
    feasible: bool
    objective: float | None
    retained_prefill_moved_s: float
    allocation: np.ndarray


def solve_replay_only(problem: ProblemData) -> BaselineResult:
    return _solve_greedy(problem, (REPLAY,), True)


def solve_state_only(problem: ProblemData) -> BaselineResult:
    return _solve_greedy(problem, (STATE,), True)


def solve_mixed_greedy(problem: ProblemData) -> BaselineResult:
    return _solve_greedy(problem, (REPLAY, STATE), True)


def solve_crossover_greedy(problem: ProblemData) -> BaselineResult:
    return _solve_greedy(problem, (REPLAY, STATE), False)


def solve_least_loaded_destination(problem: ProblemData) -> BaselineResult:
    return _solve_online(problem, "load")


def solve_online_queue_greedy(problem: ProblemData) -> BaselineResult:
    return _solve_online(problem, "queue")


def _solve_greedy(problem: ProblemData, actions: tuple[int, ...], use_load_cost: bool) -> BaselineResult:
    coeffs = compute_coefficients(problem)
    y = np.zeros((problem.G, coeffs.M + 1))
    y[:, -1] = problem.d
    remaining_retained_prefill = problem.retained_prefill_target_s

    while remaining_retained_prefill > 1e-12:
        L_net, L_prefill = capacity_loads(problem, coeffs, y)
        choices = []
        for g in range(problem.G):
            if y[g, -1] <= 1e-12:
                continue
            for m, (k, action) in enumerate(zip(coeffs.option_dest, coeffs.option_action)):
                if action not in actions:
                    continue
                b_net = coeffs.b_net[g, k, action]
                b_prefill = coeffs.b_prefill[g, k, action]
                room_net = (1.0 - 1e-9) * problem.C_net[k] - L_net[k]
                room_prefill = (1.0 - 1e-9) * problem.C_prefill[k] - L_prefill[k]
                cap_net = np.inf if b_net == 0 else room_net / b_net
                cap_prefill = np.inf if b_prefill == 0 else room_prefill / b_prefill
                amount = min(y[g, -1], remaining_retained_prefill / problem.tau[g], cap_net, cap_prefill)
                if amount > 1e-12:
                    marginal = coeffs.q[g, k, action]
                    if use_load_cost:
                        marginal += (
                            problem.w * b_net / (problem.C_net[k] - L_net[k])
                            + problem.w * b_prefill / (problem.C_prefill[k] - L_prefill[k])
                        )
                    choices.append((marginal / problem.tau[g], g, m, amount))
        if not choices:
            achieved = retained_prefill_moved_s(problem, y)
            return BaselineResult(False, None, achieved, y)
        _, g, m, amount = min(choices, key=lambda item: item[0])
        y[g, m] += amount
        y[g, -1] -= amount
        remaining_retained_prefill -= amount * problem.tau[g]

    achieved = retained_prefill_moved_s(problem, y)
    obj = objective(problem, coeffs, y)
    feasible = achieved >= problem.retained_prefill_target_s - 1e-8 and np.isfinite(obj)
    return BaselineResult(feasible, obj if feasible else None, achieved, y)


def _solve_online(problem: ProblemData, key: str) -> BaselineResult:
    coeffs = compute_coefficients(problem)
    _, lambda_avail, rho_avail = available_rates(problem)
    y = np.zeros((problem.G, coeffs.M + 1), dtype=int)
    y[:, -1] = _integer_array(problem.d, "class demand")
    L_net = problem.ell_net.copy()
    L_prefill = problem.ell_prefill.copy()
    net_done = np.zeros(problem.K)
    prefill_done = np.zeros(problem.K)
    moved = 0.0

    for g in _edf_requests(problem):
        if moved >= problem.retained_prefill_target_s - 1e-12:
            break
        choices = []
        for m, (k, action) in enumerate(zip(coeffs.option_dest, coeffs.option_action)):
            k = int(k)
            action = int(action)
            b_net = coeffs.b_net[g, k, action]
            b_prefill = coeffs.b_prefill[g, k, action]
            if (
                L_net[k] + b_net >= (1.0 - 1e-9) * problem.C_net[k]
                or L_prefill[k] + b_prefill >= (1.0 - 1e-9) * problem.C_prefill[k]
            ):
                continue
            net_complete = net_done[k] + b_net / lambda_avail[k]
            prefill_complete = max(prefill_done[k], net_complete) + b_prefill / rho_avail[k]
            completion = prefill_complete if action == REPLAY else net_complete
            load = max((L_net[k] + b_net) / problem.C_net[k], (L_prefill[k] + b_prefill) / problem.C_prefill[k])
            choices.append(((completion if key == "queue" else load), coeffs.q[g, k, action], k, action, m))
        if not choices:
            break

        _, _, k, action, m = min(choices)
        b_net = coeffs.b_net[g, k, action]
        b_prefill = coeffs.b_prefill[g, k, action]
        net_done[k] += b_net / lambda_avail[k]
        if action == REPLAY:
            prefill_done[k] = max(prefill_done[k], net_done[k]) + b_prefill / rho_avail[k]
        L_net[k] += b_net
        L_prefill[k] += b_prefill
        y[g, m] += 1
        y[g, -1] -= 1
        moved += problem.tau[g]

    achieved = retained_prefill_moved_s(problem, y)
    obj = objective(problem, coeffs, y)
    feasible = achieved >= problem.retained_prefill_target_s - 1e-8 and np.isfinite(obj)
    return BaselineResult(feasible, obj if feasible else None, achieved, y)


def _edf_requests(problem: ProblemData):
    counts = _integer_array(problem.d, "class demand")
    for g in sorted(range(problem.G), key=lambda g: (problem.deadline_s[g], g)):
        for _ in range(int(counts[g])):
            yield g


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    rounded = np.rint(values).astype(int)
    if np.any(rounded < 0) or not np.allclose(values, rounded):
        raise ValueError(f"{name} must be nonnegative integer-valued")
    return rounded
