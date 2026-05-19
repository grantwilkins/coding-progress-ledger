from __future__ import annotations

import numpy as np

from coefficients import REPLAY, STATE, Coefficients, move_view
from problem import ProblemData


def available_rates(problem: ProblemData) -> tuple[float, np.ndarray, np.ndarray]:
    windows = np.concatenate(
        [problem.C_net / problem.lambda_Bps, problem.C_prefill / problem.rho_prefill]
    )
    if not np.allclose(windows, windows[0]):
        raise ValueError("resource capacities imply inconsistent windows")
    H = float(windows[0])
    lambda_avail = (problem.C_net - problem.ell_net) / H
    rho_avail = (problem.C_prefill - problem.ell_prefill) / H
    if np.any(lambda_avail <= 0.0) or np.any(rho_avail <= 0.0):
        raise ValueError("background load leaves nonpositive service rate")
    return H, lambda_avail, rho_avail


def resource_loads(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = move_view(y, problem)
    L_net = problem.ell_net + np.sum(coeffs.b_net * x, axis=(0, 2))
    L_prefill = problem.ell_prefill + np.sum(coeffs.b_prefill * x, axis=(0, 2))
    return L_net, L_prefill


def utilization(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L_net, L_prefill = resource_loads(problem, coeffs, y)
    return L_net / problem.C_net, L_prefill / problem.C_prefill


def relief_achieved_s(problem: ProblemData, y: np.ndarray) -> float:
    moved = np.sum(move_view(y, problem), axis=(1, 2))
    return float(np.dot(problem.tau, moved))


def shed_achieved(problem: ProblemData, y: np.ndarray) -> float:
    return relief_achieved_s(problem, y)


def action_mix(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    total = float(np.sum(problem.d))
    return {
        "replay_frac": float(np.sum(x[:, :, REPLAY]) / total),
        "state_frac": float(np.sum(x[:, :, STATE]) / total),
        "stay_frac": float(np.sum(y[:, -1]) / total),
    }


def relief_action_mix(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    replay = float(np.dot(problem.tau, np.sum(x[:, :, REPLAY], axis=1)))
    state = float(np.dot(problem.tau, np.sum(x[:, :, STATE], axis=1)))
    total = replay + state
    if total == 0.0:
        return {"replay_relief_frac": 0.0, "state_relief_frac": 0.0}
    return {"replay_relief_frac": replay / total, "state_relief_frac": state / total}


def shed_action_mix(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    mix = relief_action_mix(problem, y)
    return {
        "replay_shed_frac": mix["replay_relief_frac"],
        "state_shed_frac": mix["state_relief_frac"],
    }


def relief_destination_mix(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    x = move_view(y, problem)
    moved = np.sum(x, axis=2)
    relief = problem.tau @ moved
    total = float(np.sum(relief))
    return relief / total if total > 0.0 else np.zeros(problem.K)


def shed_destination_mix(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    return relief_destination_mix(problem, y)


def deadline_load_ratios(
    problem: ProblemData, coeffs: Coefficients, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, lambda_avail, rho_avail = available_rates(problem)
    x = move_view(y, problem)
    thresholds = np.unique(problem.deadline_s)
    net = np.zeros((problem.K, thresholds.size))
    prefill = np.zeros_like(net)
    for j, deadline_s in enumerate(thresholds):
        classes = problem.deadline_s <= deadline_s
        net[:, j] = np.sum(coeffs.b_net[classes] * x[classes], axis=(0, 2)) / lambda_avail / deadline_s
        prefill[:, j] = (
            np.sum(coeffs.b_prefill[classes, :, REPLAY] * x[classes, :, REPLAY], axis=0)
            / rho_avail
            / deadline_s
        )
    return thresholds, net, prefill


def deadline_debt_ratios(
    problem: ProblemData, coeffs: Coefficients, y: np.ndarray, deadline_headroom: float
) -> np.ndarray:
    _, net, prefill = deadline_load_ratios(problem, coeffs, y)
    return np.maximum(np.r_[net.ravel(), prefill.ravel()] - deadline_headroom, 0.0)


def deadline_debt_summary(
    problem: ProblemData, coeffs: Coefficients, y: np.ndarray, deadline_headroom: float
) -> dict[str, float]:
    debt = deadline_debt_ratios(problem, coeffs, y, deadline_headroom)
    return {
        "deadline_debt_mean": float(np.mean(debt)),
        "deadline_debt_p95": float(np.percentile(debt, 95)),
        "deadline_debt_max": float(np.max(debt)),
    }


def allocation_diagnostics(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    moved_by_class = np.sum(x, axis=(1, 2))
    moved_by_dest = np.sum(x, axis=(0, 2))
    dest_share = relief_destination_mix(problem, y)
    action = relief_action_mix(problem, y)
    action_share = np.array([action["replay_relief_frac"], action["state_relief_frac"]])
    net_util, prefill_util = utilization(problem, coeffs, y)
    return {
        "active_classes_moved": float(np.sum(moved_by_class > 1e-8)),
        "active_destinations_used": float(np.sum(moved_by_dest > 1e-8)),
        "destination_entropy": _normalized_entropy(dest_share),
        "action_entropy": _normalized_entropy(action_share),
        "max_net_util": float(np.max(net_util)),
        "max_prefill_util": float(np.max(prefill_util)),
        "replay_relief_frac": action["replay_relief_frac"],
        "state_relief_frac": action["state_relief_frac"],
        "replay_shed_frac": action["replay_relief_frac"],
        "state_shed_frac": action["state_relief_frac"],
    }


def _normalized_entropy(p: np.ndarray) -> float:
    positive = p[p > 0.0]
    if positive.size <= 1:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / np.log(p.size))


def assert_feasible(problem: ProblemData, coeffs: Coefficients, y: np.ndarray, shed_tol: float) -> None:
    if not np.all(y >= -1e-8):
        raise AssertionError("allocation has negative entries")
    if not np.allclose(np.sum(y, axis=1), problem.d):
        raise AssertionError("allocation rows do not sum to class demand")
    if relief_achieved_s(problem, y) < problem.relief_target_s - shed_tol:
        raise AssertionError("relief target not met")
    net_util, prefill_util = utilization(problem, coeffs, y)
    if not (np.all(net_util < 1.0) and np.all(prefill_util < 1.0)):
        raise AssertionError("allocation is outside the barrier domain")
