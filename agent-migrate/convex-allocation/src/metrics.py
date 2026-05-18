from __future__ import annotations

import numpy as np

from coefficients import REPLAY, STATE, Coefficients, move_view
from problem import ProblemData


def resource_loads(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = move_view(y, problem)
    L_net = problem.ell_net + np.sum(coeffs.b_net * x, axis=(0, 2))
    L_prefill = problem.ell_prefill + np.sum(coeffs.b_prefill * x, axis=(0, 2))
    return L_net, L_prefill


def utilization(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L_net, L_prefill = resource_loads(problem, coeffs, y)
    return L_net / problem.C_net, L_prefill / problem.C_prefill


def shed_achieved(problem: ProblemData, y: np.ndarray) -> float:
    moved = np.sum(move_view(y, problem), axis=(1, 2))
    return float(np.dot(problem.tau, moved))


def action_mix(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    total = float(np.sum(problem.d))
    return {
        "replay_frac": float(np.sum(x[:, :, REPLAY]) / total),
        "state_frac": float(np.sum(x[:, :, STATE]) / total),
        "stay_frac": float(np.sum(y[:, -1]) / total),
    }


def shed_action_mix(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    replay = float(np.dot(problem.tau, np.sum(x[:, :, REPLAY], axis=1)))
    state = float(np.dot(problem.tau, np.sum(x[:, :, STATE], axis=1)))
    total = replay + state
    if total == 0.0:
        return {"replay_shed_frac": 0.0, "state_shed_frac": 0.0}
    return {"replay_shed_frac": replay / total, "state_shed_frac": state / total}


def shed_destination_mix(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    x = move_view(y, problem)
    moved = np.sum(x, axis=2)
    shed = problem.tau @ moved
    total = float(np.sum(shed))
    return shed / total if total > 0.0 else np.zeros(problem.K)


def allocation_diagnostics(problem: ProblemData, coeffs: Coefficients, y: np.ndarray) -> dict[str, float]:
    x = move_view(y, problem)
    moved_by_class = np.sum(x, axis=(1, 2))
    moved_by_dest = np.sum(x, axis=(0, 2))
    dest_share = shed_destination_mix(problem, y)
    action = shed_action_mix(problem, y)
    action_share = np.array([action["replay_shed_frac"], action["state_shed_frac"]])
    net_util, prefill_util = utilization(problem, coeffs, y)
    return {
        "active_classes_moved": float(np.sum(moved_by_class > 1e-8)),
        "active_destinations_used": float(np.sum(moved_by_dest > 1e-8)),
        "destination_entropy": _normalized_entropy(dest_share),
        "action_entropy": _normalized_entropy(action_share),
        "max_net_util": float(np.max(net_util)),
        "max_prefill_util": float(np.max(prefill_util)),
        "replay_shed_frac": action["replay_shed_frac"],
        "state_shed_frac": action["state_shed_frac"],
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
    if shed_achieved(problem, y) < problem.B_shed - shed_tol:
        raise AssertionError("shed target not met")
    net_util, prefill_util = utilization(problem, coeffs, y)
    if not (np.all(net_util < 1.0) and np.all(prefill_util < 1.0)):
        raise AssertionError("allocation is outside the barrier domain")
