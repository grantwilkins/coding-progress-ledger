from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from coefficients import ACTIONS, REPLAY, compute_coefficients, move_view
from problem import ProblemData


@dataclass(frozen=True)
class RequestRecord:
    g: int
    k: int
    action: str
    T: float
    slack: float
    network_demand: float
    prefill_demand: float


@dataclass(frozen=True)
class RoundedAllocation:
    y: np.ndarray
    shed_target: float
    rounded_shed: float
    records: tuple[RequestRecord, ...]


def round_allocation(problem: ProblemData, y: np.ndarray) -> RoundedAllocation:
    coeffs = compute_coefficients(problem)
    y = np.asarray(y, dtype=float)
    if y.shape != (problem.G, coeffs.M + 1):
        raise ValueError("allocation has wrong shape")
    if np.any(y < -1e-8) or not np.allclose(np.sum(y, axis=1), problem.d):
        raise ValueError("allocation must be nonnegative and preserve class demand")
    d = _integer_array(problem.d, "class demand")
    T = _integer_array(problem.T, "context length")
    moved = _rounded_moved_counts(problem, y, d, T)
    rounded = np.zeros_like(y, dtype=int)

    for g, n in enumerate(moved):
        if n:
            row = y[g, : coeffs.M]
            if np.sum(row) <= 0.0:
                raise ValueError("cannot apportion moved requests from a zero moved row")
            rounded[g, : coeffs.M] = _apportion(n, row)
        rounded[g, -1] = d[g] - n

    return RoundedAllocation(
        rounded,
        problem.B_shed,
        float(np.dot(problem.tau, moved)),
        _request_records(problem, rounded),
    )


def evaluate_static_queue(problem: ProblemData, records: tuple[RequestRecord, ...]) -> dict[str, float]:
    H, lambda_avail, rho_avail = _available_rates(problem)
    if not records:
        return {
            "mean_reconstruction_delay": 0.0,
            "p50_reconstruction_delay": 0.0,
            "p95_reconstruction_delay": 0.0,
            "p99_reconstruction_delay": 0.0,
            "p95_normalized_reconstruction_delay": 0.0,
            "deadline_miss_rate": 0.0,
            "max_network_busy_window": 0.0,
            "max_prefill_busy_window": 0.0,
            "replay_shed_frac": 0.0,
            "state_shed_frac": 0.0,
        }

    net_done = np.zeros(len(records))
    net_busy = np.zeros(problem.K)
    for k in range(problem.K):
        jobs = [
            (0.0, r.network_demand / lambda_avail[k], r.slack, (r.g, i), i)
            for i, r in enumerate(records)
            if r.k == k
        ]
        done, net_busy[k] = _schedule(jobs)
        for i, t in done.items():
            net_done[i] = t

    complete = net_done.copy()
    prefill_busy = np.zeros(problem.K)
    for k in range(problem.K):
        jobs = [
            (net_done[i], r.prefill_demand / rho_avail[k], r.slack, (r.g, i), i)
            for i, r in enumerate(records)
            if r.k == k and r.action == ACTIONS[REPLAY]
        ]
        done, prefill_busy[k] = _schedule(jobs)
        for i, t in done.items():
            complete[i] = t

    replay_shed = sum(problem.tau[r.g] for r in records if r.action == ACTIONS[REPLAY])
    state_shed = sum(problem.tau[r.g] for r in records if r.action != ACTIONS[REPLAY])
    total_shed = replay_shed + state_shed
    delay = np.asarray(complete, dtype=float)
    slack = np.asarray([r.slack for r in records], dtype=float)
    return {
        "mean_reconstruction_delay": float(np.mean(delay)),
        "p50_reconstruction_delay": float(np.percentile(delay, 50)),
        "p95_reconstruction_delay": float(np.percentile(delay, 95)),
        "p99_reconstruction_delay": float(np.percentile(delay, 99)),
        "p95_normalized_reconstruction_delay": float(np.percentile(delay / slack, 95)),
        "deadline_miss_rate": float(np.mean(delay > slack)),
        "max_network_busy_window": float(np.max(net_busy) / H),
        "max_prefill_busy_window": float(np.max(prefill_busy) / H),
        "replay_shed_frac": float(replay_shed / total_shed),
        "state_shed_frac": float(state_shed / total_shed),
    }


def queue_metrics(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    rounded = round_allocation(problem, y)
    metrics = evaluate_static_queue(problem, rounded.records)
    ratio = np.nan if rounded.shed_target == 0.0 else rounded.rounded_shed / rounded.shed_target
    metrics.update(
        {
            "rounded_shed_achieved": rounded.rounded_shed,
            "rounded_shed_target": rounded.shed_target,
            "rounded_shed_ratio": ratio,
        }
    )
    return metrics


def fractional_queue_load_proxy(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    coeffs = compute_coefficients(problem)
    H, lambda_avail, rho_avail = _available_rates(problem)
    x = move_view(y, problem)
    net = np.sum(coeffs.b_net * x, axis=(0, 2)) / lambda_avail / H
    prefill = np.sum(coeffs.b_prefill * x, axis=(0, 2)) / rho_avail / H
    return {
        "fractional_max_network_busy_window": float(np.max(net)),
        "fractional_max_prefill_busy_window": float(np.max(prefill)),
    }


def _rounded_moved_counts(
    problem: ProblemData, y: np.ndarray, d: np.ndarray, T: np.ndarray
) -> tuple[int, ...]:
    moved_float = np.sum(y[:, : y.shape[1] - 1], axis=1)
    target = max(0, int(np.ceil(problem.B_shed * problem.model.prefill_tok_s - 1e-9)))
    total = int(np.dot(d, T))
    if target > total:
        raise ValueError("shed target exceeds total class work")

    cap = min(total, target + int(np.max(T)))
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for g, (count, tokens) in enumerate(zip(d, T)):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        max_count = count if moved_float[g] > 1e-12 else 0
        for used, (dev, counts) in states.items():
            for n in range(max_count + 1):
                new_used = used + n * tokens
                if new_used > cap:
                    continue
                new_counts = counts + (n,)
                new_dev = dev + abs(n - moved_float[g])
                old = next_states.get(new_used)
                if (
                    old is None
                    or new_dev < old[0] - 1e-12
                    or (abs(new_dev - old[0]) <= 1e-12 and new_counts < old[1])
                ):
                    next_states[new_used] = (new_dev, new_counts)
        states = next_states

    eligible = [
        (used - target, dev, counts)
        for used, (dev, counts) in states.items()
        if used >= target
    ]
    if not eligible:
        raise ValueError("integer rounding cannot meet shed target within moved class support")
    return min(eligible)[2]


def _apportion(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights / np.sum(weights)
    counts = np.floor(raw).astype(int)
    remaining = total - int(np.sum(counts))
    if remaining:
        order = np.lexsort((np.arange(raw.size), -(raw - counts)))
        counts[order[:remaining]] += 1
    return counts


def _request_records(problem: ProblemData, y: np.ndarray) -> tuple[RequestRecord, ...]:
    coeffs = compute_coefficients(problem)
    records = []
    for g in range(problem.G):
        for m, count in enumerate(y[g, : coeffs.M]):
            k = int(coeffs.option_dest[m])
            action = int(coeffs.option_action[m])
            records.extend(
                RequestRecord(
                    g,
                    k,
                    ACTIONS[action],
                    float(problem.T[g]),
                    float(problem.slack[g]),
                    float(coeffs.b_net[g, k, action]),
                    float(coeffs.b_prefill[g, k, action]),
                )
                for _ in range(int(count))
            )
    return tuple(records)


def _available_rates(problem: ProblemData) -> tuple[float, np.ndarray, np.ndarray]:
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


def _schedule(jobs: list[tuple[float, float, float, tuple[int, int], int]]) -> tuple[dict[int, float], float]:
    pending = sorted(jobs)
    ready: list[tuple[float, tuple[int, int], float, float, int]] = []
    done = {}
    busy = 0.0
    time = 0.0
    i = 0
    while i < len(pending) or ready:
        if not ready:
            time = max(time, pending[i][0])
        while i < len(pending) and pending[i][0] <= time + 1e-12:
            arrival, service, slack, tie, idx = pending[i]
            heappush(ready, (slack, tie, arrival, service, idx))
            i += 1
        _, _, arrival, service, idx = heappop(ready)
        time = max(time, arrival) + service
        busy += service
        done[idx] = time
    return done, busy


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    rounded = np.rint(values).astype(int)
    if np.any(rounded < 0) or not np.allclose(values, rounded):
        raise ValueError(f"{name} must be nonnegative integer-valued")
    return rounded
