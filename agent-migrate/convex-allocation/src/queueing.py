from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from coefficients import ACTIONS, REPLAY, compute_coefficients, move_view
from metrics import available_rates
from problem import ProblemData


@dataclass(frozen=True)
class RequestRecord:
    g: int
    k: int
    action: str
    T: float
    deadline_s: float
    network_demand: float
    prefill_demand: float

    @property
    def slack(self) -> float:
        return self.deadline_s


@dataclass(frozen=True)
class RoundedAllocation:
    y: np.ndarray
    source_load_target_s: float
    source_load_moved_s: float
    records: tuple[RequestRecord, ...]

    @property
    def shed_target(self) -> float:
        return self.source_load_target_s

    @property
    def rounded_shed(self) -> float:
        return self.source_load_moved_s


@dataclass(frozen=True)
class QueueTraceRecord:
    g: int
    k: int
    action: str
    deadline_s: float
    network_queue_wait: float
    network_service_time: float
    prefill_queue_wait: float
    prefill_service_time: float
    reconstruction_delay: float
    deadline_missed: bool

    @property
    def slack(self) -> float:
        return self.deadline_s


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
        problem.source_load_target_s,
        float(np.dot(problem.tau, moved)),
        _request_records(problem, rounded),
    )


def evaluate_static_queue(problem: ProblemData, records: tuple[RequestRecord, ...]) -> dict[str, float]:
    return _evaluate_static_queue(problem, records)[0]


def evaluate_static_queue_trace(
    problem: ProblemData, records: tuple[RequestRecord, ...]
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
    return _evaluate_static_queue(problem, records)


def evaluate_rounded_queue(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    return evaluate_rounded_queue_trace(problem, y)[0]


def evaluate_rounded_queue_trace(
    problem: ProblemData, y: np.ndarray
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
    y = _integer_allocation(problem, y)
    rounded_shed = float(np.dot(problem.tau, np.sum(y[:, : y.shape[1] - 1], axis=1)))
    metrics, trace = evaluate_static_queue_trace(problem, _request_records(problem, y))
    _add_shed_metrics(metrics, problem.source_load_target_s, rounded_shed)
    return metrics, trace


def _evaluate_static_queue(
    problem: ProblemData, records: tuple[RequestRecord, ...]
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
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
        }, ()

    net_done = np.zeros(len(records))
    net_wait = np.zeros(len(records))
    net_service = np.zeros(len(records))
    net_busy = np.zeros(problem.K)
    for k in range(problem.K):
        jobs = [
            (0.0, r.network_demand / lambda_avail[k], r.deadline_s, (r.g, i), i)
            for i, r in enumerate(records)
            if r.k == k
        ]
        done, net_busy[k], spans = _schedule(jobs)
        for i, t in done.items():
            net_done[i] = t
            net_wait[i], net_service[i] = spans[i]

    complete = net_done.copy()
    prefill_wait = np.zeros(len(records))
    prefill_service = np.zeros(len(records))
    prefill_busy = np.zeros(problem.K)
    for k in range(problem.K):
        jobs = [
            (net_done[i], r.prefill_demand / rho_avail[k], r.deadline_s, (r.g, i), i)
            for i, r in enumerate(records)
            if r.k == k and r.action == ACTIONS[REPLAY]
        ]
        done, prefill_busy[k], spans = _schedule(jobs)
        for i, t in done.items():
            complete[i] = t
            prefill_wait[i], prefill_service[i] = spans[i]

    replay_shed = sum(problem.tau[r.g] for r in records if r.action == ACTIONS[REPLAY])
    state_shed = sum(problem.tau[r.g] for r in records if r.action != ACTIONS[REPLAY])
    total_shed = replay_shed + state_shed
    delay = np.asarray(complete, dtype=float)
    deadline_s = np.asarray([r.deadline_s for r in records], dtype=float)
    p95_ratio = float(np.percentile(delay / deadline_s, 95))
    miss_rate = float(np.mean(delay > deadline_s))
    metrics = {
        "mean_reconstruction_delay": float(np.mean(delay)),
        "p50_reconstruction_delay": float(np.percentile(delay, 50)),
        "p95_reconstruction_delay": float(np.percentile(delay, 95)),
        "p99_reconstruction_delay": float(np.percentile(delay, 99)),
        "p95_normalized_reconstruction_delay": p95_ratio,
        "p95_reconstruction_delay_ratio": p95_ratio,
        "deadline_miss_rate": miss_rate,
        "max_network_busy_window": float(np.max(net_busy) / H),
        "max_prefill_busy_window": float(np.max(prefill_busy) / H),
        "replay_shed_frac": float(replay_shed / total_shed),
        "state_shed_frac": float(state_shed / total_shed),
        "replay_load_frac": float(replay_shed / total_shed),
        "state_load_frac": float(state_shed / total_shed),
        "replay_relief_frac": float(replay_shed / total_shed),
        "state_relief_frac": float(state_shed / total_shed),
    }
    trace = tuple(
        QueueTraceRecord(
            r.g,
            r.k,
            r.action,
            r.deadline_s,
            float(net_wait[i]),
            float(net_service[i]),
            float(prefill_wait[i]),
            float(prefill_service[i]),
            float(delay[i]),
            bool(delay[i] > r.deadline_s),
        )
        for i, r in enumerate(records)
    )
    return metrics, trace


def queue_metrics(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    rounded = round_allocation(problem, y)
    metrics = evaluate_static_queue(problem, rounded.records)
    _add_shed_metrics(metrics, rounded.shed_target, rounded.rounded_shed)
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
    target = max(0, int(np.ceil(problem.source_load_target_s * problem.model.prefill_tok_s - 1e-9)))
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
                    float(problem.deadline_s[g]),
                    float(coeffs.b_net[g, k, action]),
                    float(coeffs.b_prefill[g, k, action]),
                )
                for _ in range(int(count))
            )
    return tuple(records)


def _add_shed_metrics(metrics: dict[str, float], target: float, achieved: float) -> None:
    ratio = np.nan if target == 0.0 else achieved / target
    metrics.update(
        {
            "source_load_moved_s": achieved,
            "source_load_target_s": target,
            "source_load_ratio": ratio,
            "load_moved_s": achieved,
            "load_target_s": target,
            "load_ratio": ratio,
            "rounded_shed_achieved": achieved,
            "rounded_shed_target": target,
            "rounded_shed_ratio": ratio,
            "rounded_relief_achieved_s": achieved,
            "relief_target_s": target,
            "relief_ratio": ratio,
        }
    )


def _integer_allocation(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    coeffs = compute_coefficients(problem)
    y = _integer_array(np.asarray(y, dtype=float), "rounded allocation").astype(int)
    if y.shape != (problem.G, coeffs.M + 1):
        raise ValueError("rounded allocation has wrong shape")
    if not np.allclose(np.sum(y, axis=1), problem.d):
        raise ValueError("rounded allocation must preserve class demand")
    return y


def _available_rates(problem: ProblemData) -> tuple[float, np.ndarray, np.ndarray]:
    return available_rates(problem)


def _schedule(
    jobs: list[tuple[float, float, float, tuple[int, int], int]]
) -> tuple[dict[int, float], float, dict[int, tuple[float, float]]]:
    pending = sorted(jobs)
    ready: list[tuple[float, tuple[int, int], float, float, int]] = []
    done = {}
    spans = {}
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
        start = max(time, arrival)
        time = start + service
        busy += service
        done[idx] = time
        spans[idx] = (start - arrival, service)
    return done, busy, spans


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    rounded = np.rint(values).astype(int)
    if np.any(rounded < 0) or not np.allclose(values, rounded):
        raise ValueError(f"{name} must be nonnegative integer-valued")
    return rounded
