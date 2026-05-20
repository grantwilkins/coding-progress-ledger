from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from coefficients import ACTIONS, REPLAY, compute_coefficients, move_view
from metrics import (
    available_rates,
    nvl72_hbm_fraction,
    resident_state_bytes,
    resident_state_moved_bytes,
    average_equivalent_state_target_bytes,
    state_tb,
    total_retained_prefill_s,
)
from problem import ProblemData

EXACT_ROUNDING_MAX_REQUESTS = 200


@dataclass(frozen=True)
class RequestRecord:
    g: int
    k: int
    action: str
    T: float
    deadline_s: float
    network_demand: float
    prefill_demand: float
    release_time_s: float = 0.0


@dataclass(frozen=True)
class RoundedAllocation:
    y: np.ndarray
    retained_prefill_target_s: float
    retained_prefill_moved_s: float


@dataclass(frozen=True)
class _CountedRequest:
    g: int
    k: int
    action: str
    deadline_s: float
    count: int
    network_service_s: float
    prefill_service_s: float
    release_rank: int


@dataclass(frozen=True)
class QueueTraceRecord:
    g: int
    k: int
    action: str
    deadline_s: float
    release_time_s: float
    network_queue_wait: float
    network_service_time: float
    prefill_queue_wait: float
    prefill_service_time: float
    reconstruction_delay: float
    deadline_missed: bool


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
        problem.retained_prefill_target_s,
        float(np.dot(problem.tau, moved)),
    )


def evaluate_static_queue(
    problem: ProblemData,
    records: tuple[RequestRecord, ...],
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> dict[str, float]:
    return _evaluate_static_queue(problem, records, drain_window_s, release_policy)[0]


def evaluate_static_queue_trace(
    problem: ProblemData,
    records: tuple[RequestRecord, ...],
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
    return _evaluate_static_queue(problem, records, drain_window_s, release_policy)


def evaluate_rounded_queue(
    problem: ProblemData,
    y: np.ndarray,
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> dict[str, float]:
    y = _integer_allocation(problem, y)
    retained_prefill_moved = float(np.dot(problem.tau, np.sum(y[:, : y.shape[1] - 1], axis=1)))
    metrics = _evaluate_counted_queue(problem, y, drain_window_s, release_policy)
    _add_retained_metrics(metrics, problem, y, retained_prefill_moved)
    _add_drain_metrics(metrics, retained_prefill_moved, drain_window_s)
    return metrics


def evaluate_rounded_allocation(
    problem: ProblemData,
    rounded: RoundedAllocation,
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> dict[str, float]:
    metrics = _evaluate_counted_queue(problem, rounded.y, drain_window_s, release_policy)
    _add_retained_metrics(metrics, problem, rounded.y, rounded.retained_prefill_moved_s)
    _add_drain_metrics(metrics, rounded.retained_prefill_moved_s, drain_window_s)
    return metrics


def evaluate_rounded_queue_trace(
    problem: ProblemData,
    y: np.ndarray,
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
    y = _integer_allocation(problem, y)
    retained_prefill_moved = float(np.dot(problem.tau, np.sum(y[:, : y.shape[1] - 1], axis=1)))
    metrics, trace = evaluate_static_queue_trace(
        problem, _request_records(problem, y), drain_window_s, release_policy
    )
    _add_retained_metrics(metrics, problem, y, retained_prefill_moved)
    _add_drain_metrics(metrics, retained_prefill_moved, drain_window_s)
    return metrics, trace


def _evaluate_static_queue(
    problem: ProblemData,
    records: tuple[RequestRecord, ...],
    drain_window_s: float,
    release_policy: str,
) -> tuple[dict[str, float], tuple[QueueTraceRecord, ...]]:
    H, lambda_avail, rho_avail = _available_rates(problem)
    records = _paced_records(records, drain_window_s, release_policy)
    if not records:
        return {
            "mean_reconstruction_delay": 0.0,
            "p50_reconstruction_delay": 0.0,
            "p95_reconstruction_delay": 0.0,
            "p99_reconstruction_delay": 0.0,
            "p95_normalized_reconstruction_delay": 0.0,
            "deadline_miss_rate": 0.0,
            "network_capacity_pressure": 0.0,
            "prefill_capacity_pressure": 0.0,
            "drain_completion_s": 0.0,
            "replay_retained_prefill_fraction": 0.0,
            "state_transfer_retained_prefill_fraction": 0.0,
        }, ()

    net_done = np.zeros(len(records))
    net_wait = np.zeros(len(records))
    net_service = np.zeros(len(records))
    net_busy = np.zeros(problem.K)
    for k in range(problem.K):
        jobs = [
            (r.release_time_s, r.network_demand / lambda_avail[k], r.deadline_s, (r.g, i), i)
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

    replay_work = sum(problem.tau[r.g] for r in records if r.action == ACTIONS[REPLAY])
    state_work = sum(problem.tau[r.g] for r in records if r.action != ACTIONS[REPLAY])
    total_work = replay_work + state_work
    release = np.asarray([r.release_time_s for r in records], dtype=float)
    delay = np.asarray(complete, dtype=float) - release
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
        "network_capacity_pressure": float(np.max(net_busy) / H),
        "prefill_capacity_pressure": float(np.max(prefill_busy) / H),
        "drain_completion_s": float(np.max(complete)),
        "replay_retained_prefill_fraction": float(replay_work / total_work),
        "state_transfer_retained_prefill_fraction": float(state_work / total_work),
    }
    trace = tuple(
        QueueTraceRecord(
            r.g,
            r.k,
            r.action,
            r.deadline_s,
            r.release_time_s,
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


def _evaluate_counted_queue(
    problem: ProblemData,
    y: np.ndarray,
    drain_window_s: float,
    release_policy: str,
) -> dict[str, float]:
    records = _counted_requests(problem, y, drain_window_s, release_policy)
    if not records:
        return _empty_queue_metrics()

    H, _, _ = _available_rates(problem)
    total = sum(record.count for record in records)
    release = [_release_times(record, total, drain_window_s) for record in records]
    net_done: list[np.ndarray | None] = [None] * len(records)
    complete: list[np.ndarray | None] = [None] * len(records)
    net_busy = np.zeros(problem.K)
    prefill_busy = np.zeros(problem.K)

    for k in range(problem.K):
        time = 0.0
        for i, record in enumerate(records):
            if record.k != k:
                continue
            done = _schedule_arrivals(release[i], time, record.network_service_s)
            net_done[i] = done
            complete[i] = done
            net_busy[k] += record.network_service_s * record.count
            time = float(done[-1])

    for k in range(problem.K):
        time = 0.0
        for i, record in enumerate(records):
            if record.k != k or record.action != ACTIONS[REPLAY]:
                continue
            done = _schedule_arrivals(net_done[i], time, record.prefill_service_s)
            complete[i] = done
            prefill_busy[k] += record.prefill_service_s * record.count
            time = float(done[-1])

    delay = np.concatenate([complete[i] - release[i] for i in range(len(records))])
    deadline_s = np.concatenate([np.full(record.count, record.deadline_s) for record in records])
    replay_work = sum(problem.tau[record.g] * record.count for record in records if record.action == ACTIONS[REPLAY])
    state_work = sum(problem.tau[record.g] * record.count for record in records if record.action != ACTIONS[REPLAY])
    total_work = replay_work + state_work
    p95_ratio = float(np.percentile(delay / deadline_s, 95))
    return {
        "mean_reconstruction_delay": float(np.mean(delay)),
        "p50_reconstruction_delay": float(np.percentile(delay, 50)),
        "p95_reconstruction_delay": float(np.percentile(delay, 95)),
        "p99_reconstruction_delay": float(np.percentile(delay, 99)),
        "p95_normalized_reconstruction_delay": p95_ratio,
        "p95_reconstruction_delay_ratio": p95_ratio,
        "deadline_miss_rate": float(np.mean(delay > deadline_s)),
        "network_capacity_pressure": float(np.max(net_busy) / H),
        "prefill_capacity_pressure": float(np.max(prefill_busy) / H),
        "drain_completion_s": float(max(np.max(done) for done in complete if done is not None)),
        "replay_retained_prefill_fraction": float(replay_work / total_work),
        "state_transfer_retained_prefill_fraction": float(state_work / total_work),
    }


def _empty_queue_metrics() -> dict[str, float]:
    return {
        "mean_reconstruction_delay": 0.0,
        "p50_reconstruction_delay": 0.0,
        "p95_reconstruction_delay": 0.0,
        "p99_reconstruction_delay": 0.0,
        "p95_normalized_reconstruction_delay": 0.0,
        "deadline_miss_rate": 0.0,
        "network_capacity_pressure": 0.0,
        "prefill_capacity_pressure": 0.0,
        "drain_completion_s": 0.0,
        "replay_retained_prefill_fraction": 0.0,
        "state_transfer_retained_prefill_fraction": 0.0,
    }


def queue_metrics(
    problem: ProblemData,
    y: np.ndarray,
    drain_window_s: float = 1800.0,
    release_policy: str = "edf",
) -> dict[str, float]:
    if np.allclose(y, np.rint(y)):
        return evaluate_rounded_queue(problem, y, drain_window_s, release_policy)
    rounded = round_allocation(problem, y)
    return evaluate_rounded_allocation(problem, rounded, drain_window_s, release_policy)


def fractional_queue_load_proxy(problem: ProblemData, y: np.ndarray) -> dict[str, float]:
    coeffs = compute_coefficients(problem)
    H, lambda_avail, rho_avail = _available_rates(problem)
    x = move_view(y, problem)
    net = np.sum(coeffs.b_net * x, axis=(0, 2)) / lambda_avail / H
    prefill = np.sum(coeffs.b_prefill * x, axis=(0, 2)) / rho_avail / H
    return {
        "fractional_network_capacity_pressure": float(np.max(net)),
        "fractional_prefill_capacity_pressure": float(np.max(prefill)),
    }


def _rounded_moved_counts(
    problem: ProblemData, y: np.ndarray, d: np.ndarray, T: np.ndarray
) -> tuple[int, ...]:
    moved_float = np.sum(y[:, : y.shape[1] - 1], axis=1)
    target = max(0, int(np.ceil(problem.retained_prefill_target_s * problem.model.prefill_tok_s - 1e-9)))
    total = int(np.dot(d, T))
    if target > total:
        raise ValueError("retained prefill target exceeds total class work")
    if int(np.sum(d)) > EXACT_ROUNDING_MAX_REQUESTS:
        return _greedy_moved_counts(moved_float, d, T, target)

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
        raise ValueError("integer rounding cannot meet retained prefill target within moved class support")
    return min(eligible)[2]


def _greedy_moved_counts(moved_float: np.ndarray, d: np.ndarray, T: np.ndarray, target: int) -> tuple[int, ...]:
    max_count = np.where(moved_float > 1e-12, d, 0)
    if target == 0:
        return tuple(0 for _ in d)
    if int(np.dot(max_count, T)) < target:
        raise ValueError("integer rounding cannot meet retained prefill target within moved class support")

    scale = min(1.0, target / max(1.0, float(np.dot(moved_float, T))))
    counts = np.minimum(np.floor(moved_float * scale), max_count).astype(int)
    used = int(np.dot(counts, T))
    while used < target:
        choices = []
        for g in np.flatnonzero(counts < max_count):
            new_used = used + int(T[g])
            delta = abs(counts[g] + 1 - moved_float[g]) - abs(counts[g] - moved_float[g])
            choices.append((new_used > target, max(0, new_used - target), delta, int(g)))
        if not choices:
            raise ValueError("integer rounding cannot meet retained prefill target within moved class support")
        g = min(choices)[3]
        counts[g] += 1
        used += int(T[g])
    return tuple(int(n) for n in counts)


def _apportion(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights / np.sum(weights)
    counts = np.floor(raw).astype(int)
    remaining = total - int(np.sum(counts))
    if remaining:
        order = np.lexsort((np.arange(raw.size), -(raw - counts)))
        counts[order[:remaining]] += 1
    return counts


def _counted_requests(
    problem: ProblemData, y: np.ndarray, drain_window_s: float, release_policy: str
) -> tuple[_CountedRequest, ...]:
    if drain_window_s < 0.0:
        raise ValueError("drain_window_s must be nonnegative")
    if release_policy != "edf":
        raise ValueError(f"unknown release policy: {release_policy}")
    coeffs = compute_coefficients(problem)
    _, lambda_avail, rho_avail = _available_rates(problem)
    cells = []
    record_index = 0
    for g in range(problem.G):
        for m, count in enumerate(y[g, : coeffs.M]):
            count = int(count)
            if count:
                k = int(coeffs.option_dest[m])
                action = int(coeffs.option_action[m])
                cells.append(
                    (
                        float(problem.deadline_s[g]),
                        int(g),
                        record_index,
                        _CountedRequest(
                            int(g),
                            k,
                            ACTIONS[action],
                            float(problem.deadline_s[g]),
                            count,
                            float(coeffs.b_net[g, k, action] / lambda_avail[k]),
                            float(coeffs.b_prefill[g, k, action] / rho_avail[k]),
                            0,
                        ),
                    )
                )
            record_index += count

    rank = 0
    records = []
    for _, _, _, record in sorted(cells):
        records.append(
            _CountedRequest(
                record.g,
                record.k,
                record.action,
                record.deadline_s,
                record.count,
                record.network_service_s,
                record.prefill_service_s,
                rank,
            )
        )
        rank += record.count
    return tuple(records)


def _release_times(record: _CountedRequest, total: int, drain_window_s: float) -> np.ndarray:
    if drain_window_s == 0.0:
        return np.zeros(record.count)
    ranks = record.release_rank + np.arange(record.count)
    return drain_window_s * ranks / total


def _schedule_arrivals(arrivals: np.ndarray, time: float, service: float) -> np.ndarray:
    i = np.arange(arrivals.size)
    if service <= 0.0:
        return np.maximum.accumulate(np.maximum(arrivals, time))
    base = np.maximum.accumulate(arrivals - i * service)
    return (i + 1.0) * service + np.maximum(base, time)


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


def _paced_records(
    records: tuple[RequestRecord, ...], drain_window_s: float, release_policy: str
) -> tuple[RequestRecord, ...]:
    if drain_window_s < 0.0:
        raise ValueError("drain_window_s must be nonnegative")
    if release_policy != "edf":
        raise ValueError(f"unknown release policy: {release_policy}")
    if not records:
        return records
    order = sorted(range(len(records)), key=lambda i: (records[i].deadline_s, records[i].g, i))
    release = np.zeros(len(records))
    if drain_window_s > 0.0:
        for rank, i in enumerate(order):
            release[i] = drain_window_s * rank / len(records)
    return tuple(
        RequestRecord(
            record.g,
            record.k,
            record.action,
            record.T,
            record.deadline_s,
            record.network_demand,
            record.prefill_demand,
            float(release[i]),
        )
        for i, record in enumerate(records)
    )


def _add_retained_metrics(
    metrics: dict[str, float], problem: ProblemData, y: np.ndarray, achieved: float
) -> None:
    target = problem.retained_prefill_target_s
    total = total_retained_prefill_s(problem)
    moved_bytes = resident_state_moved_bytes(problem, y)
    average_equivalent_target_bytes = average_equivalent_state_target_bytes(problem)
    total_bytes = resident_state_bytes(problem)
    ratio = np.nan if target == 0.0 else achieved / target
    metrics.update(
        {
            "retained_prefill_moved_s": achieved,
            "retained_prefill_target_s": target,
            "retained_prefill_ratio": ratio,
            "rounded_retained_prefill_moved_s": achieved,
            "rounded_retained_prefill_target_s": target,
            "rounded_retained_prefill_ratio": ratio,
            "retained_prefill_fraction": 0.0 if total == 0.0 else target / total,
            "retained_prefill_moved_fraction": 0.0 if total == 0.0 else achieved / total,
            "resident_state_tb": state_tb(total_bytes),
            "average_equivalent_state_target_tb": state_tb(average_equivalent_target_bytes),
            "actual_evacuated_state_tb": state_tb(moved_bytes),
            "resident_state_nvl72_hbm_fraction": nvl72_hbm_fraction(total_bytes),
            "average_equivalent_state_target_nvl72_hbm_fraction": nvl72_hbm_fraction(
                average_equivalent_target_bytes
            ),
            "actual_evacuated_nvl72_hbm_fraction": nvl72_hbm_fraction(moved_bytes),
        }
    )


def _add_drain_metrics(metrics: dict[str, float], achieved: float, drain_window_s: float) -> None:
    metrics["drain_window_s"] = drain_window_s
    if drain_window_s == 0.0:
        rate = np.inf if achieved > 0.0 else 0.0
    else:
        rate = achieved / drain_window_s
    metrics["retained_prefill_removal_rate_s_per_s"] = rate


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
            arrival, service, deadline_s, tie, idx = pending[i]
            heappush(ready, (deadline_s, tie, arrival, service, idx))
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
