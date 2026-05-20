from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratedWorkload:
    T: np.ndarray
    d: np.ndarray
    deadline_s: np.ndarray
    h_ctx: np.ndarray
    h_kv: np.ndarray


def generate_workload(
    K: int,
    seed: int | None = None,
    jobs: int = 10_000,
    classes: int = 48,
    profile: str = "agentic_retained_sessions",
) -> GeneratedWorkload:
    if profile != "agentic_retained_sessions":
        raise ValueError(f"unknown workload profile: {profile}")
    if K <= 0 or jobs <= 0 or classes <= 0:
        raise ValueError("K, jobs, and classes must be positive")

    rng = np.random.default_rng(seed)
    T = _sample_context_tokens(rng, jobs)
    deadline_s = _sample_deadline_s(rng, T)
    h_ctx, h_kv = _sample_locality(rng, jobs, K)
    return _aggregate(T, deadline_s, h_ctx, h_kv, min(classes, jobs))


def workload_quality_diagnostics(
    problem,
    cvx_y: np.ndarray,
    crossover_y: np.ndarray,
    cvx_objective: float,
    crossover_objective: float | None,
    crossover_feasible: bool,
    util_threshold: float = 0.7,
) -> dict[str, float | bool]:
    from coefficients import compute_coefficients
    from metrics import allocation_diagnostics, retained_prefill_action_mix, retained_prefill_destination_mix

    diag = allocation_diagnostics(problem, compute_coefficients(problem), cvx_y)
    cvx_action = retained_prefill_action_mix(problem, cvx_y)
    greedy_action = retained_prefill_action_mix(problem, crossover_y)
    cvx_dest = retained_prefill_destination_mix(problem, cvx_y)
    greedy_dest = retained_prefill_destination_mix(problem, crossover_y)
    gap = (
        (crossover_objective - cvx_objective) / max(1.0, abs(cvx_objective))
        if crossover_feasible and crossover_objective is not None
        else np.inf
    )
    mix_distance = float(
        abs(
            cvx_action["replay_retained_prefill_fraction"]
            - greedy_action["replay_retained_prefill_fraction"]
        )
        + np.sum(np.abs(cvx_dest - greedy_dest))
    )
    return {
        **diag,
        "crossover_relative_gap": float(gap),
        "crossover_mix_distance": mix_distance,
        "uses_multiple_classes": diag["active_classes_moved"] > 1.0,
        "uses_multiple_destinations": diag["active_destinations_used"] > 1.0,
        "uses_both_actions": min(
            diag["replay_retained_prefill_fraction"],
            diag["state_transfer_retained_prefill_fraction"],
        )
        >= 0.05,
        "has_resource_pressure": max(diag["max_net_util"], diag["max_prefill_util"])
        > util_threshold,
        "crossover_differs": bool(crossover_feasible and (gap >= 0.02 or mix_distance >= 0.10)),
    }


def assert_workload_quality(
    problem,
    cvx_y: np.ndarray,
    crossover_y: np.ndarray,
    cvx_objective: float,
    crossover_objective: float | None,
    crossover_feasible: bool,
) -> None:
    diag = workload_quality_diagnostics(
        problem, cvx_y, crossover_y, cvx_objective, crossover_objective, crossover_feasible
    )
    checks = (
        "uses_multiple_classes",
        "uses_multiple_destinations",
        "has_resource_pressure",
    )
    failed = {key: diag[key] for key in checks if not diag[key]}
    if failed:
        raise RuntimeError(f"workload is degenerate: {failed}; diagnostics={diag}")


def _sample_context_tokens(rng: np.random.Generator, jobs: int) -> np.ndarray:
    tier = rng.choice(4, size=jobs, p=(0.20, 0.45, 0.30, 0.05))
    T = np.empty(jobs)
    specs = (
        (tier == 0, 12_288.0, 0.35, 4_096.0, 24_576.0),
        (tier == 1, 32_768.0, 0.40, 12_288.0, 80_000.0),
        (tier == 2, 80_000.0, 0.35, 40_000.0, 160_000.0),
    )
    for mask, median, sigma, lo, hi in specs:
        T[mask] = np.clip(rng.lognormal(np.log(median), sigma, int(np.sum(mask))), lo, hi)
    tail = tier == 3
    T[tail] = rng.uniform(180_000.0, 256_000.0, int(np.sum(tail)))
    return np.rint(T).astype(float)


def _sample_deadline_s(rng: np.random.Generator, T: np.ndarray) -> np.ndarray:
    x = (np.log(T) - np.log(4_096.0)) / (np.log(256_000.0) - np.log(4_096.0))
    deadline_s = np.exp(np.log(12.0) + 2.2 * x + rng.normal(0.0, 0.95, T.size))
    deadline_s *= rng.choice((0.45, 1.0, 1.8), size=T.size, p=(0.25, 0.60, 0.15))
    return np.clip(deadline_s, 5.0, 600.0)


def _sample_locality(
    rng: np.random.Generator, jobs: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    h_ctx = rng.uniform(0.0, 0.08, size=(jobs, K))
    h_kv = h_ctx * rng.uniform(0.0, 0.4, size=(jobs, K))
    kind = rng.choice(4, size=jobs, p=(0.35, 0.35, 0.25, 0.05))
    dest = rng.integers(0, K, size=jobs)
    for value, kv_lo, kv_hi, mask in (
        (0.80, 0.02, 0.25, kind == 1),
        (0.86, 0.70, 0.98, kind == 2),
        (0.90, 0.30, 0.65, kind == 3),
    ):
        rows = np.flatnonzero(mask)
        ctx = np.clip(rng.normal(value, 0.08, rows.size), 0.55, 0.98)
        h_ctx[rows, dest[rows]] = ctx
        h_kv[rows, dest[rows]] = ctx * rng.uniform(kv_lo, kv_hi, rows.size)
    rows = np.flatnonzero(kind == 3)
    if rows.size and K > 1:
        other = (dest[rows] + rng.integers(1, K, rows.size)) % K
        ctx = np.clip(rng.normal(0.78, 0.10, rows.size), 0.50, 0.95)
        h_ctx[rows, other] = np.maximum(h_ctx[rows, other], ctx)
        h_kv[rows, other] = np.maximum(h_kv[rows, other], ctx * rng.uniform(0.75, 0.98, rows.size))
    return h_ctx, np.minimum(h_kv, h_ctx)


def _aggregate(
    T: np.ndarray, deadline_s: np.ndarray, h_ctx: np.ndarray, h_kv: np.ndarray, cap: int
) -> GeneratedWorkload:
    buckets = []
    for key in sorted({_bucket_key(T[i], deadline_s[i], h_ctx[i], h_kv[i]) for i in range(T.size)}):
        idx = np.array(
            [i for i in range(T.size) if _bucket_key(T[i], deadline_s[i], h_ctx[i], h_kv[i]) == key]
        )
        buckets.append(_summary(idx, T, deadline_s, h_ctx, h_kv))
    while len(buckets) > cap:
        _, i, j = min(
            (_distance(buckets[i], buckets[j]), i, j)
            for i in range(len(buckets))
            for j in range(i + 1, len(buckets))
        )
        buckets[i] = _merge(buckets[i], buckets[j])
        del buckets[j]
    buckets.sort(key=lambda b: (b[1], b[2], *b[3], *b[4]))
    d = np.array([b[0] for b in buckets], dtype=float)
    T_out = np.maximum(256.0, 256.0 * np.rint([b[1] / 256.0 for b in buckets]))
    return GeneratedWorkload(
        T_out.astype(float),
        d,
        np.array([b[2] for b in buckets], dtype=float),
        np.vstack([b[3] for b in buckets]),
        np.vstack([b[4] for b in buckets]),
    )


def _bucket_key(T: float, deadline_s: float, h_ctx: np.ndarray, h_kv: np.ndarray) -> tuple[int, int, int]:
    ctx = int(np.digitize(T, (16_000.0, 48_000.0, 96_000.0, 180_000.0)))
    deadline = int(np.digitize(deadline_s, (30.0, 90.0, 180.0)))
    if np.max(h_kv) >= 0.55:
        locality = 1 + int(np.argmax(h_kv))
    elif np.max(h_ctx) >= 0.55:
        locality = 1 + h_ctx.size + int(np.argmax(h_ctx))
    else:
        locality = 0
    return ctx, deadline, locality


def _summary(idx: np.ndarray, T, deadline_s, h_ctx, h_kv):
    return (
        float(idx.size),
        float(np.mean(T[idx])),
        float(np.mean(deadline_s[idx])),
        np.mean(h_ctx[idx], axis=0),
        np.mean(h_kv[idx], axis=0),
    )


def _merge(a, b):
    n = a[0] + b[0]
    wa, wb = a[0] / n, b[0] / n
    return n, wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3], wa * a[4] + wb * b[4]


def _distance(a, b) -> float:
    x = np.r_[np.log(a[1]) / 2.0, np.log(a[2]) / 2.0, a[3], a[4]]
    y = np.r_[np.log(b[1]) / 2.0, np.log(b[2]) / 2.0, b[3], b[4]]
    return float(np.sum((x - y) ** 2))
