from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from baselines import solve_least_loaded_destination, solve_online_queue_greedy, solve_replay_only, solve_state_only
from catalog import ModelParams
from coefficients import compute_coefficients
from cvxpy_solver import solve_soft_deadline_cvxpy
from experiments.run_integer_optimality_cases import _target_feasible_integer_allocations
from objective import objective
from problem import ProblemData
from queueing import evaluate_rounded_queue, round_allocation

DRAIN_WINDOW_S = 20.0
RETAINED_PREFILL_FRACTION = 0.25
RELEASE_POLICY = "edf"
SEEDS = tuple(range(7))
METHODS = (
    ("Deadline-aware", lambda problem: solve_soft_deadline_cvxpy(problem).y),
    ("Online queue", lambda problem: solve_online_queue_greedy(problem).allocation),
    ("Least loaded", lambda problem: solve_least_loaded_destination(problem).allocation),
    ("Replay only", lambda problem: solve_replay_only(problem).allocation),
    ("State only", lambda problem: solve_state_only(problem).allocation),
)
COLUMNS = (
    "seed",
    "policy",
    "integer_classes",
    "integer_requests",
    "enumerated_allocations",
    "drain_window_s",
    "retained_prefill_fraction",
    "release_policy",
    "target_moved_fraction",
    "network_capacity_pressure",
    "prefill_capacity_pressure",
    "absolute_p95_delay_over_deadline",
    "absolute_deadline_miss_rate",
    "integer_objective",
    "runtime_s",
    "verdict",
)
SUMMARY_COLUMNS = (
    "policy",
    "cases",
    "pass_rate_mean",
    "pass_rate_stderr",
    "target_moved_fraction_mean",
    "target_moved_fraction_stderr",
    "network_capacity_pressure_mean",
    "network_capacity_pressure_stderr",
    "prefill_capacity_pressure_mean",
    "prefill_capacity_pressure_stderr",
    "absolute_p95_delay_over_deadline_mean",
    "absolute_p95_delay_over_deadline_stderr",
    "absolute_deadline_miss_rate_mean",
    "absolute_deadline_miss_rate_stderr",
    "runtime_s_mean",
    "runtime_s_stderr",
)


def run_h1_integer_oracle(seeds: tuple[int, ...] = SEEDS) -> list[dict[str, object]]:
    rows = [row for seed in seeds for row in h1_integer_rows(make_h1_integer_problem(seed), seed)]
    summary = h1_integer_summary(rows)
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "h1_integer_oracle.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "h1_integer_oracle_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    return rows


def make_h1_integer_problem(seed: int = 0) -> ProblemData:
    rng = np.random.default_rng(seed)
    T = np.sort(rng.choice(np.array([2.0, 3.0, 4.0]), size=2, replace=False))
    d = rng.integers(3, 6, size=2).astype(float)
    lambda_Bps = rng.uniform(9.0, 12.0, size=2)
    rho_prefill = rng.uniform(2.7, 3.4, size=2)
    single_request_s = max(float(np.max(4.0 * T[:, None] / lambda_Bps)), float(np.max(T[:, None] / rho_prefill)))
    capacity_window_s = rng.uniform(1.12, 1.28) * single_request_s
    return ProblemData(
        model=ModelParams("h1-integer-oracle", 1.0, 4.0, 1.0, 0.0),
        regime="h1-integer-oracle",
        T=T,
        d=d,
        deadline_s=np.sort(rng.uniform(30.0, 38.0, size=2)),
        lambda_Bps=lambda_Bps,
        rho_prefill=rho_prefill,
        C_net=lambda_Bps * capacity_window_s,
        C_prefill=rho_prefill * capacity_window_s,
        ell_net=np.zeros(2),
        ell_prefill=np.zeros(2),
        h_ctx=np.zeros((2, 2)),
        h_kv=np.zeros((2, 2)),
        retained_prefill_target_s=RETAINED_PREFILL_FRACTION * float(np.dot(T, d)),
    )


def h1_integer_rows(problem: ProblemData, seed: int = 0) -> list[dict[str, object]]:
    return [
        h1_integer_oracle_row(problem, seed),
        *[h1_method_row(problem, seed, policy, solver) for policy, solver in METHODS],
    ]


def h1_integer_oracle_row(problem: ProblemData, seed: int = 0) -> dict[str, object]:
    coeffs = compute_coefficients(problem)
    best = None
    enumerated = 0
    start = time.perf_counter()
    for y in _target_feasible_integer_allocations(problem):
        enumerated += 1
        metrics = evaluate_rounded_queue(problem, y, DRAIN_WINDOW_S, RELEASE_POLICY)
        value = objective(problem, coeffs, y)
        key = _oracle_key(metrics, value)
        if best is None or key < best[0]:
            best = (key, metrics, value)
    if best is None:
        raise RuntimeError("no H1 integer oracle allocation")

    _, metrics, value = best
    return _row(seed, problem, "Integer feasibility oracle", enumerated, metrics, value, time.perf_counter() - start)


def h1_method_row(problem: ProblemData, seed: int, policy: str, solver) -> dict[str, object]:
    coeffs = compute_coefficients(problem)
    start = time.perf_counter()
    y = solver(problem)
    integer_y = y if np.allclose(y, np.rint(y)) else round_allocation(problem, y).y
    metrics = evaluate_rounded_queue(problem, integer_y, DRAIN_WINDOW_S, RELEASE_POLICY)
    return _row(seed, problem, policy, "", metrics, objective(problem, coeffs, integer_y), time.perf_counter() - start)


def _row(
    seed: int,
    problem: ProblemData,
    policy: str,
    enumerated: int | str,
    metrics: dict[str, float],
    value: float,
    runtime_s: float,
) -> dict[str, object]:
    target_moved_fraction = metrics["retained_prefill_moved_s"] / metrics["retained_prefill_target_s"]
    return {
        "seed": seed,
        "policy": policy,
        "integer_classes": problem.G,
        "integer_requests": int(np.sum(problem.d)),
        "enumerated_allocations": enumerated,
        "drain_window_s": DRAIN_WINDOW_S,
        "retained_prefill_fraction": RETAINED_PREFILL_FRACTION,
        "release_policy": RELEASE_POLICY,
        "target_moved_fraction": target_moved_fraction,
        "network_capacity_pressure": metrics["network_capacity_pressure"],
        "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
        "absolute_p95_delay_over_deadline": metrics["absolute_p95_delay_over_deadline"],
        "absolute_deadline_miss_rate": metrics["absolute_deadline_miss_rate"],
        "integer_objective": value,
        "runtime_s": runtime_s,
        "verdict": _verdict(metrics, target_moved_fraction),
    }


def h1_integer_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for policy in ("Integer feasibility oracle", *(name for name, _ in METHODS)):
        policy_rows = [row for row in rows if row["policy"] == policy]
        item = {"policy": policy, "cases": len(policy_rows)}
        for column in (
            "pass_rate",
            "target_moved_fraction",
            "network_capacity_pressure",
            "prefill_capacity_pressure",
            "absolute_p95_delay_over_deadline",
            "absolute_deadline_miss_rate",
            "runtime_s",
        ):
            values = (
                np.array([row["verdict"] == "Pass" for row in policy_rows], dtype=float)
                if column == "pass_rate"
                else np.array([float(row[column]) for row in policy_rows], dtype=float)
            )
            item[f"{column}_mean"] = float(np.mean(values))
            item[f"{column}_stderr"] = _stderr(values)
        summary.append(item)
    return summary


def _stderr(values: np.ndarray) -> float:
    return 0.0 if values.size <= 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))


def _oracle_key(metrics: dict[str, float], value: float) -> tuple[bool, bool, bool, bool, float, float, float, float]:
    return (
        metrics["network_capacity_pressure"] > 1.0,
        metrics["prefill_capacity_pressure"] > 1.0,
        metrics["absolute_p95_delay_over_deadline"] > 1.0,
        metrics["absolute_deadline_miss_rate"] > 0.01,
        metrics["absolute_deadline_miss_rate"],
        metrics["absolute_p95_delay_over_deadline"],
        max(metrics["network_capacity_pressure"], metrics["prefill_capacity_pressure"]),
        value,
    )


def _verdict(metrics: dict[str, float], target_moved_fraction: float) -> str:
    if target_moved_fraction < 1.0 - 1e-9:
        return "Target shortfall"
    if metrics["network_capacity_pressure"] > 1.0:
        return "Network overload"
    if metrics["prefill_capacity_pressure"] > 1.0:
        return "Prefill overload"
    if metrics["absolute_p95_delay_over_deadline"] > 1.0:
        return "P95 deadline miss"
    if metrics["absolute_deadline_miss_rate"] > 0.01:
        return "Deadline miss rate"
    return "Pass"


def _fmt(value: object) -> str:
    return f"{value:.10g}" if isinstance(value, float) and math.isfinite(value) else str(value)


if __name__ == "__main__":
    rows = run_h1_integer_oracle()
    for row in h1_integer_summary(rows):
        print(" | ".join(_fmt(row[column]) for column in SUMMARY_COLUMNS))
