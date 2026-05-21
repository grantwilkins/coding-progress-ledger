from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from catalog import ModelParams
from coefficients import compute_coefficients
from experiments.run_integer_optimality_cases import _target_feasible_integer_allocations
from objective import objective
from problem import ProblemData
from queueing import evaluate_rounded_queue

DRAIN_WINDOW_S = 20.0
RETAINED_PREFILL_FRACTION = 0.25
RELEASE_POLICY = "edf"
COLUMNS = (
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
    "verdict",
)


def run_h1_integer_oracle() -> list[dict[str, object]]:
    problem = make_h1_integer_problem()
    row = h1_integer_oracle_row(problem)
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "h1_integer_oracle.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return [row]


def make_h1_integer_problem() -> ProblemData:
    T = np.array([2.0, 4.0])
    d = np.array([4.0, 4.0])
    return ProblemData(
        model=ModelParams("h1-integer-oracle", 1.0, 4.0, 1.0, 0.0),
        regime="h1-integer-oracle",
        T=T,
        d=d,
        deadline_s=np.array([30.0, 35.0]),
        lambda_Bps=np.array([10.0, 10.0]),
        rho_prefill=np.array([3.0, 3.0]),
        C_net=np.array([200.0, 200.0]),
        C_prefill=np.array([60.0, 60.0]),
        ell_net=np.zeros(2),
        ell_prefill=np.zeros(2),
        h_ctx=np.zeros((2, 2)),
        h_kv=np.zeros((2, 2)),
        retained_prefill_target_s=RETAINED_PREFILL_FRACTION * float(np.dot(T, d)),
    )


def h1_integer_oracle_row(problem: ProblemData) -> dict[str, object]:
    coeffs = compute_coefficients(problem)
    best = None
    enumerated = 0
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
    target_moved_fraction = metrics["retained_prefill_moved_s"] / metrics["retained_prefill_target_s"]
    return {
        "policy": "Integer feasibility oracle",
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
        "verdict": _verdict(metrics, target_moved_fraction),
    }


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
    for row in run_h1_integer_oracle():
        print(" | ".join(_fmt(row[column]) for column in COLUMNS))
