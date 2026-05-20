from __future__ import annotations

import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from baselines import solve_crossover_greedy, solve_mixed_greedy, solve_replay_only, solve_state_only
from catalog import ModelParams
from coefficients import compute_coefficients
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy
from objective import objective
from problem import ProblemData
from queueing import evaluate_rounded_queue, round_allocation

from experiments.run_queue_failure_diagnostics import repair_rounded_allocation

COLUMNS = (
    "case",
    "policy",
    "fractional_objective",
    "integer_objective",
    "integer_objective_gap_to_best",
    "p95_delay",
    "p95_gap_to_best_queue",
    "miss_rate",
    "miss_rate_gap_to_best_queue",
    "movement_target_met",
)
DRAIN_WINDOW_S = 0.0


@dataclass(frozen=True)
class Case:
    name: str
    T: tuple[int, ...]
    d: tuple[int, ...]
    target_fraction: float
    deadline_s: tuple[float, ...]


CASES = (
    Case("8req_2site", (1, 2), (5, 3), 0.55, (0.8, 1.4)),
    Case("14req_2site", (1, 3), (7, 7), 0.50, (0.7, 1.8)),
    Case("20req_2site", (2,), (20,), 0.50, (1.4,)),
)
POLICIES = (
    ("CVXPY-rounded", True, lambda problem: _rounded_pair(solve_cvxpy(problem).y, problem)),
    (
        "deadline-aware-rounded",
        True,
        lambda problem: _rounded_pair(
            solve_deadline_aware_cvxpy(
                problem, deadline_margin=1.0, retained_prefill_cap=problem.retained_prefill_target_s
            ).y,
            problem,
        ),
    ),
    (
        "repaired-CVXPY-rounded",
        True,
        lambda problem: _repaired_cvxpy_pair(problem),
    ),
    ("crossover-greedy", False, lambda problem: _rounded_pair(solve_crossover_greedy(problem).allocation, problem)),
    ("mixed-greedy", False, lambda problem: _rounded_pair(solve_mixed_greedy(problem).allocation, problem)),
    ("replay-only", False, lambda problem: _rounded_pair(solve_replay_only(problem).allocation, problem)),
    ("state-only", False, lambda problem: _rounded_pair(solve_state_only(problem).allocation, problem)),
)


def run_integer_optimality_cases() -> list[dict[str, str]]:
    rows = [row for case in CASES for row in _case_rows(case)]
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "integer_optimality_cases.csv", rows)
    _print_table(rows)
    return rows


def _case_rows(case: Case) -> list[dict[str, str]]:
    problem = make_case_problem(case)
    objective_best = exact_integer_objective_optimum(problem)
    queue_best = exact_integer_queue_optimum(problem)
    queue_metrics = evaluate_rounded_queue(problem, queue_best.y, drain_window_s=DRAIN_WINDOW_S)
    rows = [
        _row(case.name, "best-integer-objective", None, objective_best.y, objective_best.objective, queue_metrics),
        _row(case.name, "best-integer-queue", None, queue_best.y, objective_best.objective, queue_metrics),
    ]
    for policy, has_fractional_objective, solver in POLICIES:
        relaxed, integer = solver(problem)
        rows.append(
            _row(
                case.name,
                policy,
                relaxed if has_fractional_objective else None,
                integer,
                objective_best.objective,
                queue_metrics,
            )
        )
    return rows


@dataclass(frozen=True)
class ExactResult:
    y: np.ndarray
    objective: float


def exact_integer_optimum(problem: ProblemData) -> ExactResult:
    return exact_integer_objective_optimum(problem)


def exact_integer_objective_optimum(problem: ProblemData) -> ExactResult:
    coeffs = compute_coefficients(problem)
    best_y = None
    best = math.inf
    for y in _target_feasible_integer_allocations(problem):
        value = objective(problem, coeffs, y)
        if value < best:
            best = value
            best_y = y
    if best_y is None:
        raise RuntimeError("no target-feasible integer allocation")
    return ExactResult(best_y, best)


def exact_integer_queue_optimum(problem: ProblemData) -> ExactResult:
    coeffs = compute_coefficients(problem)
    best_y = None
    best_key = None
    best_objective = math.inf
    for y in _target_feasible_integer_allocations(problem):
        metrics = evaluate_rounded_queue(problem, y, drain_window_s=DRAIN_WINDOW_S)
        value = objective(problem, coeffs, y)
        key = (
            metrics["deadline_miss_rate"],
            metrics["p95_reconstruction_delay"],
            metrics["mean_reconstruction_delay"],
            value,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_y = y
            best_objective = value
    if best_y is None:
        raise RuntimeError("no target-feasible integer allocation")
    return ExactResult(best_y, best_objective)


def make_case_problem(case: Case) -> ProblemData:
    model = ModelParams("integer-evidence", 1.0, 3.0, 1.0, 0.0)
    T = np.asarray(case.T, dtype=float)
    d = np.asarray(case.d, dtype=float)
    target = case.target_fraction * float(np.dot(T, d))
    return ProblemData(
        model=model,
        regime=case.name,
        T=T,
        d=d,
        deadline_s=np.asarray(case.deadline_s, dtype=float),
        lambda_Bps=np.array([16.0, 10.0]),
        rho_prefill=np.array([7.0, 11.0]),
        C_net=np.array([160.0, 100.0]),
        C_prefill=np.array([70.0, 110.0]),
        ell_net=np.zeros(2),
        ell_prefill=np.zeros(2),
        h_ctx=np.zeros((len(T), 2)),
        h_kv=np.zeros((len(T), 2)),
        retained_prefill_target_s=target,
    )


def _row(
    case: str,
    policy: str,
    fractional: np.ndarray | None,
    integer: np.ndarray,
    best_objective: float,
    best_queue_metrics: dict[str, float],
) -> dict[str, str]:
    problem = make_case_problem(next(item for item in CASES if item.name == case))
    coeffs = compute_coefficients(problem)
    metrics = evaluate_rounded_queue(problem, integer, drain_window_s=DRAIN_WINDOW_S)
    integer_objective = objective(problem, coeffs, integer)
    return {
        "case": case,
        "policy": policy,
        "fractional_objective": "NA" if fractional is None else _fmt(objective(problem, coeffs, fractional)),
        "integer_objective": _fmt(integer_objective),
        "integer_objective_gap_to_best": _fmt(integer_objective - best_objective),
        "p95_delay": _fmt(metrics["p95_reconstruction_delay"]),
        "p95_gap_to_best_queue": _fmt(
            metrics["p95_reconstruction_delay"] - best_queue_metrics["p95_reconstruction_delay"]
        ),
        "miss_rate": _fmt(metrics["deadline_miss_rate"]),
        "miss_rate_gap_to_best_queue": _fmt(
            metrics["deadline_miss_rate"] - best_queue_metrics["deadline_miss_rate"]
        ),
        "movement_target_met": str(metrics["retained_prefill_moved_s"] >= metrics["retained_prefill_target_s"] - 1e-9),
    }


def _target_feasible_integer_allocations(problem: ProblemData):
    coeffs = compute_coefficients(problem)
    choices = tuple(_row_allocations(int(n), coeffs.M + 1) for n in problem.d)
    for rows in itertools.product(*choices):
        y = np.asarray(rows, dtype=float)
        if _retained_prefill_moved(problem, y) >= problem.retained_prefill_target_s - 1e-9:
            yield y


def _row_allocations(total: int, width: int) -> tuple[tuple[int, ...], ...]:
    if width == 1:
        return ((total,),)
    rows = []
    for n in range(total + 1):
        for tail in _row_allocations(total - n, width - 1):
            rows.append((n, *tail))
    return tuple(rows)


def _retained_prefill_moved(problem: ProblemData, y: np.ndarray) -> float:
    return float(np.dot(problem.tau, np.sum(y[:, : y.shape[1] - 1], axis=1)))


def _rounded_pair(y: np.ndarray, problem: ProblemData) -> tuple[np.ndarray, np.ndarray]:
    return y, y if np.allclose(y, np.rint(y)) else round_allocation(problem, y).y


def _repaired_cvxpy_pair(problem: ProblemData) -> tuple[np.ndarray, np.ndarray]:
    relaxed = solve_cvxpy(problem).y
    rounded = round_allocation(problem, relaxed).y
    return relaxed, repair_rounded_allocation(problem, rounded, drain_window_s=DRAIN_WINDOW_S).y


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, str]]) -> None:
    widths = {col: max(len(col), *(len(row[col]) for row in rows)) for col in COLUMNS}
    print(" | ".join(col.ljust(widths[col]) for col in COLUMNS))
    print("-+-".join("-" * widths[col] for col in COLUMNS))
    for row in rows:
        print(" | ".join(row[col].ljust(widths[col]) for col in COLUMNS))


def _fmt(value: float) -> str:
    return f"{value:.10g}"


if __name__ == "__main__":
    run_integer_optimality_cases()
