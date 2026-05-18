from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines import (
    solve_crossover_greedy,
    solve_mixed_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import get_model
from coefficients import ACTIONS, compute_coefficients
from cvxpy_solver import solve_cvxpy
from metrics import utilization
from mirror_descent import solve_mirror_descent
from problem import make_problem
from queueing import evaluate_rounded_queue_trace, round_allocation

SHED_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
TIGHT_SLACK_MULTIPLIERS = (0.25, 0.50)
POLICIES = (
    ("mirror-descent-rounded", lambda problem: solve_mirror_descent(problem, eta_x0=500.0)),
    ("crossover-greedy", solve_crossover_greedy),
    ("mixed-greedy", solve_mixed_greedy),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
QUEUE_COLUMNS = (
    "policy",
    "shed_fraction",
    "slack_multiplier",
    "status",
    "safe",
    "rounded_shed_achieved",
    "rounded_shed_target",
    "rounded_shed_ratio",
    "mean_delay",
    "p50_delay",
    "p95_delay",
    "p99_delay",
    "p95_normalized_delay",
    "miss_rate",
    "max_net_busy",
    "max_prefill_busy",
    "replay_shed_frac",
    "state_shed_frac",
    "repair_move_count",
    "repair_move_summary",
)
BREAKDOWN_COLUMNS = (
    "policy",
    "shed_fraction",
    "slack_multiplier",
    "status",
    "group_type",
    "group",
    "moved_requests",
    "missed_requests",
    "miss_rate",
    "avg_missed_network_wait",
    "avg_missed_prefill_wait",
    "avg_missed_total_delay",
)


@dataclass(frozen=True)
class RepairMove:
    g: int
    from_k: int
    from_action: str
    to_k: int
    to_action: str


@dataclass(frozen=True)
class RepairResult:
    y: np.ndarray
    metrics: dict[str, float]
    trace: tuple
    moves: tuple[RepairMove, ...]


def run_queue_failure_diagnostics():
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    queue_rows = []
    breakdown_rows = []
    summary_rows = []
    model = get_model("GLM-5")

    for slack_multiplier in TIGHT_SLACK_MULTIPLIERS:
        for shed_fraction in SHED_FRACTIONS:
            problem = make_problem(
                model,
                "transition-coupled",
                shed_fraction=shed_fraction,
                slack_multiplier=slack_multiplier,
            )
            cvx = solve_cvxpy(problem)
            rounded = round_allocation(problem, cvx.y)
            metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y)
            queue_rows.append(
                _queue_row("CVXPY-rounded", shed_fraction, slack_multiplier, "OK", metrics)
            )
            breakdown_rows.extend(
                _failure_breakdown_rows(
                    "CVXPY-rounded", problem, shed_fraction, slack_multiplier, "OK", trace
                )
            )

            repair = repair_rounded_allocation(problem, rounded.y)
            queue_rows.append(
                _queue_row(
                    "repaired-CVXPY-rounded",
                    shed_fraction,
                    slack_multiplier,
                    "OK",
                    repair.metrics,
                    repair.moves,
                )
            )
            breakdown_rows.extend(
                _failure_breakdown_rows(
                    "repaired-CVXPY-rounded",
                    problem,
                    shed_fraction,
                    slack_multiplier,
                    "OK",
                    repair.trace,
                )
            )
            summary_rows.append(_summary_row(shed_fraction, slack_multiplier, metrics, repair))

            for policy, solver in POLICIES:
                row, trace = _solver_queue(policy, solver, problem, shed_fraction, slack_multiplier)
                queue_rows.append(row)
                breakdown_rows.extend(
                    _failure_breakdown_rows(
                        policy, problem, shed_fraction, slack_multiplier, row["status"], trace
                    )
                )

    _write_rows(out / "transition_coupled_repaired_queue_table.csv", queue_rows, QUEUE_COLUMNS)
    _write_rows(
        out / "transition_coupled_queue_failure_breakdown.csv",
        breakdown_rows,
        BREAKDOWN_COLUMNS,
    )
    _print_repair_summary(summary_rows)
    return queue_rows, breakdown_rows, summary_rows


def repair_rounded_allocation(problem, y, max_steps=1000):
    coeffs = compute_coefficients(problem)
    y = np.asarray(y, dtype=float)
    metrics, trace = evaluate_rounded_queue_trace(problem, y)
    y = np.rint(y).astype(int)
    moves = []

    for _ in range(max_steps):
        best = None
        best_key = _queue_key(metrics)
        for g in range(problem.G):
            for source in range(coeffs.M):
                if y[g, source] <= 0:
                    continue
                for target in range(coeffs.M):
                    if target == source:
                        continue
                    candidate = y.copy()
                    candidate[g, source] -= 1
                    candidate[g, target] += 1
                    if not _capacity_feasible(problem, candidate):
                        continue
                    candidate_metrics, candidate_trace = evaluate_rounded_queue_trace(
                        problem, candidate
                    )
                    candidate_key = _queue_key(candidate_metrics)
                    if candidate_key < best_key:
                        best_key = candidate_key
                        best = (
                            candidate,
                            candidate_metrics,
                            candidate_trace,
                            _repair_move(coeffs, g, source, target),
                        )
        if best is None:
            return RepairResult(y, metrics, trace, tuple(moves))
        y, metrics, trace, move = best
        moves.append(move)
    raise RuntimeError("rounded local repair did not converge")


def _solver_queue(policy, solver, problem, shed_fraction, slack_multiplier):
    base = _empty_queue_row(policy, shed_fraction, slack_multiplier, problem.B_shed)
    try:
        result = solver(problem)
    except RuntimeError:
        return base, ()
    if not getattr(result, "feasible", True):
        return base, ()
    y = result.allocation if hasattr(result, "allocation") else result.y
    rounded = round_allocation(problem, y)
    metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y)
    return _queue_row(policy, shed_fraction, slack_multiplier, "OK", metrics), trace


def _queue_key(metrics):
    return (
        metrics["deadline_miss_rate"],
        metrics["p95_reconstruction_delay"],
        metrics["mean_reconstruction_delay"],
    )


def _capacity_feasible(problem, y):
    coeffs = compute_coefficients(problem)
    net, prefill = utilization(problem, coeffs, y)
    return bool(np.all(net <= 1.0 + 1e-9) and np.all(prefill <= 1.0 + 1e-9))


def _repair_move(coeffs, g, source, target):
    return RepairMove(
        g,
        int(coeffs.option_dest[source]),
        ACTIONS[int(coeffs.option_action[source])],
        int(coeffs.option_dest[target]),
        ACTIONS[int(coeffs.option_action[target])],
    )


def _queue_row(policy, shed_fraction, slack_multiplier, status, metrics, moves=()):
    return {
        "policy": policy,
        "shed_fraction": shed_fraction,
        "slack_multiplier": slack_multiplier,
        "status": status,
        "safe": _safe(metrics),
        "rounded_shed_achieved": metrics["rounded_shed_achieved"],
        "rounded_shed_target": metrics["rounded_shed_target"],
        "rounded_shed_ratio": metrics["rounded_shed_ratio"],
        "mean_delay": metrics["mean_reconstruction_delay"],
        "p50_delay": metrics["p50_reconstruction_delay"],
        "p95_delay": metrics["p95_reconstruction_delay"],
        "p99_delay": metrics["p99_reconstruction_delay"],
        "p95_normalized_delay": metrics["p95_normalized_reconstruction_delay"],
        "miss_rate": metrics["deadline_miss_rate"],
        "max_net_busy": metrics["max_network_busy_window"],
        "max_prefill_busy": metrics["max_prefill_busy_window"],
        "replay_shed_frac": metrics["replay_shed_frac"],
        "state_shed_frac": metrics["state_shed_frac"],
        "repair_move_count": len(moves),
        "repair_move_summary": _move_summary(moves),
    }


def _empty_queue_row(policy, shed_fraction, slack_multiplier, shed_target):
    return {
        "policy": policy,
        "shed_fraction": shed_fraction,
        "slack_multiplier": slack_multiplier,
        "status": "INFEASIBLE",
        "safe": False,
        "rounded_shed_achieved": math.nan,
        "rounded_shed_target": shed_target,
        "rounded_shed_ratio": math.nan,
        "mean_delay": math.nan,
        "p50_delay": math.nan,
        "p95_delay": math.nan,
        "p99_delay": math.nan,
        "p95_normalized_delay": math.nan,
        "miss_rate": math.nan,
        "max_net_busy": math.nan,
        "max_prefill_busy": math.nan,
        "replay_shed_frac": math.nan,
        "state_shed_frac": math.nan,
        "repair_move_count": 0,
        "repair_move_summary": "",
    }


def _safe(metrics):
    return (
        metrics["rounded_shed_achieved"] >= metrics["rounded_shed_target"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_normalized_reconstruction_delay"] <= 1.0
    )


def _move_summary(moves):
    if not moves:
        return ""
    counts = Counter(
        (
            move.g,
            move.from_k,
            move.from_action,
            move.to_k,
            move.to_action,
        )
        for move in moves
    )
    return "; ".join(
        f"class{g}:k{src}/{src_action}->k{dst}/{dst_action} x{count}"
        for (g, src, src_action, dst, dst_action), count in sorted(counts.items())
    )


def _failure_breakdown_rows(policy, problem, shed_fraction, slack_multiplier, status, trace):
    if not trace:
        return [
            {
                "policy": policy,
                "shed_fraction": shed_fraction,
                "slack_multiplier": slack_multiplier,
                "status": status,
                "group_type": "all",
                "group": status,
                "moved_requests": 0,
                "missed_requests": 0,
                "miss_rate": math.nan,
                "avg_missed_network_wait": math.nan,
                "avg_missed_prefill_wait": math.nan,
                "avg_missed_total_delay": math.nan,
            }
        ]
    groups = (
        ("class", tuple((f"class{g}", lambda r, g=g: r.g == g) for g in range(problem.G))),
        (
            "destination",
            tuple((f"k{k}", lambda r, k=k: r.k == k) for k in range(problem.K)),
        ),
        (
            "action",
            tuple((action, lambda r, action=action: r.action == action) for action in ACTIONS),
        ),
    )
    rows = []
    for group_type, entries in groups:
        for group, predicate in entries:
            rows.append(
                _breakdown_row(
                    policy,
                    shed_fraction,
                    slack_multiplier,
                    status,
                    group_type,
                    group,
                    tuple(record for record in trace if predicate(record)),
                )
            )
    return rows


def _breakdown_row(policy, shed_fraction, slack_multiplier, status, group_type, group, records):
    missed = tuple(record for record in records if record.deadline_missed)
    return {
        "policy": policy,
        "shed_fraction": shed_fraction,
        "slack_multiplier": slack_multiplier,
        "status": status,
        "group_type": group_type,
        "group": group,
        "moved_requests": len(records),
        "missed_requests": len(missed),
        "miss_rate": 0.0 if not records else len(missed) / len(records),
        "avg_missed_network_wait": _mean(record.network_queue_wait for record in missed),
        "avg_missed_prefill_wait": _mean(record.prefill_queue_wait for record in missed),
        "avg_missed_total_delay": _mean(record.reconstruction_delay for record in missed),
    }


def _mean(values):
    values = tuple(values)
    return 0.0 if not values else float(np.mean(values))


def _summary_row(shed_fraction, slack_multiplier, original, repair):
    return {
        "shed_fraction": shed_fraction,
        "slack_multiplier": slack_multiplier,
        "original_miss_rate": original["deadline_miss_rate"],
        "repaired_miss_rate": repair.metrics["deadline_miss_rate"],
        "original_p95_delay": original["p95_reconstruction_delay"],
        "repaired_p95_delay": repair.metrics["p95_reconstruction_delay"],
        "original_mean_delay": original["mean_reconstruction_delay"],
        "repaired_mean_delay": repair.metrics["mean_reconstruction_delay"],
        "repair_moves": len(repair.moves),
        "improved": _queue_key(repair.metrics) < _queue_key(original),
        "move_summary": _move_summary(repair.moves),
    }


def _print_repair_summary(rows):
    print("\nconvex-rounded local repair summary")
    cols = (
        "slack_multiplier",
        "shed_fraction",
        "original_miss_rate",
        "repaired_miss_rate",
        "original_p95_delay",
        "repaired_p95_delay",
        "repair_moves",
        "improved",
    )
    widths = {col: max(len(col), *(len(_fmt(row[col])) for row in rows)) for col in cols}
    print(" | ".join(col.ljust(widths[col]) for col in cols))
    print("-+-".join("-" * widths[col] for col in cols))
    for row in rows:
        print(" | ".join(_fmt(row[col]).ljust(widths[col]) for col in cols))


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_rows(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_queue_failure_diagnostics()
