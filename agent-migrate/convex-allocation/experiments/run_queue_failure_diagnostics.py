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
from cvxpy_solver import solve_cvxpy, solve_deadline_aware_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import (
    retained_prefill_action_mix,
    retained_prefill_destination_mix,
    retained_prefill_moved_s,
    utilization,
)
from mirror_descent import solve_mirror_descent
from objective import objective
from problem import make_problem
from queueing import evaluate_rounded_queue_trace, round_allocation

RETAINED_PREFILL_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
TIGHT_DEADLINE_SCALES = (0.25, 0.50)
REPAIR_BUDGET_FRACTIONS = (0.05, 0.10, 0.20)
DRAIN_WINDOW_S = 1800.0
POLICIES = (
    (
        "deadline-aware-m0.8-rounded",
        lambda problem: solve_deadline_aware_cvxpy(problem, 0.8, retained_prefill_cap=problem.retained_prefill_target_s),
    ),
    (
        "deadline-aware-m1.0-rounded",
        lambda problem: solve_deadline_aware_cvxpy(problem, 1.0, retained_prefill_cap=problem.retained_prefill_target_s),
    ),
    ("mirror-descent-rounded", lambda problem: solve_mirror_descent(problem, eta_x0=500.0)),
    ("crossover-greedy", solve_crossover_greedy),
    ("mixed-greedy", solve_mixed_greedy),
    ("replay-only", solve_replay_only),
    ("state-only", solve_state_only),
)
QUEUE_COLUMNS = (
    "policy",
    "retained_prefill_fraction",
    "deadline_scale",
    "status",
    "safe",
    "rounded_retained_prefill_moved_s",
    "rounded_retained_prefill_target_s",
    "rounded_retained_prefill_ratio",
    "mean_delay",
    "p50_delay",
    "p95_delay",
    "p99_delay",
    "p95_normalized_delay",
    "miss_rate",
    "drain_window_s",
    "retained_prefill_removal_rate_s_per_s",
    "drain_completion_s",
    "network_capacity_pressure",
    "prefill_capacity_pressure",
    "replay_retained_prefill_fraction",
    "state_transfer_retained_prefill_fraction",
    "repair_move_count",
    "repair_move_summary",
)
BREAKDOWN_COLUMNS = (
    "policy",
    "retained_prefill_fraction",
    "deadline_scale",
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
REPAIR_SUMMARY_COLUMNS = (
    "retained_prefill_fraction",
    "deadline_scale",
    "moved_requests",
    "repair_steps",
    "net_changed_requests",
    "fraction_moved_requests_changed",
    "original_objective",
    "repaired_objective",
    "objective_delta",
    "objective_relative_delta",
    "original_miss_rate",
    "repaired_miss_rate",
    "miss_rate_delta",
    "original_p95_delay",
    "repaired_p95_delay",
    "p95_delay_delta",
    "original_mean_delay",
    "repaired_mean_delay",
    "mean_delay_delta",
    "original_p95_normalized_delay",
    "repaired_p95_normalized_delay",
    "p95_normalized_delay_delta",
    "original_replay_retained_prefill_fraction",
    "repaired_replay_retained_prefill_fraction",
    "replay_retained_prefill_fraction_delta",
    "original_state_transfer_retained_prefill_fraction",
    "repaired_state_transfer_retained_prefill_fraction",
    "state_transfer_retained_prefill_fraction_delta",
    "original_k0_retained_prefill_fraction",
    "repaired_k0_retained_prefill_fraction",
    "k0_retained_prefill_fraction_delta",
    "original_k1_retained_prefill_fraction",
    "repaired_k1_retained_prefill_fraction",
    "k1_retained_prefill_fraction_delta",
    "original_k2_retained_prefill_fraction",
    "repaired_k2_retained_prefill_fraction",
    "k2_retained_prefill_fraction_delta",
)
MOVE_BREAKDOWN_COLUMNS = (
    "retained_prefill_fraction",
    "deadline_scale",
    "move_type",
    "class",
    "source_destination",
    "source_action",
    "target_destination",
    "target_action",
    "move_count",
)
BUDGET_COLUMNS = (
    "retained_prefill_fraction",
    "deadline_scale",
    "budget_label",
    "budget_fraction",
    "budget_move_limit",
    "repair_steps",
    "net_changed_requests",
    "fraction_moved_requests_changed",
    "miss_rate",
    "p95_delay",
    "mean_delay",
    "p95_normalized_delay",
    "objective_delta",
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


def run_queue_failure_diagnostics(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    queue_rows = []
    breakdown_rows = []
    summary_rows = []
    repair_summary_rows = []
    move_breakdown_rows = []
    budget_rows = []
    model = get_model("GLM-5")

    for deadline_scale in TIGHT_DEADLINE_SCALES:
        for retained_prefill_fraction in RETAINED_PREFILL_FRACTIONS:
            problem = make_problem(
                model,
                "transition-coupled",
                retained_prefill_fraction=retained_prefill_fraction,
                deadline_scale=deadline_scale,
                **workload_config.problem_kwargs(),
            )
            try:
                cvx = solve_cvxpy(problem)
                rounded = round_allocation(problem, cvx.y)
                metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y)
            except (RuntimeError, ValueError):
                queue_rows.append(
                    _empty_queue_row(
                        "CVXPY-rounded", retained_prefill_fraction, deadline_scale, problem.retained_prefill_target_s
                    )
                )
                queue_rows.append(
                    _empty_queue_row(
                        "repaired-CVXPY-rounded",
                        retained_prefill_fraction,
                        deadline_scale,
                        problem.retained_prefill_target_s,
                    )
                )
            else:
                queue_rows.append(
                    _queue_row("CVXPY-rounded", retained_prefill_fraction, deadline_scale, "OK", metrics)
                )
                breakdown_rows.extend(
                    _failure_breakdown_rows(
                        "CVXPY-rounded", problem, retained_prefill_fraction, deadline_scale, "OK", trace
                    )
                )

                try:
                    repair = repair_rounded_allocation(problem, rounded.y)
                except (RuntimeError, ValueError):
                    repaired = _empty_queue_row(
                        "repaired-CVXPY-rounded",
                        retained_prefill_fraction,
                        deadline_scale,
                        problem.retained_prefill_target_s,
                    )
                    repaired["status"] = "REPAIR_FAILED"
                    queue_rows.append(repaired)
                else:
                    queue_rows.append(
                        _queue_row(
                            "repaired-CVXPY-rounded",
                            retained_prefill_fraction,
                            deadline_scale,
                            "OK",
                            repair.metrics,
                            repair.moves,
                        )
                    )
                    breakdown_rows.extend(
                        _failure_breakdown_rows(
                            "repaired-CVXPY-rounded",
                            problem,
                            retained_prefill_fraction,
                            deadline_scale,
                            "OK",
                            repair.trace,
                        )
                    )
                    summary_rows.append(_summary_row(retained_prefill_fraction, deadline_scale, metrics, repair))
                    repair_summary_rows.append(
                        _repair_summary_row(
                            problem, rounded.y, metrics, repair, retained_prefill_fraction, deadline_scale
                        )
                    )
                    move_breakdown_rows.extend(
                        _repair_move_breakdown_rows(retained_prefill_fraction, deadline_scale, repair.moves)
                    )
                    budget_rows.extend(
                        _repair_budget_rows(
                            problem, rounded.y, metrics, repair, retained_prefill_fraction, deadline_scale
                        )
                    )

            for policy, solver in POLICIES:
                row, trace = _solver_queue(policy, solver, problem, retained_prefill_fraction, deadline_scale)
                queue_rows.append(row)
                breakdown_rows.extend(
                    _failure_breakdown_rows(
                        policy, problem, retained_prefill_fraction, deadline_scale, row["status"], trace
                    )
                )

    _write_rows(out / "transition_coupled_repaired_queue_table.csv", queue_rows, QUEUE_COLUMNS)
    _write_rows(
        out / "transition_coupled_queue_failure_breakdown.csv",
        breakdown_rows,
        BREAKDOWN_COLUMNS,
    )
    _write_rows(out / "repair_summary.csv", repair_summary_rows, REPAIR_SUMMARY_COLUMNS)
    _write_rows(out / "repair_move_breakdown.csv", move_breakdown_rows, MOVE_BREAKDOWN_COLUMNS)
    _write_rows(out / "repair_budget_frontier.csv", budget_rows, BUDGET_COLUMNS)
    _print_repair_summary(summary_rows)
    _print_half_deadline_latex(summary_rows)
    return queue_rows, breakdown_rows, summary_rows


def repair_rounded_allocation(problem, y, max_steps=1000, max_changes=None, drain_window_s=DRAIN_WINDOW_S):
    coeffs = compute_coefficients(problem)
    y = np.asarray(y, dtype=float)
    metrics, trace = evaluate_rounded_queue_trace(problem, y, drain_window_s=drain_window_s)
    y = np.rint(y).astype(int)
    moves = []

    for _ in range(max_steps):
        if max_changes is not None and len(moves) >= max_changes:
            return RepairResult(y, metrics, trace, tuple(moves))
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
                        problem, candidate, drain_window_s=drain_window_s
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


def _solver_queue(policy, solver, problem, retained_prefill_fraction, deadline_scale):
    base = _empty_queue_row(policy, retained_prefill_fraction, deadline_scale, problem.retained_prefill_target_s)
    try:
        result = solver(problem)
    except RuntimeError:
        return base, ()
    if not getattr(result, "feasible", True):
        return base, ()
    y = result.allocation if hasattr(result, "allocation") else result.y
    if retained_prefill_moved_s(problem, y) < problem.retained_prefill_target_s - 1e-5:
        return base, ()
    try:
        rounded = round_allocation(problem, y)
        metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y)
    except ValueError:
        base["status"] = "ROUNDING_FAILED"
        return base, ()
    return _queue_row(policy, retained_prefill_fraction, deadline_scale, "OK", metrics), trace


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


def _repair_summary_row(problem, original_y, original_metrics, repair, retained_prefill_fraction, deadline_scale):
    coeffs = compute_coefficients(problem)
    original_obj = objective(problem, coeffs, original_y)
    repaired_obj = objective(problem, coeffs, repair.y)
    moved_requests = _moved_requests(original_y)
    net_changed = _net_changed_requests(original_y, repair.y)
    original_action = retained_prefill_action_mix(problem, original_y)
    repaired_action = retained_prefill_action_mix(problem, repair.y)
    original_dest = retained_prefill_destination_mix(problem, original_y)
    repaired_dest = retained_prefill_destination_mix(problem, repair.y)
    row = {
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
        "moved_requests": moved_requests,
        "repair_steps": len(repair.moves),
        "net_changed_requests": net_changed,
        "fraction_moved_requests_changed": 0.0 if moved_requests == 0 else net_changed / moved_requests,
        "original_objective": original_obj,
        "repaired_objective": repaired_obj,
        "objective_delta": repaired_obj - original_obj,
        "objective_relative_delta": (repaired_obj - original_obj) / max(1.0, abs(original_obj)),
        "original_miss_rate": original_metrics["deadline_miss_rate"],
        "repaired_miss_rate": repair.metrics["deadline_miss_rate"],
        "miss_rate_delta": repair.metrics["deadline_miss_rate"] - original_metrics["deadline_miss_rate"],
        "original_p95_delay": original_metrics["p95_reconstruction_delay"],
        "repaired_p95_delay": repair.metrics["p95_reconstruction_delay"],
        "p95_delay_delta": repair.metrics["p95_reconstruction_delay"] - original_metrics["p95_reconstruction_delay"],
        "original_mean_delay": original_metrics["mean_reconstruction_delay"],
        "repaired_mean_delay": repair.metrics["mean_reconstruction_delay"],
        "mean_delay_delta": repair.metrics["mean_reconstruction_delay"] - original_metrics["mean_reconstruction_delay"],
        "original_p95_normalized_delay": original_metrics["p95_normalized_reconstruction_delay"],
        "repaired_p95_normalized_delay": repair.metrics["p95_normalized_reconstruction_delay"],
        "p95_normalized_delay_delta": repair.metrics["p95_normalized_reconstruction_delay"]
        - original_metrics["p95_normalized_reconstruction_delay"],
        "original_replay_retained_prefill_fraction": original_action["replay_retained_prefill_fraction"],
        "repaired_replay_retained_prefill_fraction": repaired_action["replay_retained_prefill_fraction"],
        "replay_retained_prefill_fraction_delta": repaired_action["replay_retained_prefill_fraction"]
        - original_action["replay_retained_prefill_fraction"],
        "original_state_transfer_retained_prefill_fraction": original_action[
            "state_transfer_retained_prefill_fraction"
        ],
        "repaired_state_transfer_retained_prefill_fraction": repaired_action[
            "state_transfer_retained_prefill_fraction"
        ],
        "state_transfer_retained_prefill_fraction_delta": repaired_action[
            "state_transfer_retained_prefill_fraction"
        ]
        - original_action["state_transfer_retained_prefill_fraction"],
    }
    for k in range(problem.K):
        row[f"original_k{k}_retained_prefill_fraction"] = original_dest[k]
        row[f"repaired_k{k}_retained_prefill_fraction"] = repaired_dest[k]
        row[f"k{k}_retained_prefill_fraction_delta"] = repaired_dest[k] - original_dest[k]
    return row


def _repair_move_breakdown_rows(retained_prefill_fraction, deadline_scale, moves):
    rows = []
    for key, count in Counter(moves).items():
        rows.append(
            {
                "retained_prefill_fraction": retained_prefill_fraction,
                "deadline_scale": deadline_scale,
                "move_type": _move_type(key),
                "class": key.g,
                "source_destination": key.from_k,
                "source_action": key.from_action,
                "target_destination": key.to_k,
                "target_action": key.to_action,
                "move_count": count,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["deadline_scale"],
            row["retained_prefill_fraction"],
            row["class"],
            row["source_destination"],
            row["source_action"],
            row["target_destination"],
            row["target_action"],
        ),
    )


def _repair_budget_rows(
    problem, original_y, original_metrics, full_repair, retained_prefill_fraction, deadline_scale
):
    moved = _moved_requests(original_y)
    rows = [
        _budget_row(
            problem,
            original_y,
            original_y,
            original_metrics,
            (),
            retained_prefill_fraction,
            deadline_scale,
            "0%",
            0.0,
            0,
        )
    ]
    for budget_fraction in REPAIR_BUDGET_FRACTIONS:
        limit = int(math.floor(budget_fraction * moved))
        repair = repair_rounded_allocation(problem, original_y, max_changes=limit)
        rows.append(
            _budget_row(
                problem,
                original_y,
                repair.y,
                repair.metrics,
                repair.moves,
                retained_prefill_fraction,
                deadline_scale,
                f"{budget_fraction:.0%}",
                budget_fraction,
                limit,
            )
        )
    rows.append(
        _budget_row(
            problem,
            original_y,
            full_repair.y,
            full_repair.metrics,
            full_repair.moves,
            retained_prefill_fraction,
            deadline_scale,
            "unbounded",
            math.nan,
            math.nan,
        )
    )
    return rows


def _budget_row(
    problem,
    original_y,
    y,
    metrics,
    moves,
    retained_prefill_fraction,
    deadline_scale,
    budget_label,
    budget_fraction,
    budget_limit,
):
    coeffs = compute_coefficients(problem)
    original_obj = objective(problem, coeffs, original_y)
    moved = _moved_requests(original_y)
    net_changed = _net_changed_requests(original_y, y)
    return {
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
        "budget_label": budget_label,
        "budget_fraction": budget_fraction,
        "budget_move_limit": budget_limit,
        "repair_steps": len(moves),
        "net_changed_requests": net_changed,
        "fraction_moved_requests_changed": 0.0 if moved == 0 else net_changed / moved,
        "miss_rate": metrics["deadline_miss_rate"],
        "p95_delay": metrics["p95_reconstruction_delay"],
        "mean_delay": metrics["mean_reconstruction_delay"],
        "p95_normalized_delay": metrics["p95_normalized_reconstruction_delay"],
        "objective_delta": objective(problem, coeffs, y) - original_obj,
    }


def _moved_requests(y):
    y = np.asarray(y)
    return int(np.sum(y[:, : y.shape[1] - 1]))


def _net_changed_requests(original_y, repaired_y):
    original = np.asarray(original_y)[:, : np.asarray(original_y).shape[1] - 1]
    repaired = np.asarray(repaired_y)[:, : np.asarray(repaired_y).shape[1] - 1]
    return float(np.sum(np.abs(repaired - original)) / 2.0)


def _move_type(move):
    destination = move.from_k != move.to_k
    action = move.from_action != move.to_action
    if destination and action:
        return "destination and action switch"
    if destination:
        return "destination switch only"
    if action:
        return "action switch only"
    raise ValueError("repair move did not change destination or action")


def _queue_row(policy, retained_prefill_fraction, deadline_scale, status, metrics, moves=()):
    return {
        "policy": policy,
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
        "status": status,
        "safe": _safe(metrics),
        "rounded_retained_prefill_moved_s": metrics["rounded_retained_prefill_moved_s"],
        "rounded_retained_prefill_target_s": metrics["rounded_retained_prefill_target_s"],
        "rounded_retained_prefill_ratio": metrics["rounded_retained_prefill_ratio"],
        "mean_delay": metrics["mean_reconstruction_delay"],
        "p50_delay": metrics["p50_reconstruction_delay"],
        "p95_delay": metrics["p95_reconstruction_delay"],
        "p99_delay": metrics["p99_reconstruction_delay"],
        "p95_normalized_delay": metrics["p95_normalized_reconstruction_delay"],
        "miss_rate": metrics["deadline_miss_rate"],
        "drain_window_s": metrics["drain_window_s"],
        "retained_prefill_removal_rate_s_per_s": metrics["retained_prefill_removal_rate_s_per_s"],
        "drain_completion_s": metrics["drain_completion_s"],
        "network_capacity_pressure": metrics["network_capacity_pressure"],
        "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
        "replay_retained_prefill_fraction": metrics["replay_retained_prefill_fraction"],
        "state_transfer_retained_prefill_fraction": metrics["state_transfer_retained_prefill_fraction"],
        "repair_move_count": len(moves),
        "repair_move_summary": _move_summary(moves),
    }


def _empty_queue_row(policy, retained_prefill_fraction, deadline_scale, retained_prefill_target_s):
    return {
        "policy": policy,
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
        "status": "INFEASIBLE",
        "safe": False,
        "rounded_retained_prefill_moved_s": math.nan,
        "rounded_retained_prefill_target_s": retained_prefill_target_s,
        "rounded_retained_prefill_ratio": math.nan,
        "mean_delay": math.nan,
        "p50_delay": math.nan,
        "p95_delay": math.nan,
        "p99_delay": math.nan,
        "p95_normalized_delay": math.nan,
        "miss_rate": math.nan,
        "drain_window_s": DRAIN_WINDOW_S,
        "retained_prefill_removal_rate_s_per_s": math.nan,
        "drain_completion_s": math.nan,
        "network_capacity_pressure": math.nan,
        "prefill_capacity_pressure": math.nan,
        "replay_retained_prefill_fraction": math.nan,
        "state_transfer_retained_prefill_fraction": math.nan,
        "repair_move_count": 0,
        "repair_move_summary": "",
    }


def _safe(metrics):
    return (
        metrics["rounded_retained_prefill_moved_s"] >= metrics["rounded_retained_prefill_target_s"] - 1e-9
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


def _failure_breakdown_rows(policy, problem, retained_prefill_fraction, deadline_scale, status, trace):
    if not trace:
        return [
            {
                "policy": policy,
                "retained_prefill_fraction": retained_prefill_fraction,
                "deadline_scale": deadline_scale,
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
                    retained_prefill_fraction,
                    deadline_scale,
                    status,
                    group_type,
                    group,
                    tuple(record for record in trace if predicate(record)),
                )
            )
    return rows


def _breakdown_row(policy, retained_prefill_fraction, deadline_scale, status, group_type, group, records):
    missed = tuple(record for record in records if record.deadline_missed)
    return {
        "policy": policy,
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
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


def _summary_row(retained_prefill_fraction, deadline_scale, original, repair):
    return {
        "retained_prefill_fraction": retained_prefill_fraction,
        "deadline_scale": deadline_scale,
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
    if not rows:
        print("no feasible CVXPY-rounded rows to repair")
        return
    cols = (
        "deadline_scale",
        "retained_prefill_fraction",
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


def _print_half_deadline_latex(rows):
    print("\n0.5x deadline miss-rate repair table (LaTeX)")
    print("\\begin{tabular}{rrrr}")
    print("retained-prefill fraction & original miss & repaired miss & repair moves \\\\")
    print("\\hline")
    for row in rows:
        if row["deadline_scale"] == 0.50:
            print(
                f"{row['retained_prefill_fraction']:.2f} & "
                f"{row['original_miss_rate']:.4f} & "
                f"{row['repaired_miss_rate']:.4f} & "
                f"{row['repair_moves']} \\\\"
            )
    print("\\end{tabular}")


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
    run_queue_failure_diagnostics(parse_workload_config("Run queue failure diagnostics."))
