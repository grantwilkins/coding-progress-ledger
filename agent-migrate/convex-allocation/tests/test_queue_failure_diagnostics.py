"""
Claim:
Queue failure diagnostics preserve rounded allocations while tracing misses by
request group, and the local repair pass only accepts one-request moves that
improve miss rate, then p95 delay, then mean delay.

Plausible wrong implementations:
- Re-run fractional rounding during repair and change class shed counts.
- Accept a move that improves mean delay while leaving a higher miss rate.
- Aggregate failures at the wrong level, such as shed-weighted instead of request-counted.
- Attribute prefill wait to state-transfer requests.
- Try to repair a CVXPY row after the generated workload makes that shed point infeasible.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import experiments.run_queue_failure_diagnostics as queue_diag
from catalog import ModelParams
from evaluation import WorkloadConfig
from experiments.run_queue_failure_diagnostics import (
    _failure_breakdown_rows,
    _move_type,
    _queue_key,
    _repair_budget_rows,
    _repair_move_breakdown_rows,
    _repair_summary_row,
    RepairMove,
    RepairResult,
    repair_rounded_allocation,
)
from problem import ProblemData
from queueing import evaluate_rounded_queue_trace


def repair_problem():
    model = ModelParams("repair-test", 1.0, 3.0, 1.0, 0.0)
    lambda_Bps = np.array([10.0, 100.0])
    rho_prefill = np.array([100.0, 100.0])
    return ProblemData(
        model=model,
        regime="repair-test",
        T=np.array([10.0]),
        d=np.array([2.0]),
        slack=np.array([4.0]),
        lambda_Bps=lambda_Bps,
        rho_prefill=rho_prefill,
        C_net=lambda_Bps * 10.0,
        C_prefill=rho_prefill * 10.0,
        ell_net=np.zeros(2),
        ell_prefill=np.zeros(2),
        h_ctx=np.zeros((1, 2)),
        h_kv=np.zeros((1, 2)),
        B_shed=20.0,
    )


def test_local_repair_improves_queue_key_without_changing_class_totals_or_shed():
    problem = repair_problem()
    y = np.array([[0, 2, 0, 0, 0]])
    original, _ = evaluate_rounded_queue_trace(problem, y)

    repair = repair_rounded_allocation(problem, y)

    assert _queue_key(repair.metrics) < _queue_key(original)
    assert repair.metrics["deadline_miss_rate"] == 0.0
    assert repair.metrics["rounded_shed_achieved"] == original["rounded_shed_achieved"]
    assert repair.y[0, 1] < y[0, 1]
    assert repair.y.sum(axis=1).tolist() == y.sum(axis=1).tolist()
    assert repair.moves


def test_local_repair_respects_zero_move_budget():
    problem = repair_problem()
    y = np.array([[0, 2, 0, 0, 0]])
    original, _ = evaluate_rounded_queue_trace(problem, y)

    repair = repair_rounded_allocation(problem, y, max_changes=0)

    np.testing.assert_array_equal(repair.y, y)
    assert _queue_key(repair.metrics) == _queue_key(original)
    assert not repair.moves


def test_failure_breakdown_counts_misses_by_class_destination_and_action():
    problem = repair_problem()
    metrics, trace = evaluate_rounded_queue_trace(problem, np.array([[0, 2, 0, 0, 0]]))

    rows = _failure_breakdown_rows("toy", problem, 0.2, 0.25, "OK", trace)
    by_group = {(row["group_type"], row["group"]): row for row in rows}

    assert metrics["deadline_miss_rate"] == 0.5
    assert by_group[("class", "class0")]["missed_requests"] == 1
    assert by_group[("destination", "k0")]["missed_requests"] == 1
    assert by_group[("destination", "k1")]["moved_requests"] == 0
    assert by_group[("action", "state")]["missed_requests"] == 1
    assert by_group[("action", "replay")]["avg_missed_prefill_wait"] == 0.0


def test_repair_summary_reports_net_changed_requests_and_destination_shift():
    problem = repair_problem()
    original_y = np.array([[0, 2, 0, 0, 0]])
    repaired_y = np.array([[0, 1, 0, 1, 0]])
    original_metrics, _ = evaluate_rounded_queue_trace(problem, original_y)
    repaired_metrics, repaired_trace = evaluate_rounded_queue_trace(problem, repaired_y)
    repair = RepairResult(
        repaired_y,
        repaired_metrics,
        repaired_trace,
        (RepairMove(0, 0, "state", 1, "state"),),
    )

    row = _repair_summary_row(problem, original_y, original_metrics, repair, 0.2, 0.5)

    assert row["repair_steps"] == 1
    assert row["net_changed_requests"] == 1
    assert row["fraction_moved_requests_changed"] == 0.5
    assert row["original_k0_shed_frac"] == 1.0
    assert row["repaired_k0_shed_frac"] == 0.5
    assert row["k1_shed_frac_delta"] == 0.5
    assert row["objective_delta"] != 0.0


def test_repair_move_breakdown_classifies_switch_types_and_counts_repeats():
    moves = (
        RepairMove(0, 0, "state", 1, "state"),
        RepairMove(0, 0, "state", 1, "state"),
        RepairMove(1, 0, "state", 0, "replay"),
        RepairMove(2, 0, "state", 1, "replay"),
    )

    rows = _repair_move_breakdown_rows(0.2, 0.5, moves)
    by_type = {row["move_type"]: row for row in rows}

    assert _move_type(moves[0]) == "destination switch only"
    assert _move_type(moves[2]) == "action switch only"
    assert _move_type(moves[3]) == "destination and action switch"
    assert by_type["destination switch only"]["move_count"] == 2


def test_repair_budget_rows_include_capped_and_unbounded_results():
    problem = repair_problem()
    y = np.array([[0, 2, 0, 0, 0]])
    original_metrics, _ = evaluate_rounded_queue_trace(problem, y)
    full = repair_rounded_allocation(problem, y)

    rows = _repair_budget_rows(problem, y, original_metrics, full, 0.2, 0.5)
    by_label = {row["budget_label"]: row for row in rows}

    assert by_label["0%"]["repair_steps"] == 0
    assert by_label["5%"]["budget_move_limit"] == 0
    assert by_label["20%"]["budget_move_limit"] == 0
    assert by_label["unbounded"]["repair_steps"] == len(full.moves)


def test_queue_diagnostics_reports_infeasible_cvx_rows_without_repair(monkeypatch, tmp_path):
    captured = {}

    def fail_solve(problem):
        raise RuntimeError("infeasible")

    monkeypatch.setattr(queue_diag, "ROOT", tmp_path)
    monkeypatch.setattr(queue_diag, "TIGHT_SLACK_MULTIPLIERS", (0.25,))
    monkeypatch.setattr(queue_diag, "SHED_FRACTIONS", (0.2,))
    monkeypatch.setattr(queue_diag, "POLICIES", ())
    monkeypatch.setattr(
        queue_diag, "make_problem", lambda *args, **kwargs: SimpleNamespace(B_shed=1.0)
    )
    monkeypatch.setattr(queue_diag, "solve_cvxpy", fail_solve)
    monkeypatch.setattr(
        queue_diag,
        "_write_rows",
        lambda path, rows, columns: captured.setdefault(path, rows),
    )
    monkeypatch.setattr(queue_diag, "_print_repair_summary", lambda rows: None)
    monkeypatch.setattr(queue_diag, "_print_half_slack_latex", lambda rows: None)

    queue_diag.run_queue_failure_diagnostics(WorkloadConfig(source="generated", seed=7))

    queue_rows = next(
        rows for path, rows in captured.items() if path.name.endswith("queue_table.csv")
    )
    assert [row["policy"] for row in queue_rows] == [
        "CVXPY-rounded",
        "repaired-CVXPY-rounded",
    ]
    assert {row["status"] for row in queue_rows} == {"INFEASIBLE"}
    assert all("generated_seed7" in str(path) for path in captured)
