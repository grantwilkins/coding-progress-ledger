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
"""

from __future__ import annotations

import numpy as np

from catalog import ModelParams
from experiments.run_queue_failure_diagnostics import (
    _failure_breakdown_rows,
    _queue_key,
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
