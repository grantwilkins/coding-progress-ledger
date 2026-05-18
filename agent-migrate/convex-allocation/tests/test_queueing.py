"""
Claim:
The static queue evaluator turns fractional shed allocations into integer
requests, preserves the shed target by minimum overshed, and computes
nonpreemptive network-then-prefill EDF reconstruction delays.

Plausible wrong implementations:
- Round to the nearest fractional counts and allow integer under-shed.
- Tie-break shed-equivalent class counts without respecting fractional moved counts.
- Treat replay requests as complete after network transfer instead of prefill.
- Schedule by arrival or input order instead of earliest class deadline.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from catalog import ModelParams
from queueing import RequestRecord, evaluate_static_queue, evaluate_static_queue_trace, round_allocation
from problem import ProblemData


def queue_problem(T, d, slack, B_shed) -> ProblemData:
    model = ModelParams("queue-test", 1.0, 1.0, 1.0, 0.0)
    K = 1
    return ProblemData(
        model=model,
        regime="queue-test",
        T=np.asarray(T, dtype=float),
        d=np.asarray(d, dtype=float),
        slack=np.asarray(slack, dtype=float),
        lambda_Bps=np.full(K, 10.0),
        rho_prefill=np.full(K, 5.0),
        C_net=np.full(K, 100.0),
        C_prefill=np.full(K, 50.0),
        ell_net=np.zeros(K),
        ell_prefill=np.zeros(K),
        h_ctx=np.zeros((len(T), K)),
        h_kv=np.zeros((len(T), K)),
        B_shed=B_shed,
    )


def test_rounding_meets_target_with_minimum_overshed_before_deviation():
    problem = queue_problem([6, 10], [2, 1], [1, 1], 10.0)
    y = np.array([[1.7, 0.0, 0.3], [0.0, 0.1, 0.9]])

    rounded = round_allocation(problem, y)

    assert rounded.rounded_shed == 10.0
    assert_allclose(np.sum(rounded.y, axis=1), problem.d)
    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [0, 1])


def test_rounding_tie_breaks_by_fractional_moved_counts_and_apportions_cells():
    problem = queue_problem([5, 5], [2, 2], [1, 1], 10.0)
    y = np.array([[1.2, 0.6, 0.2], [0.2, 0.0, 1.8]])

    rounded = round_allocation(problem, y)

    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [2, 0])
    assert_allclose(rounded.y[0], [1, 1, 0])
    assert rounded.rounded_shed == 10.0


def test_rounding_does_not_invent_movement_for_zero_moved_classes():
    problem = queue_problem([6, 10], [2, 1], [1, 1], 10.0)
    y = np.array([[1.7, 0.0, 0.3], [0.0, 0.0, 1.0]])

    rounded = round_allocation(problem, y)

    assert rounded.rounded_shed == 12.0
    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [2, 0])


def test_static_queue_uses_network_then_prefill_with_edf():
    problem = queue_problem([1, 1], [1, 1], [3.5, 10.0], 0.0)
    records = (
        RequestRecord(1, 0, "state", 1.0, 10.0, 20.0, 0.0),
        RequestRecord(0, 0, "replay", 1.0, 3.5, 10.0, 15.0),
    )

    metrics = evaluate_static_queue(problem, records)
    _, trace = evaluate_static_queue_trace(problem, records)

    assert_allclose(metrics["mean_reconstruction_delay"], 3.5)
    assert_allclose(metrics["p50_reconstruction_delay"], 3.5)
    assert_allclose(metrics["p95_reconstruction_delay"], 3.95)
    assert_allclose(metrics["p99_reconstruction_delay"], 3.99)
    assert_allclose(metrics["p95_normalized_reconstruction_delay"], 1.1007142857142855)
    assert_allclose(metrics["deadline_miss_rate"], 0.5)
    assert_allclose(metrics["max_network_busy_window"], 0.3)
    assert_allclose(metrics["max_prefill_busy_window"], 0.3)
    assert_allclose(metrics["replay_shed_frac"], 0.5)
    assert_allclose(metrics["state_shed_frac"], 0.5)

    state, replay = trace
    assert replay.g == 0
    assert replay.action == "replay"
    assert_allclose(
        [
            replay.network_queue_wait,
            replay.network_service_time,
            replay.prefill_queue_wait,
            replay.prefill_service_time,
            replay.reconstruction_delay,
        ],
        [0.0, 1.0, 0.0, 3.0, 4.0],
    )
    assert replay.deadline_missed
    assert state.g == 1
    assert state.action == "state"
    assert_allclose(
        [
            state.network_queue_wait,
            state.network_service_time,
            state.prefill_queue_wait,
            state.prefill_service_time,
            state.reconstruction_delay,
        ],
        [1.0, 2.0, 0.0, 0.0, 3.0],
    )
    assert not state.deadline_missed
