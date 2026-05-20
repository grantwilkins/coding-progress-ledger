"""
Claim:
The queue evaluator turns fractional retained-prefill allocations into integer
requests, preserves the retained-prefill target by minimum overshoot, releases
moved requests through a deterministic EDF drain, and computes nonpreemptive
network-then-prefill reconstruction delays relative to request release time.

Plausible wrong implementations:
- Round to the nearest fractional counts and allow integer target shortfall.
- Tie-break target-equivalent class counts without respecting fractional moved counts.
- Let production-sized rounding hang or use classes with zero moved support.
- Treat replay requests as complete after network transfer instead of prefill.
- Schedule by arrival or input order instead of earliest class deadline.
- Count drain wait as reconstruction delay after choosing release-relative deadlines.
- Drop the burst-at-zero baseline when drain_window_s is explicitly zero.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from catalog import ModelParams
from queueing import RequestRecord, evaluate_static_queue, evaluate_static_queue_trace, queue_metrics, round_allocation
from problem import ProblemData


def queue_problem(T, d, deadline_s, retained_prefill_target_s) -> ProblemData:
    model = ModelParams("queue-test", 1.0, 1.0, 1.0, 0.0)
    K = 1
    return ProblemData(
        model=model,
        regime="queue-test",
        T=np.asarray(T, dtype=float),
        d=np.asarray(d, dtype=float),
        deadline_s=np.asarray(deadline_s, dtype=float),
        lambda_Bps=np.full(K, 10.0),
        rho_prefill=np.full(K, 5.0),
        C_net=np.full(K, 100.0),
        C_prefill=np.full(K, 50.0),
        ell_net=np.zeros(K),
        ell_prefill=np.zeros(K),
        h_ctx=np.zeros((len(T), K)),
        h_kv=np.zeros((len(T), K)),
        retained_prefill_target_s=retained_prefill_target_s,
    )


def test_rounding_meets_target_with_minimum_overshoot_before_deviation():
    problem = queue_problem([6, 10], [2, 1], [1, 1], 10.0)
    y = np.array([[1.7, 0.0, 0.3], [0.0, 0.1, 0.9]])

    rounded = round_allocation(problem, y)

    assert rounded.retained_prefill_moved_s == 10.0
    assert_allclose(np.sum(rounded.y, axis=1), problem.d)
    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [0, 1])


def test_rounding_tie_breaks_by_fractional_moved_counts_and_apportions_cells():
    problem = queue_problem([5, 5], [2, 2], [1, 1], 10.0)
    y = np.array([[1.2, 0.6, 0.2], [0.2, 0.0, 1.8]])

    rounded = round_allocation(problem, y)

    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [2, 0])
    assert_allclose(rounded.y[0], [1, 1, 0])
    assert rounded.retained_prefill_moved_s == 10.0


def test_rounding_does_not_invent_movement_for_zero_moved_classes():
    problem = queue_problem([6, 10], [2, 1], [1, 1], 10.0)
    y = np.array([[1.7, 0.0, 0.3], [0.0, 0.0, 1.0]])

    rounded = round_allocation(problem, y)

    assert rounded.retained_prefill_moved_s == 12.0
    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [2, 0])


def test_large_rounding_meets_target_without_zero_support_classes():
    problem = queue_problem([5, 8, 13], [120, 120, 120], [1, 1, 1], 1000.0)
    y = np.array([[60.4, 0.0, 59.6], [50.2, 0.0, 69.8], [0.0, 0.0, 120.0]])

    rounded = round_allocation(problem, y)

    assert 1000.0 <= rounded.retained_prefill_moved_s < 1008.0
    assert_allclose(np.sum(rounded.y, axis=1), problem.d)
    assert_allclose(np.sum(rounded.y[:, :2], axis=1)[2], 0)


def test_static_queue_uses_network_then_prefill_with_edf():
    problem = queue_problem([1, 1], [1, 1], [3.5, 10.0], 0.0)
    records = (
        RequestRecord(1, 0, "state", 1.0, 10.0, 20.0, 0.0),
        RequestRecord(0, 0, "replay", 1.0, 3.5, 10.0, 15.0),
    )

    metrics = evaluate_static_queue(problem, records, drain_window_s=0.0)
    _, trace = evaluate_static_queue_trace(problem, records, drain_window_s=0.0)

    assert_allclose(metrics["mean_reconstruction_delay"], 3.5)
    assert_allclose(metrics["p50_reconstruction_delay"], 3.5)
    assert_allclose(metrics["p95_reconstruction_delay"], 3.95)
    assert_allclose(metrics["p99_reconstruction_delay"], 3.99)
    assert_allclose(metrics["p95_normalized_reconstruction_delay"], 1.1007142857142855)
    assert_allclose(metrics["deadline_miss_rate"], 0.5)
    assert_allclose(metrics["network_capacity_pressure"], 0.3)
    assert_allclose(metrics["prefill_capacity_pressure"], 0.3)
    assert_allclose(metrics["replay_retained_prefill_fraction"], 0.5)
    assert_allclose(metrics["state_transfer_retained_prefill_fraction"], 0.5)

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


def test_queue_drains_requests_by_edf_release_order():
    problem = queue_problem([1, 1], [1, 1], [3.5, 10.0], 0.0)
    records = (
        RequestRecord(1, 0, "state", 1.0, 10.0, 20.0, 0.0),
        RequestRecord(0, 0, "replay", 1.0, 3.5, 10.0, 15.0),
    )

    metrics, trace = evaluate_static_queue_trace(problem, records, drain_window_s=60.0)

    state, replay = trace
    assert_allclose([replay.release_time_s, state.release_time_s], [0.0, 30.0])
    assert_allclose(metrics["mean_reconstruction_delay"], 3.0)
    assert_allclose(metrics["drain_completion_s"], 32.0)
    assert_allclose(
        [
            replay.network_queue_wait,
            replay.prefill_queue_wait,
            replay.reconstruction_delay,
            state.network_queue_wait,
            state.reconstruction_delay,
        ],
        [0.0, 0.0, 4.0, 0.0, 2.0],
    )
    assert replay.deadline_missed
    assert not state.deadline_missed


def test_default_drain_window_is_thirty_minutes():
    problem = queue_problem([1, 1], [1, 1], [10.0, 20.0], 0.0)
    records = (
        RequestRecord(0, 0, "state", 1.0, 10.0, 0.0, 0.0),
        RequestRecord(1, 0, "state", 1.0, 20.0, 0.0, 0.0),
    )

    _, trace = evaluate_static_queue_trace(problem, records)

    assert_allclose([record.release_time_s for record in trace], [0.0, 900.0])


def test_zero_window_zero_load_has_zero_removal_rate():
    problem = queue_problem([1], [1], [3.5], 0.0)
    y = np.array([[0, 0, 1]])

    metrics = queue_metrics(problem, y, drain_window_s=0.0)

    assert metrics["retained_prefill_removal_rate_s_per_s"] == 0.0


def test_queue_metrics_report_resident_state_tb_and_nvl72_fraction():
    problem = queue_problem([10, 30], [1, 1], [10.0, 10.0], 10.0)
    y = np.array([[1, 0, 0], [0, 0, 1]])

    metrics = queue_metrics(problem, y)

    assert_allclose(metrics["resident_state_tb"], 40.0 / 1e12)
    assert_allclose(metrics["average_equivalent_state_target_tb"], 10.0 / 1e12)
    assert_allclose(metrics["actual_evacuated_state_tb"], 10.0 / 1e12)
    assert_allclose(metrics["actual_evacuated_nvl72_hbm_fraction"], 10.0 / (13.4e12))
    assert not {
        "source_" + "working_set_fraction",
        "source_" + "working_set_moved_fraction",
        "retained_" + "state_target_tb",
        "evacuated_" + "state_tb",
    } & metrics.keys()
