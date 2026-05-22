"""
Claim:
The queue evaluator turns fractional retained-prefill allocations into integer
requests, preserves the retained-prefill target by minimum overshoot, releases
moved requests through deterministic release policies, and computes
nonpreemptive network-then-prefill reconstruction delays relative to request
release time.

Plausible wrong implementations:
- Round to the nearest fractional counts and allow integer target shortfall.
- Tie-break target-equivalent class counts without respecting fractional moved counts.
- Let production-sized rounding hang or use classes with zero moved support.
- Treat replay requests as complete after network transfer instead of prefill.
- Schedule by arrival or input order instead of earliest class deadline.
- Treat shortest-context-first as service-time shortest-job-first.
- Make random release order unseeded or inconsistent between counted and trace paths.
- Count drain wait as reconstruction delay after choosing release-relative deadlines.
- Reuse release-relative completion when reporting absolute deadline metrics.
- Drop the burst-at-zero baseline when drain_window_s is explicitly zero.
- Re-round an already-integer online baseline allocation and erase its chosen requests.
- Keep metric-only rounded queue evaluation dependent on per-request records.
- Change percentile or EDF tie semantics while compressing counted requests.
- Switch large allocations to a request-count heuristic that overshoots the
  retained-prefill target when exact rounding is cheap.
- Accept public RequestRecord release times that the evaluator will overwrite.
- Treat misspelled actions as state-transfer requests.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import queueing
from catalog import ModelParams
from queueing import (
    RELEASE_POLICIES,
    RequestRecord,
    evaluate_rounded_allocation,
    evaluate_rounded_queue,
    evaluate_rounded_queue_trace,
    evaluate_static_queue,
    evaluate_static_queue_trace,
    queue_metrics,
    round_allocation,
)
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


def test_large_rounding_uses_exact_minimum_overshoot_when_state_space_is_small():
    problem = queue_problem([6, 10], [201, 1], [1, 1], 10.0)
    y = np.array([[1.7, 0.0, 199.3], [0.1, 0.0, 0.9]])

    rounded = round_allocation(problem, y)

    assert rounded.retained_prefill_moved_s == 10.0
    assert_allclose(np.sum(rounded.y[:, :2], axis=1), [0, 1])


def test_rounding_does_not_materialize_request_records():
    problem = queue_problem([5, 8, 13], [120, 120, 120], [1, 1, 1], 1000.0)
    y = np.array([[60.4, 0.0, 59.6], [50.2, 0.0, 69.8], [0.0, 0.0, 120.0]])

    rounded = round_allocation(problem, y)

    assert not hasattr(rounded, "records")


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


def test_queue_supports_shortest_context_first_release_order():
    problem = queue_problem([100, 10, 20], [1, 1, 1], [30.0, 50.0, 40.0], 0.0)
    records = (
        RequestRecord(0, 0, "state", 100.0, 30.0, 0.0, 0.0),
        RequestRecord(1, 0, "state", 10.0, 50.0, 0.0, 0.0),
        RequestRecord(2, 0, "state", 20.0, 40.0, 0.0, 0.0),
    )

    _, trace = evaluate_static_queue_trace(
        problem, records, drain_window_s=60.0, release_policy="shortest-context-first"
    )

    assert _release_order(trace) == [1, 2, 0]


def test_random_release_order_is_seeded():
    problem = queue_problem([1, 1, 1, 1], [1, 1, 1, 1], [10.0, 20.0, 30.0, 40.0], 0.0)
    records = tuple(RequestRecord(g, 0, "state", 1.0, 10.0 * (g + 1), 0.0, 0.0) for g in range(4))

    first = evaluate_static_queue_trace(
        problem, records, drain_window_s=40.0, release_policy="random", release_seed=7
    )[1]
    repeat = evaluate_static_queue_trace(
        problem, records, drain_window_s=40.0, release_policy="random", release_seed=7
    )[1]
    other = evaluate_static_queue_trace(
        problem, records, drain_window_s=40.0, release_policy="random", release_seed=8
    )[1]

    assert _release_order(first) == [0, 2, 1, 3]
    assert _release_order(repeat) == _release_order(first)
    assert _release_order(other) != _release_order(first)


def test_absolute_deadline_metrics_count_late_release_time():
    problem = queue_problem([1, 1], [1, 1], [10.0, 10.0], 0.0)
    records = (
        RequestRecord(0, 0, "state", 1.0, 10.0, 10.0, 0.0),
        RequestRecord(1, 0, "state", 1.0, 10.0, 10.0, 0.0),
    )

    metrics = evaluate_static_queue(problem, records, drain_window_s=60.0)

    assert metrics["deadline_miss_rate"] == 0.0
    assert_allclose(metrics["p95_reconstruction_delay_ratio"], 0.1)
    assert metrics["absolute_deadline_miss_rate"] == 0.5
    assert metrics["absolute_p95_delay_over_deadline"] > 1.0


def test_counted_rounded_metrics_match_expanded_trace_metrics():
    problem = queue_problem([1, 1, 2], [3, 2, 4], [4.0, 8.0, 6.0], 0.0)
    y = np.array(
        [
            [2, 1, 0],
            [0, 2, 0],
            [3, 0, 1],
        ]
    )

    for release_policy in RELEASE_POLICIES:
        counted = evaluate_rounded_queue(
            problem, y, drain_window_s=12.0, release_policy=release_policy, release_seed=8
        )
        expanded, trace = evaluate_rounded_queue_trace(
            problem, y, drain_window_s=12.0, release_policy=release_policy, release_seed=8
        )

        assert len(trace) == 8
        for key in (
            "mean_reconstruction_delay",
            "p50_reconstruction_delay",
            "p95_reconstruction_delay",
            "p99_reconstruction_delay",
            "p95_normalized_reconstruction_delay",
            "deadline_miss_rate",
            "absolute_p95_delay_over_deadline",
            "absolute_deadline_miss_rate",
            "network_capacity_pressure",
            "prefill_capacity_pressure",
            "drain_completion_s",
            "replay_retained_prefill_fraction",
            "state_transfer_retained_prefill_fraction",
            "retained_prefill_moved_s",
            "retained_prefill_removal_rate_s_per_s",
        ):
            assert_allclose(counted[key], expanded[key])


def test_counted_metrics_match_expanded_edf_after_non_edf_release_inversion():
    problem = queue_problem([10, 100], [3, 1], [10.0, 1.0], 0.0)
    y = np.array([[3, 0, 0], [1, 0, 0]])

    counted = evaluate_rounded_queue(
        problem, y, drain_window_s=1.0, release_policy="shortest-context-first"
    )
    expanded, trace = evaluate_rounded_queue_trace(
        problem, y, drain_window_s=1.0, release_policy="shortest-context-first"
    )

    assert [record.g for record in sorted(trace, key=lambda record: record.reconstruction_delay)] != [0, 0, 0, 1]
    assert_allclose(counted["p95_reconstruction_delay"], expanded["p95_reconstruction_delay"])
    assert_allclose(counted["deadline_miss_rate"], expanded["deadline_miss_rate"])


def test_empty_queue_metrics_include_normalized_delay_aliases():
    problem = queue_problem([1], [1], [10.0], 0.0)
    y = np.array([[0, 0, 1]])

    metrics = evaluate_rounded_queue(problem, y, drain_window_s=10.0)

    assert metrics["p95_normalized_reconstruction_delay"] == 0.0
    assert metrics["p95_reconstruction_delay_ratio"] == 0.0
    assert metrics["absolute_p95_delay_over_deadline"] == 0.0


def test_counted_metrics_preserve_zero_window_edf_tie_order():
    problem = queue_problem([1, 1, 1], [2, 2, 2], [5.0, 3.0, 5.0], 0.0)
    y = np.array(
        [
            [2, 0, 0],
            [2, 0, 0],
            [2, 0, 0],
        ]
    )

    counted = evaluate_rounded_queue(problem, y, drain_window_s=0.0)
    expanded, trace = evaluate_rounded_queue_trace(problem, y, drain_window_s=0.0)

    assert len(trace) == 6
    assert_allclose(counted["p95_reconstruction_delay"], expanded["p95_reconstruction_delay"])
    assert_allclose(counted["deadline_miss_rate"], expanded["deadline_miss_rate"])


def test_metric_only_paths_do_not_build_request_records(monkeypatch):
    problem = queue_problem([10, 6], [2, 2], [10.0, 10.0], 6.0)
    integer = np.array([[1, 0, 1], [1, 0, 1]])
    fractional = np.array([[1.2, 0.0, 0.8], [0.2, 0.0, 1.8]])

    monkeypatch.setattr(
        queueing,
        "_request_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expanded records")),
    )

    integer_metrics = queue_metrics(problem, integer, drain_window_s=0.0)
    fractional_metrics = queue_metrics(problem, fractional, drain_window_s=0.0)
    rounded_metrics = evaluate_rounded_allocation(problem, round_allocation(problem, fractional), drain_window_s=0.0)

    assert integer_metrics["retained_prefill_moved_s"] == 16.0
    assert fractional_metrics["retained_prefill_moved_s"] >= problem.retained_prefill_target_s
    assert rounded_metrics["retained_prefill_moved_s"] == fractional_metrics["retained_prefill_moved_s"]


def test_default_drain_window_is_thirty_minutes():
    problem = queue_problem([1, 1], [1, 1], [10.0, 20.0], 0.0)
    records = (
        RequestRecord(0, 0, "state", 1.0, 10.0, 0.0, 0.0),
        RequestRecord(1, 0, "state", 1.0, 20.0, 0.0, 0.0),
    )

    _, trace = evaluate_static_queue_trace(problem, records)

    assert_allclose([record.release_time_s for record in trace], [0.0, 900.0])


def test_static_queue_rejects_preassigned_release_times():
    problem = queue_problem([1], [1], [10.0], 0.0)
    records = (RequestRecord(0, 0, "state", 1.0, 10.0, 0.0, 0.0, release_time_s=1.0),)

    with pytest.raises(ValueError, match="assigns release times"):
        evaluate_static_queue(problem, records)


def test_static_queue_rejects_unknown_actions():
    problem = queue_problem([1], [1], [10.0], 0.0)
    records = (RequestRecord(0, 0, "typo", 1.0, 10.0, 0.0, 0.0),)

    with pytest.raises(ValueError, match="unknown queue action"):
        evaluate_static_queue(problem, records)


def test_zero_window_zero_load_has_zero_removal_rate():
    problem = queue_problem([1], [1], [3.5], 0.0)
    y = np.array([[0, 0, 1]])

    metrics = queue_metrics(problem, y, drain_window_s=0.0)

    assert metrics["retained_prefill_removal_rate_s_per_s"] == 0.0


def test_queue_metrics_preserves_integer_allocations_without_rerounding():
    problem = queue_problem([10, 6], [1, 1], [10.0, 10.0], 6.0)
    y = np.array([[1, 0, 0], [1, 0, 0]])

    metrics = queue_metrics(problem, y, drain_window_s=0.0)

    assert metrics["retained_prefill_moved_s"] == 16.0


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


def _release_order(trace):
    return [record.g for record in sorted(trace, key=lambda record: record.release_time_s)]
