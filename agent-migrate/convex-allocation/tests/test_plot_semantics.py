"""
Claim:
The report figure script emits exactly one simple artifact per hypothesis plus a
compact integer benchmark table. The H1/H2 plots use queue safety from absolute
event-start deadline metrics. The H2 frontier isolates release policy for the
main allocation policy, the H2 CDF is request-level, and the H4 heatmap exposes
per-class state locality rather than only destination load.

Plausible wrong implementations:
- Reintroduce crowded diagnostic plots instead of the five report figures.
- Keep using release-relative reconstruction metrics after switching drain
  frontier safety to event-start deadlines.
- Keep the old cumulative max frontier after switching to event-start deadlines.
- Plot all allocation-policy by release-policy pairs in H2.
- Average class delays before building the CDF.
- Keep unrelated integer policies in the compact summary table.
- Drop context/KV locality from the manifest heatmap labels.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from catalog import ModelParams
from experiments.plot_queue_centered import (
    FRONTIER_POLICY,
    FRONTIER_RELEASE_POLICIES,
    INTEGER_TABLE_POLICIES,
    OUTPUT_FILES,
    REPORT_POLICIES,
    _allocation_heatmap,
    _cdf_points,
    _integer_summary_rows,
    _max_waiting_depth_points,
    _resource_pressure_frame,
    _safe_frontier,
    _safe_series,
)
from problem import ProblemData

import numpy as np


def test_report_outputs_are_one_plot_per_hypothesis_plus_summary_table():
    assert set(OUTPUT_FILES) == {
        "h1_resource_pressure.pdf",
        "h2_safe_frontier.pdf",
        "h2_delay_cdf.pdf",
        "h3_action_mix_by_model.pdf",
        "h4_state_manifest_heatmap.pdf",
        "integer_benchmark_summary.csv",
    }
    assert not any("queue_depth" in path or "busy_scatter" in path for path in OUTPUT_FILES)


def test_report_policies_are_small_enough_to_read_without_overplotting():
    assert REPORT_POLICIES == (
        "deadline-penalty-rounded",
        "online-queue-greedy",
        "least-loaded-destination",
        "replay-only",
        "state-only",
    )
    assert len(REPORT_POLICIES) == 5


def test_safe_series_requires_absolute_deadline_bounds():
    df = pd.DataFrame(
        {
            "absolute_deadline_miss_rate": [0.01, 0.011, 0.0],
            "absolute_p95_delay_over_deadline": [1.0, 0.5, 1.001],
        }
    )

    assert _safe_series(df).tolist() == [True, False, False]


def test_safe_series_ignores_release_relative_deadline_metrics_when_present():
    df = pd.DataFrame(
        {
            "deadline_miss_rate": [0.5],
            "p95_delay_over_deadline": [3.0],
            "absolute_deadline_miss_rate": [0.0],
            "absolute_p95_delay_over_deadline": [0.1],
        }
    )

    assert _safe_series(df).tolist() == [True]


def test_safe_frontier_uses_largest_safe_fraction_by_drain_window():
    rows = [
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.2, 0.0, 0.8),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.4, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.6, 0.2, 0.9),
        _sweep_row("deadline-penalty-rounded", "edf", 1800.0, 0.6, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "edf", 3600.0, 0.3, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "random", 900.0, 0.3, 0.0, 0.9),
        _sweep_row("replay-only", "edf", 900.0, 0.9, 0.0, 0.9),
    ]

    frontier = _safe_frontier(rows)
    by_release_window = {
        (row.release_policy, row.drain_window_s): row.max_safe_retained_prefill_fraction
        for row in frontier.itertuples()
    }

    assert FRONTIER_POLICY == "deadline-penalty-rounded"
    assert FRONTIER_RELEASE_POLICIES == ("edf", "shortest-context-first", "random")
    assert by_release_window[("edf", 900.0)] == 0.4
    assert by_release_window[("edf", 1800.0)] == 0.6
    assert by_release_window[("edf", 3600.0)] == 0.3
    assert by_release_window[("random", 900.0)] == 0.3
    assert set(frontier["policy"]) == {"deadline-penalty-rounded"}


def test_resource_pressure_uses_frontier_rows_for_selected_workload():
    rows = [
        _frontier_row("deadline-penalty-rounded", "edf", 20.0, 0.25, 0.8, 0.7),
        _frontier_row("replay-only", "edf", 20.0, 0.13, 0.0, 1.1),
        _frontier_row("state-only", "edf", 20.0, 0.27, 1.6, 0.0),
        _frontier_row("online-queue-greedy", "edf", 20.0, 0.0, 0.0, 0.0),
        _frontier_row("least-loaded-destination", "edf", 20.0, 0.0, 0.0, 0.0),
        _frontier_row("deadline-penalty-rounded", "random", 20.0, 0.03, 0.2, 0.2),
        _frontier_row("deadline-penalty-rounded", "edf", 1200.0, 0.0, 0.0, 0.0),
    ]

    df = _resource_pressure_frame(rows, deadline_scale=1.0)

    assert set(df["policy"]) == set(REPORT_POLICIES)
    assert set(df["release_policy"]) == {"edf"}
    assert set(df["drain_window_s"]) == {20.0}
    by_policy = {row.policy: row for row in df.itertuples()}
    assert by_policy["deadline-penalty-rounded"].network_capacity_pressure == 0.8
    assert by_policy["replay-only"].prefill_capacity_pressure == 1.1


def test_cdf_points_are_request_level_empirical_cdf():
    assert _cdf_points([3.0, 1.0, 3.0]) == [(1.0, 1 / 3), (3.0, 2 / 3), (3.0, 1.0)]


def test_integer_summary_keeps_only_compact_methodology_rows():
    rows = [
        _integer_row("case-a", policy)
        for policy in (*INTEGER_TABLE_POLICIES, "online-queue-greedy", "state-only")
    ]

    summary = _integer_summary_rows(rows)

    assert [row["policy"] for row in summary] == list(INTEGER_TABLE_POLICIES)
    assert set(summary[0]) == {"case", "policy", "integer_objective_gap_to_best", "p95_delay", "miss_rate"}


def test_manifest_heatmap_labels_include_context_deadline_and_locality():
    model = ModelParams("manifest", 1.0, 2.0, 1.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="manifest",
        T=np.array([1000.0, 2000.0]),
        d=np.array([2.0, 2.0]),
        deadline_s=np.array([10.0, 5.0]),
        lambda_Bps=np.array([10.0]),
        rho_prefill=np.array([10.0]),
        C_net=np.array([100.0]),
        C_prefill=np.array([100.0]),
        ell_net=np.zeros(1),
        ell_prefill=np.zeros(1),
        h_ctx=np.array([[0.2], [0.8]]),
        h_kv=np.array([[0.1], [0.5]]),
        retained_prefill_target_s=1.0,
    )
    allocation = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    heatmap, row_labels, col_labels = _allocation_heatmap(problem, allocation)

    assert heatmap.shape == (2, 3)
    assert col_labels == ["site 0\nreplay", "site 0\nstate", "stay"]
    assert all("T=" in label and "ddl=" in label and "ctx=" in label and "kv=" in label for label in row_labels)


def test_queue_depth_helper_still_counts_waiting_requests_for_bandwidth_plot():
    trace = (
        _record(k=0, network_wait=0.0, network_service=1.0),
        _record(k=0, network_wait=2.0, network_service=1.0),
        _record(k=0, network_wait=4.0, network_service=1.0),
        _record(k=1, network_wait=3.0, network_service=1.0),
    )

    assert _max_waiting_depth_points(trace, "network") == [
        (0.0, 0.0),
        (0.0, 2.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]


def _sweep_row(policy, release_policy, drain_window_s, retained_prefill_fraction, miss_rate, delay_ratio):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "retained_prefill_fraction": str(retained_prefill_fraction),
        "deadline_scale": "0.5",
        "drain_window_s": str(drain_window_s),
        "network_capacity_pressure": "0.5",
        "prefill_capacity_pressure": "0.5",
        "deadline_miss_rate": str(miss_rate),
        "p95_delay_over_deadline": str(delay_ratio),
        "absolute_deadline_miss_rate": str(miss_rate),
        "absolute_p95_delay_over_deadline": str(delay_ratio),
    }


def _frontier_row(policy, release_policy, drain_window_s, fraction, net, prefill):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "drain_window_s": str(drain_window_s),
        "deadline_scale": "1.0",
        "max_safe_retained_prefill_fraction": str(fraction),
        "network_capacity_pressure_at_frontier": str(net),
        "prefill_capacity_pressure_at_frontier": str(prefill),
    }


def _integer_row(case, policy):
    return {
        "case": case,
        "policy": policy,
        "integer_objective_gap_to_best": "0.1",
        "p95_delay": "2.0",
        "miss_rate": "0.0",
        "integer_objective": "1.0",
    }


def _record(k, network_wait, network_service, release_time=0.0):
    return SimpleNamespace(
        k=k,
        release_time_s=release_time,
        network_queue_wait=network_wait,
        network_service_time=network_service,
        prefill_queue_wait=0.0,
        prefill_service_time=0.0,
    )
