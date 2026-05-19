"""
Claim:
Queue-centered plots read finite queue metrics, omit infeasible points, use
request-level hard-case CDFs, and derive waiting queue depth from queue traces.

Plausible wrong implementations:
- Plot infeasible rows as zero and create false frontier points.
- Build CDFs from aggregated classes instead of per-request trace records.
- Count requests in service as waiting queue depth.
- Keep emitting retired heatmap, objective-ratio, or summary plot artifacts.
"""

from __future__ import annotations

from types import SimpleNamespace

from experiments.plot_queue_centered import (
    OUTPUT_FILES,
    _cdf_points,
    _max_waiting_depth_points,
    _policy_points,
)


def test_policy_points_omit_infeasible_and_rounding_failed_rows():
    rows = [
        {"policy": "CVXPY-rounded", "slack_multiplier": "0.5", "shed_fraction": "0.2", "miss_rate": "0.4"},
        {"policy": "CVXPY-rounded", "slack_multiplier": "0.5", "shed_fraction": "0.3", "miss_rate": "nan"},
        {"policy": "replay-only", "slack_multiplier": "0.5", "shed_fraction": "0.2", "miss_rate": "0.0"},
    ]

    assert _policy_points(
        rows,
        "CVXPY-rounded",
        "shed_fraction",
        "miss_rate",
        {"slack_multiplier": 0.5},
    ) == [(0.2, 0.4)]


def test_cdf_points_are_request_level_empirical_cdf():
    assert _cdf_points([3.0, 1.0, 3.0]) == [(1.0, 1 / 3), (3.0, 2 / 3), (3.0, 1.0)]


def test_queue_depth_counts_waiting_requests_not_requests_in_service():
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


def test_requested_outputs_exclude_retired_png_artifacts():
    retired = {
        "headline_action_mix.png",
        "allocation_heatmap_per_scenario.png",
        "utilization_vs_policy.png",
        "objective_vs_policy.png",
        "convergence_one_scenario.png",
        "crossover_recovery.png",
    }
    requested = {
        "safe_shed_frontier_lines.pdf",
        "miss_rate_frontier_lines.pdf",
        "delay_cdf_hard_case.pdf",
        "queue_depth_hard_case.pdf",
        "resource_pressure_scatter.pdf",
    }

    assert set(OUTPUT_FILES) == requested
    assert retired.isdisjoint(OUTPUT_FILES)


def _record(k, network_wait, network_service):
    return SimpleNamespace(
        k=k,
        network_queue_wait=network_wait,
        network_service_time=network_service,
        prefill_queue_wait=0.0,
        prefill_service_time=0.0,
    )
