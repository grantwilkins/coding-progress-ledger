"""
Claim:
Queue-centered plots read finite queue metrics, omit infeasible points, use
request-level hard-case CDFs, and derive waiting queue depth from release-time
queue traces.

Plausible wrong implementations:
- Plot infeasible rows as zero and create false frontier points.
- Build CDFs from aggregated classes instead of per-request trace records.
- Count requests in service as waiting queue depth.
- Plot drained requests as though every request arrived at time zero.
- Keep emitting retired heatmap, objective-ratio, or summary plot artifacts.
"""

from __future__ import annotations

from types import SimpleNamespace

from experiments.plot_queue_centered import (
    OUTPUT_FILES,
    POLICY_LABELS,
    PLOT_POLICIES,
    _cdf_points,
    _max_waiting_depth_points,
    _policy_points,
)


def test_policy_points_omit_infeasible_and_rounding_failed_rows():
    rows = [
        {
            "policy": "CVXPY-rounded",
            "deadline_scale": "0.5",
            "evacuated_state_tb": "2.0",
            "deadline_miss_rate": "0.4",
        },
        {
            "policy": "CVXPY-rounded",
            "deadline_scale": "0.5",
            "evacuated_state_tb": "3.0",
            "deadline_miss_rate": "nan",
        },
        {
            "policy": "replay-only",
            "deadline_scale": "0.5",
            "evacuated_state_tb": "2.0",
            "deadline_miss_rate": "0.0",
        },
    ]

    assert _policy_points(
        rows,
        "CVXPY-rounded",
        "evacuated_state_tb",
        "deadline_miss_rate",
        {"deadline_scale": 0.5},
    ) == [(2.0, 0.4)]


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


def test_queue_depth_uses_release_time_for_drained_requests():
    trace = (_record(k=0, network_wait=2.0, network_service=1.0, release_time=10.0),)

    assert _max_waiting_depth_points(trace, "network") == [(0.0, 0.0), (10.0, 1.0), (12.0, 0.0)]


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
        "retained_state_frontier.pdf",
        "deadline_miss_frontier.pdf",
        "deadline_delay_cdf.pdf",
        "queue_depth_example.pdf",
        "network_prefill_busy_scatter.pdf",
    }

    assert set(OUTPUT_FILES) == requested
    assert retired.isdisjoint(OUTPUT_FILES)


def test_policy_labels_are_short_enough_for_paper_legends():
    assert set(POLICY_LABELS) == {
        "CVXPY-rounded",
        "deadline-penalty-rounded",
        "mirror-descent-rounded",
        "crossover-greedy",
        "replay-only",
        "state-only",
    }
    assert max(len(label) for label in POLICY_LABELS.values()) <= 16


def test_plot_legend_uses_only_main_unambiguous_policies():
    assert len(PLOT_POLICIES) == 6
    assert "deadline-aware-m0.8-rounded" not in PLOT_POLICIES
    assert "deadline-aware-m1.0-rounded" not in PLOT_POLICIES
    assert set(PLOT_POLICIES) == {
        "CVXPY-rounded",
        "deadline-penalty-rounded",
        "mirror-descent-rounded",
        "crossover-greedy",
        "replay-only",
        "state-only",
    }


def test_plot_outputs_use_retained_state_names():
    text = "\n".join(OUTPUT_FILES)

    assert "retained_state_frontier.pdf" in OUTPUT_FILES
    for stale in ("source_" + "load", "source_" + "prefill", "sh" + "ed"):
        assert stale not in text


def _record(k, network_wait, network_service, release_time=0.0):
    return SimpleNamespace(
        k=k,
        release_time_s=release_time,
        network_queue_wait=network_wait,
        network_service_time=network_service,
        prefill_queue_wait=0.0,
        prefill_service_time=0.0,
    )
