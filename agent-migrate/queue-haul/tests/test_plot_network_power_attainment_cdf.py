"""
Claim:
The frontier ECDF reports the earliest common-epoch time at which completed
sessions reach the nonlinear modeled power target, with one denominator entry
per matched episode, late events retained, and unattained targets as missing mass.

Plausible wrong implementations:
- Count completed sessions instead of evaluating their modeled power shed.
- Measure each request from its own start instead of the shared policy epoch.
- Condition on attained episodes or clip late events at the deadline.
- Bound the plot by attainment events instead of the full observed episode horizon.
- Retain bespoke scheduler colors instead of the requested Tab10 mapping.
"""

import numpy as np
from matplotlib import pyplot as plt

from plot_network_power_attainment_cdf import (
    COLORS,
    POLICIES,
    attainment_curve,
    attainment_time,
    completion_times,
    plot_horizon,
)


def test_attainment_time_uses_power_not_completed_session_fraction():
    weights = {"heavy": 9, "a": 1, "b": 1, "c": 1}
    event = attainment_time(
        [(4, "c"), (1, "heavy"), (3, "b"), (2, "a")], 9,
        lambda sessions: sum(weights[session] for session in sessions),
    )
    assert event == 1


def test_completion_times_use_common_epoch_and_exclude_failed_requests():
    assert completion_times({
        "started_ns": 1_000_000_000,
        "requests": [
            {"session_id": "complete", "request": {
                "start_ns": 5_000_000_000, "end_ns": 7_000_000_000,
            }},
            {"session_id": "failed", "error": "timeout"},
        ],
    }) == [(6, "complete")]


def test_attainment_curve_keeps_late_events_and_missing_mass():
    rows = [
        {"policy": "queue_haul", "attainment_s": value}
        for value in (35, None, 10)
    ] + [{"policy": "greedy", "attainment_s": 1}]
    x, y = attainment_curve(rows, "queue_haul")
    np.testing.assert_array_equal(x, [0, 10, 35])
    np.testing.assert_allclose(y, [0, 1 / 3, 2 / 3])


def test_plot_uses_full_episode_horizon_and_tab10_policy_order():
    rows = [
        {"episode_end_s": 10, "attainment_s": 8},
        {"episode_end_s": 37.6, "attainment_s": None},
    ]
    assert plot_horizon(rows, 30) == 40
    assert plot_horizon([{"episode_end_s": 20}], 30) == 30
    assert [COLORS[policy] for policy in POLICIES] \
        == list(plt.get_cmap("tab10").colors[:len(POLICIES)])
