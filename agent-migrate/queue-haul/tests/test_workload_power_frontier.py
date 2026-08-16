"""
Claim:
The workload frontier pairs every Queue-Haul request within each sampled
workload/calibration draw, credits only safe shed by 30 seconds, normalizes by
that draw's removable power, and weights each draw-by-factor case once.

Plausible wrong implementations:
- Resample the workload, timing, or power model independently by request.
- Divide attained power by requested power or a global maximum.
- Credit an unsafe higher-request plan instead of retaining the last safe plan.
- Overweight a case because it contributes extra rows or policies.
- Use a different Queue-Haul draw or solver at the shared two-thirds point.
- Plot target-following policies against their own target instead of the
  distribution of maximum safely attainable power.
- Collapse bandwidth states without proving they are paired null effects.
- Omit or duplicate a factorial state when labeling bandwidth-null curve pairs.
"""

import numpy as np
import pytest

import workload_adaptation_campaign as adaptation
from workload_power_frontier import (
    DISPLAY_GROUPS, SOLVERS, assert_bandwidth_null, capacity_summary,
    power_summary, request_grid, sweep,
)


def test_display_groups_partition_all_eight_factorial_states():
    grouped = sum(DISPLAY_GROUPS.values(), ())
    expected = tuple(case_id for case_id, _, _ in adaptation.factorial_cases())

    assert len(grouped) == len(set(grouped)) == 8
    assert set(grouped) == set(expected)


def test_capacity_summary_uses_one_maximum_per_paired_draw():
    rows = [{
        "replicate": replicate, "factor_case_id": state,
        "policy": "queue_haul_lp", "requested_fraction": 1,
        "safely_attained_fraction": capacity,
    } for state in ("none", "hbm", "dest_compute",
                    "dest_compute-hbm")
            for replicate, capacity in enumerate((.25, .75))]

    summary = capacity_summary(rows, grid=(0, .5, 1))
    none = [row for row in summary if row["constraint_state"] == "none"]

    assert [row["attainment_rate"] for row in none] == [1, .5, 0]
    assert all(row["cases"] == 2 for row in summary)
    with pytest.raises(RuntimeError, match="one maximum"):
        capacity_summary([*rows, rows[0]], grid=(0, .5, 1))


def test_bandwidth_null_must_hold_for_every_paired_frontier_point():
    pairs = (("none", "bandwidth"), ("hbm", "bandwidth-hbm"),
             ("dest_compute", "bandwidth-dest_compute"),
             ("dest_compute-hbm", "bandwidth-dest_compute-hbm"))
    rows = [{
        "replicate": 0, "factor_case_id": state,
        "policy": "queue_haul_lp", "requested_fraction": fraction,
        "safely_attained_fraction": fraction / 2,
    } for pair in pairs for state in pair for fraction in (0, 1)]

    assert_bandwidth_null(rows)
    rows[-1]["safely_attained_fraction"] = .6
    with pytest.raises(RuntimeError, match="bandwidth state is not null"):
        assert_bandwidth_null(rows)


def test_power_summary_weights_cases_once_and_keeps_watts():
    rows = [{
        "case_id": case, "policy": "queue_haul_lp", "requested_fraction": .5,
        "maximum_removable_w": maximum, "safely_attained_shed_w": attained,
    } for case, maximum, attained in (
        ("a", 10, 5), ("b", 20, 10), ("c", 30, 15), ("d", 40, 20),
    )]

    summary = power_summary(rows)[0]
    assert summary["cases"] == 4
    assert summary["maximum_removable_w_median"] == 25
    assert summary["safely_attained_w_median"] == 12.5

    with pytest.raises(RuntimeError, match="weight each case once"):
        power_summary([*rows, rows[0]])


def test_sampled_frontier_is_paired_normalized_and_monotone():
    rows, _ = sweep(samples=1, points=2, seed=3)
    action_rows, _ = adaptation.simulate(samples=1, seed=3)
    fractions = request_grid(2)

    assert len(rows) == 8 * len(SOLVERS) * len(fractions)
    assert len({row["power_bootstrap_index"] for row in rows}) == 1
    assert len({row["timing_fit_sha256"] for row in rows}) == 1
    assert len({row["maximum_removable_w"] for row in rows}) == 1
    assert all(np.isclose(
        row["requested_shed_w"],
        row["requested_fraction"] * row["maximum_removable_w"],
    ) for row in rows)
    assert all(np.isclose(
        row["safely_attained_fraction"],
        row["safely_attained_shed_w"] / row["maximum_removable_w"],
    ) for row in rows)
    for case in {row["case_id"] for row in rows}:
        for policy in SOLVERS:
            attained = [row["safely_attained_shed_w"] for row in rows
                        if row["case_id"] == case and row["policy"] == policy]
            assert all(right >= left for left, right in zip(attained, attained[1:]))
    frontier = {row["factor_case_id"]: row for row in rows
                if row["policy"] == "queue_haul_lp"
                and np.isclose(row["requested_fraction"], 2 / 3)}
    action = {row["case_id"]: row for row in action_rows}
    assert frontier.keys() == action.keys()
    assert all(np.isclose(frontier[case]["requested_shed_w"], row["target_w"])
               and np.isclose(frontier[case]["raw_safe_shed_w"],
                              row["planned_shed_w"])
               and frontier[case]["target_met_by_30s"] == row["target_met"]
               for case, row in action.items())
