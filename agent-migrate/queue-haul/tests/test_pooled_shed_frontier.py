"""
Claim:
The pooled frontier uses attained/removable power within each equally weighted
case, reports the median and interquartile range at each policy/request, and
maps independent-fastest to the canonical True Greedy identity.

Plausible wrong implementations:
- Pool raw watts so high-power cases dominate the summary.
- Divide attained shed by requested or policy-maximum power.
- Weight a case more because it has more sessions or policies.
- Mix requested-shed coordinates while computing uncertainty bands.
- Compute quartiles across policies rather than across cases.
- Preserve the obsolete independent-fastest display identity.
"""

import csv

import matplotlib.pyplot as plt
import pytest

from plot_pooled_shed_frontier import (
    hardware_point, POLICY_STYLE_IDS, pooled_summary, queue_haul_cutoff,
    write_plot,
)


def test_pooled_summary_normalizes_cases_and_keeps_policy_coordinates():
    rows = [
        {"case_id": case, "policy": policy, "requested_fraction": request,
         "safely_attained_fraction": value}
        for policy, request, values in (
            ("queue_haul_lp", .5, (.2, .4, .6, .8)),
            ("queue_haul_greedy", .5, (.1, .2, .3, .4)),
        )
        for case, value in zip("abcd", values)
    ]

    summary = {(row["policy"], row["requested_fraction"]): row
               for row in pooled_summary(rows)}
    assert summary["queue_haul_lp", .5] == {
        "policy": "queue_haul_lp", "requested_fraction": .5,
        "lower_quartile": pytest.approx(.35), "median": pytest.approx(.5),
        "upper_quartile": pytest.approx(.65), "cases": 4,
    }
    assert summary["queue_haul_greedy", .5]["median"] == pytest.approx(.25)

    duplicated = [*rows, next(row for row in rows
                              if row["policy"] == "queue_haul_lp"
                              and row["requested_fraction"] == .5)]
    with pytest.raises(RuntimeError, match="weight each case once"):
        pooled_summary(duplicated)


def test_pooled_frontier_maps_true_greedy_identity():
    assert POLICY_STYLE_IDS["independent_fastest"] == "isolated_fastest"


def test_queue_haul_cutoff_uses_nonzero_interpolated_parity_crossing():
    rows = [
        {"policy": "queue_haul_lp", "requested_fraction": x, "median": y}
        for x, y in ((0, 0), (.5, .7), (.75, .7), (1, .7))
    ]
    assert queue_haul_cutoff(rows) == pytest.approx(.7)


def test_pooled_plot_shows_crossing_inside_80_percent_x_limit(
        tmp_path, monkeypatch):
    rows = [
        {"policy": policy, "requested_fraction": x,
         "lower_quartile": y, "median": y, "upper_quartile": y}
        for policy in (
            "queue_haul_lp", "queue_haul_greedy", "independent_fastest",
            "replay_only", "kv_only", "power_blind", "deadline_blind",
        ) for x, y in ((0, 0), (.5, .7), (.75, .7), (1, .7))
    ]
    monkeypatch.setattr(plt, "close", lambda _: None)
    write_plot(rows, tmp_path / "frontier")
    axis = plt.gca()
    assert axis.get_xlim() == pytest.approx((0, .8))
    assert axis.get_ylim() == pytest.approx((0, 1))
    assert axis.lines[-1].get_xdata()[-1] == pytest.approx(.7)


def test_hardware_point_pools_matched_queue_haul_runs(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "policy", "condition_index", "repeat", "status", "deadline_met",
            "realized_shed_w", "requested_shed_w"))
        writer.writeheader()
        for condition in range(5):
            for repeat in range(3):
                writer.writerow({
                    "policy": "queue_haul_robust", "condition_index": condition,
                    "repeat": repeat, "status": "complete", "deadline_met": True,
                    "realized_shed_w": 21, "requested_shed_w": 20,
                })
    assert hardware_point(path, .4) == {
        "requested_fraction": .4, "attained_fraction": pytest.approx(.42),
        "observations": 15,
    }
