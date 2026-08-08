"""
Claim:
Operating curves preserve each campaign cell, count only complete full-target
episodes, include the trailing power window, report deadline shed in watts,
normalize every planned action over all offered sessions, and show only observed
bandwidth/prefill action cells using the campaign's prefill calibration.

Plausible wrong implementations:
- Treat partial work as full attainment or omit the trailing power window.
- Fill unmeasured heatmap cells by interpolating between the two sweeps.
- Use total load or divide by, rather than multiply by, prefill capacity.
- Mix 19/30-second cohorts or pool unbalanced context profiles.
- Normalize actions over moved sessions and hide the not-moved choice.
"""

import pytest

from plot_capacity_operating_curves import (
    _cdf,
    action_heatmap_rows,
    action_fractions,
    bandwidth_observations,
    summarize_bandwidth,
    summarize_full_drain,
    summarize_load,
)


def scenario(identifier, policy="lp", load=.8, deadline=30, bandwidth=1000,
             profile="small", methods=("replay",), sessions=2):
    return {
        "scenario_id": identifier, "policy": policy,
        "load_fraction": load, "required_deadline_s": deadline,
        "bandwidth_mbps": bandwidth, "context_profile": profile,
        "sessions": [{"session_id": f"s{i}"} for i in range(sessions)],
        "moves": [{"session_id": f"s{i}", "method": method}
                  for i, method in enumerate(methods)],
    }


def test_action_fractions_include_sessions_the_scheduler_did_not_move():
    fractions = action_fractions([
        scenario("a", methods=("replay", "kv_transfer"), sessions=4),
        scenario("b", methods=("replay",), sessions=4),
    ])
    assert fractions == {
        "replay": 2 / 8, "kv_transfer": 1 / 8, "not_moved": 5 / 8,
    }
    assert sum(fractions.values()) == 1


def test_load_curve_rejects_fast_partial_work_as_full_power():
    plans = [scenario(str(i), methods=("replay",) * moved, sessions=2)
             for i, moved in enumerate((2, 1, 0))]
    rows = [{
        "scenario_id": str(i), "policy": "lp", "load_fraction": .8,
        "offered_rho": offered, "requested_shed_w": 100,
        "achieved_shed_w": shed, "full_drain_s": drain,
        "planned_sessions": moved, "credited_sessions": moved,
    } for i, (offered, shed, drain, moved) in enumerate((
        (.7, 100, 12, 2), (.8, 50, 1, 1), (.95, 0, 0, 0),
    ))]
    point = summarize_load(rows, plans, power_window_s=5)[0]
    assert point["independent_value"] == .8
    assert point["watts_shed_by_deadline"] == 50
    assert point["time_to_full_power_s"] == 17
    assert point["deadline_attainment_fraction"] == pytest.approx(1 / 3)


def test_bandwidth_curves_keep_campaign_splits_and_convert_to_watts():
    plans = [
        scenario("a", policy="queue_haul", profile="small", deadline=30,
                 methods=("replay", "kv_transfer")),
        scenario("b", policy="queue_haul", profile="small", deadline=30,
                 methods=("replay", "kv_transfer")),
        scenario("c", policy="queue_haul", profile="large", deadline=19,
                 methods=("kv_transfer", "kv_transfer")),
    ]
    episodes = [{
        "scenario_id": row["scenario_id"], "commit_100_s": commit,
        "planned_migrations": 2, "completed_migrations": 2,
    } for row, commit in zip(plans, (10, 28, 20))]
    attainment = [{
        "scenario_id": row["scenario_id"], "power_attainment_fraction": value,
    } for row, value in zip(plans, (1, .5, .25))]
    points = summarize_bandwidth(
        {"scenarios": plans}, episodes, attainment,
        target_w=120, power_window_s=5,
    )
    assert {(row["split"], row["deadline_s"]) for row in points} == {
        ("small", 30), ("large", 19),
    }
    small = next(row for row in points if row["split"] == "small")
    assert small["time_to_full_power_s"] == 24
    assert small["deadline_attainment_fraction"] == .5
    assert small["watts_shed_by_deadline"] == 90
    assert small["replay_action_fraction"] == .5
    assert small["kv_transfer_action_fraction"] == .5


def test_full_drain_keeps_eventual_time_when_deadline_shed_is_partial():
    plans = [scenario(str(i), policy="replay_only", load=.9, deadline=30,
                      methods=("replay", "replay")) for i in range(2)]
    rows = [{
        "scenario_id": str(i), "policy": "replay_only", "load_fraction": .9,
        "configured_goodput_mbps": 1000, "requested_shed_w": 100,
        "achieved_shed_w": shed, "full_drain_s": drain,
        "planned_sessions": 2,
    } for i, (shed, drain) in enumerate(((100, 28), (50, 32)))]
    point = summarize_full_drain(rows, plans, power_window_s=5)[0]
    assert point["time_to_full_power_s"] == 35
    assert point["deadline_attainment_fraction"] == 0
    assert point["watts_shed_by_deadline"] == 75


def test_pooled_cdf_keeps_failures_as_missing_mass_and_requires_balance():
    assert _cdf([2, None, 1], 3) == ([0, 1, 2], [0, 1 / 3, 2 / 3])
    plans = [
        scenario("a", policy="queue_haul", profile="small"),
        scenario("b", policy="queue_haul", profile="small"),
        scenario("c", policy="queue_haul", profile="large"),
    ]
    episodes = [{"scenario_id": row["scenario_id"], "commit_100_s": 10,
                 "planned_migrations": 2, "completed_migrations": 2}
                for row in plans]
    attainment = [{"scenario_id": row["scenario_id"],
                   "power_attainment_fraction": 1} for row in plans]
    with pytest.raises(ValueError, match="balanced"):
        bandwidth_observations(
            {"scenarios": plans}, episodes, attainment, 100, 5)


def test_action_heatmap_keeps_sparse_observations_and_calibrates_prefill():
    plans = [
        scenario("zero", policy="lp", load=0, sessions=4, methods=()),
        scenario("loaded", policy="lp", load=.5, sessions=4,
                 methods=("replay", "kv_transfer", "kv_transfer")),
    ]
    load = [
        {"scenario_id": "zero", "policy": "lp", "load_fraction": 0,
         "configured_goodput_mbps": 10000, "offered_rho_prefill": 0},
        {"scenario_id": "loaded", "policy": "lp", "load_fraction": .5,
         "configured_goodput_mbps": 10000, "offered_rho_prefill": .3,
         "offered_rho": .9},
    ]
    bandwidth = [{
        "deadline_s": deadline, "scheduler": "queue_haul", "split": "small",
        "independent_value": bandwidth_gbps, "observations": 2,
        "replay_action_fraction": replay, "kv_transfer_action_fraction": 1 - replay,
        "not_moved_action_fraction": 0,
    } for deadline, bandwidth_gbps, replay in (
        (30, 1, 1), (30, 10, .5), (19, 1, 0),
    )]
    rows = action_heatmap_rows(load, plans, bandwidth, prefill_capacity_tps=100)
    coordinates = {(row["bandwidth_gbps"],
                    row["destination_prefill_tokens_per_s"])
                   for row in rows}
    assert coordinates == {(1, 0), (10, 0), (10, 30)}
    assert (1, 30) not in coordinates
    assert {row["campaign"] for row in rows
            if row["bandwidth_gbps"] == 10
            and row["destination_prefill_tokens_per_s"] == 0} == {"bandwidth"}
    cells = {}
    for row in rows:
        cells.setdefault((row["bandwidth_gbps"],
                          row["destination_prefill_tokens_per_s"]), {})[
                              row["action"]] = row["action_fraction"]
    assert cells[1, 0] == {"replay": 1, "kv_transfer": 0, "not_moved": 0}
    assert cells[10, 30] == {
        "replay": .25, "kv_transfer": .5, "not_moved": .25,
    }
    assert all(sum(fractions.values()) == 1 for fractions in cells.values())
