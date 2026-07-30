"""
Claim:
The simulated Pareto campaign maximizes attained shed within each time budget,
reports completion for only the admitted actions, compares policies only within
matched scenario-budget pairs, and exposes interpolation or extrapolation.

Plausible wrong implementations:
- Reverse either Pareto objective.
- Let an equal point dominate another point.
- Compare policies from different episodes in the paired result.
- Label non-anchor contexts or contexts outside the measured range as measured.
- Apply aggregate replay/KV capacity independently to every concurrent stream.
- Mix protocol-wire bytes with the simulator's sealed KV byte units.
- Append cleanup migrations that were outside the deadline-admitted set.
- Normalize attained shed by admitted sessions instead of all source sessions.
"""

import json
from types import SimpleNamespace

import simulated_pareto_campaign as campaign
from simulated_pareto_campaign import (
    admitted_moves, aggregate_planning_profile, context_evidence,
    frontier_metrics, full_attainment_cdf, measured_kv_caps,
    measured_replay_caps, meets_deadline, parallel_profile, pareto_flags,
    shared_kv_profile,
)
from policy_hardware_campaign import _moves, _problem
from simulate import PlannedMove
from test_execution_simulator import model


def test_pareto_direction_and_pairing():
    rows = [
        {"match": "a", "power_attainment_fraction": .8,
         "completion_s": .8},
        {"match": "a", "power_attainment_fraction": .7,
         "completion_s": 1},
        {"match": "a", "power_attainment_fraction": .9,
         "completion_s": 1.2},
        {"match": "b", "power_attainment_fraction": 1,
         "completion_s": .1},
    ]

    pareto_flags(rows, ("match",))

    assert [row["pareto"] for row in rows] == [True, False, True, True]


def test_context_evidence_marks_nonanchors_and_extrapolation():
    anchors = {2048, 4096, 8192, 16384}

    assert context_evidence((2048, 8192), anchors) == "measured"
    assert context_evidence((4096, 12288), anchors) == "interpolated"
    assert context_evidence((1024, 4096), anchors) == "extrapolated"


def test_full_attainment_detail_filters_and_normalizes_per_policy():
    rows = [
        {"policy": "a", "power_attainment_fraction": 1,
         "completion_budget_ratio": .8},
        {"policy": "a", "power_attainment_fraction": .98,
         "completion_budget_ratio": .2},
        {"policy": "a", "power_attainment_fraction": .99,
         "completion_budget_ratio": .4},
        {"policy": "b", "power_attainment_fraction": 1,
         "completion_budget_ratio": .1},
    ]

    x, y = full_attainment_cdf(rows, "a")

    assert x.tolist() == [.4, .8]
    assert y.tolist() == [.5, 1]


def test_deadline_boundary_tolerates_roundoff_but_not_real_misses():
    assert meets_deadline(1 - 1e-12, 10 + 1e-12, 10)
    assert not meets_deadline(.99, 9, 10)
    assert not meets_deadline(1, 11, 10)


def test_frontier_uses_only_admitted_moves_and_total_source_sessions(
        tmp_path, monkeypatch):
    base = model(tmp_path)
    move = PlannedMove("a", "destination", "replay", 0, ("link",))
    monkeypatch.setattr(
        campaign, "plan",
        lambda *args, **kwargs: SimpleNamespace(moves=(move,)),
    )

    selected = admitted_moves("queue_haul", None, None, None, 0)
    attainment, completion = frontier_metrics(
        [1], 2, 10, base.case().power_curve, base.power_window_s
    )

    assert selected == (move,)
    assert 0 < attainment < 1
    assert completion == 1


def test_width8_contract_does_not_silently_serialize_destination(tmp_path):
    base = model(tmp_path, tp=1)
    context = base.case().replay.by_concurrency[1][0][0]
    serial = base.case().replay.rate(context, 1)
    profile = parallel_profile(base, 8, {"central": serial / 2})

    assert profile.max_destination_replays == 8
    assert profile.max_destination_kv_streams == 8
    assert all(
        curve.concurrency[-1] == 8
        for case in profile.cases.values()
        for curve in case.action_power_w.values()
    )
    assert all(set(case.replay.by_concurrency) == set(range(1, 9))
               for case in profile.cases.values())
    assert profile.case().replay.rate(context, 1) == serial
    assert 8 * profile.case().replay.rate(context, 8) == serial / 2


def test_replay_cap_uses_aggregate_episode_tokens(tmp_path):
    (tmp_path / "plan.json").write_text("""{
      "scenarios": [
        {"episode": 0, "policy": "control",
         "sessions": [{"initial_tokens": 40}, {"initial_tokens": 60}]},
        {"episode": 1, "policy": "control",
         "sessions": [{"initial_tokens": 80}, {"initial_tokens": 120}]}
      ]
    }""")
    (tmp_path / "policy_episodes.csv").write_text(
        "episode,policy,commit_100_s\n"
        "0,replay_only,10\n1,replay_only,20\n"
    )

    caps, count = measured_replay_caps(tmp_path)

    assert caps == {"central": 10, "faster": 10, "slower": 10}
    assert count == 2


def test_kv_cap_is_shared_without_changing_replay(tmp_path):
    base = parallel_profile(model(tmp_path), 8, {"central": 10})
    replay = base.case().replay.rate(10, 4)
    capped = shared_kv_profile(
        base, 5000, 4, {"central": {5000.0: 80}}
    )

    assert capped.case().kv_transfer.destination_bytes_per_s == 80
    assert capped.case().replay.rate(10, 4) == replay
    assert shared_kv_profile(
        base, 5000, 1, {"central": {5000.0: 80}}
    ).case().kv_transfer.destination_bytes_per_s \
        == base.case().kv_transfer.destination_bytes_per_s


def test_kv_cap_uses_sealed_bytes_and_correct_bandwidth_source(tmp_path):
    base, crossover = model(tmp_path), tmp_path / "crossover"
    crossover.mkdir()
    block = base.case().kv_transfer.block_tokens
    size = base.case().kv_transfer.block_bytes
    (tmp_path / "plan.json").write_text(json.dumps({"scenarios": [
        {"scenario_id": "w5", "policy": "kv_only", "bandwidth_mbps": 5000,
         "sessions": [{"initial_tokens": block}]},
        {"scenario_id": "w10", "policy": "kv_only", "bandwidth_mbps": 10000,
         "sessions": [{"initial_tokens": block}]},
    ]}))
    (tmp_path / "policy_episodes.csv").write_text(
        "scenario_id,policy,commit_100_s\n"
        "w5,kv_only,2\nw10,kv_only,1\n"
    )
    (crossover / "migrations.csv").write_text(
        "scenario_id,method,bandwidth_mbps,measured_kv_bytes\n"
        "x1,kv_transfer,1000,100\nx25,kv_transfer,2500,300\n"
    )
    (crossover / "scenarios.csv").write_text(
        "scenario_id,migration_s\nx1,2\nx25,3\n"
    )

    caps, counts = measured_kv_caps(tmp_path, crossover, base)

    assert caps["central"] == {
        1000.0: 50, 2500.0: 100,
        5000.0: size / 2, 10000.0: size,
    }
    assert counts == {"serial": 2, "width8": 2}


def test_queue_haul_replans_when_aggregate_bottleneck_changes(tmp_path):
    base = model(tmp_path, tp=1)
    context = max(
        int(base.case().replay.by_concurrency[1][0][0]),
        base.case().kv_transfer.block_tokens,
    )
    scenario, routes = _problem(
        base, [{"session_id": "a", "initial_tokens": context}], 1000, 100
    )
    replay = {"central": 1e9}
    kv = {"central": {1000.0: 1}}
    replay_plan = aggregate_planning_profile(base, 1000, replay, kv)
    kv_plan = aggregate_planning_profile(
        base, 1000, {"central": .001}, {"central": {1000.0: 1e9}}
    )

    assert _moves(
        "queue_haul", scenario, routes, replay_plan, 1
    )[0]["method"] == "replay"
    assert _moves(
        "queue_haul", scenario, routes, kv_plan, 1
    )[0]["method"] == "kv_transfer"
