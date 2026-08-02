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
- Keep a hidden width-eight endpoint cap in the full-width profile.
- Scale session count in metadata without expanding the simulated episode.
- Replace fluid link sharing with one serialized transfer per session.
- Append cleanup migrations that were outside the deadline-admitted set.
- Normalize attained shed by admitted sessions instead of all source sessions.
- Resample workloads by policy/budget or count repeated plan cells as variation.
- Aggregate raw completion points into a median or one point per budget.
- Run coupled greedy without its pool architecture or discard its pool assignment.
"""

from types import SimpleNamespace

import simulated_pareto_campaign as campaign
from simulated_pareto_campaign import (
    admitted_moves, context_evidence,
    coupled_architecture,
    aggregate_profile, expand_moves, expand_sessions, fluid_profile, frontier_metrics,
    full_attainment_cdf,
    meets_deadline, pareto_flags, policy_coordinates, workload_grid,
)
from policy_hardware_campaign import _problem
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


def test_coupled_greedy_uses_the_single_destination_pool(tmp_path):
    base = model(tmp_path, tp=1)
    context = max(
        int(base.case().replay.by_concurrency[1][0][0]),
        base.case().kv_transfer.block_tokens,
    )
    scenario, routes = _problem(
        base, [{"session_id": "a", "initial_tokens": context}], 1000, 100
    )
    destination = coupled_architecture(base)

    moves = admitted_moves(
        "greedy_coupled", scenario, routes, base, 0, destination,
    )

    assert len(moves) == 1
    assert moves[0].destination_pool == "coupled-pool"
    assert moves[0].destination_instance == "destination"


def test_workload_grid_deduplicates_samples_and_crosses_same_bandwidths():
    def control(sample, profile, bandwidth, tokens):
        return {
            "policy": "control", "sample_id": sample,
            "context_profile": profile, "bandwidth_mbps": bandwidth,
            "sessions": [{"initial_tokens": tokens}],
        }

    fixed = {"scenarios": [
        control("fixed-a", "tiny", 1000, 2048),
        control("fixed-b", "tiny", 2500, 2048),
    ]}
    hardware = {"scenarios": [
        control("mixed", "coding", 5000, 14080),
        control("mixed", "coding", 10000, 14080),
    ]}

    grid = workload_grid(fixed, hardware)

    assert [(source, row["sessions"][0]["initial_tokens"], bandwidth)
            for source, row, bandwidth in grid] == [
        ("fixed_anchor", 2048, 1000),
        ("fixed_anchor", 2048, 2500),
        ("measured_workload_mix", 14080, 1000),
        ("measured_workload_mix", 14080, 2500),
    ]


def test_raw_plot_coordinates_keep_every_completion():
    rows = [
        {"policy": "queue_haul", "power_attainment_fraction": .5,
         "completion_s": 7, "completion_budget_ratio": .7},
        {"policy": "queue_haul", "power_attainment_fraction": .8,
         "completion_s": 11, "completion_budget_ratio": .55},
        {"policy": "greedy", "power_attainment_fraction": 1,
         "completion_s": 3, "completion_budget_ratio": .1},
    ]

    assert policy_coordinates(rows, "queue_haul", False) == ([50, 80], [7, 11])
    assert policy_coordinates(rows, "queue_haul", True) == ([50, 80], [.7, .55])


def test_fluid_profile_supports_full_width_without_aggregate_cap(tmp_path):
    base = model(tmp_path, tp=1)
    context = base.case().replay.by_concurrency[1][0][0]
    serial = base.case().replay.rate(context, 1)
    profile = fluid_profile(base, 32, 32 * int(context))

    assert profile.max_destination_replays == 32
    assert profile.max_destination_kv_streams == 32
    assert profile.kv_capacity_tokens >= 32 * context
    assert all(
        curve.concurrency[-1] == 32
        for case in profile.cases.values()
        for curve in case.action_power_w.values()
    )
    assert all(set(case.replay.by_concurrency) == set(range(1, 33))
               for case in profile.cases.values())
    assert profile.case().replay.rate(context, 1) == serial
    assert profile.case().replay.rate(context, 32) == serial


def test_aggregate_profile_preserves_replicated_fluid_work(tmp_path):
    base = model(tmp_path, tp=1)
    full = fluid_profile(base, 32, 32 * 100)
    grouped = aggregate_profile(base, base, 8, 4, 8 * 100)

    assert grouped.max_destination_replays == 8
    assert grouped.case().kv_transfer.destination_bytes_per_s \
        == base.case().kv_transfer.destination_bytes_per_s / 4
    assert grouped.case().action_power_w["replay"].source_w[-1] \
        == full.case().action_power_w["replay"].source_w[-1]


def test_expand_sessions_repeats_templates_at_requested_width():
    base = {
        "sample_id": "sample",
        "sessions": [
            {"job_class": "a", "initial_tokens": 2},
            {"job_class": "b", "initial_tokens": 4},
        ],
    }

    expanded = expand_sessions(base, 5)

    assert len(expanded) == 5
    assert [row["initial_tokens"] for row in expanded] == [2, 4, 2, 4, 2]
    assert [row["session_id"] for row in expanded] == [
        "sample-0", "sample-1", "sample-2", "sample-3", "sample-4",
    ]


def test_expand_moves_replicates_template_admissions():
    base = {
        "sample_id": "sample",
        "sessions": [{"job_class": "a", "initial_tokens": 2},
                     {"job_class": "b", "initial_tokens": 4}],
    }
    template = expand_sessions(base, 2)
    full = expand_sessions(base, 6)
    moves = (PlannedMove("sample-1", "destination", "replay", 0, ("link",)),)

    expanded = expand_moves(moves, template, full)

    assert [move.session_id for move in expanded] == [
        "sample-1", "sample-3", "sample-5",
    ]
    assert [move.order for move in expanded] == [0, 1, 2]
