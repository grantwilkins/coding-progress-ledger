"""
Claim:
Action views select one Queue-Haul plan per case. Controlled bars report the
total replay/KV composition of selected actions from matched 28-session packs
under exactly one HBM, bandwidth, or prefill bottleneck.

Plausible wrong implementations:
- Pool raw action counts, overweighting cases with more sessions.
- Swap a regional action column or fail to combine both regions.
- Mix policies or duplicate a designed case at a requested-shed coordinate.
- Normalize controlled bars by all source sessions instead of selected actions.
- Include unaccounted selected sessions or a mismatched pack.
- Leave a second constraint bound in a nominally single-bottleneck case.
"""

import pytest

from plot_pooled_action_adaptation import (
    ACTION_MIX_CASES, ACTION_MIX_FIGSIZE, ACTION_MIX_LABEL_SIZE,
    ACTION_MIX_LEGEND_SIZE, ACTION_MIX_TICK_SIZE, at_fraction,
    controlled_action_mixes, pooled_composition, single_bottleneck_scenarios,
)


def row(case, sessions, counts, policy="queue_haul_lp", fraction=.5):
    actions = ("east_replay", "east_kv_transfer", "germany_replay",
               "germany_kv_transfer")
    return {"case_id": case, "policy": policy,
            "requested_fraction": fraction, "sessions": sessions,
            "selected_sessions": sum(counts),
            **dict(zip(actions, counts))}


def test_composition_equal_weights_cases_and_conserves_sessions():
    rows = [row("small", 2, (1, 0, 0, 0)),
            row("large", 8, (0, 2, 2, 0))]
    result = pooled_composition(rows)[0]

    assert result["east_replay"] == pytest.approx(.25)
    assert result["east_kv_transfer"] == pytest.approx(.125)
    assert result["germany_replay"] == pytest.approx(.125)
    assert result["germany_kv_transfer"] == 0
    assert result["not_moved"] == pytest.approx(.5)
    assert sum(result[action] for action in (
        "east_replay", "east_kv_transfer", "germany_replay",
        "germany_kv_transfer", "not_moved")) == pytest.approx(1)


def test_fraction_selection_rejects_duplicate_case_and_other_policy():
    chosen = row("a", 2, (1, 0, 0, 0))
    assert at_fraction([chosen, row("ignored", 2, (0, 1, 0, 0), "greedy")], .5) \
        == [chosen]
    with pytest.raises(RuntimeError, match="one Queue-Haul row per case"):
        at_fraction([chosen, chosen], .5)


def test_controlled_mix_normalizes_selected_actions_without_n(monkeypatch):
    monkeypatch.setattr("plot_pooled_action_adaptation.ACTION_MIX_CASES",
                        (("case", "Bandwidth"),))
    rows = [row("case", 28, (3, 1, 7, 4), fraction=2 / 3)]

    assert controlled_action_mixes(rows) == [{
        "bound_constraint": "Bandwidth",
        "replay": 10 / 15, "kv_transfer": 5 / 15,
    }]

    rows[0]["selected_sessions"] = 16
    with pytest.raises(RuntimeError, match="accounted 28-session pack"):
        controlled_action_mixes(rows)


def test_single_bottleneck_scenarios_release_the_other_two_constraints():
    released = {
        "background": {"east": [.25, 0], "germany": [.25, 0]},
        "kv_capacity_fraction": {"east": 1, "germany": 1},
        "bandwidth": "natural", "bandwidth_mbps": {"east": 2, "germany": 8},
    }
    bound = {
        **released,
        "background": {"east": [.25, .9], "germany": [.75, 0]},
        "kv_capacity_fraction": {"east": .1, "germany": 1},
        "bandwidth": "controlled", "bandwidth_mbps": {"east": 1, "germany": 3},
    }

    cases = single_bottleneck_scenarios(bound, released)
    assert cases["single/hbm"]["background"] == {
        "east": [.25, .9], "germany": [.25, 0]}
    assert cases["single/hbm"]["bandwidth"] == "natural"
    assert cases["single/hbm"]["kv_capacity_fraction"]["east"] == .1
    assert cases["single/bandwidth"]["background"] == released["background"]
    assert cases["single/bandwidth"]["bandwidth"] == "controlled"
    assert cases["single/bandwidth"]["kv_capacity_fraction"]["east"] == 1
    assert cases["single/prefill"]["background"] == {
        "east": [.25, 0], "germany": [.75, 0]}
    assert cases["single/prefill"]["kv_capacity_fraction"]["east"] == 1


def test_action_mix_uses_five_resource_states_with_bound_extremes_adjacent():
    assert len(ACTION_MIX_CASES) == 5
    assert ACTION_MIX_CASES[-2:] == (
        ("single/all-bound", "All bound"),
        ("single/none-bound", "None bound"),
    )
    assert all(case != "constraint/quota-30" for case, _ in ACTION_MIX_CASES)
    assert ACTION_MIX_FIGSIZE == (5.5, 3)
    assert (ACTION_MIX_TICK_SIZE, ACTION_MIX_LABEL_SIZE,
            ACTION_MIX_LEGEND_SIZE) == (11, 12, 10)
