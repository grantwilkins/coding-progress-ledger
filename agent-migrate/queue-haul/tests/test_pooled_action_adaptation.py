"""
Claim:
Action views select one Queue-Haul plan per case and pool case-normalized session
fractions, so cases with more sessions do not receive more weight.

Plausible wrong implementations:
- Pool raw action counts, overweighting cases with more sessions.
- Normalize by selected sessions and hide changes in how many sessions move.
- Swap an action column or omit sessions that remain at the source.
- Mix policies or duplicate a designed case at a requested-shed coordinate.
"""

import pytest

from plot_pooled_action_adaptation import at_fraction, pooled_composition


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
