"""
Claim:
The timing sweep gives every solver the same fixed 100k-session scenario and
interprets 1--3600 seconds as migration time before the trailing power window.
Rows report the chosen action partition and simulated deadline outcome.

Plausible wrong implementations:
- Treat the migration budget as the deadline and silently lose the power window.
- Rebuild or mutate the workload between time points.
- Count replay and KV actions inconsistently with selected sessions.
- Report planned power instead of simulated deadline-window power.
- Mix random trials from different migration budgets in uncertainty bounds.
"""

from types import SimpleNamespace

import pytest

from simulate import ExecutionScenario, NetworkLink, PlannedMove, PowerNode, ServingInstance
from time_scaling_experiment import bounds, summarize, timed_scenario


def scenario():
    return ExecutionScenario(
        5, 5, 60, "awake", 0, (PowerNode("n"),),
        (ServingInstance("i", ("n",)),), (), (NetworkLink("l", 1),),
    )


def test_timed_scenario_adds_the_power_window_without_rebuilding_inputs():
    base = scenario()
    timed = timed_scenario(base, migration_s=1, power_window_s=5)

    assert timed.deadline_s == timed.end_s == 6
    assert timed.nodes is base.nodes
    assert timed.instances is base.instances
    assert timed.sessions is base.sessions
    assert timed.links is base.links


def test_summary_partitions_actions_and_uses_simulated_deadline_power():
    moves = (
        PlannedMove("a", "d", "replay", 0, ("l",)),
        PlannedMove("b", "d", "kv_transfer", 1, ("l",)),
    )
    plan = SimpleNamespace(
        solver="lp", seed=3, moves=moves, initial_source_power_w=100,
        planned_source_power_w=50, feasible=False, solve_s=2,
    )
    result = SimpleNamespace(
        sessions=(SimpleNamespace(committed_s=.5), SimpleNamespace(committed_s=7)),
        modeled_source_power_at_deadline_w=60,
    )
    row = summarize(scenario(), plan, result, migration_s=1, minimum_power_w=20,
                    execute_s=3)

    assert row["planned_moves"] == row["replay_moves"] + row["kv_moves"] == 2
    assert row["moves_completed_by_budget"] == 1
    assert row["modeled_source_drop_at_deadline_w"] == 40
    assert row["requested_power_fraction_achieved"] == pytest.approx(1)
    assert row["last_completed_migration_s"] == 7


def test_timing_bounds_group_replicates_by_migration_budget():
    rows = [
        {"migration_s": 10, "value": 2}, {"migration_s": 1, "value": 7},
        {"migration_s": 10, "value": 6}, {"migration_s": 1, "value": 7},
    ]

    assert bounds(rows, lambda row: row["value"]) == ([1, 10], [7, 4], [7, 2], [7, 6])
