"""
Claim:
The requirement solver jointly chooses one replay or KV action per session,
meets the source-power target when possible, and reports raw destination and
transport requirements without inventing destination capacity.

Plausible wrong implementations:
- Choose one migration method globally or select both methods for one session.
- Optimize bytes instead of migration work after satisfying the power target.
- Round aggregate resident KV instead of rounding each session to paging blocks.
- Treat RTT as a bandwidth penalty or omit its one fixed per-action term.
- Change conserved resource totals when only source-stream concurrency changes.
- Hide replay reconstruction inside total duration instead of reporting it.
"""

from dataclasses import replace

import pytest

from destination import MigrationComponents
from requirement_frontier import (
    RequirementAction, _actions, _baseline, _greedy, _solve, requirement_frontier,
    sweep_frontier,
)
from test_execution_simulator import model
from test_planner import problem
from test_pool_planner import architecture


def action(session, method, gain, duration):
    return RequirementAction(
        session, session, method, gain, duration, 1, (1, 0), 1,
    )


def destination_type():
    q = architecture(block=16).types[0]
    components = MigrationComponents((1, 1000), (1, 1000), "hand")
    return replace(q, migration={
        "replay": components, "kv_transfer": components,
    })


def test_exact_solver_jointly_selects_replay_and_kv():
    actions = (
        action("a", "replay", 6, 1),
        action("a", "kv_transfer", 6, 4),
        action("b", "replay", 4, 100),
        action("b", "kv_transfer", 4, 2),
    )

    selected, maximum = _solve(actions, 10, 200, 1, 100)
    infeasible, _ = _solve(actions, 11, 200, 1, 100)

    assert selected == (0, 3)
    assert infeasible == selected
    assert maximum == 10


def test_source_streams_are_indivisible_deadline_bins():
    actions = tuple(
        replace(action(str(i), "replay", 1, 6), source_instance="source")
        for i in range(3)
    )

    selected, maximum = _solve(actions, 3, 10, 2, 100)

    assert len(selected) == 2
    assert maximum == 2


def test_solver_minimizes_work_after_crossing_target():
    actions = (
        action("a", "replay", 6, 1),
        action("b", "kv_transfer", 5, 2),
        action("c", "kv_transfer", 5, 3),
    )

    selected, _ = _solve(actions, 10, 10, 1, 100)

    assert selected == (0, 1)


def test_greedy_is_joint_and_stream_feasible_but_not_labeled_exact():
    actions = (
        action("a", "replay", 6, 1),
        action("a", "kv_transfer", 6, 4),
        replace(action("b", "replay", 4, 100), source_instance="a"),
        replace(action("b", "kv_transfer", 4, 2), source_instance="a"),
    )

    assert _greedy(actions, 10, 10, 1, 100) == _solve(
        actions, 10, 10, 1, 100,
    )[0] == (0, 3)
    crowded = tuple(
        replace(action(str(i), "replay", 1, 6), source_instance="source")
        for i in range(3)
    )
    assert len(_greedy(crowded, 3, 10, 2, 100)) == 2


def test_greedy_caps_overshoot_and_accounts_for_wan_and_source():
    overshoot = (
        action("large", "replay", 10, 5),
        action("precise", "kv_transfer", 6, 4),
    )
    assert _greedy(overshoot, 6, 10, 1, 100) == (1,)

    wan = (
        replace(action("a", "replay", 1, 1), route_bytes=101),
        replace(action("a", "kv_transfer", 1, 2), route_bytes=10),
    )
    assert _greedy(wan, 1, 10, 1, 10) == (1,)

    sources = (
        replace(action("a", "replay", 1, 6), source_instance="source-a"),
        replace(action("b", "replay", 1, 6), source_instance="source-a"),
        replace(action("c", "replay", 1, 6), source_instance="source-b"),
    )
    first = _greedy(sources, 3, 10, 1, 100)
    assert first == _greedy(sources, 3, 10, 1, 100)
    assert first == (0, 2)


def test_baselines_change_only_the_declared_ranking_policy():
    actions = (
        replace(action("a", "replay", 4, 2), route_bytes=1,
                service_work=(10, 0)),
        replace(action("a", "kv_transfer", 4, 1), route_bytes=10,
                service_work=(0, 0)),
        replace(action("b", "replay", 6, 3), route_bytes=1,
                service_work=(1, 0)),
        replace(action("b", "kv_transfer", 6, 4), route_bytes=10,
                service_work=(0, 0)),
    )

    assert _baseline(actions, 4, 10, 1, 100, "all_replay") == (0,)
    assert _baseline(actions, 4, 10, 1, 100, "all_kv") == (1,)
    assert _baseline(actions, 4, 10, 1, 100, "isolated_fastest") == (1,)
    assert _baseline(actions, 4, 10, 1, 100, "network_greedy") == (2,)
    assert _baseline(actions, 4, 10, 1, 100, "service_greedy") == (1,)
    assert _baseline(actions, 4, 10, 1, 100, "power_first") == (2,)


def test_frontier_reports_physical_requirements_and_stream_invariance(tmp_path):
    profile = model(tmp_path, switch=0, tp=1)
    base = problem()
    sessions = (
        replace(base.sessions[0], source_instance="s0", log_bytes=1),
        replace(base.sessions[1], source_instance="s0", context_tokens=17,
                log_bytes=1000),
    )
    scenario = replace(base, sessions=sessions)

    one, two = sweep_frontier(
        scenario, profile, destination_type(), (20,), (1, 2), 100, .5,
    )

    assert one.target_met and dict(one.method_mix) == {"replay": 1, "kv_transfer": 1}
    assert one.destination_service_work == pytest.approx((.5, 0))
    assert one.destination_transition_work[0] > 0
    assert one.destination_kv_blocks == 3
    assert one.destination_kv_tokens == 48
    assert one.wan_bytes == 101
    assert one.actions[1].route_bytes == 100  # one sealed 10-token transfer block
    conserved = (
        "actions", "destination_service_work", "destination_transition_work",
        "destination_kv_blocks",
        "destination_kv_tokens", "replay_migration_slot_s",
        "kv_migration_slot_s", "wan_bytes", "source_stream_occupancy_s",
        "method_mix",
    )
    assert all(getattr(one, field) == getattr(two, field) for field in conserved)
    assert one.makespan_lower_bound_s > two.makespan_lower_bound_s


def test_target_status_uses_exact_posthoc_power_gain(tmp_path):
    profile, base = model(tmp_path, switch=0, tp=1), problem(final="off")
    scenario = replace(
        base, sessions=tuple(replace(s, source_instance="s0") for s in base.sessions),
    )

    result = requirement_frontier(
        scenario, profile, destination_type(), 25, 100, 0, 2,
    )

    assert result.maximum_modeled_source_power_gain_w == 20
    assert result.achieved_source_power_reduction_w == 30
    assert result.target_met


def test_destination_capacity_does_not_gate_requirements(tmp_path):
    profile, scenario, q = model(tmp_path, switch=0, tp=1), problem(), destination_type()
    scenario = replace(scenario, nodes=scenario.nodes[:2], instances=scenario.instances[:2])
    tiny = replace(
        q, bounds={mode: (1e-9,) for mode in ("normal", "emergency", "stable")},
        kv_capacity_tokens=1,
    )

    result = requirement_frontier(scenario, profile, tiny, 20, 100, 0, 2)

    assert result.target_met
    assert result.destination_service_work[0] > tiny.bounds["stable"][0]
    assert result.destination_kv_tokens > tiny.kv_capacity_tokens


def test_public_greedy_result_disclaims_optimality(tmp_path):
    result = requirement_frontier(
        problem(), model(tmp_path, switch=0, tp=1), destination_type(),
        20, 100, 0, 2, solver_mode="greedy",
    )

    assert result.target_met
    assert result.solver_mode == "greedy"
    assert result.solver_status == "approximate_target_met"
    assert result.solver_mip_gap is None
    assert result.maximum_modeled_source_power_gain_w is None


def test_requirement_path_keeps_kv_destination_ingest_floor(tmp_path):
    profile = model(tmp_path, switch=0, destination_rate=50, tp=1)
    scenario = replace(problem(), sessions=(problem().sessions[0],))

    actions = _actions(scenario, profile, destination_type(), 100, 0, 9, "central")

    kv = next(action for action in actions if action.method == "kv_transfer")
    assert kv.route_bytes == 100
    assert kv.duration_s == pytest.approx(2)


def test_rtt_is_one_additive_term_not_a_throughput_change(tmp_path):
    profile, scenario, q = model(tmp_path, switch=0, tp=1), problem(), destination_type()

    zero = requirement_frontier(scenario, profile, q, 10, 100, 0, 1)
    delayed = requirement_frontier(scenario, profile, q, 10, 100, .25, 1)

    assert delayed.wan_bytes == zero.wan_bytes
    assert delayed.actions[0].duration_s == pytest.approx(
        zero.actions[0].duration_s + .25
    )
