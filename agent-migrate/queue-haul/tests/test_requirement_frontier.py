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
"""

from dataclasses import replace

import pytest

from destination import MigrationComponents
from requirement_frontier import (
    RequirementAction, _solve, requirement_frontier, sweep_frontier,
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
    assert one.destination_kv_blocks == 3
    assert one.destination_kv_tokens == 48
    assert one.wan_bytes == 101
    assert one.actions[1].route_bytes == 100  # one sealed 10-token transfer block
    conserved = (
        "actions", "destination_service_work", "destination_kv_blocks",
        "destination_kv_tokens", "replay_migration_slot_s",
        "kv_migration_slot_s", "wan_bytes", "source_stream_occupancy_s",
        "method_mix",
    )
    assert all(getattr(one, field) == getattr(two, field) for field in conserved)
    assert one.makespan_lower_bound_s > two.makespan_lower_bound_s


def test_rtt_is_one_additive_term_not_a_throughput_change(tmp_path):
    profile, scenario, q = model(tmp_path, switch=0, tp=1), problem(), destination_type()

    zero = requirement_frontier(scenario, profile, q, 10, 100, 0, 1)
    delayed = requirement_frontier(scenario, profile, q, 10, 100, .25, 1)

    assert delayed.wan_bytes == zero.wan_bytes
    assert delayed.actions[0].duration_s == pytest.approx(
        zero.actions[0].duration_s + .25
    )
