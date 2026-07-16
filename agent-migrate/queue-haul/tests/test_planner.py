"""
Claim:
Planners move whole sessions, use local node power, do not inspect sampled future
requests, restrict methods by GPU residency, and keep placement separate from selection.

Plausible wrong implementations:
- Give a planner sampled request times that are unavailable when it acts.
- Treat source and destination power as one shared budget.
- Stop halfway through a node-drain group.
- Reorder sessions while reconstructing an already ordered node-drain group.
- Place every selected session on the first destination.
- Defer replay for an active session or transfer nonexistent KV for a cold session.
"""

from dataclasses import replace

from planner import METHODS, plan
from simulate import (ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimRequest,
                      SimSession)
from test_execution_simulator import model


def problem(requests=(), limit=20, final="awake"):
    sessions = (
        SimSession("a", "s0", 10, 25, 0, 100, requests=requests),
        SimSession("b", "s1", 10, 25, 0, 100),
    )
    return ExecutionScenario(
        10, 20, limit, final, 0,
        (PowerNode("n0", 1, True), PowerNode("n1", 1, True),
         PowerNode("d0", 1, False), PowerNode("d1", 1, False)),
        (ServingInstance("s0", ("n0",)), ServingInstance("s1", ("n1",)),
         ServingInstance("t0", ("d0",)), ServingInstance("t1", ("d1",))),
        sessions, (NetworkLink("wan", 100),),
    )


PATHS = {(source, dest): ("wan",) for source in ("s0", "s1") for dest in ("t0", "t1")}


def test_plan_does_not_read_sampled_future_requests(tmp_path):
    profile = model(tmp_path, tp=1)
    a = plan(problem(), profile, PATHS, "load_only")
    b = plan(problem((SimRequest(0, 10, 0),)), profile, PATHS, "load_only")
    assert [(m.session_id, m.method) for m in a.moves] == [
        (m.session_id, m.method) for m in b.moves
    ]
    assert a.profile_id == profile.profile_id


def test_destination_placement_balances_whole_sessions(tmp_path):
    result = plan(problem(limit=0), model(tmp_path, tp=1), PATHS, "load_only")
    assert {move.destination_instance for move in result.moves} == {"t0", "t1"}
    assert {move.session_id for move in result.moves} == {"a", "b"}
    assert all(move.method in METHODS for move in result.moves)

    def route(source, destination):
        return PATHS[source, destination]
    assert plan(problem(limit=0), model(tmp_path, tp=1), route, "load_only").moves == result.moves


def test_planner_only_transfers_active_kv_and_defers_cold_replay(tmp_path):
    topology = replace(problem(limit=0, final="off"), sessions=(
        SimSession("a", "s0", 10, 25, 0, 100),
        SimSession("b", "s1", 10, 0, 0, 100, wake_probability=1, state="cold"),
    ))
    result = plan(topology, model(tmp_path, tp=1), PATHS, "node_drain")

    assert {move.session_id: move.method for move in result.moves} == {
        "a": "kv_transfer", "b": "replay_on_request",
    }


def test_node_drain_counts_sleep_only_after_the_whole_node_is_selected(tmp_path):
    scenario = problem(limit=5, final="sleep")
    shared = replace(scenario, nodes=(
        PowerNode("n0", 2, True), PowerNode("d0", 1, False), PowerNode("d1", 1, False),
    ), instances=(
        ServingInstance("s0", ("n0",)), ServingInstance("s1", ("n0",)),
        ServingInstance("t0", ("d0",)), ServingInstance("t1", ("d1",)),
    ))
    result = plan(shared, model(tmp_path, tp=1), PATHS, "node_drain")
    assert {move.session_id for move in result.moves} == {"a", "b"}
    assert result.feasible


def test_node_drain_orders_groups_then_sessions_by_move_time(tmp_path):
    sessions = (
        SimSession("a", "s0", 10, 25, 0, 100),
        SimSession("b", "s1", 20, 25, 0, 100),
        SimSession("c", "s2", 30, 25, 0, 100),
    )
    scenario = ExecutionScenario(
        10, 20, 6, "sleep", 0,
        (
            PowerNode("n0", 2, True), PowerNode("n1", 1, True),
            PowerNode("d0", 1, False), PowerNode("d1", 1, False),
            PowerNode("d2", 1, False),
        ),
        (
            ServingInstance("s0", ("n0",)), ServingInstance("s1", ("n0",)),
            ServingInstance("s2", ("n1",)), ServingInstance("t0", ("d0",)),
            ServingInstance("t1", ("d1",)), ServingInstance("t2", ("d2",)),
        ),
        sessions, (NetworkLink("wan", 100),),
    )
    paths = {
        (source, destination): ("wan",)
        for source in ("s0", "s1", "s2")
        for destination in ("t0", "t1", "t2")
    }

    result = plan(scenario, model(tmp_path, tp=1), paths, "node_drain")

    assert [move.session_id for move in result.moves] == ["a", "b", "c"]


def test_random_skips_sessions_that_cannot_finish_by_the_deadline(tmp_path):
    result = plan(replace(problem(), deadline_s=1), model(tmp_path, tp=1), PATHS, "random")
    assert result.moves == ()
    assert not result.feasible


def test_collective_link_contention_can_make_a_plan_infeasible(tmp_path):
    result = plan(
        replace(problem(limit=0), deadline_s=2.5), model(tmp_path, tp=1), PATHS, "load_only"
    )
    assert len(result.moves) == 2
    assert not result.feasible
