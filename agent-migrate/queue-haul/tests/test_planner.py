"""
Claim:
Planners move whole sessions, use local node power, do not inspect sampled future
requests, restrict methods by GPU residency, and keep placement separate from selection.

Plausible wrong implementations:
- Give a planner sampled request times that are unavailable when it acts.
- Treat source and destination power as one shared budget.
- Choose each session's fastest method before enforcing shared replay and KV capacity.
- Apply a fleet-wide migration budget instead of one budget per source instance.
- Count external replay bytes against source egress.
- Use the deadline instead of the deadline minus the trailing power window.
- Ignore the resource budget while trying to drain a node.
- Reorder sessions while reconstructing an already ordered node-drain group.
- Place every selected session on the first destination.
- Defer replay for an active session or transfer nonexistent KV for a cold session.
- Admit active KV that fits source instances but overfills a destination.
- Add pipelined destination ingestion time to network transfer time.
"""

from dataclasses import replace

import pytest

from planner import METHODS, _duration, plan
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


def test_kv_duration_uses_the_slower_pipeline_stage(tmp_path):
    profile = model(tmp_path, switch=0, tp=1, destination_rate=50)
    session = SimSession("a", "s0", 10, 0, 0, 1)

    assert _duration(session, "kv_transfer", profile.case(), ("wan",), {"wan": 100}) \
        == pytest.approx(2)


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


def test_planner_rejects_destination_kv_overcommit(tmp_path):
    topology = replace(problem(limit=0), instances=problem().instances[:-1])
    paths = {(source, "t0"): ("wan",) for source in ("s0", "s1")}

    with pytest.raises(ValueError, match="destination compute or KV capacity"):
        plan(topology, model(tmp_path, tp=1, kv_capacity=15), paths, "load_only")


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


def test_node_drain_reserves_source_time_and_power_window(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100, parallel_moves=1,
        replay_rate={"1": [[1, 50], [1000, 50]], "2": [[1, 25], [1000, 25]]},
    )
    scenario = ExecutionScenario(
        4, 5, 0, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 1, False),
         SimSession("b", "s", 100, 10, 0, 1, False)),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "node_drain")

    assert [(move.session_id, move.method) for move in result.moves] == [("a", "replay")]
    assert result.planned_source_power_w > scenario.power_limit_w
    assert not result.feasible


def test_node_drain_uses_kv_when_shared_replay_time_is_full(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=500, parallel_moves=1,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    sessions = tuple(SimSession(str(i), f"s{i}", 100, 10, 0, 1, False)
                     for i in range(4))
    scenario = ExecutionScenario(
        4.2, 5, 40, "awake", 0,
        (PowerNode("n0", 2, True), PowerNode("n1", 2, True),
         PowerNode("d0", 1, False)),
        tuple(ServingInstance(f"s{i}", (f"n{i // 2}",)) for i in range(4))
        + (ServingInstance("d", ("d0",)),),
        sessions, (NetworkLink("wan", 10_000),),
    )
    paths = {(f"s{i}", "d"): ("wan",) for i in range(4)}

    result = plan(scenario, profile, paths, "node_drain")

    assert result.feasible
    assert [move.method for move in result.moves].count("replay") == 3
    assert [move.method for move in result.moves].count("kv_transfer") == 1


def test_node_drain_prefers_power_reduction_per_predicted_second(tmp_path):
    profile = model(tmp_path, switch=0, tp=1, destination_rate=10, parallel_moves=1)
    sessions = (
        SimSession("fast", "sf", 100, 25, 0, 1, False),
        SimSession("slow", "ss", 300, 25, 0, 1, False),
    )
    scenario = ExecutionScenario(
        10, 11, 35, "awake", 0,
        (PowerNode("nf", 1, True), PowerNode("ns", 1, True),
         PowerNode("df", 1, False), PowerNode("ds", 1, False)),
        (ServingInstance("sf", ("nf",)), ServingInstance("ss", ("ns",)),
         ServingInstance("tf", ("df",)), ServingInstance("ts", ("ds",))),
        sessions, (NetworkLink("wan", 10_000),),
    )
    paths = {(source, destination): ("wan",)
             for source in ("sf", "ss") for destination in ("tf", "ts")}

    result = plan(scenario, profile, paths, "node_drain")

    assert [move.session_id for move in result.moves] == ["fast"]


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


def test_lp_uses_kv_when_shared_replay_time_is_full(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=500, parallel_moves=1,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    sessions = tuple(SimSession(str(i), f"s{i}", 100, 10, 0, 1, False)
                     for i in range(4))
    scenario = ExecutionScenario(
        4.1, 5, 40, "awake", 0,
        (PowerNode("n0", 2, True), PowerNode("n1", 2, True),
         PowerNode("d0", 1, False)),
        tuple(ServingInstance(f"s{i}", (f"n{i // 2}",)) for i in range(4))
        + (ServingInstance("d", ("d0",)),),
        sessions, (NetworkLink("wan", 10_000),),
    )
    paths = {(f"s{i}", "d"): ("wan",) for i in range(4)}

    result = plan(scenario, profile, paths, "lp")

    assert result.feasible
    assert [move.method for move in result.moves].count("replay") == 3
    assert [move.method for move in result.moves].count("kv_transfer") == 1


def test_lp_enforces_each_source_instance_queue(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100, parallel_moves=1,
        replay_rate={"1": [[1, 50], [1000, 50]], "2": [[1, 25], [1000, 25]]},
    )
    scenario = ExecutionScenario(
        4, 5, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 1, False),
         SimSession("b", "s", 100, 10, 0, 1, False)),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "lp")

    assert len(result.moves) == 1
    assert result.planned_source_power_w > scenario.power_limit_w
    assert not result.feasible


def test_lp_external_replay_bypasses_source_egress(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100, parallel_moves=2,
        replay_rate={"1": [[1, 1000], [1000, 1000]], "2": [[1, 500], [1000, 500]]},
    )
    scenario = ExecutionScenario(
        2, 3, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 100, True),
         SimSession("b", "s", 100, 10, 0, 100, True)),
        (NetworkLink("source", 1), NetworkLink("destination", 1000)),
    )

    result = plan(
        scenario, profile, {("s", "t"): ("source", "destination")}, "lp"
    )

    assert result.feasible
    assert len(result.moves) == 2
    assert {move.method for move in result.moves} == {"replay"}


def test_lp_reserves_the_trailing_power_window(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=1, parallel_moves=1,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    scenario = ExecutionScenario(
        4, 5, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 301, 10, 0, 1, False),),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "lp")

    assert result.moves == ()
    assert not result.feasible
    assert plan(
        replace(scenario, deadline_s=1, end_s=1), profile,
        {("s", "t"): ("wan",)}, "lp",
    ).moves == ()
