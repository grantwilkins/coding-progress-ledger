"""
Claim:
Planners move whole sessions, use local node power, do not inspect sampled future
requests, and keep destination placement separate from selection.

Plausible wrong implementations:
- Give a planner sampled request times that are unavailable when it acts.
- Treat source and destination power as one shared budget.
- Round fractional sessions or stop halfway through a node-drain group.
- Place every selected session on the first destination.
"""

from dataclasses import replace

from planner import METHODS, plan
from simulate import (ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimRequest,
                      SimSession)
from test_execution_simulator import model


def problem(requests=(), limit=20, final="awake"):
    sessions = (
        SimSession("a", "s0", 10, 25, 0, 100, requests=requests, wake_probability=0.1),
        SimSession("b", "s1", 10, 25, 0, 100, wake_probability=0.1),
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
    profile = model(tmp_path)
    a = plan(problem(), profile, PATHS, "load_only")
    b = plan(problem((SimRequest(0, 10, 0),)), profile, PATHS, "load_only")
    assert [(m.session_id, m.method) for m in a.moves] == [
        (m.session_id, m.method) for m in b.moves
    ]
    assert a.profile_id == profile.profile_id


def test_destination_placement_balances_whole_sessions(tmp_path):
    result = plan(problem(limit=0), model(tmp_path), PATHS, "load_only")
    assert {move.destination_instance for move in result.moves} == {"t0", "t1"}
    assert {move.session_id for move in result.moves} == {"a", "b"}
    assert all(move.method in METHODS for move in result.moves)


def test_node_drain_counts_sleep_only_after_the_whole_node_is_selected(tmp_path):
    scenario = problem(limit=5, final="sleep")
    shared = replace(scenario, nodes=(
        PowerNode("n0", 2, True), PowerNode("d0", 1, False), PowerNode("d1", 1, False),
    ), instances=(
        ServingInstance("s0", ("n0",)), ServingInstance("s1", ("n0",)),
        ServingInstance("t0", ("d0",)), ServingInstance("t1", ("d1",)),
    ))
    result = plan(shared, model(tmp_path), PATHS, "node_drain")
    assert {move.session_id for move in result.moves} == {"a", "b"}
    assert result.feasible


def test_rounded_lp_returns_only_whole_moves(tmp_path):
    result = plan(problem(), model(tmp_path), PATHS, "rounded_lp")
    assert result.moves
    assert len({move.session_id for move in result.moves}) == len(result.moves)
    assert all(isinstance(move.order, int) for move in result.moves)


def test_random_skips_sessions_that_cannot_finish_by_the_deadline(tmp_path):
    result = plan(replace(problem(), deadline_s=1), model(tmp_path), PATHS, "random")
    assert result.moves == ()
    assert not result.feasible
