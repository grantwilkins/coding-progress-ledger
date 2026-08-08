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
- Rank by summed work instead of the largest scarcity-weighted resource use.
- Admit a session after one of its resource capacities is exhausted.
- Place every selected session on the first destination.
- Defer replay for an active session or transfer nonexistent KV for a cold session.
- Admit active KV that fits source instances but overfills a destination.
- Add pipelined destination ingestion time to network transfer time.
- Merge distinct route-resource summaries while caching repeated routes.
- Require every migration before the power window instead of the migration deadline.
- Accept a late migration or a missed measured power target.
- Minimize peak resource use before migration work in the old Queue-Haul LP.
- Round every positive fraction after enough power reduction has been selected.
- Fail instead of maximizing achievable power reduction when the target is infeasible.
- Spend setup/completion time as transfer time or clamp an impossible KV pace.
- Omit the mandatory final KV catch-up and partial-tail reconstruction.
- Hold replay log bytes fixed while expected session state grows.
- Award an idle-node bonus while an unmovable session remains.
- Reject a valid tail-only KV move because its background byte rate is zero.
- Reintroduce power gain into the resource-only ranking.
- Evaluate migration methods that are illegal for a session's residency state.
- Rebuild an already-static expected scenario.
- Change seeded random methods or move order while batching random choices.
- Let a fixed-method baseline silently use the other migration mechanism.
- Choose one globally fastest method for the isolated-fastest baseline.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import planner
from planner import (
    METHODS, _duration, _expected_scenario, _greedy, _migration_resources,
    _required_kv_rate, _round_lp, _route_resources, _solve_lp, plan,
)
from power_model import ExpectedPower
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
        2 if final == "off" else None,
    )


PATHS = {(source, dest): ("wan",) for source in ("s0", "s1") for dest in ("t0", "t1")}


def test_lp_objective_variants_have_the_stated_priority():
    gains = np.ones(2)
    work = np.array([1.0, 2.0, 100.0, 100.0])
    valid = np.array([True, True, False, False])
    resources = csr_matrix([[1.0, 0.5, 0.0, 0.0]])

    old = _solve_lp("lp", gains, work, valid, resources, 1)
    peak = _solve_lp("lp_peak_first", gains, work, valid, resources, 1)
    switched = _solve_lp("lp_work_first", gains, work, valid, resources, 1)

    assert old[0] > 1 - 1e-5
    assert switched[0] > 1 - 1e-5
    assert peak[1] > 1 - 1e-5


def test_power_blind_lp_uses_uniform_pack_average_gains(monkeypatch, tmp_path):
    scenario = replace(problem(), sessions=(
        SimSession("a", "s0", 10, 5, 0, 100),
        SimSession("b", "s1", 10, 45, 0, 100),
    ))
    seen = {}
    original = planner._solve_lp

    def capture(solver, gains, *args, **kwargs):
        seen.update(solver=solver, gains=gains.copy())
        return original(solver, gains, *args, **kwargs)

    monkeypatch.setattr(planner, "_solve_lp", capture)
    plan(scenario, model(tmp_path, tp=1), PATHS, "lp_power_blind")

    expected = ExpectedPower(scenario, model(tmp_path, tp=1)).drain_gain(("a", "b")) / 2
    assert seen["solver"] == "lp_work_first"
    assert seen["gains"] == pytest.approx([expected, expected])


def test_lexicographic_lp_retains_last_feasible_stage(monkeypatch):
    original, calls = planner.cp.Problem.solve, 0

    def solve(problem, *args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(problem, *args, **kwargs)
        if calls == 2:
            problem._status = planner.cp.INFEASIBLE_INACCURATE
        return value

    monkeypatch.setattr(planner.cp.Problem, "solve", solve)
    values = _solve_lp(
        "lp_work_first", np.ones(2), np.ones(4), np.ones(4, bool),
        csr_matrix([[1.0, 0.5, 1.0, 0.5]]), 1,
    )

    assert calls == 2
    assert values is not None


def test_old_lp_maximizes_power_when_target_is_infeasible():
    values = _solve_lp(
        "lp", np.array([2.0, 1.0]), np.ones(4), np.ones(4, bool),
        csr_matrix([[1.0, 1.0, 1.0, 1.0]]), 4,
    )

    assert values[0] + values[2] > 1 - 1e-6
    assert values[1] + values[3] < 1e-6


def test_lp_rounding_stops_after_reaching_the_power_target():
    chosen, _ = _round_lp(
        np.array([0.6, 0.6, 0.0, 0.0]), np.ones(4, bool),
        csr_matrix((0, 4)), np.ones(2), np.ones(4), 1,
    )

    assert np.count_nonzero(chosen >= 0) == 1


def test_route_resource_cache_reuses_only_identical_path_sets():
    destinations = (ServingInstance("t0", ("d",)), ServingInstance("t1", ("d",)))
    routes = {
        ("same-a", "t0"): ("fast",), ("same-a", "t1"): ("fast",),
        ("same-b", "t0"): ("fast",), ("same-b", "t1"): ("fast",),
        ("different", "t0"): ("slow",), ("different", "t1"): ("slow",),
    }
    cache = {}

    same_a = _route_resources("same-a", destinations, routes, {"fast": 100, "slow": 10},
                              cache)
    same_b = _route_resources("same-b", destinations, routes, {"fast": 100, "slow": 10},
                              cache)
    different = _route_resources(
        "different", destinations, routes, {"fast": 100, "slow": 10}, cache,
    )

    assert same_a == same_b
    assert same_a[0][2] == 100
    assert different[0][2] == 10
    assert len(cache) == 2


def test_equivalent_destination_routes_are_evaluated_once():
    destinations = (
        ServingInstance("t0", ("d",)), ServingInstance("t1", ("d",))
    )
    calls = []

    def routes(source, destination):
        calls.append((source, destination))
        return ("wan",)

    routes.destinations_equivalent = True
    _route_resources("s0", destinations, routes, {"wan": 100}, {})

    assert calls == [("s0", "t0")]


def test_kv_duration_uses_the_slower_pipeline_stage(tmp_path):
    profile = model(tmp_path, switch=0, tp=1, destination_rate=50)
    session = SimSession("a", "s0", 10, 0, 0, 1)

    assert _duration(session, "kv_transfer", profile.case(), ("wan",), {"wan": 100}) \
        == pytest.approx(2)


def test_kv_duration_includes_final_tail_and_catch_up(tmp_path):
    profile = model(tmp_path, switch=0, tp=1, destination_rate=100)
    case = replace(
        profile.case(),
        kv_transfer=replace(
            profile.case().kv_transfer, catch_up_fixed_s=.4, tail_replay_tps=10,
        ),
    )
    session = SimSession("a", "s0", 11, 0, 0, 1)

    assert _duration(session, "kv_transfer", case, ("wan",), {"wan": 100}) \
        == pytest.approx(1.5)


def test_required_kv_rate_reserves_fixed_completion_and_rejects_overload(tmp_path):
    case = model(tmp_path, switch=0, tp=1).case()
    case = replace(
        case,
        kv_transfer=replace(case.kv_transfer, initial_completion_s=2),
    )
    session = SimSession("a", "s0", 10, 0, 0, 1)

    assert _required_kv_rate(session, case, 10, 0, 100) == pytest.approx(12.5)
    with pytest.raises(ValueError, match="physical capacity"):
        _required_kv_rate(session, case, 10, 0, 12)


def test_expected_prediction_materializes_growth_at_quiescence():
    scenario = replace(
        problem(), sessions=(SimSession(
            "a", "s0", 10, 0, 0, 1,
            requests=(SimRequest(1, 99, 0),),
            expected_growth_tokens_per_s=2,
        ),),
    )
    expected = _expected_scenario(
        scenario,
        (planner.PlannedMove(
            "a", "t0", "kv_transfer", 0, ("wan",), quiesce_s=3,
        ),),
    )

    assert expected.sessions[0].context_tokens == 16
    assert expected.sessions[0].log_bytes == 2
    assert expected.sessions[0].requests == ()
    assert expected.sessions[0].expected_growth_tokens_per_s == 0


def test_expected_prediction_reuses_a_static_scenario():
    scenario = problem()
    move = planner.PlannedMove("a", "t0", "replay", 0, ("wan",))

    assert _expected_scenario(scenario, (move,)) is scenario


def test_planner_only_evaluates_methods_allowed_by_session_state(tmp_path, monkeypatch):
    original = planner._duration

    def allowed(session, method, *args, **kwargs):
        assert method in planner.MOVE_METHODS_BY_STATE[session.state]
        return original(session, method, *args, **kwargs)

    monkeypatch.setattr(planner, "_duration", allowed)

    plan(problem(), model(tmp_path, tp=1), PATHS, "greedy")


def test_random_batching_preserves_scalar_seed_sequence(tmp_path):
    scenario = problem(limit=0)
    rng = np.random.default_rng(7)
    order = rng.permutation(2)
    choices = [rng.choice(2) for _ in scenario.sessions]

    result = plan(scenario, model(tmp_path, tp=1), PATHS, "random", seed=7)

    assert [(move.session_id, move.method) for move in result.moves] == [
        (scenario.sessions[j].session_id, METHODS[choices[j]]) for j in order
    ]


def test_replay_duration_scales_durable_log_with_expected_growth(tmp_path):
    session = SimSession(
        "a", "s0", 10, 0, 0, 100, expected_growth_tokens_per_s=10,
    )

    assert _duration(
        session, "replay", model(tmp_path, switch=0, tp=1).case(),
        ("wan",), {"wan": 100}, 1,
    ) == pytest.approx(2.2)


def test_node_gain_does_not_idle_a_node_with_an_unmovable_session(tmp_path):
    scenario = replace(problem(final="sleep"), nodes=(
        PowerNode("n", 2, True), PowerNode("d0", 1, False),
        PowerNode("d1", 1, False),
    ), instances=(
        ServingInstance("s0", ("n",)), ServingInstance("s1", ("n",)),
        ServingInstance("t0", ("d0",)), ServingInstance("t1", ("d1",)),
    ), sessions=(
        SimSession("a", "s0", 10, 25, 0, 100),
        SimSession("b", "s1", 10, 25, 0, 100, movable=False),
    ))
    power = ExpectedPower(scenario, model(tmp_path, tp=1))

    assert power.drain_gain(["a"]) == pytest.approx(power.marginal("a"))


def test_tail_only_kv_move_needs_no_background_rate(tmp_path):
    scenario = ExecutionScenario(
        10, 10, 0, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 5, 25, 0, 10_000),),
        (NetworkLink("wan", 100),),
    )

    result = plan(
        scenario, model(tmp_path, switch=0, tp=1),
        {("s", "t"): ("wan",)}, "greedy",
    )

    assert result.moves[0].method == "kv_transfer"
    assert result.moves[0].rate_limit_bytes_per_s is None


def test_plan_does_not_read_sampled_future_requests(tmp_path):
    profile = model(tmp_path, tp=1)
    a = plan(problem(), profile, PATHS, "greedy")
    b = plan(problem((SimRequest(0, 10, 0),)), profile, PATHS, "greedy")
    assert [(m.session_id, m.method) for m in a.moves] == [
        (m.session_id, m.method) for m in b.moves
    ]
    assert a.profile_id == profile.profile_id


def test_destination_placement_balances_whole_sessions(tmp_path):
    result = plan(problem(limit=0), model(tmp_path, tp=1), PATHS, "greedy")
    assert {move.destination_instance for move in result.moves} == {"t0", "t1"}
    assert {move.session_id for move in result.moves} == {"a", "b"}
    assert all(move.method in METHODS for move in result.moves)

    def route(source, destination):
        return PATHS[source, destination]
    assert plan(problem(limit=0), model(tmp_path, tp=1), route, "greedy").moves == result.moves


def test_planner_only_transfers_active_kv_and_defers_cold_replay(tmp_path):
    topology = replace(problem(limit=0, final="off"), sessions=(
        SimSession("a", "s0", 10, 25, 0, 100),
        SimSession("b", "s1", 10, 0, 0, 100, wake_probability=1, state="cold"),
    ))
    result = plan(topology, model(tmp_path, tp=1), PATHS, "random")

    assert {move.session_id: move.method for move in result.moves} == {
        "a": "kv_transfer", "b": "replay_on_request",
    }


def test_greedy_prevents_destination_kv_overcommit(tmp_path):
    topology = replace(problem(limit=0), instances=problem().instances[:-1])
    paths = {(source, "t0"): ("wan",) for source in ("s0", "s1")}

    result = plan(topology, model(tmp_path, tp=1, kv_capacity=15), paths, "greedy")

    assert len(result.moves) == 1
    assert not result.feasible


class UnlimitedPower:
    def __init__(self):
        self.removed = []

    def power(self, _local):
        return 1

    def remove(self, session):
        self.removed.append(session)


def test_greedy_uses_bottleneck_pressure_and_preserves_capacity():
    sessions = [SimpleNamespace(session_id=str(j)) for j in range(3)]
    resources = csr_matrix([
        [0.7, 0.4, 0.0, 0, 0, 0],
        [0.7, 0.0, 0.4, 0, 0, 0],
    ])

    selected, chosen, usage = _greedy(
        sessions, np.ones(3), np.array([True] * 3 + [False] * 3), resources,
        UnlimitedPower(), 0,
    )

    assert selected == [1, 2]
    assert chosen.tolist() == [-1, 0, 0]
    assert np.all(usage <= 1)


def test_greedy_picks_the_lower_pressure_action():
    selected, chosen, _ = _greedy(
        [SimpleNamespace(session_id="a")], np.ones(1), np.ones(2, bool),
        csr_matrix([[0.6, 0.2], [0.0, 0.2]]), UnlimitedPower(), 0,
    )

    assert selected == [0]
    assert chosen.tolist() == [1]


def test_greedy_prioritizes_power_gain_over_small_resource_use():
    sessions = [SimpleNamespace(session_id=str(j)) for j in range(11)]
    resources = csr_matrix([np.r_[np.repeat(0.1, 10), 0.2, np.zeros(11)]])
    gains = np.r_[np.ones(10), 100]
    valid = np.r_[np.ones(11, bool), np.zeros(11, bool)]

    selected, _, usage = _greedy(
        sessions, gains, valid, resources, UnlimitedPower(), 0,
    )
    lp = _solve_lp("lp", gains, np.ones(22), valid, resources, 100)

    assert selected[0] == 10
    assert lp[10] > 1 - 1e-6
    assert len(selected) == 9
    assert usage[0] == pytest.approx(1)


def test_greedy_prices_demand_from_one_action_per_session():
    resources = csr_matrix([
        [0.5, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.5, 0.0, 0.0],
    ])

    selected, chosen, _ = _greedy(
        [SimpleNamespace(session_id=str(j)) for j in range(3)],
        np.array([10, 9, 9]), np.array([True, True, True, True, False, False]),
        resources, UnlimitedPower(), 0,
    )

    assert selected[0] == 0
    assert chosen[0] == 1


def test_greedy_reserves_replay_time_and_power_window(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100,
        replay_rate={"1": [[1, 50], [1000, 50]], "2": [[1, 25], [1000, 25]]},
    )
    scenario = ExecutionScenario(
        4, 5, 0, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 1),
         SimSession("b", "s", 100, 10, 0, 1)),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "greedy")

    assert [(move.session_id, move.method) for move in result.moves] == [("a", "replay")]
    assert result.planned_source_power_w > scenario.power_limit_w
    assert not result.feasible


def test_greedy_uses_kv_when_shared_replay_time_is_full(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=500,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    sessions = tuple(SimSession(str(i), f"s{i}", 100, 10, 0, 1)
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

    result = plan(scenario, profile, paths, "greedy")

    assert result.feasible
    assert [move.method for move in result.moves].count("replay") == 3
    assert [move.method for move in result.moves].count("kv_transfer") == 1


def test_fixed_and_isolated_baselines_enforce_their_method_contract(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100,
        replay_rate={
            "1": [[1, 1], [10, 1], [100, 1000], [1000, 1000]],
            "2": [[1, .5], [10, .5], [100, 500], [1000, 500]],
        },
    )
    scenario = replace(
        problem(limit=0),
        sessions=(
            SimSession("a", "s0", 10, 25, 0, 1),
            SimSession("b", "s1", 100, 25, 0, 1),
        ),
        links=(NetworkLink("wan", 100),),
        deadline_s=20,
    )

    replay = plan(scenario, profile, PATHS, "replay_only")
    kv = plan(scenario, profile, PATHS, "kv_only")
    isolated = plan(scenario, profile, PATHS, "isolated_fastest")

    assert {move.method for move in replay.moves} == {"replay"}
    assert {move.method for move in kv.moves} == {"kv_transfer"}
    assert {move.session_id: move.method for move in isolated.moves} == {
        "a": "kv_transfer", "b": "replay",
    }


def test_random_skips_sessions_that_cannot_finish_by_the_deadline(tmp_path):
    result = plan(replace(problem(), deadline_s=1), model(tmp_path, tp=1), PATHS, "random")
    assert result.moves == ()
    assert not result.feasible


def test_collective_link_contention_can_make_a_plan_infeasible(tmp_path):
    result = plan(
        replace(problem(limit=0), deadline_s=2.5), model(tmp_path, tp=1), PATHS, "random"
    )
    assert len(result.moves) == 2
    assert not result.feasible


def test_lp_uses_kv_when_shared_replay_time_is_full(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=500,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    sessions = tuple(SimSession(str(i), f"s{i}", 100, 10, 0, 1)
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


def test_lp_enforces_destination_replay_capacity(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100,
        replay_rate={"1": [[1, 50], [1000, 50]], "2": [[1, 25], [1000, 25]]},
    )
    scenario = ExecutionScenario(
        4, 5, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 1),
         SimSession("b", "s", 100, 10, 0, 1)),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "lp")

    assert len(result.moves) == 1
    assert result.planned_source_power_w > scenario.power_limit_w
    assert not result.feasible


def test_lp_source_local_replay_uses_source_egress(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=100,
        replay_rate={"1": [[1, 1000], [1000, 1000]], "2": [[1, 500], [1000, 500]]},
    )
    scenario = ExecutionScenario(
        2, 3, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 100, 10, 0, 100),
         SimSession("b", "s", 100, 10, 0, 100)),
        (NetworkLink("source", 1), NetworkLink("destination", 1000)),
    )

    result = plan(
        scenario, profile, {("s", "t"): ("source", "destination")}, "lp"
    )

    assert not result.feasible
    assert result.moves == ()


def test_destination_capacity_reserves_expected_context_growth(tmp_path):
    session = SimSession(
        "a", "s0", 10, 25, 0, 100, expected_growth_tokens_per_s=1,
    )
    topology = replace(problem(limit=0), sessions=(session,))

    with pytest.raises(ValueError, match="destination compute or KV capacity"):
        plan(topology, model(tmp_path, tp=1, kv_capacity=15), PATHS, "greedy")


def test_lp_reserves_the_trailing_power_window(tmp_path):
    profile = model(
        tmp_path, switch=0, tp=1, destination_rate=1,
        replay_rate={"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]},
    )
    scenario = ExecutionScenario(
        4, 5, 10, "awake", 0,
        (PowerNode("n", 1, True), PowerNode("d", 1, False)),
        (ServingInstance("s", ("n",)), ServingInstance("t", ("d",))),
        (SimSession("a", "s", 301, 10, 0, 1),),
        (NetworkLink("wan", 10_000),),
    )

    result = plan(scenario, profile, {("s", "t"): ("wan",)}, "lp")

    assert result.moves == ()
    assert not result.feasible
    assert plan(
        replace(scenario, deadline_s=1, end_s=1), profile,
        {("s", "t"): ("wan",)}, "lp",
    ).moves == ()


@pytest.mark.parametrize("solver", ("lp", "greedy"))
@pytest.mark.parametrize(("commit_s", "power_met", "feasible"), (
    (10, True, True),
    (10.000001, True, False),
    (9, False, False),
))
def test_planner_separates_migration_deadline_from_power_window(
        tmp_path, monkeypatch, solver, commit_s, power_met, feasible):
    def predict(_scenario, _profile, moves, _case_id):
        return SimpleNamespace(
            deadline_met=power_met,
            sessions=tuple(SimpleNamespace(committed_s=commit_s) for _ in moves),
            modeled_source_power_at_deadline_w=0,
        )

    monkeypatch.setattr(planner, "predict", predict)
    result = plan(problem(), model(tmp_path, tp=1), PATHS, solver)

    assert result.moves
    assert result.feasible is feasible
