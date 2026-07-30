"""
Claim:
Pool-aware planning selects at most one method/pool candidate per session, charges
resources to the exact pool and route, packs whole sessions on concrete replicas,
and distinguishes normal, emergency, and valid target-unmet outcomes. KV transfer
time is limited by the slower of route transfer and destination ingestion.
Temporary reconstruction debt is measured in replica-seconds and can recover
only from post-migration spare service.

Plausible wrong implementations:
- Borrow residual service or KV capacity across pools or replicas.
- Inflate greedy scarcity prices by counting every duplicate pool candidate.
- Use migration growth for long-lived KV residency or count destination state twice.
- Ignore destination ingestion or add its time to the overlapping route transfer.
- Admit an aggregate-feasible set that cannot be packed on replicas.
- Label emergency rescue or maximum-shed best effort as normal success.
- Validate execution against the admission envelope instead of stable capacity.
- Treat a service percentage as sessions or omit the migration-window units.
- Divide debt by total capacity instead of post-migration spare capacity.
- Mark positive debt feasible when post-migration service has no recovery spare.
- Lose physical units or pool/facet identity in normalized planner rows.
- Treat unused early capacity as if it could serve transition work arriving late.
- Credit node shutdown during session selection instead of only after planning.
- Rank sessions only by initial marginal power and miss a feasible full-drain bundle.
- Assign every coupled prefix member to the same cheap pool and fail concrete packing.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from destination import (DESTINATION_SCHEMA, CompatibilityFingerprint, ContextRate,
                         DestinationArchitecture, DestinationPool, DestinationReplica,
                         DestinationType, LoadedCoefficients, MigrationComponents)
from planner import plan
from pool_planner import (Candidate, CandidateTable, _destination_duration, _event_bounds,
                          _greedy, _greedy_bundle, _greedy_coupled,
                          _coupled_source_pattern, _mode_boundary_rho,
                          _recover_coupled, _service_trace, candidate_table,
                          destination_service_execution, exact_replica_assignment,
                          service_debt, validate_destination_execution)
from power_model import ExpectedPower
from simulate import (PlannedMove, PowerNode, ServingInstance, SessionExecution,
                      SimSession)
from test_execution_simulator import model
from test_planner import PATHS, problem


FP = CompatibilityFingerprint("m", "t", "log", "kv")


def test_full_drain_bundle_crosses_power_knee(tmp_path):
    profile = model(tmp_path)
    sessions = tuple(
        SimSession(str(i), "s0" if i < 3 else f"s{i - 2}", 1,
                   30 if i < 3 else 20, 0, 1)
        for i in range(6)
    )
    scenario = replace(
        problem(), sessions=sessions,
        nodes=tuple(PowerNode(f"n{i}", 1, True) for i in range(4)),
        instances=tuple(ServingInstance(f"s{i}", (f"n{i}",)) for i in range(4)),
    )
    power = ExpectedPower(scenario, profile)
    costs = (.25, .25, .25, .3, .3, .3)
    candidates = tuple(
        Candidate(i, "replay", 0, power.marginal(str(i)), 1, 1, (), 0, (0, 0), 0)
        for i in range(6)
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(6)),
        csr_matrix((costs, (np.zeros(6), np.arange(6))), shape=(1, 6)),
        ("route",), (1,), ("fraction",), 1,
    )

    assert _greedy(table, 26) == {3, 4, 5}
    assert _greedy_bundle(table, 26, power) == {0, 1, 2}
    assert power.drain_gain(("3", "4", "5")) == pytest.approx(24)
    assert power.drain_gain(("0", "1", "2")) == pytest.approx(28)


def architecture(*, normal=.3, emergency=.5, stable=1, baselines=((0, 0), (0, 0)),
                 kv=1000, methods=("replay", "kv_transfer"), compatibility=FP,
                 residency=None, routes=(("wan",), ("wan",)), block=1,
                 flex=None, debt=0):
    loaded = LoadedCoefficients((0, 2), (1, 1), (1, 1000), (1, 1000), "hand")
    q = DestinationType(
        "q", compatibility, ContextRate((1, 1000), (100, 100)),
        ContextRate((1, 1000), (100, 100)), ((1, 1),),
        {"normal": (normal,), "emergency": (emergency,), "stable": (stable,)},
        kv, {"replay": loaded, "kv_transfer": loaded}, (0, 1), "hand",
        kv_block_tokens=block,
    )
    pools = tuple(DestinationPool(
        f"p{i}", "q", (DestinationReplica(f"t{i}", baseline),),
        f"r{i}", route, methods, flex, debt,
    ) for i, (baseline, route) in enumerate(zip(baselines, routes)))
    return DestinationArchitecture(DESTINATION_SCHEMA, FP, (q,), pools, residency)


def test_coupled_prefix_crosses_knee_with_packable_mixed_pools(tmp_path):
    profile = model(tmp_path)
    sessions = tuple(SimSession(str(i), "s0", 1, 30, 0, 1) for i in range(3))
    nodes = (PowerNode("n0", 1, True),) + tuple(
        PowerNode(f"d{i}", 1, False) for i in range(3)
    )
    scenario = replace(
        problem(), sessions=sessions, nodes=nodes,
        instances=(ServingInstance("s0", ("n0",)),) + tuple(
            ServingInstance(f"t{i}", (f"d{i}",)) for i in range(3)
        ),
    )
    q = architecture(normal=1, emergency=1).types[0]
    arch = DestinationArchitecture(
        DESTINATION_SCHEMA, FP, (q,), (
            DestinationPool(
                "p0", "q", (DestinationReplica("t0"), DestinationReplica("t1")),
                "r0", ("wan",), ("replay",),
            ),
            DestinationPool(
                "p1", "q", (DestinationReplica("t2"),),
                "r1", ("wan",), ("replay",),
            ),
        ),
    )
    power = ExpectedPower(scenario, profile)
    candidates = tuple(
        Candidate(j, "replay", pool, power.marginal(str(j)), 1, 1, ("wan",),
                  1, (.6, 0), 0)
        for j in range(3) for pool in range(2)
    )
    table = CandidateTable(
        sessions, candidates,
        csr_matrix((np.ones(6), (np.repeat(np.arange(3), 2), np.arange(6)))),
        csr_matrix((np.full(6, .2), (np.zeros(6), np.arange(6))), shape=(1, 6)),
        ("route",), (1,), ("fraction",), 10,
    )

    selected = _greedy_coupled(table, 28, power, arch, scenario, "normal")

    assert _coupled_source_pattern(
        table, power, ((0, 1), (2, 1.5), (4, 15)), 1, 1,
    ) == (0, 2)  # objectives: 0, -5, -13.5, -10.5
    assert len(selected) == 3
    assert {candidates[i].session for i in selected} == {0, 1, 2}
    assert sorted(candidates[i].pool for i in selected) == [0, 0, 1]
    assert exact_replica_assignment(table, selected, arch, scenario, "normal") is not None


def test_coupled_recovery_caps_one_watt_overshoot_before_work(tmp_path):
    profile, scenario = model(tmp_path), problem()
    sessions = (
        replace(scenario.sessions[0], expected_f=5, expected_g=0),
        replace(scenario.sessions[1], expected_f=2.5, expected_g=0),
    )
    scenario = replace(scenario, sessions=sessions)
    power = ExpectedPower(scenario, profile)
    candidates = (
        Candidate(0, "replay", 0, 2, 1, 1, ("wan",), 1, (0, 0), 0),
        Candidate(1, "replay", 0, 1, .75, 1, ("wan",), 1, (0, 0), 0),
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(2)),
        csr_matrix((np.full(2, .1), (np.zeros(2), np.arange(2))), shape=(1, 2)),
        ("route",), (1,), ("fraction",), 10,
    )

    selected = _recover_coupled(
        table, power, {"s0": {(), (0,)}, "s1": {(1,)}}, 1,
        replace(architecture(normal=1, emergency=1), pools=(
            architecture(normal=1, emergency=1).pools[0],
        )), scenario, "normal",
    )

    assert selected == {1}

    upgrade = replace(
        table,
        candidates=(
            replace(candidates[0], migration_work_s=.01),
            candidates[1],
        ),
    )
    target = power.drain_gain([s.session_id for s in sessions])
    selected = _recover_coupled(
        upgrade, power, {"s0": {(0,), (0, 1)}}, target,
        replace(architecture(normal=1, emergency=1), pools=(
            architecture(normal=1, emergency=1).pools[0],
        )), scenario, "normal",
    )

    assert selected == {0, 1}

    swap_candidates = (
        replace(candidates[0], pool=0),
        replace(candidates[0], pool=1),
        replace(candidates[1], pool=0),
    )
    swap_table = replace(
        table, candidates=swap_candidates,
        incidence=csr_matrix((
            np.ones(3), ((0, 0, 1), np.arange(3)),
        ), shape=(2, 3)),
        resources=csr_matrix((
            (.6, .6, .6), ((0, 1, 0), np.arange(3)),
        ), shape=(2, 3)),
        resource_names=("pool:a", "pool:b"),
        resource_capacities=(1, 1),
        resource_units=("fraction", "fraction"),
    )
    selected = _recover_coupled(
        swap_table, power, {"s0": {(0,), (1,)}, "s1": {(2,)}}, target,
        architecture(normal=1, emergency=1), scenario, "normal",
    )

    assert selected == {1, 2}


def test_coupled_recovery_preserves_aggregate_boundary(tmp_path):
    profile, scenario = model(tmp_path), problem()
    power = ExpectedPower(scenario, profile)
    candidates = tuple(
        Candidate(i, "replay", 0, power.marginal(session.session_id), 1, 1,
                  ("wan",), 1, (0, 0), 0)
        for i, session in enumerate(scenario.sessions)
    )
    limit = 1 + 1e-8

    def recover(second):
        table = CandidateTable(
            scenario.sessions, candidates, csr_matrix(np.eye(2)),
            csr_matrix(((.6, second), ((0, 0), (0, 1))), shape=(1, 2)),
            ("route",), (1,), ("fraction",), 10,
        )
        return _recover_coupled(
            table, power, {"s0": {(0,)}, "s1": {(1,)}},
            power.drain_gain(("a", "b")),
            replace(architecture(normal=1, emergency=1), pools=(
                architecture(normal=1, emergency=1).pools[0],
            )), scenario, "normal",
        )

    boundary = limit - .6
    assert recover(boundary) == {0, 1}
    assert recover(boundary + 1e-12) == {0}


def test_coupled_recovery_preserves_prefix_tie_order(tmp_path):
    profile, scenario = model(tmp_path), problem()
    sessions = (scenario.sessions[0], replace(
        scenario.sessions[1], source_instance="s0",
    ))
    scenario = replace(scenario, sessions=sessions)
    power = ExpectedPower(scenario, profile)
    candidates = (
        Candidate(0, "replay", 0, 1, 1, 1, ("wan",), 1, (0, 0), 0),
        Candidate(1, "replay", 0, 1, 0, 1, ("wan",), 1, (0, 0), 0),
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(2)),
        csr_matrix((np.full(2, .1), (np.zeros(2), np.arange(2))), shape=(1, 2)),
        ("route",), (1,), ("fraction",), 10,
    )

    selected = _recover_coupled(
        table, power, {"s0": {(0,), (0, 1)}}, .1,
        replace(architecture(normal=1, emergency=1), pools=(
            architecture(normal=1, emergency=1).pools[0],
        )), scenario, "normal",
    )

    assert selected == {0, 1}


def test_loaded_lookup_boundary_tracks_selected_admission_mode():
    q = architecture(normal=.4, emergency=.6, stable=.8).types[0]
    assert _mode_boundary_rho(q, "normal") == 1
    assert _mode_boundary_rho(q, "emergency") == pytest.approx(1.5)


def test_five_percent_flex_adds_stable_capacity_not_sessions():
    arch = architecture(normal=.8, emergency=.8, stable=1, flex=.05)

    assert _event_bounds(arch.types[0], arch.pools[0], "normal") == pytest.approx((.85,))


def test_service_debt_and_recovery_use_replica_seconds():
    debt, recovery = service_debt(
        baseline=(.6,), ongoing=(.2,), transition=(30,), stable=(1,), horizon=100,
    )

    assert debt == pytest.approx((10,))
    assert recovery == pytest.approx((50,))


def test_positive_debt_without_spare_capacity_never_recovers():
    debt, recovery = service_debt(
        baseline=(.8,), ongoing=(.2,), transition=(1,), stable=(1,), horizon=100,
    )

    assert debt == pytest.approx((1,))
    assert recovery[0] == float("inf")


def test_late_transition_work_cannot_use_early_idle_capacity():
    trace = _service_trace(0, 1, ((9, 2),), 0, 10)

    assert trace[-1] == (10, 2, 1, 1)
    assert _service_trace(0, 1, ((9, 2),), 0, 10, False) == trace[-1:]


def test_realized_pool_trace_rejects_late_debt_above_budget(tmp_path):
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((.6, 0),),
        routes=(("wan",),), methods=("replay",), flex=0, debt=.05,
    )
    move = PlannedMove(
        "a", "t0", "replay", 0, ("wan",), destination_pool="p0",
    )
    execution = SimpleNamespace(sessions=(
        SessionExecution(
            "a", "replay", 0, 9, 9, 9, None, None, 9, 9, None, None, 8,
        ),
    ))

    rows = destination_service_execution(
        problem(), model(tmp_path, switch=0, tp=1), arch, (move,), execution,
    )

    final = rows[-1]
    assert final.peak_queued_replica_s == pytest.approx(.6)
    assert final.debt_budget_replica_s == pytest.approx(.45)
    assert not final.within_contract


def test_prediction_rejects_realized_service_queue_violation(tmp_path):
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((.6, 0),),
        routes=(("wan",),), methods=("replay",), flex=0, debt=.02,
    )
    scenario = replace(problem(limit=40), controller_delay_s=7)

    result = plan(
        scenario, model(tmp_path, switch=0, tp=1), PATHS, "lp",
        destination=arch,
    )

    assert result.moves
    assert result.failure_reason == "destination_service_queue"
    assert not result.feasible


def test_plan_rejects_positive_debt_without_recovery_spare(tmp_path):
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((.75, 0),),
        routes=(("wan",),), methods=("replay",), flex=0, debt=.2,
    )
    result = plan(
        problem(limit=40), model(tmp_path, switch=0, tp=1), PATHS, "lp",
        destination=arch,
    )

    assert result.moves and result.service_debt_replica_s > 0
    assert result.required_recovery_s == float("inf")
    assert not result.feasible
    assert result.failure_reason == "service_debt_unrecoverable"


def test_replay_charges_transition_debt_but_kv_does_not(tmp_path):
    arch = architecture(normal=1, emergency=1, stable=1, flex=0, debt=.2)
    arch = replace(arch, pools=(arch.pools[0],))
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    table = candidate_table(scenario, profile, arch, "normal",
                            ExpectedPower(scenario, profile))
    row = next(i for i, name in enumerate(table.resource_names)
               if name.startswith("service-debt:"))
    choices = {
        candidate.method: float(table.resources[row, i])
        for i, candidate in enumerate(table.candidates)
        if candidate.session == 0
    }

    assert choices["replay"] > choices["kv_transfer"]


def test_plan_preserves_physical_resource_and_debt_rows(tmp_path):
    result = plan(
        problem(limit=40), model(tmp_path, switch=0, tp=1), PATHS, "lp",
        destination=architecture(
            normal=1, emergency=1, stable=1, baselines=((.6, 0),),
            routes=(("wan",),), methods=("replay",), flex=0, debt=.2,
        ),
    )

    route = next(row for row in result.resource_uses if row.name == "route:wan")
    assert route.unit == "bytes"
    assert route.used == 100
    assert route.capacity == 900
    assert route.utilization == pytest.approx(1 / 9)
    assert len(result.service_debts) == 1
    debt = result.service_debts[0]
    assert (debt.pool_id, debt.facet) == ("p0", 0)
    assert debt.debt_replica_s == 0
    assert debt.spare_replicas == pytest.approx(.15)
    assert debt.recovery_s == 0


def test_v1_resource_rows_match_the_documented_contract(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    arch = architecture(normal=1, emergency=1, stable=1, flex=0, debt=.2)
    arch = replace(arch, pools=(arch.pools[0],))

    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )

    assert table.resource_names == (
        "route:wan", "service:p0:0", "service-debt:p0:0",
        "kv:p0", "migration:p0",
    )


def test_absent_architecture_is_exact_legacy_adapter(tmp_path):
    profile, scenario = model(tmp_path, tp=1), problem()
    old, adapted = plan(scenario, profile, PATHS, "lp"), plan(
        scenario, profile, PATHS, "lp", destination=None,
    )

    assert old.moves == adapted.moves
    assert old.planned_source_power_w == adapted.planned_source_power_w
    assert old.feasible == adapted.feasible


def test_normal_success_and_emergency_rescue_are_distinct(tmp_path):
    profile, scenario = model(tmp_path, switch=0, tp=1), problem()

    assert plan(scenario, profile, PATHS, "lp", destination=architecture()).admission_mode == "normal"
    rescued = plan(
        scenario, profile, PATHS, "lp",
        destination=architecture(normal=.2, emergency=.3),
    )
    assert rescued.feasible and rescued.admission_mode == "emergency"


def test_target_unmet_returns_valid_maximum_shed_plan(tmp_path):
    result = plan(
        problem(), model(tmp_path, switch=0, tp=1), PATHS, "greedy",
        destination=architecture(methods=("kv_transfer",),
                                 compatibility=replace(FP, kv_abi="other")),
    )

    assert not result.feasible and result.failure_reason == "target_unmet"
    assert result.power_shortfall_w > 0 and result.moves == ()


def test_shutdown_is_realized_after_selection_not_credited_to_the_plan(tmp_path):
    profile = model(tmp_path, switch=0, tp=1)
    scenario = replace(problem(final="off"), assumed_shutdown_s=1)
    result = plan(
        scenario, profile, PATHS, "lp",
        destination=architecture(normal=1, emergency=1),
    )

    assert result.moves
    assert result.expected_source_power_at_deadline_w < result.planned_source_power_w


def test_pool_capacity_cannot_be_borrowed(tmp_path):
    arch = architecture(normal=.2, emergency=.2, baselines=((.1, 0), (.1, 0)))
    scenario = replace(problem(), sessions=(problem().sessions[0],))
    result = plan(scenario, model(tmp_path, switch=0, tp=1), PATHS, "lp", destination=arch)

    assert result.moves == () and result.failure_reason == "target_unmet"


def test_destination_background_and_explicit_baseline_are_rejected(tmp_path):
    background = SimSession("bg", "t0", 10, 1, 0, 10, movable=False)
    scenario = replace(problem(), sessions=problem().sessions + (background,))

    with pytest.raises(ValueError, match="exclusive"):
        plan(scenario, model(tmp_path, tp=1), PATHS, "lp",
             destination=architecture(baselines=((.1, 0), (0, 0))))


def test_residency_horizon_is_independent_of_migration_horizon(tmp_path):
    growing = replace(problem().sessions[0], context_tokens=10,
                      expected_growth_tokens_per_s=1)
    scenario = replace(problem(), sessions=(growing,))
    profile = model(tmp_path, switch=0, tp=1)
    long = plan(scenario, profile, PATHS, "lp", destination=architecture(kv=25))
    short = plan(scenario, profile, PATHS, "lp",
                 destination=architecture(kv=25, residency=5))

    assert not long.moves and short.moves


def test_private_kv_is_rounded_per_session_not_after_aggregation(tmp_path):
    scenario = replace(problem(), sessions=tuple(
        replace(s, context_tokens=17, expected_growth_tokens_per_s=0)
        for s in problem().sessions
    ))
    profile = model(tmp_path, switch=0, tp=1)
    arch = architecture(kv=48, block=16, methods=("replay",))
    arch = replace(arch, pools=(arch.pools[0],))
    result = plan(
        scenario, profile, PATHS, "lp", destination=arch,
    )

    assert len(result.moves) == 1


def test_physical_destination_timing_keeps_route_time_unscaled(tmp_path):
    profile = model(tmp_path, switch=0, tp=1)
    session = replace(problem().sessions[0], context_tokens=10, log_bytes=100,
                      expected_growth_tokens_per_s=0)
    components = MigrationComponents((5, 20), (50, 200), "hand", .5, 2)

    replay = _destination_duration(
        session, "replay", profile.case(), ("wan",), {"wan": 100}, 0, components,
    )
    kv = _destination_duration(
        session, "kv_transfer", profile.case(), ("wan",), {"wan": 100}, 0,
        components,
    )

    assert replay == pytest.approx(1 + .5 * .1)
    assert kv == pytest.approx(1 + 2)


def test_kv_destination_timing_uses_ingest_floor(tmp_path):
    profile = model(tmp_path, switch=0, destination_rate=50, tp=1)
    session = replace(problem().sessions[0], context_tokens=10,
                      expected_growth_tokens_per_s=0)
    components = MigrationComponents((5, 20), (50, 200), "hand", residual_s=2)

    duration = _destination_duration(
        session, "kv_transfer", profile.case(), ("wan",), {"wan": 100}, 0,
        components,
    )

    assert duration == pytest.approx(2 + 2)


def test_static_kv_snapshot_has_no_fake_deadline_catch_up(tmp_path):
    result = plan(
        problem(), model(tmp_path, switch=0, tp=1), PATHS, "lp",
        destination=architecture(methods=("kv_transfer",)),
    )

    assert result.moves
    assert all(move.quiesce_s is None for move in result.moves)


def test_exact_route_rows_choose_the_route_with_capacity(tmp_path):
    scenario = replace(
        problem(), sessions=(problem().sessions[0],),
        links=(replace(problem().links[0], link_id="slow", bytes_per_s=1),
               replace(problem().links[0], link_id="fast", bytes_per_s=100)),
    )
    arch = architecture(routes=(("slow",), ("fast",)))
    result = plan(scenario, model(tmp_path, switch=0, tp=1), {}, "lp", destination=arch)

    assert result.moves[0].destination_pool == "p1"
    assert result.moves[0].path == ("fast",)


def test_aggregate_feasibility_does_not_override_replica_packing(tmp_path):
    arch = architecture(normal=1, emergency=1, stable=1,
                        baselines=((.2, 0), (.4, 0)))
    arch = replace(arch, pools=(DestinationPool(
        "p", "q", (DestinationReplica("t0", (.2, 0)),
                    DestinationReplica("t1", (.4, 0))), "r", ("wan",), ("replay",),
    ),))
    sessions = tuple(replace(s, expected_f=70, expected_g=0) for s in problem().sessions)
    scenario = replace(problem(), sessions=sessions)
    table = candidate_table(scenario, model(tmp_path, tp=1), arch, "normal",
                            ExpectedPower(scenario, model(tmp_path, tp=1)))

    assert exact_replica_assignment(table, {0, 1}, arch, scenario, "normal") is None
    result = plan(scenario, model(tmp_path, switch=0, tp=1), PATHS, "lp", destination=arch)
    assert result.packing_repair_count >= 1 and result.failure_reason == "target_unmet"


def test_exact_oracle_enforces_one_migration_at_a_time_per_replica(tmp_path):
    arch = architecture(normal=1, emergency=1, stable=1)
    arch = replace(arch, pools=(DestinationPool(
        "p", "q", (DestinationReplica("t0"), DestinationReplica("t1", (.9, 0))),
        "r", ("wan",), ("replay",),
    ),))
    sessions = tuple(replace(s, expected_f=5, log_bytes=500) for s in problem().sessions)
    scenario = replace(problem(), sessions=sessions)
    profile = model(tmp_path, switch=0, tp=1)
    table = candidate_table(scenario, profile, arch, "normal", ExpectedPower(scenario, profile))
    assignment = exact_replica_assignment(table, {0, 1}, arch, scenario, "normal")

    assert assignment is not None
    assert len(set(assignment.values())) == 2


def test_execution_independently_rejects_stable_overflow():
    arch = architecture(normal=.3, emergency=.3, stable=.3,
                        baselines=((.2, 0), (0, 0)))
    move = PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0")

    with pytest.raises(ValueError, match="stable envelope"):
        validate_destination_execution(problem(), arch, (move,))
