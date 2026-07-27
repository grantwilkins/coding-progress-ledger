"""
Claim:
Pool-aware planning selects at most one method/pool candidate per session, charges
resources to the exact pool and route, packs whole sessions on concrete replicas,
and distinguishes normal, emergency, and valid target-unmet outcomes. KV transfer
time is limited by the slower of route transfer and destination ingestion.

Plausible wrong implementations:
- Borrow residual service or KV capacity across pools or replicas.
- Inflate greedy scarcity prices by counting every duplicate pool candidate.
- Use migration growth for long-lived KV residency or count destination state twice.
- Ignore destination ingestion or add its time to the overlapping route transfer.
- Admit an aggregate-feasible set that cannot be packed on replicas.
- Label emergency rescue or maximum-shed best effort as normal success.
- Validate execution against the admission envelope instead of stable capacity.
"""

from dataclasses import replace

import pytest

from destination import (DESTINATION_SCHEMA, CompatibilityFingerprint, ContextRate,
                         DestinationArchitecture, DestinationPool, DestinationReplica,
                         DestinationType, LoadedCoefficients, MigrationComponents)
from planner import plan
from pool_planner import (_destination_duration, _mode_boundary_rho, candidate_table,
                          exact_replica_assignment, validate_destination_execution)
from power_model import ExpectedPower
from simulate import PlannedMove, SimSession
from test_execution_simulator import model
from test_planner import PATHS, problem


FP = CompatibilityFingerprint("m", "t", "log", "kv")


def architecture(*, normal=.3, emergency=.5, stable=1, baselines=((0, 0), (0, 0)),
                 kv=1000, methods=("replay", "kv_transfer"), compatibility=FP,
                 residency=None, routes=(("wan",), ("wan",)), block=1):
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
        f"r{i}", route, methods,
    ) for i, (baseline, route) in enumerate(zip(baselines, routes)))
    return DestinationArchitecture(DESTINATION_SCHEMA, FP, (q,), pools, residency)


def test_loaded_lookup_boundary_tracks_selected_admission_mode():
    q = architecture(normal=.4, emergency=.6, stable=.8).types[0]
    assert _mode_boundary_rho(q, "normal") == 1
    assert _mode_boundary_rho(q, "emergency") == pytest.approx(1.5)


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
