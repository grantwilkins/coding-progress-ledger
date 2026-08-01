"""
Claim:
Pool-aware planning selects at most one method/pool candidate per session, charges
resources to the exact pool and route, packs whole sessions on concrete replicas,
and distinguishes normal, emergency, and valid target-unmet outcomes. KV transfer
time is limited by the slower of route transfer and destination ingestion.
Temporary reconstruction debt is measured in replica-seconds and can recover
only from post-migration spare service.
The experimental HiGHS backend solves the same target-first LP and uses the same
rounder without replacing the default Clarabel backend.
The column-generation backend uses an always-feasible shortfall phase, includes
session prices in reduced costs, and certifies every phase with complete pricing.

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
- Serialize replay and KV work despite having separate measured aggregate caps.
- Relax aggregate method constraints but retain cross-method serialization in packing.
- Choose one cheap action per session before exploring feasible mixed-method patterns.
- Materialize every source prefix even though recovery retains a bounded frontier.
- Reverse or mis-scale the HiGHS target row or skip maximum-gain fallback.
- Route the existing `lp` solver through HiGHS instead of keeping it additive.
- Charge migration work during shortfall minimization or omit the session dual.
- Run Phase II at an infeasible requested target instead of maximum attainable gain.
- Reuse candidate physics across pools with different types, routes, or source loads.
- Double-count the session dual while repairing a tolerated reduced-cost violation.
- Omit the Phase-I shortfall dual cap or stop before the global gap closes.
- Merge pool variables that share physics or misalign SoA resource templates.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

import pool_planner

from destination import (DESTINATION_SCHEMA, CompatibilityFingerprint, ContextRate,
                         DestinationArchitecture, DestinationPool, DestinationReplica,
                         DestinationType, LoadedCoefficients, MigrationComponents)
from planner import plan
from pool_planner import (Candidate, CandidateTable, _destination_duration, _event_bounds,
                          _candidate_oracle,
                          _greedy, _greedy_bundle, _greedy_coupled,
                          _greedy_prefix, _coupled_source_pattern,
                          _lp_column_generation, _lp_column_generation_lazy,
                          _lp_column_generation_native,
                          _lp_column_generation_persistent,
                          _lp_highs,
                          _mode_boundary_rho,
                          _native_pricing_oracle, _pricing_soa,
                          _dual_resource_limits, _retained_prefixes,
                          _source_removed_gain,
                          _recover_coupled, _service_trace, candidate_table,
                          destination_service_execution, exact_replica_assignment,
                          service_debt, validate_destination_execution)
from power_model import ExpectedPower
from simulate import (PlannedMove, PowerNode, ServingInstance, SessionExecution,
                      SimSession)
from test_execution_simulator import model
from test_planner import PATHS, problem


FP = CompatibilityFingerprint("m", "t", "log", "kv")


def _hand_lp_table():
    candidates = (
        Candidate(0, "replay", 0, 2, 2, 2, (), 0, (0, 0), 0),
        Candidate(0, "kv_transfer", 0, 2, 1, 1, (), 0, (0, 0), 0),
        Candidate(1, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
    )
    return CandidateTable(
        (), candidates,
        csr_matrix((np.ones(3), ((0, 0, 1), range(3))), shape=(2, 3)),
        csr_matrix((np.array([.6, .8, .2]),
                    (np.zeros(3, int), np.arange(3))), shape=(1, 3)),
        ("resource",), (1,), ("fraction",), 10,
    )


def _table_oracle(table):
    matrix, candidates = pool_planner.csc_matrix(table.resources), table.candidates
    columns = {id(candidate): i for i, candidate in enumerate(candidates)}
    sessions = range(table.incidence.shape[0])
    return SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=str(i)) for i in sessions),
        pools=tuple(
            SimpleNamespace(pool_id=f"p{i}", methods=("replay", "kv_transfer"))
            for i in range(1 + max((c.pool for c in candidates), default=-1))
        ),
        migration_horizon_s=table.migration_horizon_s,
        specs=tuple(zip(table.resource_capacities, table.resource_names,
                        table.resource_units)),
        choices=lambda session: tuple(
            candidate for candidate in candidates if candidate.session == session
        ),
        column=lambda candidate: tuple(zip(
            matrix.indices[matrix.indptr[columns[id(candidate)]]:
                           matrix.indptr[columns[id(candidate)] + 1]],
            matrix.data[matrix.indptr[columns[id(candidate)]]:
                        matrix.indptr[columns[id(candidate)] + 1]],
        )),
    )


def test_highs_lp_matches_target_problem_and_max_gain_fallback():
    table = _hand_lp_table()

    assert _lp_highs(table, 3) == {1, 2}
    assert _lp_highs(table, 4) == {1, 2}


@pytest.mark.parametrize("target, shortfall", ((3, 0), (4, 1)))
def test_column_generation_matches_flat_lp_and_certifies_both_phases(
    target, shortfall,
):
    table, stats = _hand_lp_table(), {}

    selected = _lp_column_generation(table, target, stats)

    assert selected == _lp_highs(table, target) == {1, 2}
    assert stats["phase1_shortfall"] == pytest.approx(shortfall)
    assert stats["effective_target"] == pytest.approx(3)
    assert stats["phase1"]["gap"] == pytest.approx(0, abs=1e-7)
    assert stats["phase2"]["gap"] == pytest.approx(0, abs=1e-7)


def test_column_pricing_uses_session_dual_to_avoid_duplicate_equivalent_choices():
    candidates = tuple(
        Candidate(0, method, 0, 1, 1, 1, (), 0, (0, 0), 0)
        for method in ("replay", "kv_transfer")
    )
    table = CandidateTable(
        (), candidates, csr_matrix(np.ones((1, 2))), csr_matrix((0, 2)),
        (), (), (), 1,
    )
    stats = {}

    _lp_column_generation(table, 1, stats)

    assert stats["active_columns"] == 1
    assert stats["phase1"]["gap"] == pytest.approx(0)


def test_column_phase_one_ignores_migration_work():
    candidates = (
        Candidate(0, "replay", 0, 1, 100, 1, (), 0, (0, 0), 0),
        Candidate(0, "kv_transfer", 0, 1, 1, 1, (), 0, (0, 0), 0),
    )
    table = CandidateTable(
        (), candidates, csr_matrix(np.ones((1, 2))), csr_matrix((0, 2)),
        (), (), (), 1,
    )
    stats = {}

    selected = _lp_column_generation(table, 1, stats)

    assert stats["phase1"]["columns"] == 1
    assert stats["active_columns"] == 2
    assert selected == {1}


def test_column_certificate_repairs_reduced_cost_inside_tolerance(monkeypatch):
    epsilon = 1e-5
    table = CandidateTable(
        (), (
            Candidate(0, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
            Candidate(0, "kv_transfer", 0, 1 + epsilon, 1, 1, (), 0, (0, 0), 0),
        ),
        csr_matrix(np.ones((1, 2))), csr_matrix((0, 2)), (), (), (), 1,
    )
    monkeypatch.setattr(pool_planner, "COLUMN_TOLERANCE", 2 * epsilon)
    stats = {}

    pool_planner._column_phase(
        table, 2, np.zeros(2), {0}, True, stats,
    )

    assert stats["upper"] == pytest.approx(1)
    assert stats["lower"] == pytest.approx(1 - epsilon)
    assert stats["lower"] <= 1 - epsilon + 1e-12


def test_persistent_certificate_repairs_reduced_cost_inside_tolerance(monkeypatch):
    epsilon = 1e-5
    table = CandidateTable(
        (), (
            Candidate(0, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
            Candidate(0, "kv_transfer", 0, 1 + epsilon, 1, 1, (), 0, (0, 0), 0),
        ),
        csr_matrix(np.ones((1, 2))), csr_matrix((0, 2)), (), (), (), 1,
    )
    monkeypatch.setattr(pool_planner, "COLUMN_TOLERANCE", 2 * epsilon)
    highs = pool_planner.highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.addRows(
        1, np.array([2.0]), np.array([pool_planner.highspy.kHighsInf]),
        0, np.zeros(2, np.int32), np.array([], np.int32), np.array([], float),
    )
    highs.addCol(1, 0, pool_planner.highspy.kHighsInf, 1,
                 np.array([0], np.int32), np.array([1.0]))
    columns, rows = np.full(2, -1, np.int32), np.full(1, -1, np.int32)
    pool_planner._add_priced_columns(
        highs, table, pool_planner.csc_matrix(table.resources), [0],
        np.zeros(2), columns, rows,
    )
    stats = {}

    pool_planner._persistent_column_phase(
        highs, table, pool_planner.csc_matrix(table.resources), 2,
        np.zeros(2), columns, rows, stats,
    )

    assert stats["upper"] == pytest.approx(1)
    assert stats["lower"] == pytest.approx(1 - epsilon)


def test_lazy_certificate_closes_aggregate_subthreshold_gap(monkeypatch):
    epsilon = 1e-5
    table = CandidateTable(
        (), (
            Candidate(0, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
            Candidate(0, "kv_transfer", 0, 1 + epsilon, 1, 1, (), 0, (0, 0), 0),
        ),
        csr_matrix(np.ones((1, 2))), csr_matrix((0, 2)), (), (), (), 1,
    )
    oracle = _table_oracle(table)
    monkeypatch.setattr(pool_planner, "COLUMN_TOLERANCE", 2 * epsilon)
    highs = pool_planner.highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.addRows(
        1, np.array([2.0]), np.array([pool_planner.highspy.kHighsInf]),
        0, np.zeros(2, np.int32), np.array([], np.int32), np.array([], float),
    )
    highs.addCol(1, 0, pool_planner.highspy.kHighsInf, 1,
                 np.array([0], np.int32), np.array([1.0]))
    rows, active, candidates = np.full(1, -1, np.int32), set(), []
    first = table.candidates[0]
    pool_planner._lazy_add_columns(
        highs, oracle,
        [pool_planner._PricedColumn(0, ("0", "p0", "0"), first, (), 0)],
        rows, active, candidates,
    )
    stats = {}

    pool_planner._lazy_column_phase(
        highs, oracle, 2, False, rows, active, candidates, stats,
    )

    assert stats["columns"] == 2
    assert stats["upper"] == pytest.approx(1 - epsilon)
    assert stats["lower"] <= 1 - epsilon + 1e-12
    assert stats["gap"] <= pool_planner.COLUMN_GAP_TOLERANCE


@pytest.mark.parametrize("target, shortfall", ((3, 0), (4, 1)))
def test_persistent_column_master_matches_rebuilding_certificate(target, shortfall):
    table, persistent, rebuilding = _hand_lp_table(), {}, {}

    selected = _lp_column_generation_persistent(table, target, persistent)
    reference = _lp_column_generation(table, target, rebuilding)

    assert selected == reference == {1, 2}
    assert persistent["phase1_shortfall"] == pytest.approx(shortfall)
    assert persistent["phase1"]["gap"] == pytest.approx(0, abs=1e-7)
    assert persistent["phase2"]["gap"] == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("target", (3, 4))
def test_lazy_column_master_matches_complete_persistent_lp(target):
    table = _hand_lp_table()
    lazy, persistent = {}, {}

    restricted, selected = _lp_column_generation_lazy(_table_oracle(table), target, lazy)
    reference = _lp_column_generation_persistent(table, target, persistent)

    semantic = lambda source, indices: {
        (source.candidates[i].session, source.candidates[i].method) for i in indices
    }
    assert semantic(restricted, selected) == semantic(table, reference)
    assert lazy["phase1_shortfall"] == pytest.approx(
        persistent["phase1_shortfall"], abs=1e-7,
    )
    assert lazy["phase2"]["upper"] == pytest.approx(
        persistent["phase2"]["upper"], abs=1e-7,
    )
    assert lazy["phase1"]["gap"] == pytest.approx(0, abs=1e-7)
    assert lazy["phase2"]["gap"] == pytest.approx(0, abs=1e-7)
    assert lazy["completion_columns"] == 0


@pytest.mark.parametrize("target", (4, 20))
@pytest.mark.parametrize("reverse", (False, True))
def test_lazy_matches_complete_lp_with_mixed_asymmetric_pools(target, reverse):
    candidates = [
        Candidate(0, "replay", 0, 2, 3, 3, (), 0, (0, 0), 0),
        Candidate(0, "kv_transfer", 1, 2, 1, 1, (), 0, (0, 0), 0),
        Candidate(1, "replay", 1, 3, 2, 2, (), 0, (0, 0), 0),
        Candidate(2, "kv_transfer", 0, 1, 1, 1, (), 0, (0, 0), 0),
        Candidate(4, "replay", 0, 4, 4, 4, (), 0, (0, 0), 0),
        Candidate(4, "kv_transfer", 1, 4, 2, 2, (), 0, (0, 0), 0),
    ]
    resources = np.array((
        (.2, .2, .1, .3, .4, .4),
        (.5, 0, 0, .4, .2, 0),
        (0, .3, .6, 0, 0, .4),
    ))
    if reverse:
        candidates.reverse()
        resources = resources[:, ::-1]
    incidence = csr_matrix((
        np.ones(len(candidates)),
        ([candidate.session for candidate in candidates], range(len(candidates))),
    ), shape=(5, len(candidates)))
    table = CandidateTable(
        (), tuple(candidates), incidence, csr_matrix(resources),
        ("shared", "pool-0", "pool-1"), (1, 1, 1),
        ("fraction",) * 3, 10,
    )
    lazy, persistent = {}, {}

    _lp_column_generation_lazy(_table_oracle(table), target, lazy)
    _lp_column_generation_persistent(table, target, persistent)

    assert lazy["phase1_shortfall"] == pytest.approx(
        persistent["phase1_shortfall"], abs=1e-7,
    )
    assert lazy["effective_target"] == pytest.approx(
        persistent["effective_target"], abs=1e-7,
    )
    assert lazy["phase2"]["upper"] == pytest.approx(
        persistent["phase2"]["upper"], abs=1e-7,
    )
    assert lazy["phase1"]["gap"] <= pool_planner.COLUMN_GAP_TOLERANCE
    assert lazy["phase2"]["gap"] <= pool_planner.COLUMN_GAP_TOLERANCE


@pytest.mark.parametrize("seed", range(3))
def test_randomized_persistent_master_matches_complete_lp(seed):
    rng, candidates, session_rows = np.random.default_rng(seed), [], []
    for session in range(300):
        if session % 17 == 0:
            continue
        gain = rng.uniform(.5, 2)
        for choice in range(1 + session % 3):
            candidates.append(Candidate(
                session, "replay" if choice % 2 == 0 else "kv_transfer", 0,
                gain, rng.uniform(.1, 3), 1, (), 0, (0, 0), 0,
            ))
            session_rows.append(session)
    n = len(candidates)
    resources = csr_matrix(rng.uniform(.001, .02, (2, n)))
    incidence = csr_matrix((np.ones(n), (session_rows, np.arange(n))),
                           shape=(300, n))
    table = CandidateTable(
        (), tuple(candidates), incidence, resources, ("a", "b"), (1, 1),
        ("fraction", "fraction"), 10,
    )
    gains = np.array([c.gain_w for c in candidates])
    work = np.array([c.migration_work_s for c in candidates]) / 10
    common = csr_matrix(np.vstack((incidence.toarray(), resources.toarray())))
    maximum = linprog(-gains, A_ub=common, b_ub=np.ones(common.shape[0]),
                      bounds=(0, None), method="highs-ipm")
    maximum_gain = -maximum.fun
    target = maximum_gain * (1.1 if seed % 2 else .8)
    effective = min(target, maximum_gain)
    target_row = csr_matrix((-gains).reshape(1, -1))
    exact = linprog(
        work, A_ub=csr_matrix(np.vstack((common.toarray(), target_row.toarray()))),
        b_ub=np.append(np.ones(common.shape[0]), -(effective - 1e-7)),
        bounds=(0, None), method="highs-ipm",
    )
    stats = {}

    selected = _lp_column_generation_persistent(table, target, stats)

    assert stats["phase1_shortfall"] == pytest.approx(
        max(0, target - maximum_gain), abs=1e-7,
    )
    assert stats["phase2"]["upper"] == pytest.approx(exact.fun, abs=1e-7)
    assert stats["phase1"]["gap"] == pytest.approx(0, abs=1e-7)
    assert stats["phase2"]["gap"] == pytest.approx(0, abs=1e-7)
    assert stats["phase1"]["sweeps"] > 1
    chosen = sorted(selected)
    assert np.max(np.asarray(incidence[:, chosen].sum(1)), initial=0) <= 1
    assert np.max(np.asarray(resources[:, chosen].sum(1)), initial=0) <= 1 + 1e-8


def test_highs_lp_is_additive_and_does_not_replace_clarabel(monkeypatch):
    table, called = _hand_lp_table(), []
    monkeypatch.setattr(pool_planner, "candidate_table", lambda *args: table)
    monkeypatch.setattr(pool_planner, "_lp", lambda *args: called.append("lp") or set())
    monkeypatch.setattr(pool_planner, "_lp_highs",
                        lambda *args: called.append("lp_highs") or set())
    monkeypatch.setattr(pool_planner, "_lp_column_generation",
                        lambda *args: called.append("lp_column_generation") or set())
    monkeypatch.setattr(
        pool_planner, "_lp_column_generation_persistent",
        lambda *args: called.append("lp_column_generation_persistent") or set(),
    )
    monkeypatch.setattr(pool_planner, "_candidate_oracle", lambda *args: object())
    monkeypatch.setattr(
        pool_planner, "_lp_column_generation_lazy",
        lambda *args: (table, called.append("lp_column_generation_lazy") or set()),
    )
    monkeypatch.setattr(
        pool_planner, "_lp_column_generation_native",
        lambda *args: (table, called.append("lp_column_generation_native") or set()),
    )
    monkeypatch.setattr(pool_planner, "_pack", lambda *args: ({}, None))

    pool_planner._mode_plan(None, None, None, "lp", "normal", None, 0)
    pool_planner._mode_plan(None, None, None, "lp_highs", "normal", None, 0)
    pool_planner._mode_plan(
        None, None, None, "lp_column_generation", "normal", None, 0,
    )
    pool_planner._mode_plan(
        None, None, None, "lp_column_generation_persistent", "normal", None, 0,
    )
    monkeypatch.setattr(
        pool_planner, "candidate_table",
        lambda *args: pytest.fail("lazy solver materialized the full table"),
    )
    pool_planner._mode_plan(
        None, None, None, "lp_column_generation_lazy", "normal", None, 0,
    )
    pool_planner._mode_plan(
        None, None, None, "lp_column_generation_native", "normal", None, 0,
    )

    assert called == [
        "lp", "lp_highs", "lp_column_generation",
        "lp_column_generation_persistent", "lp_column_generation_lazy",
        "lp_column_generation_native",
    ]


def test_dual_retention_is_bounded_and_keeps_safeguards():
    ordered, best = tuple(range(100)), tuple(range(37))

    retained = _retained_prefixes(best, ordered)

    assert {(0,), ordered, ordered[:36], best, ordered[:38]} <= retained
    assert len(retained) <= pool_planner.DUAL_PREFIX_BUCKETS + 4


def test_dual_route_limit_reserves_the_largest_post_route_tail():
    table = CandidateTable(
        (), (Candidate(0, "replay", 0, 1, 1, 3, (), 1, (0, 0), 0),),
        csr_matrix(np.zeros((0, 1))), csr_matrix(np.array(((.2,), (.4,)))),
        ("route:wan", "service:p:0"), (1, 1), ("bytes", "replica-s/s"), 10,
    )

    assert _dual_resource_limits(table) == pytest.approx((.9, 1))


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
    removed = 0
    for count, session in enumerate(sessions, 1):
        removed += power.ell[session.session_id]
        assert _source_removed_gain(power, "s0", removed) == pytest.approx(
            power.drain_gain(row.session_id for row in sessions[:count])
        )
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


def test_coupled_oracle_finds_the_only_feasible_method_mix(tmp_path):
    profile, scenario = model(tmp_path), problem()
    scenario = replace(scenario, sessions=(
        scenario.sessions[0],
        replace(scenario.sessions[1], source_instance="s0"),
    ))
    power = ExpectedPower(scenario, profile)
    candidates = tuple(
        Candidate(session, method, 0, 1, work, 6, ("wan",), 1, (0, 0), 0)
        for session in range(2)
        for method, work in (("replay", 1), ("kv_transfer", 2))
    )
    table = CandidateTable(
        scenario.sessions, candidates, csr_matrix((
            np.ones(4), ((0, 0, 1, 1), np.arange(4)),
        )), csr_matrix((
            np.full(4, .6), ((0, 1, 0, 1), np.arange(4)),
        ), shape=(2, 4)), ("replay", "kv"), (10, 10), ("s", "s"), 10,
    )
    arch = architecture(normal=1, emergency=1, stable=1)
    arch = replace(arch, pools=(DestinationPool(
        "p", "q", (DestinationReplica("t0"),), "r", ("wan",),
    ),))

    selected = _greedy_coupled(
        table, power.drain_gain(("a", "b")), power, arch, scenario, "normal",
    )
    prefix = _greedy_prefix(
        table, power.drain_gain(("a", "b")), power, arch, scenario, "normal",
    )

    assert {candidates[i].session for i in selected} == {0, 1}
    assert {candidates[i].method for i in selected} == {
        "replay", "kv_transfer",
    }
    assert {candidates[i].session for i in prefix} == {0, 1}
    assert {candidates[i].method for i in prefix} == {"replay", "kv_transfer"}


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


def test_coupled_recovery_packs_once_and_falls_back(tmp_path, monkeypatch):
    profile, scenario = model(tmp_path), problem()
    power = ExpectedPower(scenario, profile)
    candidates = tuple(
        Candidate(i, "replay", 0, 1, 1, 1, ("wan",), 1, (0, 0), 0)
        for i in range(2)
    )
    table = CandidateTable(
        scenario.sessions, candidates, csr_matrix(np.eye(2)),
        csr_matrix((np.full(2, .1), (np.zeros(2), np.arange(2))), shape=(1, 2)),
        ("route",), (1,), ("fraction",), 10,
    )
    arch = replace(architecture(normal=1, emergency=1), pools=(
        architecture(normal=1, emergency=1).pools[0],
    ))
    patterns = {"s0": {(0,)}, "s1": {(1,)}}
    target = power.drain_gain(("a", "b"))
    real, calls = pool_planner._pack, 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return real(*args)

    monkeypatch.setattr(pool_planner, "_pack", counted)
    lazy = _recover_coupled(
        table, power, patterns, target, arch, scenario, "normal",
    )
    lazy_calls, calls = calls, 0
    eager = _recover_coupled(
        table, power, patterns, target, arch, scenario, "normal",
        eager_pack=True,
    )

    assert lazy == eager == {0, 1}
    assert lazy_calls == 1
    assert calls == 2

    calls = 0
    def reject_final_once(*args):
        nonlocal calls
        calls += 1
        return (None, ()) if calls == 1 else real(*args)

    monkeypatch.setattr(pool_planner, "_pack", reject_final_once)
    assert _recover_coupled(
        table, power, patterns, target, arch, scenario, "normal",
    ) == eager
    assert calls == 3


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
        "kv:p0", "migration:p0:replay", "migration:p0:kv_transfer",
    )
    for method in ("replay", "kv_transfer"):
        row = table.resource_names.index(f"migration:p0:{method}")
        assert {
            candidate.method for i, candidate in enumerate(table.candidates)
            if table.resources[row, i]
        } == {method}


def test_equivalent_pools_share_candidate_physics_not_capacity_rows(tmp_path, monkeypatch):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    calls, duration = 0, pool_planner._duration

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return duration(*args, **kwargs)

    monkeypatch.setattr(pool_planner, "_duration", counted)
    table = candidate_table(
        scenario, profile, architecture(normal=1, emergency=1, stable=1),
        "normal", ExpectedPower(scenario, profile),
    )

    choices = {
        (candidate.session, candidate.method, candidate.pool): candidate
        for candidate in table.candidates
    }
    for session in range(len(table.sessions)):
        for method in ("replay", "kv_transfer"):
            left, right = choices[session, method, 0], choices[session, method, 1]
            assert replace(left, pool=0) == replace(right, pool=0)
    assert calls == 2 * len(table.sessions)
    assert any(name.startswith("service:p0") for name in table.resource_names)
    assert any(name.startswith("service:p1") for name in table.resource_names)


def test_streamed_candidates_and_columns_equal_exhaustive_table(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    arch = architecture(normal=1, emergency=1, stable=1)
    power = ExpectedPower(scenario, profile)
    oracle = _candidate_oracle(scenario, profile, arch, "normal", power)
    table = candidate_table(scenario, profile, arch, "normal", power)
    streamed = tuple(
        candidate for j in range(len(oracle.sessions))
        for candidate in oracle.choices(j)
    )
    matrix = pool_planner.csc_matrix(table.resources)
    table_rows = {name: row for row, name in enumerate(table.resource_names)}

    assert streamed == table.candidates
    for column, candidate in enumerate(streamed):
        expected = {
            table.resource_names[row]: value
            for row, value in zip(
                matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]],
                matrix.data[matrix.indptr[column]:matrix.indptr[column + 1]],
            )
        }
        actual = {
            oracle.specs[row][1]: value for row, value in oracle.column(candidate)
            if oracle.specs[row][1] in table_rows
        }
        assert actual == pytest.approx(expected)


def test_pricing_soa_reconstructs_every_candidate_column(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    arch = architecture(normal=1, emergency=1, stable=1)
    oracle = _candidate_oracle(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    soa = _pricing_soa(oracle)

    assert len(oracle.options) == 4
    assert len(oracle.signatures) == 2
    assert oracle.option_signatures[0] == oracle.option_signatures[2]
    assert oracle.option_signatures[1] == oracle.option_signatures[3]
    expected_masks = np.zeros(len(oracle.sessions), np.uint16)
    for j in range(len(oracle.sessions)):
        for candidate in oracle.choices(j):
            option = oracle.option_for[candidate.pool, candidate.method]
            expected_masks[j] |= np.uint16(1 << option)
            signature = soa.option_signatures[option]
            assert soa.features[j, signature] == pytest.approx(
                oracle.feature(candidate),
            )
            start, end = soa.option_starts[option:option + 2]
            actual = tuple(zip(
                soa.resource_rows[start:end],
                soa.resource_coefficients[start:end] @ soa.features[j, signature],
            ))
            expected = oracle.column(candidate)
            assert tuple(row for row, _ in actual) == tuple(row for row, _ in expected)
            assert tuple(value for _, value in actual) == pytest.approx(
                tuple(value for _, value in expected),
            )
    assert np.array_equal(soa.feasible, expected_masks)
    assert soa.resource_rows[
        soa.option_starts[0]:soa.option_starts[1]
    ].tolist() != soa.resource_rows[
        soa.option_starts[2]:soa.option_starts[3]
    ].tolist()


def test_native_pricing_matches_complete_python_sweep(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    oracle = _candidate_oracle(
        scenario, profile, architecture(normal=1, emergency=1, stable=1),
        "normal", ExpectedPower(scenario, profile),
    )
    native = _native_pricing_oracle(oracle)
    resource_duals = np.linspace(.01, .03, len(oracle.specs))
    session_duals = np.linspace(0, .02, len(oracle.sessions))
    eta = .7
    expected, repair, minimum, evaluated = [], 0.0, np.inf, 0
    for j in range(len(oracle.sessions)):
        choices = []
        for candidate in oracle.choices(j):
            option = oracle.option_for[candidate.pool, candidate.method]
            reduced = (
                candidate.duration_s / oracle.migration_horizon_s
                + sum(resource_duals[row] * value
                      for row, value in oracle.column(candidate))
                - eta * oracle.gains[j] + session_duals[j]
            )
            choices.append((reduced, option, candidate))
            evaluated += 1
        session_minimum = min(value for value, _, _ in choices)
        repair += max(0, -session_minimum)
        minimum = min(minimum, session_minimum)
        reduced, option, candidate = min(choices)
        if reduced < 0:
            expected.append((reduced, j, option, candidate))
    ranks = {session.session_id: rank for rank, session in enumerate(sorted(
        oracle.sessions, key=lambda session: session.session_id,
    ))}
    expected.sort(key=lambda row: (row[0], ranks[oracle.sessions[row[1]].session_id], row[2]))
    sweep = native.price(2, eta, resource_duals, session_duals, len(oracle.sessions), 0)

    assert sweep["session_indices"].tolist() == [row[1] for row in expected]
    assert sweep["option_indices"].tolist() == [row[2] for row in expected]
    assert sweep["candidate_ids"].tolist() == [
        (row[1] << 4) | row[2] for row in expected
    ]
    assert sweep["reduced_costs"] == pytest.approx([row[0] for row in expected])
    assert sweep["phase2_costs"] == pytest.approx([
        row[3].duration_s / oracle.migration_horizon_s for row in expected
    ])
    assert sweep["candidate_features"].reshape(-1, 7) == pytest.approx(
        np.asarray([oracle.feature(row[3]) for row in expected]),
    )
    assert sweep["repair_sum"] == pytest.approx(repair)
    assert sweep["minimum_reduced_cost"] == pytest.approx(minimum)
    assert sweep["evaluated_choices"] == evaluated
    starts = sweep["resource_starts"]
    for column, row in enumerate(expected):
        start, end = starts[column:column + 2]
        expected_column = oracle.column(row[3])
        assert sweep["resource_rows"][start:end].tolist() == [
            resource for resource, _ in expected_column
        ]
        assert sweep["resource_values"][start:end] == pytest.approx([
            value for _, value in expected_column
        ])


def test_native_column_master_matches_python_lazy_solver(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    arch = architecture(normal=1, emergency=1, stable=1)
    power = ExpectedPower(scenario, profile)
    target = sum(power.marginal(session.session_id)
                 for session in scenario.sessions) / 2
    python_stats, native_stats = {}, {}
    python_table, python_selected = _lp_column_generation_lazy(
        _candidate_oracle(scenario, profile, arch, "normal", power),
        target, python_stats,
    )
    native_table, native_selected = _lp_column_generation_native(
        _candidate_oracle(scenario, profile, arch, "normal", power),
        target, native_stats,
    )

    totals = lambda table, selected: (
        sum(table.candidates[i].gain_w for i in selected),
        sum(table.candidates[i].migration_work_s for i in selected),
    )
    assert totals(native_table, native_selected) == pytest.approx(
        totals(python_table, python_selected),
    )
    assert native_stats["phase1_shortfall"] == pytest.approx(
        python_stats["phase1_shortfall"], abs=1e-7,
    )
    assert native_stats["phase2"]["upper"] == pytest.approx(
        python_stats["phase2"]["upper"], abs=1e-7,
    )
    assert native_stats["phase1"]["gap"] <= pool_planner.COLUMN_GAP_TOLERANCE
    assert native_stats["phase2"]["gap"] <= pool_planner.COLUMN_GAP_TOLERANCE
    assert native_stats["phase1"]["evaluated_choices"] > 0
    assert native_stats["phase2"]["materialize_s"] >= 0
    assert native_stats["completion_s"] >= 0
    assert native_stats["table_s"] >= 0


def test_native_pricing_is_stable_certified_and_transactional():
    from _queue_haul_native import PricingOracle

    coefficients = np.zeros((2, 7))
    coefficients[:, 0] = (1, 2)
    native = PricingOracle(
        3, 1, 2, 1, 10.0,
        np.array([5., 5., 2.]),
        np.array([[1., 0, 0, 0, 2, 0, 0],
                  [1., 0, 0, 0, 4, 0, 0],
                  [3., 0, 0, 0, 1, 0, 0]]).ravel(),
        np.array([3, 3, 1], np.uint16), np.zeros(2, np.uint16),
        np.array([0, 1, 2], np.int32), np.zeros(2, np.int32),
        coefficients.ravel(), np.array([2, 0, 1], np.uint32),
    )
    sweep = native.price(2, 1, np.array([.5]), np.array([0., 1., 0.]), 2, 0)
    assert sweep["candidate_ids"].tolist() == [0, 16]
    assert sweep["reduced_costs"] == pytest.approx([-4.3, -3.1])
    assert sweep["repair_sum"] == pytest.approx(7.8)
    assert sweep["minimum_reduced_cost"] == pytest.approx(-4.3)
    assert sweep["violating_sessions"] == 3
    assert sweep["evaluated_choices"] == 5
    with pytest.raises(ValueError, match="uncommitted"):
        native.price(2, 1, np.array([.5]), np.array([0., 1., 0.]), 2, 0)
    with pytest.raises(ValueError, match="does not match"):
        native.commit(sweep["epoch"], np.array([0], np.uint64))
    native.discard(sweep["epoch"])

    tied = native.price(1, 10, np.zeros(1), np.zeros(3), 1, 0)
    assert tied["effective_eta"] == 1
    assert tied["candidate_ids"].tolist() == [16]
    assert tied["repair_sum"] == pytest.approx(12)
    native.commit(tied["epoch"], tied["candidate_ids"])
    remaining = native.price(1, 1, np.zeros(1), np.zeros(3), 1, 0)
    assert remaining["candidate_ids"].tolist() == [17]
    assert remaining["repair_sum"] == pytest.approx(12)
    native.discard(remaining["epoch"])


@pytest.mark.parametrize("argument,value", [
    (9, np.array([1, 1, 2], np.int32)),
    (9, np.array([0, -1, 2], np.int32)),
    (7, np.array([4, 3, 1], np.uint16)),
    (12, np.array([0, 0, 2], np.uint32)),
])
def test_native_pricing_rejects_malformed_soa(argument, value):
    from _queue_haul_native import PricingOracle

    coefficients = np.zeros((2, 7))
    coefficients[:, 0] = (1, 2)
    arguments = [
        3, 1, 2, 1, 10., np.array([5., 5., 2.]), np.zeros(21),
        np.array([3, 3, 1], np.uint16), np.zeros(2, np.uint16),
        np.array([0, 1, 2], np.int32), np.zeros(2, np.int32),
        coefficients.ravel(), np.arange(3, dtype=np.uint32),
    ]
    arguments[argument] = value
    with pytest.raises(ValueError, match="invalid pricing SoA"):
        PricingOracle(*arguments)


def test_native_pricing_requires_complete_nonoverlapping_chunks():
    from _queue_haul_native import PricingOracle

    native = PricingOracle.allocate(
        2, 1, 1, 1, 10., np.zeros(1, np.uint16),
        np.array([0, 1], np.int32), np.zeros(1, np.int32), np.zeros(7),
    )
    with pytest.raises(ValueError, match="incomplete"):
        native.price(1, 1, np.zeros(1), np.zeros(2), 1, 0)
    chunk = (np.ones(1), np.zeros(7), np.ones(1, np.uint16))
    native.load(0, *chunk, np.zeros(1, np.uint32))
    with pytest.raises(ValueError, match="invalid pricing SoA chunk"):
        native.load(0, *chunk, np.zeros(1, np.uint32))
    native.load(1, *chunk, np.ones(1, np.uint32))
    assert native.price(1, 1, np.zeros(1), np.zeros(2), 1, 0)[
        "evaluated_choices"
    ] == 2


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


def test_exact_oracle_enforces_each_method_budget_per_replica(tmp_path):
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


def test_replica_allows_replay_and_kv_work_to_overlap():
    scenario = problem()
    arch = architecture(normal=1, emergency=1, stable=1)
    arch = replace(arch, pools=(DestinationPool(
        "p", "q", (DestinationReplica("t0"),), "r", ("wan",),
    ),))
    candidates = tuple(
        Candidate(session, method, 0, 1, 6, 6, ("wan",), 1, (0, 0), 1)
        for session in range(2) for method in ("replay", "kv_transfer")
    )
    table = CandidateTable(
        scenario.sessions, candidates, csr_matrix((
            np.ones(4), ((0, 0, 1, 1), np.arange(4)),
        )), csr_matrix((0, 4)), (), (), (), 10,
    )

    assert exact_replica_assignment(
        table, {0, 3}, arch, scenario, "normal",
    ) == {0: "t0", 3: "t0"}
    assert exact_replica_assignment(
        table, {0, 2}, arch, scenario, "normal",
    ) is None


def test_execution_independently_rejects_stable_overflow():
    arch = architecture(normal=.3, emergency=.3, stable=.3,
                        baselines=((.2, 0), (0, 0)))
    move = PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0")

    with pytest.raises(ValueError, match="stable envelope"):
        validate_destination_execution(problem(), arch, (move,))
