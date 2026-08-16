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
- Let a matched oracle silently retry under a different admission mode.
- Validate execution against the admission envelope instead of stable capacity.
- Treat a service percentage as sessions or omit the migration-window units.
- Divide debt by total capacity instead of post-migration spare capacity.
- Mark positive debt feasible when post-migration service has no recovery spare.
- Lose physical units or pool/facet identity in normalized planner rows.
- Compare Replay seconds with KV bytes in the work-first objective.
- Reuse endpoint replica-seconds as the objective after route overlap, making
  route bandwidth invisible even when predicted action duration changes.
- Treat unused early capacity as if it could serve transition work arriving late.
- Credit node shutdown during session selection instead of only after planning.
- Rank sessions only by initial marginal power and miss a feasible source prefix.
- Sum initial marginal watts instead of inverting the nonlinear one-source phase curve.
- Apply one additive phase-load target across multiple source instances.
- Correct the LP power target but leave fixed-action baselines on marginal watts.
- Assign every Lagrangian prefix member to the same cheap pool and fail concrete packing.
- Serialize replay and KV work despite having separate measured aggregate caps.
- Relax aggregate method constraints but retain cross-method serialization in packing.
- Choose one cheap action per session before exploring feasible mixed-method patterns.
- Materialize every source prefix even though recovery retains a bounded frontier.
- Reverse or mis-scale the HiGHS target row or skip maximum-gain fallback.
- Route the existing `lp` solver through HiGHS instead of keeping it additive.
- Charge migration work during shortfall minimization or omit the session dual.
- Run Phase II at an infeasible requested target instead of maximum attainable gain.
- Reuse candidate physics across pools with different types, routes, or source loads.
- Drop Replay inside regional timing support when its base rate curve is narrower.
- Hide an internal timing error by treating it as an unsupported candidate.
- Enforce regional timing support for Replay execution but not KV execution.
- Apply loaded slowdown to WAN bytes or the mode boundary instead of endpoint
  work at the actual incumbent load.
- Charge measured loaded wall time again as transient service debt.
- Double-count the session dual while repairing a tolerated reduced-cost violation.
- Omit the Phase-I shortfall dual cap or stop before the global gap closes.
- Merge pool variables that share physics or misalign SoA resource templates.
- Re-scan every candidate for every session in the feasible-random baseline.
- Repack every retained action after dropping one fragmented packing candidate.
- Scale the wrong method or physical resource when applying migration headroom.
- Report a capacity shadow price with the wrong sign or normalization.
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
                         DestinationType, FluidMigrationService, LoadedCoefficients,
                         MigrationComponents)
from planner import plan, source_power
from repair_controller import (Assignment, Attempt, LedgerSnapshot, PrefillCapacity,
                               RepairRequest)
from pool_planner import (Candidate, CandidateTable, _destination_duration, _event_bounds,
                          _baseline_policy,
                          _candidate_oracle,
                          _greedy, _greedy_lagrangian,
                          _lagrangian_source_prefix,
                          _lp,
                          _lp_column_generation, _lp_column_generation_lazy,
                          _lp_column_generation_native,
                          _lp_column_generation_persistent,
                          _lp_highs,
                          _mode_boundary_rho,
                          _native_pricing_oracle, _pricing_soa,
                          _dual_resource_limits, _retained_prefixes,
                          _assignment_valid, _pack,
                          _source_removed_gain,
                          _recover_lagrangian, _service_trace, candidate_table,
                          destination_service_execution, exact_replica_assignment,
                          fractional_power_opportunity, phase_one_capacity_duals,
                          service_debt,
                          validate_destination_execution)
from power_model import ExpectedPower
from profiles import PhasePower
from simulate import (PlannedMove, PowerNode, ServingInstance, SessionExecution,
                      SimSession, NetworkLink, execute)
from test_execution_simulator import model
from test_planner import PATHS, problem


FP = CompatibilityFingerprint("m", "t", "log", "kv")


def _phase_target_case(tmp_path, split_sources=False):
    profile = model(tmp_path, tp=1)
    phase = PhasePower(
        10, 90, 1, 1, ((0, 0), (3, 0), (0, 3)), 0, 1, (), "0" * 64,
    )
    profile = replace(
        profile, cases={"central": replace(profile.case(), phase_power=phase)},
        max_power_load=3,
    )
    base = problem()
    sessions = (
        replace(base.sessions[0], session_id="large-a", expected_f=1, expected_g=0),
        replace(base.sessions[1], session_id="large-b",
                source_instance="s1" if split_sources else "s0",
                expected_f=1, expected_g=0),
        replace(base.sessions[0], session_id="small", expected_f=.1, expected_g=0),
    )
    scenario = replace(base, sessions=sessions)
    initial = source_power(scenario, profile)
    return profile, replace(scenario, power_limit_w=initial - 30), initial


@pytest.mark.parametrize("solver", (
    "lp_highs", "greedy", "isolated_fastest", "replay_only", "kv_only",
))
def test_phase_power_target_uses_exact_removed_load_for_all_power_aware_policies(
        tmp_path, solver):
    profile, scenario, initial = _phase_target_case(tmp_path)

    result = plan(
        scenario, profile, PATHS, solver, destination=architecture(),
        admission_mode="normal",
    )

    # P(2.1)-P(.1)=52.79 W, while either one-session removal sheds <16 W.
    assert {move.session_id for move in result.moves} == {"large-a", "large-b"}
    assert initial - result.planned_source_power_w >= 30


def test_phase_power_target_rejects_multiple_source_instances(tmp_path):
    profile, scenario, _ = _phase_target_case(tmp_path, split_sources=True)

    with pytest.raises(ValueError, match="one source instance"):
        plan(
            scenario, profile, PATHS, "lp_highs", destination=architecture(),
            admission_mode="normal",
        )


def test_fractional_phase_opportunity_reports_joint_nonlinear_watts(tmp_path):
    profile, scenario, _ = _phase_target_case(tmp_path)
    power = ExpectedPower(scenario, profile)
    sessions = scenario.sessions
    candidates = tuple(
        Candidate(i, "replay", 0, power.marginal(session.session_id),
                  1, 1, (), 0, (0, 0), 0)
        for i, session in enumerate(sessions)
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(len(sessions))),
        csr_matrix((0, len(sessions))), (), (), (), 9,
    )
    expected = power.drain_gain(session.session_id for session in sessions)

    assert sum(candidate.gain_w for candidate in candidates) < expected
    assert fractional_power_opportunity(table, power) == pytest.approx(expected)


def test_public_solver_surface_hard_fails_retired_greedies(tmp_path):
    profile, scenario = model(tmp_path), problem()
    with pytest.raises(ValueError, match="requires a destination architecture"):
        plan(scenario, profile, PATHS, "greedy_lagrangian")
    for retired in ("greedy_bundle", "greedy_coupled", "greedy_prefix"):
        with pytest.raises(ValueError, match="supports pool-aware LP and greedy"):
            plan(
                scenario, profile, PATHS, retired,
                destination=architecture(),
            )


def test_exact_max_shed_uses_phase_power_load_not_candidate_credit():
    sessions = (SimpleNamespace(session_id="prefill"),
                SimpleNamespace(session_id="decode"))
    candidates = tuple(Candidate(i, "replay", 0, 100 - i, 1, 1, (), 0, (0, 0), 0)
                       for i in range(2))
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(2)), csr_matrix([[1, 1]]),
        ("only-one",), (1,), ("count",), 1,
    )
    power = SimpleNamespace(route={"prefill": "source", "decode": "source"},
                            ell={"prefill": .1, "decode": .9})
    assert pool_planner._max_shed(table, power) == {1}


def test_pool_power_blind_lp_uses_uniform_pack_average_gains(monkeypatch, tmp_path):
    scenario = replace(problem(), sessions=(
        replace(problem().sessions[0], expected_f=5),
        replace(problem().sessions[1], expected_f=45),
    ))
    profile, seen = model(tmp_path, tp=1), []
    original = pool_planner._lp_highs

    def capture(table, target):
        seen.append([candidate.gain_w for candidate in table.candidates])
        return original(table, target)

    monkeypatch.setattr(pool_planner, "_lp_highs", capture)
    plan(scenario, profile, PATHS, "lp_power_blind", destination=architecture())

    expected = ExpectedPower(scenario, profile).drain_gain(("a", "b")) / 2
    assert seen and all(gains == pytest.approx([expected] * len(gains))
                        for gains in seen)


def _reference_marginal(power, session_id):
    source = power.route[session_id]
    owned = power.instance_slots[source]
    by_node, share = {}, power.ell[session_id] / len(owned)
    for node_id, slot in owned:
        by_node.setdefault(node_id, list(power.slots[node_id]))[slot] -= share
    return sum(
        power.node_power[node_id] - power._power(
            node_id, slots,
            power.scenario.final_state
            if power.dependents[node_id] - power.removed == {session_id}
            else "awake",
        )
        for node_id, slots in by_node.items() if power.nodes[node_id].local
    )


@pytest.mark.parametrize("scope", ("gpu", "server"))
@pytest.mark.parametrize("final,count", (("awake", 2), ("sleep", 1), ("off", 1)))
def test_fast_marginal_matches_full_node_recomputation(tmp_path, scope, final, count):
    profile, base = replace(model(tmp_path, tp=2), power_scope=scope), problem(final=final)
    sessions = tuple(
        replace(base.sessions[0], session_id=str(index), source_instance="s0")
        for index in range(count)
    )
    scenario = replace(
        base, sessions=sessions, nodes=(PowerNode("n0", 2, True),),
        instances=(ServingInstance("s0", ("n0", "n0")),), links=(),
    )
    power = ExpectedPower(scenario, profile)

    for session in sessions:
        assert power.marginal(session.session_id) == pytest.approx(
            _reference_marginal(power, session.session_id), abs=1e-12,
        )


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


def test_max_shed_selects_the_exact_source_load_optimum():
    table = CandidateTable(
        tuple(SimpleNamespace(session_id=name) for name in "abc"),
        (
            Candidate(0, "replay", 0, 100, 1, 1, (), 0, (0, 0), 0),
            Candidate(1, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
            Candidate(2, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
        ),
        csr_matrix(np.eye(3)), csr_matrix(np.array(((1, .5, .5),))),
        ("capacity",), (1,), ("fraction",), 1,
    )
    power = SimpleNamespace(
        ell={"a": 4, "b": 3, "c": 3},
        route={name: "source" for name in "abc"},
    )

    assert pool_planner._max_shed(table, power) == {1, 2}


@pytest.mark.parametrize("policy,method", (
    ("replay_only", "replay"), ("kv_only", "kv_transfer"),
))
def test_fixed_method_baselines_share_candidate_resources(policy, method):
    table = _hand_lp_table()

    selected = _baseline_policy(table, 3, policy, 0)

    assert {table.candidates[i].method for i in selected} == {method}
    assert np.asarray(table.resources[:, list(selected)].sum(1)).max() <= 1


def test_isolated_fastest_locks_method_before_shared_admission():
    table = _hand_lp_table()

    selected = _baseline_policy(table, 3, "isolated_fastest", 0)

    assert selected == {1, 2}


def test_seeded_random_baseline_never_exceeds_shared_resources():
    table = _hand_lp_table()

    first = _baseline_policy(table, 3, "random", 7)
    second = _baseline_policy(table, 3, "random", 7)

    assert first == second
    assert np.asarray(table.resources[:, list(first)].sum(1)).max(initial=0) <= 1
    assert len({table.candidates[i].session for i in first}) == len(first)


def test_random_groups_candidates_in_one_pass():
    class Counted:
        def __init__(self, rows):
            self.rows, self.iterations = rows, 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    table = _hand_lp_table()
    candidates = Counted(table.candidates)

    _baseline_policy(replace(table, candidates=candidates), 3, "random", 7)

    assert candidates.iterations == 1


def test_random_chooses_among_currently_feasible_methods():
    table = _hand_lp_table()
    table = replace(table, resources=csr_matrix(np.array([[.4, .8, .6]])))

    selected = _baseline_policy(table, 3, "random", 3)

    assert selected == {0, 2}


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


@pytest.mark.parametrize("solve", (_lp, _lp_highs, _lp_column_generation))
def test_lp_objective_uses_duration_not_endpoint_capacity_work(solve):
    candidates = (
        Candidate(0, "replay", 0, 1, 2, 3, (), 0, (0, 0), 0),
        Candidate(0, "kv_transfer", 0, 1, 1, 10, (), 0, (0, 0), 0),
    )
    table = CandidateTable(
        (), candidates, csr_matrix(np.ones((1, 2))),
        csr_matrix(np.array(((.2, .1),))), ("endpoint",), (10,),
        ("replica-s",), 10,
    )

    assert solve(table, 1) == {0}
    assert candidates[0].migration_work_s > candidates[1].migration_work_s
    assert candidates[0].objective_cost_s < candidates[1].objective_cost_s


@pytest.mark.parametrize("status, values", [
    ("failed", np.zeros(3)),
    (pool_planner.cp.OPTIMAL, None),
    (pool_planner.cp.OPTIMAL, np.full(3, np.nan)),
])
def test_clarabel_invalid_final_solution_falls_back_to_highs(
    monkeypatch, status, values,
):
    table, stats, called = _hand_lp_table(), {}, []

    class FailedProblem:
        solver_stats = SimpleNamespace(solve_time=0, num_iters=0)

        def __init__(self, objective, constraints):
            self.objective, self.status, self.value = objective, status, None

        def solve(self, **kwargs):
            for variable in self.objective.args[0].variables():
                variable.save_value(values)

    expected = {1, 2}
    monkeypatch.setattr(pool_planner.cp, "Problem", FailedProblem)
    monkeypatch.setattr(pool_planner, "_lp_highs", lambda t, target, s: (
        called.append((t, target, s)) or expected))

    assert pool_planner._lp(table, 3, stats) == expected
    assert called == [(table, 3, stats)]


def test_clarabel_solver_error_falls_back_to_highs(monkeypatch):
    table, expected = _hand_lp_table(), {1, 2}
    monkeypatch.setattr(pool_planner.cp.Problem, "solve", lambda *args, **kwargs: (
        _ for _ in ()).throw(pool_planner.cp.error.SolverError()))
    monkeypatch.setattr(pool_planner, "_lp_highs", lambda *args: expected)
    assert pool_planner._lp(table, 3) == expected

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
        Candidate(0, "replay", 0, 1, 100, 100, (), 0, (0, 0), 0),
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

    def semantic(source, indices):
        return {(source.candidates[i].session, source.candidates[i].method)
                for i in indices}
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
    work = np.array([c.objective_cost_s for c in candidates]) / 10
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
    monkeypatch.setattr(pool_planner, "_pack", lambda *args, **kwargs: ({}, ()))

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


def test_phase_one_capacity_dual_matches_hand_worked_fractional_knapsack():
    candidates = (
        Candidate(0, "replay", 0, 2, 1, 1, (), 0, (0, 0), 0),
        Candidate(1, "replay", 0, 1, 1, 1, (), 0, (0, 0), 0),
    )
    table = CandidateTable(
        (SimpleNamespace(session_id="a"), SimpleNamespace(session_id="b")),
        candidates, csr_matrix(np.eye(2)),
        csr_matrix(np.array(((.6, .6),))),
        ("migration:p:replay",), (10,), ("replica-s",), 10,
    )

    maximum, duals = phase_one_capacity_duals(table)

    assert maximum == pytest.approx(8 / 3)
    assert duals == pytest.approx((5 / 3,))


def architecture(*, normal=.3, emergency=.5, stable=1, baselines=((0, 0), (0, 0)),
                 kv=1000, methods=("replay", "kv_transfer"), compatibility=FP,
                 residency=None, routes=(("wan",), ("wan",)), block=1,
                 flex=None, debt=0, fluid=None):
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
        f"r{i}", route, methods, flex, debt, fluid,
    ) for i, (baseline, route) in enumerate(zip(baselines, routes)))
    return DestinationArchitecture(DESTINATION_SCHEMA, FP, (q,), pools, residency)


def repair_request(*, target, deadline, attempts=(), routes=()):
    return RepairRequest(1, "soft:1", 0, LedgerSnapshot(
        5, target, deadline, 0, 0, 0, frozenset(), frozenset(("a", "b")),
        attempts, routes, (),
    ))


def test_repair_continues_remaining_work_on_the_current_replica(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    current = Attempt(
        "a", 0, Assignment("replay", "t0", "p0"), "running",
        100, 99, 5, 105, rate=1,
    )

    result = pool_planner.repair_destination(
        scenario, profile, architecture(),
        repair_request(target=10, deadline=9, attempts=(current,), routes=(("wan", 1),)),
    )

    assert result.reaches_target
    assert result.moves[0].assignment == current.assignment
    assert result.moves[0].duration_s == pytest.approx(1)


def test_repair_keeps_locked_running_work_as_a_fixed_move(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    current = Attempt(
        "a", 0, Assignment("replay", "t0", "p0"), "running",
        100, 99, 5, 8, rate=1, repairable=False,
    )

    result = pool_planner.repair_destination(
        scenario, profile, architecture(),
        repair_request(target=10, deadline=9, attempts=(current,),
                       routes=(("wan", 1),)),
    )

    assert result.reaches_target
    assert len(result.moves) == 1
    assert result.moves[0].assignment == current.assignment
    assert result.moves[0].duration_s == pytest.approx(1)


def test_repair_prefill_observation_scales_only_the_named_pool():
    original = architecture()
    east = original.pools[0]
    reference = original.type_by_id[east.type_id].prefill.at(512)

    repaired = pool_planner._repair_architecture(
        original, (PrefillCapacity(east.pool_id, 512, reference / 10),))

    east_type = repaired.type_by_id[repaired.pools[0].type_id]
    west_type = repaired.type_by_id[repaired.pools[1].type_id]
    assert east_type.prefill.at(512) == pytest.approx(reference / 10)
    assert repaired.pools[0].replicas[0].baseline_work[0] \
        == pytest.approx(original.pools[0].replicas[0].baseline_work[0] * 10)
    assert west_type.prefill.at(512) == pytest.approx(reference)


def test_repair_does_not_retain_progress_across_replicas(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    arch = architecture(routes=(("wan",),), baselines=((0, 0),), methods=("replay",))
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    current = Attempt(
        "a", 0, Assignment("replay", "t0", "p0"), "running",
        100, 99, 5, 105, rate=1,
    )

    result = pool_planner.repair_destination(
        scenario, profile, arch,
        repair_request(target=10, deadline=200, attempts=(current,), routes=(("wan", 100),)),
    )

    assert result.moves[0].assignment.destination in {"t0", "t1"}
    assert result.moves[0].duration_s > 1


def test_repair_reports_the_revised_maximum_for_an_impossible_target(tmp_path):
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)

    result = pool_planner.repair_destination(
        scenario, profile, architecture(normal=1, emergency=1),
        repair_request(target=100, deadline=20),
    )

    assert not result.reaches_target
    assert result.attainable_watts == pytest.approx(20)


def test_fluid_pool_converts_method_work_to_replica_seconds(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 1, "kv_transfer": 1},
        {"replay": 1, "kv_transfer": 1}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    table = candidate_table(scenario, profile, arch, "normal", ExpectedPower(scenario, profile))
    single = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"),),
    ),))
    single_table = candidate_table(
        scenario, profile, single, "normal", ExpectedPower(scenario, profile),
    )
    row = table.resource_names.index("migration:p0:replay")
    kv_row = table.resource_names.index("migration:p0:kv_transfer")
    wan_row = table.resource_names.index("route:wan")
    replay = [i for i, candidate in enumerate(table.candidates)
              if candidate.method == "replay"]
    kv = [i for i, candidate in enumerate(table.candidates)
          if candidate.method == "kv_transfer"]

    assert table.resource_capacities[row] == pytest.approx(2 * 9)
    assert table.resource_capacities[kv_row] == pytest.approx(2 * 9)
    assert table.resource_capacities[wan_row] == pytest.approx(100 * 9)
    assert table.resource_capacities[kv_row] == pytest.approx(
        2 * single_table.resource_capacities[
            single_table.resource_names.index("migration:p0:kv_transfer")
        ]
    )
    assert all(table.candidates[i].migration_work_s == pytest.approx(
        table.candidates[i].route_bytes / service.kv_ingest_bytes_per_s
    ) for i in kv)
    assert table.resource_capacities[wan_row] == pytest.approx(
        single_table.resource_capacities[
            single_table.resource_names.index("route:wan")
        ]
    )
    assert np.asarray(table.resources[row, replay].sum()) == pytest.approx(
        sum(table.candidates[i].migration_work_s for i in replay)
        / table.resource_capacities[row]
    )


def test_fluid_coupling_is_a_dimensionally_common_linear_envelope(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", .5,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    replay = next(i for i, candidate in enumerate(table.candidates)
                  if candidate.session == 0 and candidate.method == "replay")
    kv = next(i for i, candidate in enumerate(table.candidates)
              if candidate.session == 1 and candidate.method == "kv_transfer")
    rows = [table.resource_names.index(f"migration:p0:{method}")
            for method in ("replay", "kv_transfer")]
    used = np.asarray(table.resources[rows][:, [replay, kv]].sum(1)).ravel()
    r, k = (table.candidates[index].migration_work_s for index in (replay, kv))

    assert used * np.asarray(table.resource_capacities)[rows] == pytest.approx(
        (r + .5 * k, .5 * r + k)
    )


def test_fluid_serial_route_charges_route_and_switch_to_shared_work(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    scenario, profile = problem(), model(tmp_path, switch=.25, tp=1)
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    selected = [next(i for i, candidate in enumerate(table.candidates)
                     if candidate.session == session and candidate.method == method)
                for session, method in ((0, "replay"), (1, "kv_transfer"))]
    row = table.resource_names.index("migration:p0:replay")
    used = float(table.resources[row, selected].sum()) \
        * table.resource_capacities[row]
    compute = sum(
        table.candidates[index].migration_work_s
        - table.candidates[index].route_bytes / 100 - .25
        for index in selected
    )

    assert used == pytest.approx(
        compute + sum(table.candidates[index].route_bytes / 100 + .25
                      for index in selected)
    )


def test_fluid_execution_uses_the_planner_coupling_envelope(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", .5,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    selected = [next(candidate for candidate in table.candidates
                     if candidate.session == session and candidate.method == method)
                for session, method in ((0, "replay"), (1, "kv_transfer"))]
    moves = tuple(PlannedMove(
        table.sessions[candidate.session].session_id, "t0", candidate.method,
        order, ("wan",), destination_pool="p0",
    ) for order, candidate in enumerate(selected))
    r, k = (candidate.migration_work_s for candidate in selected)
    network = sum(candidate.route_bytes for candidate in selected) / 100

    result = execute(scenario, profile, moves, destination=arch)

    assert result.migration_makespan_s == pytest.approx(
        max(network, r + .5 * k, .5 * r + k)
    )


def test_fluid_loaded_slowdown_scales_endpoint_work_at_actual_load(tmp_path):
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    base = architecture(
        normal=1, emergency=2, stable=2, baselines=((.5, 0),),
        routes=(("wan",),), fluid=service,
    )
    loaded = LoadedCoefficients(
        (0, 1), (1, 2), (1, 1000), (1, 1000), "hand",
    )
    components = MigrationComponents(
        (1, 1000), (1, 1000), "hand", residual_s=2,
        kv_ingest_bytes_per_s=100,
    )
    arch = replace(
        base, types=(replace(
            base.types[0], loaded={"replay": loaded, "kv_transfer": loaded},
            migration={"replay": components, "kv_transfer": components},
        ),),
    )
    scenario, profile = problem(), model(
        tmp_path, switch=0, tp=1, replay_completion=2,
    )
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    replay = next(candidate for candidate in table.candidates
                  if candidate.session == 0 and candidate.method == "replay")
    kv = next(candidate for candidate in table.candidates
              if candidate.session == 0 and candidate.method == "kv_transfer")
    route = replay.route_bytes / 100
    replay_endpoint = replay.migration_work_s - route
    kv_route = kv.route_bytes / 100
    kv_endpoint = kv.migration_work_s - kv_route

    assert replay_endpoint == pytest.approx(1.5 * (10 / 100 + 2))
    assert kv_endpoint == pytest.approx(1.5 * (kv.route_bytes / 100 + 2))
    assert kv.duration_s == pytest.approx(
        max(kv_route, 1.5 * kv.route_bytes / 100) + 1.5 * 2
    )
    assert replay.transition_work == pytest.approx((10 / 100 + 2, 0))
    assert replay.duration_s == pytest.approx(
        max(route, 1.5 * 10 / 100) + 1.5 * 2
    )

    move = PlannedMove(
        "a", "t0", "replay", 0, ("wan",), destination_pool="p0",
    )
    result = execute(scenario, profile, (move,), destination=arch)

    assert result.migration_makespan_s == pytest.approx(
        replay.migration_work_s
    )


def test_fluid_loaded_slowdown_preserves_idle_columns(tmp_path):
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    base = architecture(
        normal=1, emergency=2, stable=2, baselines=((.5, 0),),
        routes=(("wan",),), fluid=service,
    )
    components = MigrationComponents(
        (1, 1000), (1, 1000), "hand", residual_s=2,
        kv_ingest_bytes_per_s=100,
    )
    identity = LoadedCoefficients(
        (0, 1), (1, 1), (1, 1000), (1, 1000), "identity",
    )
    loaded = replace(identity, slowdown=(1, 2), provenance="loaded")
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1,
                                        replay_completion=2)

    def table(coefficients):
        return candidate_table(
            scenario, profile, replace(base, types=(replace(
                base.types[0],
                loaded={method: coefficients for method in (
                    "replay", "kv_transfer")},
                migration={method: components for method in (
                    "replay", "kv_transfer")},
            ),)), "normal", ExpectedPower(scenario, profile),
        )

    idle, busy = table(identity), table(loaded)
    for method in ("replay", "kv_transfer"):
        a = next(candidate for candidate in idle.candidates
                 if candidate.session == 0 and candidate.method == method)
        b = next(candidate for candidate in busy.candidates
                 if candidate.session == 0 and candidate.method == method)
        assert b.route_bytes == a.route_bytes
        assert b.service_work == a.service_work
        assert b.kv_tokens == a.kv_tokens
        assert b.transition_work == a.transition_work
        assert b.migration_work_s > a.migration_work_s


def test_fluid_loaded_mixed_plan_matches_execution(tmp_path):
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    base = architecture(
        normal=1, emergency=2, stable=2, baselines=((.5, 0),),
        routes=(("wan",),), fluid=service,
    )
    loaded = LoadedCoefficients(
        (0, 1), (1, 2), (1, 1000), (1, 1000), "hand",
    )
    components = MigrationComponents(
        (1, 1000), (1, 1000), "hand", residual_s=2,
        kv_ingest_bytes_per_s=100,
    )
    identity = replace(loaded, slowdown=(1, 1), provenance="identity")
    arch = replace(base, types=(replace(
        base.types[0],
        loaded={"replay": loaded, "kv_transfer": identity},
        migration={method: components for method in ("replay", "kv_transfer")},
    ),))
    scenario, profile = problem(), model(
        tmp_path, switch=.25, tp=1, replay_completion=2,
    )
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    selected = [next(candidate for candidate in table.candidates
                     if candidate.session == session
                     and candidate.method == method)
                for session, method in ((0, "replay"), (1, "kv_transfer"))]
    moves = tuple(PlannedMove(
        table.sessions[candidate.session].session_id, "t0", candidate.method,
        order, ("wan",), destination_pool="p0",
    ) for order, candidate in enumerate(selected))

    result = execute(scenario, profile, moves, destination=arch)

    assert result.migration_makespan_s == pytest.approx(
        sum(candidate.migration_work_s for candidate in selected)
    )

    zero = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0", (0, 0)),),
    ),))
    identity_type = replace(
        zero.types[0], loaded={method: identity for method in (
            "replay", "kv_transfer")},
    )
    idle = candidate_table(
        scenario, profile, replace(zero, types=(identity_type,)), "normal",
        ExpectedPower(scenario, profile),
    )
    loaded_idle = candidate_table(
        scenario, profile, zero, "normal", ExpectedPower(scenario, profile),
    )
    assert [(row.method, row.migration_work_s, row.duration_s)
            for row in loaded_idle.candidates] == [
        (row.method, row.migration_work_s, row.duration_s)
        for row in idle.candidates
    ]

    with pytest.raises(ValueError, match="destination architecture"):
        replace(arch, pools=(replace(
            arch.pools[0], event_flex_fraction=0,
        ),))


def test_fluid_execution_serializes_route_and_switch_with_shared_work(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    scenario, profile = problem(), model(tmp_path, switch=.25, tp=1)
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    selected = [next(candidate for candidate in table.candidates
                     if candidate.session == session and candidate.method == method)
                for session, method in ((0, "replay"), (1, "kv_transfer"))]
    moves = tuple(PlannedMove(
        table.sessions[candidate.session].session_id, "t0", candidate.method,
        order, ("wan",), destination_pool="p0",
    ) for order, candidate in enumerate(selected))

    result = execute(scenario, profile, moves, destination=arch)

    assert result.migration_makespan_s == pytest.approx(
        sum(candidate.migration_work_s for candidate in selected)
    )


def test_fluid_execution_runs_disjoint_destination_routes_in_parallel(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0), (0, 0)),
        routes=(("wan",), ("wan2",)), fluid=service,
    )
    scenario = replace(problem(), links=(NetworkLink("wan", 100),
                                         NetworkLink("wan2", 50)))
    profile = model(tmp_path, switch=0, tp=1)
    moves = (
        PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0"),
        PlannedMove("b", "t1", "replay", 1, ("wan2",), destination_pool="p1"),
    )

    result = execute(scenario, profile, moves, destination=arch)

    assert result.migration_makespan_s == pytest.approx(2.025)


def test_fluid_serial_route_rejects_cross_pool_link_sharing():
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1, False,
    )

    with pytest.raises(ValueError, match="destination architecture"):
        architecture(normal=1, emergency=1, stable=1, fluid=service)


def test_migration_headroom_scales_only_its_pool_method_window(tmp_path):
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),),
    )
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    power = ExpectedPower(scenario, profile)
    full = candidate_table(scenario, profile, arch, "normal", power)
    limited = candidate_table(
        scenario, profile, replace(arch, pools=(replace(
            arch.pools[0], migration_headroom={"replay": .25}),)),
        "normal", power,
    )
    capacities = dict(zip(full.resource_names, full.resource_capacities))
    limited_capacities = dict(zip(
        limited.resource_names, limited.resource_capacities))

    assert limited_capacities["migration:p0:replay"] == pytest.approx(
        capacities["migration:p0:replay"] / 4)
    assert limited_capacities["migration:p0:kv_transfer"] == pytest.approx(
        capacities["migration:p0:kv_transfer"])
    assert limited_capacities["route:wan"] == pytest.approx(
        capacities["route:wan"])


def test_fluid_execution_uses_whole_pool_and_fitted_action_power(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 5, "kv_transfer": 1},
        {"replay": 7, "kv_transfer": 1}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    scenario, profile = problem(), model(
        tmp_path, switch=0, tp=1, replay_completion=2,
    )
    move = PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0")
    baseline = ExpectedPower(scenario, profile)

    result = execute(scenario, profile, (move,), destination=arch)
    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    planned = next(candidate for candidate in table.candidates
                   if candidate.session == 0 and candidate.method == "replay")

    assert result.sessions[0].committed_s == pytest.approx(1 + 2 / 8)
    assert planned.duration_s == pytest.approx(result.sessions[0].committed_s)
    assert result.power[0][1:] == pytest.approx((baseline.power(True), baseline.power(False)))
    assert result.power[1][1:] == pytest.approx(
        (baseline.power(True) + 5, baseline.power(False) + 2 * 7)
    )


def test_fluid_kv_execution_includes_post_ingest_residual(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    components = MigrationComponents((1, 1000), (1, 1000), "hand", residual_s=3)
    arch = replace(
        arch, types=(replace(arch.types[0], migration={
            "replay": components, "kv_transfer": components,
        }),), pools=(replace(
            arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
        ),),
    )
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    move = PlannedMove(
        "a", "t0", "kv_transfer", 0, ("wan",), destination_pool="p0",
    )

    result = execute(scenario, profile, (move,), destination=arch)

    assert result.sessions[0].committed_s == pytest.approx(4)


def test_fluid_execution_is_split_invariant_and_power_is_not_per_flow(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 5, "kv_transfer": 1},
        {"replay": 7, "kv_transfer": 1}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    base, profile = problem(), model(tmp_path, switch=0, tp=1)
    whole = replace(base, sessions=(SimSession("a", "s0", 10, 25, 0, 100),))
    split = replace(base, sessions=(
        SimSession("a", "s0", 5, 12.5, 0, 50),
        SimSession("b", "s1", 5, 12.5, 0, 50),
    ))
    one = execute(whole, profile, (
        PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0"),
    ), destination=arch)
    two = execute(split, profile, (
        PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0"),
        PlannedMove("b", "t1", "replay", 1, ("wan",), destination_pool="p0"),
    ), destination=arch)
    baseline = ExpectedPower(split, profile)

    assert one.migration_makespan_s == pytest.approx(two.migration_makespan_s)
    assert two.power[1][1:] == pytest.approx(
        (baseline.power(True) + 2 * 5, baseline.power(False) + 2 * 7)
    )


def test_fluid_replay_and_kv_move_simultaneously_on_fixed_wan(tmp_path):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    scenario, profile = problem(), model(tmp_path, switch=0, tp=1)
    moves = (
        PlannedMove("a", "t0", "replay", 0, ("wan",), destination_pool="p0"),
        PlannedMove("b", "t1", "kv_transfer", 1, ("wan",), destination_pool="p0"),
    )
    first = execute(scenario, profile, moves, destination=arch)
    second = execute(scenario, profile, (
        replace(moves[0], order=1), replace(moves[1], order=0),
    ), destination=arch)

    assert {row.start_s for row in first.network} == {0}
    assert {row.end_s for row in first.network} == {2}
    assert {row.session_id: row.committed_s for row in first.sessions} == pytest.approx(
        {row.session_id: row.committed_s for row in second.sessions}
    )


def test_fluid_deadline_repair_is_separate_and_recomputes_shortfall(
        tmp_path, monkeypatch):
    service = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    arch = architecture(normal=1, emergency=1, stable=1, fluid=service)

    def predicted(scenario, profile, moves, *args, **kwargs):
        makespan = 10 if len(moves) > 1 else 8
        return SimpleNamespace(
            migration_makespan_s=makespan, deadline_met=True,
            modeled_source_power_at_deadline_w=source_power(
                scenario, profile, [move.session_id for move in moves],
            ), pool_service=(),
        )

    pack, calls = pool_planner._pack, []

    def counted_pack(*args, **kwargs):
        calls.append(None)
        return pack(*args, **kwargs)

    monkeypatch.setattr(pool_planner, "predict", predicted)
    monkeypatch.setattr(pool_planner, "_pack", counted_pack)
    result = plan(problem(), model(tmp_path, switch=0, tp=1), PATHS, "greedy",
                  destination=arch)

    assert len(calls) == 1
    assert result.deadline_repair_count == 1
    assert result.packing_repair_count == 0
    assert result.predicted_migration_makespan_s <= 9
    assert result.power_shortfall_w > 0 and result.failure_reason == "target_unmet"


def test_fluid_replay_capacity_has_no_width_eight_admission_ceiling():
    candidates = tuple(
        Candidate(i, "replay", 0, 1, 5, 1, (), 0, (0, 0), 0)
        for i in range(16)
    )
    table = CandidateTable(
        (), candidates, csr_matrix(np.eye(16)),
        csr_matrix(np.full((1, 16), 1 / 16)),
        ("migration:p:replay",), (80,), ("serial-replay-s",), 10,
    )

    selected = _greedy(table, 16)

    assert len(selected) == 16
    assert np.asarray(table.resources[:, list(selected)].sum()).item() == pytest.approx(1)


def test_fluid_resource_boundaries_conserve_replay_and_mixed_wan():
    candidates = (
        Candidate(0, "replay", 0, 1, 30, 1, (), 60, (0, 0), 0),
        Candidate(1, "replay", 0, 1, 50, 1, (), 0, (0, 0), 0),
        Candidate(2, "kv_transfer", 0, 1, 40, 1, (), 40, (0, 0), 0),
        Candidate(3, "replay", 0, 1, 1e-6, 1, (), 1e-4, (0, 0), 0),
    )
    table = CandidateTable(
        (), candidates, csr_matrix(np.eye(4)),
        csr_matrix(np.array(((30 / 80, 50 / 80, 0, 1e-6 / 80),
                             (0, 0, 1, 0), (.6, 0, .4, 1e-6)))),
        ("migration:p:replay", "migration:p:kv_transfer", "route:wan"),
        (80, 40, 100), ("serial-replay-s", "bytes", "bytes"), 10,
    )
    def usage(selected):
        return np.asarray(table.resources[:, selected].sum(1)).ravel()

    assert usage([0, 1, 2]) == pytest.approx((1, 1, 1))
    assert np.any(usage([0, 1, 2, 3]) > 1)
    assert np.all(usage(sorted(_greedy(table, 4))) <= 1 + 1e-8)


def test_lagrangian_crosses_knee_with_packable_mixed_pools(tmp_path):
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

    selected = _greedy_lagrangian(table, 28, power, arch, scenario, "normal")

    assert _lagrangian_source_prefix(
        table, power, ((0, 1), (2, 1.5), (4, 15)), 1, 1,
    ) == (0, 2)  # objectives: 0, -5, -13.5, -10.5
    assert len(selected) == 3
    assert {candidates[i].session for i in selected} == {0, 1, 2}
    assert sorted(candidates[i].pool for i in selected) == [0, 0, 1]
    assert exact_replica_assignment(table, selected, arch, scenario, "normal") is not None


def test_lagrangian_finds_the_only_feasible_method_mix(tmp_path):
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

    selected = _greedy_lagrangian(
        table, power.drain_gain(("a", "b")), power, arch, scenario, "normal",
    )

    assert {candidates[i].session for i in selected} == {0, 1}
    assert {candidates[i].method for i in selected} == {
        "replay", "kv_transfer",
    }


def test_lagrangian_recovery_caps_one_watt_overshoot_before_work(tmp_path):
    profile, scenario = model(tmp_path), problem()
    sessions = (
        replace(scenario.sessions[0], expected_f=5, expected_g=0),
        replace(scenario.sessions[1], expected_f=2.5, expected_g=0),
    )
    scenario = replace(scenario, sessions=sessions)
    power = ExpectedPower(scenario, profile)
    candidates = (
        Candidate(0, "replay", 0, 2, 1, 1, ("wan",), 1, (0, 0), 0),
        Candidate(1, "replay", 0, 1, .75, .75, ("wan",), 1, (0, 0), 0),
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(2)),
        csr_matrix((np.full(2, .1), (np.zeros(2), np.arange(2))), shape=(1, 2)),
        ("route",), (1,), ("fraction",), 10,
    )

    selected = _recover_lagrangian(
        table, power, {"s0": {(), (0,)}, "s1": {(1,)}}, 1,
        replace(architecture(normal=1, emergency=1), pools=(
            architecture(normal=1, emergency=1).pools[0],
        )), scenario, "normal",
    )

    assert selected == {1}

    upgrade = replace(
        table,
        candidates=(
            replace(candidates[0], migration_work_s=.01, duration_s=.01),
            candidates[1],
        ),
    )
    target = power.drain_gain([s.session_id for s in sessions])
    selected = _recover_lagrangian(
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
    selected = _recover_lagrangian(
        swap_table, power, {"s0": {(0,), (1,)}, "s1": {(2,)}}, target,
        architecture(normal=1, emergency=1), scenario, "normal",
    )

    assert selected == {1, 2}


def test_lagrangian_recovery_packs_once_and_falls_back(tmp_path, monkeypatch):
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
    lazy = _recover_lagrangian(
        table, power, patterns, target, arch, scenario, "normal",
    )
    lazy_calls, calls = calls, 0
    eager = _recover_lagrangian(
        table, power, patterns, target, arch, scenario, "normal",
        eager_pack=True,
    )

    assert lazy == eager == {0, 1}
    assert lazy_calls == 1
    assert calls == 2

    calls = 0
    gain_calls, removed_gain = 0, pool_planner._source_removed_gain

    def counted_gain(*args):
        nonlocal gain_calls
        gain_calls += 1
        return removed_gain(*args)

    def reject_final_once(*args):
        nonlocal calls
        calls += 1
        return (None, ()) if calls == 1 else real(*args)

    monkeypatch.setattr(pool_planner, "_source_removed_gain", counted_gain)
    monkeypatch.setattr(pool_planner, "_pack", reject_final_once)
    assert _recover_lagrangian(
        table, power, patterns, target, arch, scenario, "normal",
    ) == eager
    assert calls == 3
    assert gain_calls == 2


def test_lagrangian_recovery_preserves_aggregate_boundary(tmp_path):
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
        return _recover_lagrangian(
            table, power, {"s0": {(0,)}, "s1": {(1,)}},
            power.drain_gain(("a", "b")),
            replace(architecture(normal=1, emergency=1), pools=(
                architecture(normal=1, emergency=1).pools[0],
            )), scenario, "normal",
        )

    boundary = limit - .6
    assert recover(boundary) == {0, 1}
    assert recover(boundary + 1e-12) == {0}


def test_lagrangian_recovery_preserves_prefix_tie_order(tmp_path):
    profile, scenario = model(tmp_path), problem()
    sessions = (scenario.sessions[0], replace(
        scenario.sessions[1], source_instance="s0",
    ))
    scenario = replace(scenario, sessions=sessions)
    power = ExpectedPower(scenario, profile)
    candidates = (
        Candidate(0, "replay", 0, 1, 1, 1, ("wan",), 1, (0, 0), 0),
        Candidate(1, "replay", 0, 1, 0, 0, ("wan",), 1, (0, 0), 0),
    )
    table = CandidateTable(
        sessions, candidates, csr_matrix(np.eye(2)),
        csr_matrix((np.full(2, .1), (np.zeros(2), np.arange(2))), shape=(1, 2)),
        ("route",), (1,), ("fraction",), 10,
    )

    selected = _recover_lagrangian(
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

    power = ExpectedPower(scenario, profile)
    assert len(_candidate_oracle(
        scenario, profile, architecture(normal=1, emergency=1, stable=1),
        "normal", power,
    ).pool_groups) == 1
    assert len(_candidate_oracle(
        scenario, profile, architecture(
            normal=1, emergency=1, stable=1, baselines=((0, 0), (.1, 0)),
        ), "normal", power,
    ).pool_groups) == 2
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand", 1,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1,
        routes=(("wan",), ("wan2",)), fluid=service,
    )
    arch = replace(arch, pools=(arch.pools[0], replace(
        arch.pools[1], fluid_migration=replace(service, route_overlap=False),
    )))
    scenario = replace(scenario, links=scenario.links + (NetworkLink("wan2", 100),))
    assert len(_candidate_oracle(
        scenario, profile, arch, "normal", power,
    ).pool_groups) == 2
    table = candidate_table(scenario, profile, arch, "normal", power)
    work = {candidate.pool: candidate.migration_work_s
            for candidate in table.candidates
            if candidate.session == 0 and candidate.method == "replay"}
    assert work[1] > work[0]


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

    def totals(table, selected):
        return (
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


def test_normal_success_and_emergency_rescue_are_distinct(tmp_path, monkeypatch):
    profile, scenario = model(tmp_path, switch=0, tp=1), problem()
    mode_plan, calls = pool_planner._mode_plan, []

    def counted(*args, **kwargs):
        calls.append(args[4])
        return mode_plan(*args, **kwargs)

    monkeypatch.setattr(pool_planner, "_mode_plan", counted)

    assert plan(scenario, profile, PATHS, "lp", destination=architecture()).admission_mode == "normal"
    calls.clear()
    rescued = plan(
        scenario, profile, PATHS, "lp",
        destination=architecture(normal=.2, emergency=.3),
    )
    assert calls == ["normal", "emergency"]
    assert rescued.feasible and rescued.admission_mode == "emergency"


def test_admission_mode_can_be_frozen_for_matched_oracles(tmp_path, monkeypatch):
    profile, scenario = model(tmp_path, switch=0, tp=1), problem()
    mode_plan, calls = pool_planner._mode_plan, []

    def counted(*args, **kwargs):
        calls.append(args[4])
        return mode_plan(*args, **kwargs)

    monkeypatch.setattr(pool_planner, "_mode_plan", counted)
    result = plan(
        scenario, profile, PATHS, "lp",
        destination=architecture(normal=.2, emergency=.3),
        admission_mode="normal",
    )

    assert calls == ["normal"] and result.admission_mode == "normal"


def test_target_unmet_returns_valid_maximum_shed_plan(tmp_path, monkeypatch):
    mode_plan, calls = pool_planner._mode_plan, []

    def counted(*args, **kwargs):
        calls.append(args[4])
        return mode_plan(*args, **kwargs)

    monkeypatch.setattr(pool_planner, "_mode_plan", counted)
    result = plan(
        problem(), model(tmp_path, switch=0, tp=1), PATHS, "greedy",
        destination=architecture(normal=.3, emergency=.3, methods=("kv_transfer",),
                                 compatibility=replace(FP, kv_abi="other")),
    )

    assert calls == ["normal"] and result.admission_mode == "emergency"
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


@pytest.mark.parametrize(("context", "rate"), ((9, 100), (10, 200)))
def test_fluid_replay_uses_regional_support_across_base_curve_boundary(
        tmp_path, context, rate):
    profile = model(
        tmp_path, switch=0, tp=1,
        replay_rate={"1": [[10, 200], [20, 100]]}, replay_completion=2,
    )
    session = replace(
        problem().sessions[0], context_tokens=context, log_bytes=100,
        expected_growth_tokens_per_s=0,
    )
    scenario = replace(problem(), sessions=(session,))
    service = FluidMigrationService(
        2, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    components = MigrationComponents(
        (5, 25), (50, 200), "hand", .5, kv_ingest_bytes_per_s=100,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=service,
    )
    arch = replace(
        arch, types=(replace(arch.types[0], migration={
            "replay": components, "kv_transfer": components,
        }),),
    )

    table = candidate_table(
        scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
    )
    by_method = {candidate.method: candidate for candidate in table.candidates}
    replay_work = .5 * (context / rate + 2)

    assert set(by_method) == {"replay", "kv_transfer"}
    assert by_method["replay"].migration_work_s == pytest.approx(replay_work)
    assert by_method["replay"].duration_s == pytest.approx(
        max(1, .5 * context / rate) + 1,
    )
    assert _destination_duration(
        session, "replay", profile.case(), ("wan",), {"wan": 100}, 0,
        components,
    ) == pytest.approx(1 + replay_work)


def test_unexpected_fluid_replay_timing_error_is_not_silently_dropped(
        tmp_path, monkeypatch):
    profile = model(tmp_path, switch=0, tp=1)
    scenario = replace(problem(), sessions=(problem().sessions[0],))
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    components = MigrationComponents(
        (1, 1000), (50, 200), "hand", kv_ingest_bytes_per_s=100,
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), methods=("replay",), fluid=service,
    )
    arch = replace(
        arch, types=(replace(arch.types[0], migration={
            "replay": components, "kv_transfer": components,
        }),),
    )

    def broken_rate(*_args):
        raise ValueError("internal replay model error")

    monkeypatch.setattr(type(profile.case().replay), "conservative_rate", broken_rate)

    with pytest.raises(ValueError, match="internal replay model error"):
        candidate_table(
            scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
        )


def test_physical_destination_timing_rejects_unmeasured_extrapolation(tmp_path):
    profile = model(tmp_path, switch=0, tp=1)
    session = replace(problem().sessions[0], context_tokens=10, log_bytes=100,
                      expected_growth_tokens_per_s=0)
    components = MigrationComponents((5, 20), (50, 200), "hand")

    with pytest.raises(ValueError, match="outside calibrated bandwidth range"):
        _destination_duration(
            session, "replay", profile.case(), ("wan",), {"wan": 10}, 0,
            components,
        )

    extrapolated = _destination_duration(
        session, "replay", profile.case(), ("wan",), {"wan": 10}, 0,
        replace(components, allow_extrapolation=True),
    )
    assert extrapolated > 0


def test_fluid_candidate_respects_regional_extrapolation_flag(tmp_path):
    profile = model(tmp_path, switch=0, tp=1)
    scenario = replace(problem(), sessions=(problem().sessions[0],))
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    components = MigrationComponents(
        (20, 30), (50, 200), "hand", kv_ingest_bytes_per_s=100,
    )

    def candidates(allow):
        value = replace(components, allow_extrapolation=allow)
        arch = architecture(
            normal=1, emergency=1, stable=1, baselines=((0, 0),),
            routes=(("wan",),), methods=("replay",), fluid=service,
        )
        arch = replace(arch, types=(replace(arch.types[0], migration={
            "replay": value, "kv_transfer": value,
        }),))
        return candidate_table(
            scenario, profile, arch, "normal", ExpectedPower(scenario, profile),
        ).candidates

    assert not candidates(False)
    assert [candidate.method for candidate in candidates(True)] == ["replay"]


@pytest.mark.parametrize("method", ("replay", "kv_transfer"))
def test_fluid_execution_respects_regional_extrapolation_flag(tmp_path, method):
    profile = model(tmp_path, switch=0, tp=1)
    scenario = replace(problem(), sessions=(problem().sessions[0],))
    service = FluidMigrationService(
        1, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    components = MigrationComponents(
        (20, 30), (50, 200), "hand", kv_ingest_bytes_per_s=100,
    )

    def architecture_for(allow):
        value = replace(components, allow_extrapolation=allow)
        arch = architecture(
            normal=1, emergency=1, stable=1, baselines=((0, 0),),
            routes=(("wan",),), fluid=service,
        )
        return replace(arch, types=(replace(arch.types[0], migration={
            "replay": value, "kv_transfer": value,
        }),))

    move = PlannedMove(
        "a", "t0", method, 0, ("wan",), destination_pool="p0",
    )
    with pytest.raises(ValueError, match="outside calibrated context range"):
        execute(scenario, profile, (move,), destination=architecture_for(False))
    assert execute(
        scenario, profile, (move,), destination=architecture_for(True),
    ).sessions[0].committed_s is not None


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
    fluid = FluidMigrationService(
        4, 100, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    arch = architecture(normal=1, emergency=1, stable=1,
                        baselines=((.2, 0), (.4, 0)), fluid=fluid)
    arch = replace(arch, pools=(DestinationPool(
        "p", "q", (DestinationReplica("t0", (.2, 0)),
                    DestinationReplica("t1", (.4, 0))), "r", ("wan",), ("replay",),
        fluid_migration=fluid,
    ),))
    sessions = tuple(replace(s, expected_f=70, expected_g=0) for s in problem().sessions)
    scenario = replace(problem(), sessions=sessions)
    table = candidate_table(scenario, model(tmp_path, tp=1), arch, "normal",
                            ExpectedPower(scenario, model(tmp_path, tp=1)))

    assert exact_replica_assignment(table, {0, 1}, arch, scenario, "normal") is None
    assignment, rejected = _pack(
        table, {0, 1}, arch, scenario, "normal", repair=True,
    )

    assert set(assignment) | set(rejected) == {0, 1}
    assert set(assignment).isdisjoint(rejected)
    assert len(assignment) == len(rejected) == 1
    costs = np.asarray(table.resources.sum(0)).ravel()
    assert set(assignment) == {min((0, 1), key=lambda i: costs[i] / table.candidates[i].gain_w)}
    assert _assignment_valid(table, assignment, arch, scenario, "normal")
    result = plan(scenario, model(tmp_path, switch=0, tp=1), PATHS, "lp", destination=arch)
    assert result.packing_repair_count == 1 and result.failure_reason == "target_unmet"


def test_packing_repair_keeps_power_efficient_maximal_greedy_subset():
    fluid = FluidMigrationService(
        1, 1, {"replay": 0, "kv_transfer": 0},
        {"replay": 0, "kv_transfer": 0}, "hand",
    )
    arch = architecture(
        normal=1, emergency=1, stable=1, baselines=((0, 0),),
        routes=(("wan",),), fluid=fluid,
    )
    arch = replace(arch, pools=(replace(
        arch.pools[0], replicas=(DestinationReplica("t0"), DestinationReplica("t1")),
    ),))
    candidates = tuple(
        Candidate(i, "replay", 0, 4 - i, 1, 1, (), 0, (work, 0), 0)
        for i, work in enumerate((.6, .6, .5, .3))
    )
    table = CandidateTable(
        tuple(SimpleNamespace(session_id=str(i)) for i in range(4)), candidates,
        csr_matrix(np.eye(4)), csr_matrix(np.full((1, 4), .25)),
        ("aggregate",), (1,), ("fraction",), 1,
    )

    forward = _pack(table, list(range(4)), arch, problem(), "normal", repair=True)
    reverse = _pack(table, list(reversed(range(4))), arch, problem(), "normal", repair=True)

    assert forward == reverse
    assignment, rejected = forward
    assert set(assignment) == {0, 1, 3} and rejected == (2,)
    assert _assignment_valid(table, assignment, arch, problem(), "normal")


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
