"""Claim:
Node-knee exploration values source-node concentration with a conservative tangent LP.

Plausible wrong implementations:
- Treat node-knee shed as additive per job, losing increasing returns around the knee.
- Use a tangent that is not a lower bound because the removed-load value is not convex.
- Let fixed-region active-knee LPs cross inactive nodes with stale plateau slopes.
- Let node-knee methods move pinned jobs that the dispatch LP would block.
- Report infeasible movement as feasible because the solver path usually enforces budgets.
- Compare active-knee LP against whole-job baselines without an integer counterpart.
- Search only a capped subset of active-node regions while calling the result a MILP.
- Use the certified active-work target as the conservative active floor.
- Infer node placement silently or ignore it when computing expected node shed.
- Randomize at the wrong aggregation level or ignore budgets in a random baseline.
- Compare heuristics without an exact tiny oracle on hand-checkable cases.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import Event
from impact import Impact, Movement
from instance import JobPopulation
from node_knee import (
    _knee_candidates,
    _lp,
    _region_affine,
    _result,
    _tangent,
    comparison_rows,
    evaluate_node_expected_w,
    node_loads,
    place_source_nodes,
    removed_loads,
    solve_active_knee_milp,
    solve_active_knee_lp,
    solve_exact_oracle,
    evaluate_active_floor_w,
    solve_live_greedy,
    solve_node_aware_greedy,
    solve_node_drain_greedy,
    solve_power_function_lp_rounded,
    solve_power_unaware,
    solve_random_jobs,
    solve_random_nodes,
    solve_single_source_milp,
    solve_tangent_lp,
)
from power import ETA_BYTES_PER_TOK, PoolPower, rho_replay

SLACK_E = Event(D=1e9, dest_nodes=10**7)
SLACK_M = replace(Movement(), lambda_src=1e18, mu_in=1e18)


def _pop(ell, source_node=None):
    ell = np.asarray(ell, float)
    n = len(ell)
    return JobPopulation(
        np.array(["agentic"] * n),
        np.array(["agentic_tool_loop"] * n),
        np.array(["active"] * n),
        np.zeros(n, bool),
        np.full(n, 1000.0),
        np.ones(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n, bool),
        ell,
        np.zeros(n),
        ETA_BYTES_PER_TOK * np.full(n, 1000.0),
        "bf16",
        0.35,
        None if source_node is None else np.asarray(source_node, int),
    )


def _imp(pop, costs):
    n = len(pop)
    costs = np.asarray(costs, float)
    return Impact(
        np.zeros(n),
        PoolPower().s_plat * pop.ell,
        PoolPower().base_w_per_load * pop.ell,
        PoolPower().p_bar * pop.ell,
        np.zeros(n),
        costs,
        1000.0 * costs,
        np.zeros(n),
        np.zeros(n),
        "load",
    )


def _movement_slacks(pop, pool, imp, res, event, move):
    y = res.y
    held = pop.T / pool.mean_context_tokens * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)
    return np.array([
        move.lambda_src * (event.D - event.tau_src) - (imp.b_replay @ res.y_R + imp.b_transfer @ res.y_S),
        np.floor(event.spare_frac * event.dest_nodes) * (event.D - event.tau_pre) - (pop.T / rho_replay(pop.T, pop.mfu)) @ res.y_R,
        np.floor(event.spare_frac * event.dest_nodes) * move.mu_in * (event.D - event.tau_in) - imp.b_transfer @ res.y_S,
        event.l_dest(pool) - pop.ell @ y,
        event.s_dest(pool) - held @ y,
    ])


def test_removed_load_value_is_convex_for_ramp_plateau_curve():
    pool = PoolPower()
    assert pool.power_knee < pool.rho_star < pool.latency_knee
    assert pool.ramp_slope > 10 * pool.s_plat
    for L in (0.05, 0.2, 0.8, 1.2):
        r = np.linspace(0.0, L, 200)
        F = pool.node_power(L) - pool.node_power(L - r)
        assert np.diff(F, 2).min() >= -1e-8


def test_node_expected_requires_explicit_source_placement():
    with pytest.raises(ValueError, match="source_node"):
        evaluate_node_expected_w(_pop([0.1]), PoolPower(), np.ones(1))


def test_source_placement_policies_are_deterministic_and_complete():
    pool = PoolPower()
    pop = _pop([0.3, 0.2, 0.1, 0.05])
    for policy in ("memory", "load", "balanced"):
        a = place_source_nodes(pop, pool, 2, policy)
        b = place_source_nodes(pop, pool, 2, policy)
        assert np.array_equal(a, b)
        assert sorted(a.tolist()) == [0, 0, 1, 1]


def test_active_knee_lp_finds_crossing_missed_by_initial_tangent():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30], [0, 0, 0, 1, 1, 1, 1])
    imp = _imp(pop, [10, 10, 10, 1, 1, 1, 1000])
    target = 500.0
    initial = solve_tangent_lp(pop, pool, imp, target, SLACK_E, SLACK_M, max_iter=1)
    active = solve_active_knee_lp(pop, pool, imp, target, SLACK_E, SLACK_M)
    assert not initial.method_target_feasible
    assert active.method_target_feasible and active.true_expected_feasible
    assert active.cost < initial.cost
    assert pop.ell[:3].sum() - pop.ell[:3] @ active.y[:3] <= pool.power_knee + 1e-6


def test_active_knee_region_affine_is_exact_inside_fixed_regions():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08], [0, 0, 1, 1])
    y = np.array([1.0, 0.0, 0.5, 0.0])
    w, b = _region_affine(pop, pool, active_nodes=(0,))
    residual = node_loads(pop) - removed_loads(pop, y)
    assert residual[0] <= pool.power_knee
    assert residual[1] >= pool.power_knee
    assert b + w @ y == pytest.approx(evaluate_node_expected_w(pop, pool, y))


def test_fixed_region_lp_keeps_inactive_nodes_above_knee():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08], [0, 0, 1, 1])
    imp = _imp(pop, [100, 100, 1, 1])
    load = node_loads(pop)
    r0 = np.zeros_like(load)
    r0[0] = load[0] - pool.power_knee
    w, b = _tangent(pop, pool, r0)
    res = _lp(pop, pool, imp, 900.0, w, b, r0, SLACK_E, SLACK_M,
              active_nodes=(0,), method="test", region_consistent=True)
    residual = load - removed_loads(pop, res.y)
    assert residual[0] <= pool.power_knee + 1e-6
    assert residual[1] >= pool.power_knee - 1e-6


def test_active_knee_candidates_can_cover_all_four_nodes():
    pool = PoolPower()
    pop = _pop([0.08] * 8, [0, 0, 1, 1, 2, 2, 3, 3])
    imp = _imp(pop, np.ones(8))
    cand = _knee_candidates(pop, pool, imp)
    assert () in cand
    assert (0, 2) in cand
    assert max(len(c) for c in cand) == 4


def test_active_knee_milp_exhausts_small_source_node_regions():
    pool = PoolPower()
    pop = _pop([0.08] * 10, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    imp = _imp(pop, np.ones(10))
    target = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
    milp = solve_active_knee_milp(pop, pool, imp, target, SLACK_E, SLACK_M)
    oracle = solve_exact_oracle(pop, pool, imp, target, SLACK_E, SLACK_M, max_jobs=14)
    assert milp.true_expected_feasible
    assert milp.node_expected_w == pytest.approx(target)
    assert milp.cost == pytest.approx(oracle.cost)


def test_active_knee_hard_fails_beyond_exhaustive_region_cap():
    pool = PoolPower()
    pop = _pop([0.16] * 9, np.arange(9))
    imp = _imp(pop, np.ones(9))
    with pytest.raises(ValueError, match="at most 8"):
        solve_active_knee_milp(pop, pool, imp, 1.0, SLACK_E, SLACK_M)


def test_active_floor_uses_guaranteed_not_certified_token_work():
    imp = Impact(
        np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.zeros(2), np.zeros(2),
        np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), "load"
    )
    y = np.array([1.0, 0.5])
    assert evaluate_active_floor_w(imp, y) == pytest.approx(2.0)


def test_active_knee_milp_is_whole_job_and_bounded_by_lp_relaxation():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30], [0, 0, 0, 1, 1, 1, 1])
    imp = _imp(pop, [10, 10, 10, 1, 1, 1, 1000])
    lp = solve_active_knee_lp(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    mi = solve_active_knee_milp(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert mi.true_expected_feasible
    assert np.allclose(mi.y, np.round(mi.y))
    assert lp.cost <= mi.cost + 1e-6


def test_result_reports_actual_movement_feasibility():
    pool = PoolPower()
    pop = _pop([0.08], [0])
    imp = Impact(
        np.zeros(1), pool.s_plat * pop.ell, pool.base_w_per_load * pop.ell,
        pool.p_bar * pop.ell, np.zeros(1), np.ones(1), np.ones(1),
        np.array([100.0]), np.zeros(1), "load"
    )
    event = Event(D=10, dest_nodes=10**6)
    res = _result(pop, pool, imp, np.ones(1), np.zeros(1), 1.0, "test", 1.0,
                  event, replace(Movement(), lambda_src=1.0))
    assert not res.movement_feasible
    assert not res.feasible


def test_node_knee_methods_respect_pinned_jobs():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [1, 1, 1])
    event = replace(SLACK_E, pinned=("agentic",))
    solvers = (
        solve_active_knee_milp,
        solve_live_greedy,
        solve_node_aware_greedy,
        solve_node_drain_greedy,
        solve_random_jobs,
        solve_random_nodes,
        solve_exact_oracle,
    )
    for solver in solvers:
        res = solver(pop, pool, imp, 500.0, event, SLACK_M)
        assert np.array_equal(res.y, np.zeros(len(pop)))
        assert res.movement_feasible


def test_comparison_rows_keep_additive_target_feasibility():
    pool = PoolPower()
    pop = _pop([0.08, 0.08], [0, 0])
    imp = _imp(pop, [1, 1])
    rows = comparison_rows(pop, pool, imp, 1e9, SLACK_E, SLACK_M)
    additive = next(r for r in rows if r["method"] == "additive_lp")
    assert not additive["method_target_feasible"]


def test_node_drain_greedy_beats_live_marginal_on_knee_bundle_case():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30], [0, 0, 0, 1, 1, 1, 1])
    imp = _imp(pop, [10, 10, 10, 1, 1, 1, 1000])
    live = solve_live_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    drain = solve_node_drain_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert live.true_expected_feasible and drain.true_expected_feasible
    assert drain.cost < live.cost


def test_node_aware_greedy_uses_finite_difference_gain():
    pool = PoolPower()
    pop = _pop([0.08, 0.02], [0, 0])
    imp = _imp(pop, [1, 100])
    res = solve_node_aware_greedy(pop, pool, imp, 1.0, SLACK_E, SLACK_M)
    j = res.order[0]
    before = node_loads(pop)[0]
    direct = pool.node_power(before) - pool.node_power(before - pop.ell[j])
    assert res.node_expected_w == pytest.approx(direct)
    assert np.array_equal(res.y, np.array([1.0, 0.0]))


def test_node_aware_greedy_takes_knee_bundle_missed_by_marginal_greedy():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30], [0, 0, 0, 1, 1, 1, 1])
    imp = _imp(pop, [10, 10, 10, 1, 1, 1, 1000])
    live = solve_live_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    aware = solve_node_aware_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert aware.true_expected_feasible
    assert np.allclose(aware.y, np.round(aware.y))
    assert aware.cost < live.cost
    assert aware.order[:3] == (3, 4, 5)


def test_node_aware_greedy_tries_replay_when_kv_does_not_fit():
    pool = PoolPower()
    pop = _pop([0.08], [0])
    imp = replace(_imp(pop, [1]), c_replay=np.array([5.0]), c_transfer=np.array([1.0]),
                  b_replay=np.array([1.0]), b_transfer=np.array([100.0]))
    event = Event(D=20, dest_nodes=10**6, tau_src=0, tau_pre=0, tau_in=0)
    move = replace(SLACK_M, lambda_src=0.1)
    res = solve_node_aware_greedy(pop, pool, imp, 1.0, event, move)
    assert np.array_equal(res.y_R, np.ones(1))
    assert np.array_equal(res.y_S, np.zeros(1))
    assert res.movement_feasible


def test_node_aware_greedy_matches_tiny_oracle_and_is_deterministic():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [100, 1, 1])
    a = solve_node_aware_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    b = solve_node_aware_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    oracle = solve_exact_oracle(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert a.cost == pytest.approx(oracle.cost)
    assert np.array_equal(a.y, oracle.y)
    assert a.order == b.order
    assert np.array_equal(a.y_R, b.y_R) and np.array_equal(a.y_S, b.y_S)


def test_lp_rounding_is_whole_feasible_and_deterministic():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [100, 1, 1])
    a = solve_power_function_lp_rounded(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    b = solve_power_function_lp_rounded(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert a.true_expected_feasible and a.movement_feasible
    assert np.allclose(a.y, np.round(a.y))
    assert np.array_equal(a.y, b.y)
    assert a.order == b.order


def test_power_unaware_orders_by_disruption_per_removed_load():
    pool = PoolPower()
    pop = _pop([0.10, 0.20], [0, 1])
    imp = _imp(pop, [1, 3])
    result = solve_power_unaware(pop, pool, imp, 1.0, SLACK_E, SLACK_M)
    assert result.order == (0,)
    assert np.array_equal(result.y, np.array([1.0, 0.0]))
    assert result.true_expected_feasible


def test_single_source_milp_is_whole_and_matches_oracle():
    pool = PoolPower(power_curve="saturating", p_idle_w=67.1, p_busy_w=424.4,
                     power_knee=2.05, rho_star=0.535)
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [100, 1, 1])
    target = 20.0
    milp = solve_single_source_milp(pop, pool, imp, target, SLACK_E, SLACK_M)
    oracle = solve_exact_oracle(pop, pool, imp, target, SLACK_E, SLACK_M)
    assert np.array_equal(milp.y, oracle.y)
    assert milp.cost == pytest.approx(oracle.cost)


def test_node_drain_bundle_respects_joint_resource_budget():
    pool = PoolPower()
    pop = _pop([0.5, 0.5, 0.5], [0, 0, 0])  # one whole spare node admits 1 job (0.5 ≤ 0.8 < 1.0)
    imp = _imp(pop, [1, 1, 1])
    event = Event(D=1e9, dest_nodes=1, spare_frac=1.0)
    drain = solve_node_drain_greedy(pop, pool, imp, 500.0, event, SLACK_M)
    assert pop.ell @ drain.y <= event.l_dest(pool) + 1e-9


def test_random_baselines_randomize_at_declared_aggregation_level():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08], [0, 0, 1, 1])
    imp = _imp(pop, [1, 100, 1, 100])
    job = solve_random_jobs(pop, pool, imp, 900.0, SLACK_E, SLACK_M, seed=0)
    node = solve_random_nodes(pop, pool, imp, 900.0, SLACK_E, SLACK_M, seed=0)
    assert np.array_equal(job.y, np.array([0.0, 0.0, 1.0, 0.0]))
    assert np.array_equal(node.y, np.array([1.0, 0.0, 0.0, 0.0]))


def test_random_baselines_are_seeded_and_budget_respecting():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08], [0, 0, 1, 1])
    imp = _imp(pop, [1, 2, 3, 4])
    event = Event(D=1e9, dest_nodes=1, spare_frac=1.0)
    for solver in (solve_random_jobs, solve_random_nodes):
        a = solver(pop, pool, imp, 10_000.0, event, SLACK_M, seed=2)
        b = solver(pop, pool, imp, 10_000.0, event, SLACK_M, seed=2)
        assert np.array_equal(a.y_R, b.y_R)
        assert np.array_equal(a.y_S, b.y_S)
        assert _movement_slacks(pop, pool, imp, a, event, SLACK_M).min() >= -1e-9


def test_exact_oracle_picks_cheapest_knee_crossing_bundle():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [100, 1, 1])
    oracle = solve_exact_oracle(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert oracle.true_expected_feasible
    assert oracle.cost == pytest.approx(2.0)
    assert np.array_equal(oracle.y, np.array([0.0, 1.0, 1.0]))
