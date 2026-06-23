"""Claim:
Node-knee exploration values source-node concentration with a conservative tangent LP.

Plausible wrong implementations:
- Treat node-knee shed as additive per job, losing increasing returns around the knee.
- Use a tangent that is not a lower bound because the removed-load value is not convex.
- Infer node placement silently or ignore it when computing expected node shed.
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
    evaluate_node_expected_w,
    place_source_nodes,
    solve_active_knee_lp,
    solve_exact_oracle,
    solve_live_greedy,
    solve_node_drain_greedy,
    solve_tangent_lp,
)
from power import ETA_BYTES_PER_TOK, PoolPower

SLACK_E = Event(D=1e9, W=10**7, dest_nodes=10**7)
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
    assert not initial.surrogate_feasible
    assert active.surrogate_feasible and active.true_expected_feasible
    assert active.cost < initial.cost
    assert pop.ell[:3].sum() - pop.ell[:3] @ active.y[:3] <= pool.power_knee + 1e-6


def test_node_drain_greedy_beats_live_marginal_on_knee_bundle_case():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30], [0, 0, 0, 1, 1, 1, 1])
    imp = _imp(pop, [10, 10, 10, 1, 1, 1, 1000])
    live = solve_live_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    drain = solve_node_drain_greedy(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert live.true_expected_feasible and drain.true_expected_feasible
    assert drain.cost < live.cost


def test_node_drain_bundle_respects_joint_resource_budget():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [1, 1, 1])
    event = Event(D=1e9, W=10**7, dest_nodes=1, spare_frac=0.125)
    drain = solve_node_drain_greedy(pop, pool, imp, 500.0, event, SLACK_M)
    assert pop.ell @ drain.y <= event.l_dest(pool) + 1e-9


def test_exact_oracle_picks_cheapest_knee_crossing_bundle():
    pool = PoolPower()
    pop = _pop([0.08, 0.08, 0.08], [0, 0, 0])
    imp = _imp(pop, [100, 1, 1])
    oracle = solve_exact_oracle(pop, pool, imp, 500.0, SLACK_E, SLACK_M)
    assert oracle.true_expected_feasible
    assert oracle.cost == pytest.approx(2.0)
    assert np.array_equal(oracle.y, np.array([0.0, 1.0, 1.0]))
