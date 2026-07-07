from __future__ import annotations

"""
Claim:
The pilot node-power formulation uses an increasing concave P(L), with modeled
shed P(L)-P(L-r). Tangent LP/MILP surrogates must be conservative lower bounds.

Plausible wrong implementations:
- Fail to anchor the log curve at idle and the operating reference load.
- Use a convex or increasing-slope power function, losing concentration value.
- Treat the tangent as an upper bound, falsely certifying power shed.
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
from node_knee import _tangent, evaluate_node_expected_w, node_loads, solve_power_function_lp
from power import ETA_BYTES_PER_TOK, PoolPower


def _pop(ell, source_node):
    ell = np.asarray(ell, float)
    n = len(ell)
    z = np.zeros(n)
    return JobPopulation(
        np.array(["agentic"] * n), np.array(["agentic_tool_loop"] * n), np.array(["active"] * n),
        np.zeros(n, bool), np.full(n, 1000.0), np.ones(n), z, z, z, z, np.ones(n, bool),
        ell, z, ETA_BYTES_PER_TOK * np.full(n, 1000.0), "bf16", 0.35, np.asarray(source_node, int)
    )


def _imp(pop):
    n = len(pop)
    z = np.zeros(n)
    return Impact(z, z, z, z, z, np.ones(n), np.ones(n), z, z, "load")


def test_log_power_curve_is_anchored_and_concave():
    pool = replace(PoolPower(), power_curve="log", p_idle_w=10.0, p_busy_w=110.0, rho_star=1.0, log_shape=3.0)

    assert pool.node_power([0.0, 1.0]) == pytest.approx([10.0, 110.0])
    assert np.all(np.diff(pool.node_power_slope([0.1, 0.5, 0.9])) < 0)


def test_concave_power_values_concentrated_removed_load_more_than_spread():
    pool = replace(PoolPower(), power_curve="log", p_idle_w=10.0, p_busy_w=110.0, rho_star=1.0, log_shape=3.0)
    pop = _pop([0.2, 0.2, 0.2, 0.2], [0, 0, 1, 1])

    concentrated = evaluate_node_expected_w(pop, pool, [1, 1, 0, 0])
    spread = evaluate_node_expected_w(pop, pool, [1, 0, 1, 0])

    assert concentrated > spread


def test_power_function_tangent_is_conservative_lower_bound():
    pool = replace(PoolPower(), power_curve="log", p_idle_w=10.0, p_busy_w=110.0, rho_star=1.0, log_shape=3.0)
    pop = _pop([0.1, 0.2, 0.3], [0, 0, 0])
    r0 = np.array([0.2])
    w, b = _tangent(pop, pool, r0)

    for y in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])):
        assert b + w @ y <= evaluate_node_expected_w(pop, pool, y) + 1e-9


def test_power_function_lp_iterates_past_single_tangent_overdrain():
    pool = replace(PoolPower(), power_curve="log", p_idle_w=0.0, p_busy_w=100.0, rho_star=1.0, log_shape=3.0)
    pop = _pop([0.5, 0.5], [0, 0])
    target = evaluate_node_expected_w(pop, pool, [1.0, 0.0])

    res = solve_power_function_lp(pop, pool, _imp(pop), target, Event(D=1e9, dest_nodes=10**6), Movement(lambda_src=1e18))

    assert res.true_expected_feasible
    assert res.cost == pytest.approx(1.0)
