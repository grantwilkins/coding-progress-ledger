"""Semantic tests for the three swappable Stage 1 objectives.

Targets believable errors in the objective wiring and the Stage 2 link:
  1. all objectives conserve mass
  2. throughput is the unique total-evacuation maximizer (smallest sum z)
  3. max_min raises the worst-class floor, and alpha* == that floor
  4. Stage 2 preserves Z* and the per-objective fairness floors
  5. prop_fair solves and yields a smooth, starvation-free middle ground
"""

from __future__ import annotations

import numpy as np

from instance import build_instance
from objective_metrics import evac_fraction
from stage1 import solve_stage1
from stage2 import solve_stage2

# Tight deadline so the objectives genuinely diverge (capacity binds).
INST = build_instance(D=80.0, n_bins=3, total_jobs=2000, seed=1)
RUNS = {o: solve_stage1(INST, o) for o in ("throughput", "max_min", "prop_fair")}


def test_all_objectives_conserve():
    for s1 in RUNS.values():
        total = s1.x_R.sum(axis=1) + s1.x_S.sum(axis=1) + s1.z
        np.testing.assert_allclose(total, INST.n, atol=1e-6)


def test_throughput_minimizes_z():
    zt = RUNS["throughput"].Z_star
    assert zt <= RUNS["max_min"].Z_star + 1e-4
    assert zt <= RUNS["prop_fair"].Z_star + 1e-4


def test_maxmin_raises_floor():
    floor_t = evac_fraction(INST, RUNS["throughput"].z).min()
    floor_m = evac_fraction(INST, RUNS["max_min"].z).min()
    assert floor_m >= floor_t - 1e-6
    np.testing.assert_allclose(RUNS["max_min"].alpha_star, floor_m, atol=1e-4)
    assert floor_m > 1e-3  # the tight instance actually starves throughput


def test_stage2_preserves_links():
    for o, s1 in RUNS.items():
        s2 = solve_stage2(INST, s1)
        np.testing.assert_allclose(s2.z.sum(), s1.Z_star, atol=1e-3)
        if o == "max_min":
            assert np.all(s2.z <= (1.0 - s1.alpha_star) * INST.n + 1e-3 * INST.n)
        if o == "prop_fair":
            u = evac_fraction(INST, s2.z)
            w = INST.n
            U = float((w * np.log(s1.utility_epsilon + u)).sum())
            assert U >= s1.U_star - s1.utility_delta - 1e-3 * abs(s1.U_star)


def test_propfair_runs():
    s1 = RUNS["prop_fair"]
    assert np.isfinite(s1.U_star) and s1.utility_weights == "population"
    u = evac_fraction(INST, s1.z)
    assert u.min() > 1e-6  # no starvation
    # prop_fair total evacuation lies between max_min and throughput
    assert RUNS["max_min"].Z_star >= s1.Z_star - 1e-4 >= RUNS["throughput"].Z_star - 1e-4
