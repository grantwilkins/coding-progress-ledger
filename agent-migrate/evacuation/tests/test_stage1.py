"""Semantic tests for Stage 1 of the evacuation LP.

The five tests target believable errors in the LP wiring:
  1. conservation             — `sum(x_R[q]) + sum(x_S[q]) + z[q] == n_q`
  2. loose capacity ⇒ Z* = 0  — sign of the objective, capacity orientation
  3. zero capacity ⇒ Z* = N   — min/max confusion, ill-conditioned RHS
  4. hand-worked tiny LP      — coefficient swap β↔η, per-job arithmetic
  5. monotonicity in D        — capacity-coefficient orientation w.r.t. deadline
"""

from __future__ import annotations

import numpy as np

from instance import build_instance, ProblemInstance
from stage1 import solve_stage1


def test_conservation():
    inst = build_instance()
    res = solve_stage1(inst)
    total = res.x_R.sum(axis=1) + res.x_S.sum(axis=1) + res.z
    np.testing.assert_allclose(total, inst.n, atol=1e-6)


def test_loose_capacity_evacuates_everything():
    inst = build_instance(D=1e6, total_jobs=200, seed=0)
    res = solve_stage1(inst)
    assert res.Z_star < 1e-4


def test_zero_capacity_strands_everything():
    inst = build_instance(D=0.0, total_jobs=200, seed=0)
    res = solve_stage1(inst)
    np.testing.assert_allclose(res.Z_star, inst.n.sum(), atol=1e-6)


def test_hand_worked_tiny_instance():
    """One model, one destination, 20 jobs, D=5s.

    Per-job costs:     β·T = 4·1000 = 4000 B, η·T = 1e9 B, τ = T/ρ = 1 GPU-s.
    Capacities:        C_net = 5e15 B (slack), C_pfill = 5 GPU-s, C_ing = 5e9 B.
    Replay budget:     bound by C_pfill → 5 jobs.
    State budget:      bound by C_ing  → 5 jobs.
    Network is shared but slack at this scale, so does not bind.
    Movable = 10, Z* = 20 - 10 = 10.
    """
    Q = 20
    inst = ProblemInstance(
        model_idx=np.zeros(Q, dtype=int),
        T=np.full(Q, 1000.0),
        beta=np.full(Q, 4.0),
        eta=np.full(Q, 1e6),
        rho=np.full(Q, 1000.0),
        n=np.ones(Q),
        lambda_bps=np.array([1e15]),
        W=np.array([[1.0]]),
        mu_ing=1e9,
        D=5.0,
        M_names=("toy",),
        L_names=("toy_dst",),
    )
    res = solve_stage1(inst)
    np.testing.assert_allclose(res.Z_star, 10.0, atol=1e-4)


def test_Z_monotone_in_deadline():
    Z = [solve_stage1(build_instance(D=D, total_jobs=500, seed=1)).Z_star
         for D in (10.0, 60.0, 300.0)]
    assert Z[0] >= Z[1] - 1e-4 >= Z[2] - 2e-4
