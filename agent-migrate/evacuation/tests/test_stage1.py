"""Semantic tests for Stage 1 of the evacuation LP.

The tests target believable errors in the LP wiring:
  1. conservation             — `sum(x_R[q]) + sum(x_S[q]) + z[q] == n_q`
  2. loose capacity ⇒ Z* = 0  — sign of the objective, capacity orientation
  3. zero capacity ⇒ Z* = N   — min/max confusion, ill-conditioned RHS
  4. hand-worked tiny LP      — coefficient swap β↔η, per-job arithmetic
  5. monotonicity in D        — capacity-coefficient orientation w.r.t. deadline
  6. residency wall           — at o > 1 the stranded KV equals the HBM excess
  7. replay mask              — solo-infeasible replays (T/ρ > D) never replay
"""

from __future__ import annotations

import numpy as np

from instance import build_instance, ProblemInstance
from loads import replay_infeasible
from stage1 import solve_stage1


def test_conservation():
    inst = build_instance()
    res = solve_stage1(inst)
    total = res.x_R.sum(axis=1) + res.x_S.sum(axis=1) + res.z
    np.testing.assert_allclose(total, inst.n, atol=1e-6)


def test_loose_capacity_evacuates_everything():
    # o = 0.5 keeps the residency stock slack, so a huge deadline clears all.
    inst = build_instance(D=1e6, occupancy=0.5, seed=0)
    res = solve_stage1(inst)
    assert res.Z_star < 1e-4


def test_zero_capacity_strands_everything():
    inst = build_instance(D=0.0, occupancy=0.5, seed=0)
    res = solve_stage1(inst)
    np.testing.assert_allclose(res.Z_star, inst.n.sum(), atol=1e-6)


def test_hand_worked_tiny_instance():
    """One model, one destination, 20 jobs, D=5s.

    Per-job costs:     β·T = 4·1000 = 4000 B, η·T = 1e9 B, τ = T/ρ = 1 GPU-s.
    Capacities:        C_net = 5e15 B (slack), C_pfill = 5 GPU-s, C_ing = 5e9 B,
                       C_res = 1e18 B (slack residency).
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
        W_ing=np.array([[1.0]]),
        C_res=np.array([1e18]),
        mu_ing=1e9,
        D=5.0,
        M_names=("toy",),
        L_names=("toy_dst",),
    )
    res = solve_stage1(inst)
    np.testing.assert_allclose(res.Z_star, 10.0, atol=1e-4)


def test_Z_monotone_in_deadline():
    Z = [solve_stage1(build_instance(D=D, seed=1)).Z_star
         for D in (10.0, 60.0, 300.0)]
    assert Z[0] >= Z[1] - 1e-4 >= Z[2] - 2e-4


def test_residency_wall_at_high_occupancy():
    # Deadline never binds; the decode-HBM stock does. The LP strands exactly
    # the KV that does not fit (it picks the largest-KV jobs to minimize count).
    inst = build_instance(D=50_000.0, occupancy=1.5, seed=0)
    res = solve_stage1(inst)
    kv_total = (inst.eta * inst.T * inst.n).sum()
    stranded = (inst.eta * inst.T * res.z).sum()
    assert res.Z_star > 0
    np.testing.assert_allclose(stranded, kv_total - inst.C_res[0], rtol=1e-3)


def test_replay_mask_blocks_solo_infeasible_jobs():
    inst = build_instance(D=100.0, seed=0)
    bad = replay_infeasible(inst)
    assert bad.any()  # the snapshot tail contains T/rho > 100s jobs
    res = solve_stage1(inst)
    assert res.x_R[bad].sum() < 1e-6
