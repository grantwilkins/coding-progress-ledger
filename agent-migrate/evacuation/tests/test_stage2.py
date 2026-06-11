"""Semantic tests for Stage 2 of the evacuation LP.

The tests target believable errors in the LP wiring:
  1. conservation             — `sum(x_R[q]) + sum(x_S[q]) + z[q] == n_q`
  2. Stage 1 link             — `sum(z) == Z*`
  3. hand-worked tiny LP      — balanced split, phi* = 0.5
  4. phi* == realized peak    — objective matches the binding pressure
  5. monotonicity in D        — phi*(D) non-increasing
  6. residency excluded from phi — utilization ~ o while phi* stays below it
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2


def test_conservation():
    inst = build_instance()
    s1 = solve_stage1(inst)
    res = solve_stage2(inst, s1)
    total = res.x_R.sum(axis=1) + res.x_S.sum(axis=1) + res.z
    np.testing.assert_allclose(total, inst.n, atol=1e-6)


def test_stage1_optimum_preserved():
    inst = build_instance()
    s1 = solve_stage1(inst)
    res = solve_stage2(inst, s1)
    np.testing.assert_allclose(res.z.sum(), s1.Z_star, atol=1e-6)


def test_hand_worked_tiny_instance():
    """One model, one destination, 20 jobs, D=20s.

    Per-job:      tau = 1 GPU-s, eta*T = 1e5 B, beta*T = 4e3 B.
    Capacities:   C_pfill = 20 GPU-s, C_ing = 2e6 B (= 20 state jobs),
                  C_net = 1e8 B (slack), C_res = 1e9 B (slack residency).
    Stage 1 evacuates all 20 jobs (Z* = 0). Pfill-job-capacity and ingest-
    job-capacity are both 20, so balanced split x_R = x_S = 10 gives
    p_pfill = p_ing = 0.5 and phi* = 0.5. Constants are kept moderate so
    the SCIPY linprog backend stays in its well-conditioned regime.
    """
    Q = 20
    inst = ProblemInstance(
        model_idx=np.zeros(Q, dtype=int),
        T=np.full(Q, 1000.0),
        beta=np.full(Q, 4.0),
        eta=np.full(Q, 100.0),
        rho=np.full(Q, 1000.0),
        n=np.ones(Q),
        lambda_bps=np.array([5e6]),
        W=np.array([[1.0]]),
        W_ing=np.array([[1.0]]),
        C_res=np.array([1e9]),
        mu_ing=1e5,
        D=20.0,
        M_names=("toy",),
        L_names=("toy_dst",),
    )
    s1 = solve_stage1(inst)
    np.testing.assert_allclose(s1.Z_star, 0.0, atol=1e-6)
    res = solve_stage2(inst, s1)
    np.testing.assert_allclose(res.phi_star, 0.5, atol=1e-4)
    np.testing.assert_allclose(res.x_R.sum(), 10.0, atol=1e-3)
    np.testing.assert_allclose(res.x_S.sum(), 10.0, atol=1e-3)


def test_phi_equals_realized_peak():
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    res = solve_stage2(inst, s1)
    np.testing.assert_allclose(max(res.pressures.values()), res.phi_star, atol=1e-6)


def test_phi_monotone_in_deadline():
    phi = []
    for D in (10.0, 60.0, 300.0):
        inst = build_instance(D=D, seed=1)
        s1 = solve_stage1(inst)
        phi.append(solve_stage2(inst, s1).phi_star)
    assert phi[0] >= phi[1] - 2e-4 >= phi[2] - 4e-4


def test_residency_reported_not_pressured():
    # At full evacuation the destination holds the whole pod's KV, so
    # utilization ~ realized KV / C_res, while phi* (rate resources at a slack
    # deadline) stays strictly below it.
    inst = build_instance(D=2000.0, occupancy=0.75, seed=0)
    s1 = solve_stage1(inst)
    res = solve_stage2(inst, s1)
    expect = (inst.eta * inst.T * (inst.n - res.z)).sum() / inst.C_res[0]
    np.testing.assert_allclose(res.residency_utilization, expect, rtol=1e-6)
    assert all(not k.startswith("res|") for k in res.pressures)
    assert res.phi_star < res.residency_utilization
