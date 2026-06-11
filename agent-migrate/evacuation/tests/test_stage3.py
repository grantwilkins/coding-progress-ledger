"""Semantic tests for Stage 3 of the evacuation LP.

  1. hand-worked tiny instance  — closed-form H* on an instance whose c_R/c_S
                                  asymmetry forces a specific R/S assignment.
  2. dominates Stage 2          — H*(stage 3) <= max_q r_q on stage 2's plan.
  3. respects prior stages      — sum(z) == Z* and all pressures <= phi*.
  4. monotone in D + 2b path    — H*(D) non-increasing; psi-ceiling honored
                                  when `stage2b` is supplied.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2b import solve_stage2b
from stage3 import recon_costs, solve_stage3


def _eval_r_q(inst, x_R, x_S, z):
    c_R, c_S = recon_costs(inst)
    return ((c_R * x_R).sum(axis=1)
            + (c_S * x_S).sum(axis=1)
            + inst.d_miss * z) / inst.n


def _eval_pressures(inst, x_R, x_S):
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)
    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]
    b_net_R = inst.beta * inst.T
    b_net_S = inst.eta * inst.T
    L_net = b_net_R @ x_R + b_net_S @ x_S
    L_pfill = S_pfill @ x_R
    L_ing = S_ing @ x_S
    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W_ing.T * inst.mu_ing * inst.D
    with np.errstate(divide="ignore", invalid="ignore"):
        p_net = np.where(C_net > 0, L_net / np.where(C_net > 0, C_net, 1.0), 0.0)
        p_pfill = np.where(C_pfill > 0, L_pfill / np.where(C_pfill > 0, C_pfill, 1.0), 0.0)
        p_ing = np.where(C_ing > 0, L_ing / np.where(C_ing > 0, C_ing, 1.0), 0.0)
    return p_net, p_pfill, p_ing


def test_hand_worked_tiny_instance():
    """Two jobs, one model, one destination.

    c_R[0]=10, c_S[0]=1.0001 (job 0 prefers state by ~10x).
    c_R[1]=0.1, c_S[1]=10.001 (job 1 prefers replay by ~100x).
    Stage 1 evacuates both (Z*=0). Stage 2's phi* makes both the pfill and
    ing ceilings bind simultaneously at the unique fractional split
    x_R[0]=10.0818-10*x_R[1], x_R[1]=1. Solving:
        20*phi* = 0.9182  (pfill cap),  2e6*phi* = 91818  (ing cap)
        => phi* = 0.04591, x_R[0] = 0.0818, x_R[1] = 1.0.
    With zero remaining freedom under the phi* ceiling, Stage 3 inherits
    that assignment, so H* = r_0 = 10*0.0818 + 1*0.9182 = 1.7364.
    """
    inst = ProblemInstance(
        model_idx=np.array([0, 0]),
        T=np.array([1000.0, 100.0]),
        beta=np.array([4.0, 4.0]),
        eta=np.array([100.0, 1e4]),
        rho=np.array([100.0, 1000.0]),
        n=np.ones(2),
        lambda_bps=np.array([1e9]),
        W=np.array([[1.0]]),
        W_ing=np.array([[1.0]]),
        C_res=np.array([1e9]),
        mu_ing=1e5,
        D=20.0,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=40.0,
    )
    s1 = solve_stage1(inst)
    np.testing.assert_allclose(s1.Z_star, 0.0, atol=1e-6)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    np.testing.assert_allclose(s3.H_star, 1.7364, atol=2e-3)
    np.testing.assert_allclose(s3.x_R[1, 0], 1.0, atol=2e-3)


def test_dominates_stage2():
    """Stage 3 only adds constraints over Stage 2's feasible set, so its
    worst-class r_q can only be <= the worst-class r_q on Stage 2's plan."""
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    r_s2 = _eval_r_q(inst, s2.x_R, s2.x_S, s2.z)
    assert s3.H_star <= r_s2.max() + 1e-6


def test_respects_prior_stages():
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    np.testing.assert_allclose(s3.z.sum(), s2.Z_star, atol=1e-5)
    p_net, p_pfill, p_ing = _eval_pressures(inst, s3.x_R, s3.x_S)
    tol = 1e-6
    assert p_net.max() <= s2.phi_star + tol
    assert p_pfill.max() <= s2.phi_star + tol
    assert p_ing.max() <= s2.phi_star + tol


def test_h_monotone_and_2b_ceiling():
    # d_miss is pinned across the sweep: the default 2D couples the stranded-job
    # penalty to the deadline, which dominates H* below the evacuation frontier
    # and would mask the capacity orientation this test targets.
    H = []
    for D in (60.0, 120.0, 300.0):
        inst = build_instance(D=D, d_miss=600.0, seed=1)
        s1 = solve_stage1(inst)
        s2 = solve_stage2(inst, s1)
        H.append(solve_stage3(inst, s2).H_star)
    assert H[0] >= H[1] - 1e-4 >= H[2] - 2e-4

    inst = build_instance(D=300.0, seed=1)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s2b = solve_stage2b(inst, s1, stage2=s2)  # formulation Stage 2b (phi*-aware)
    s3 = solve_stage3(inst, s2, stage2b=s2b)
    p_net, p_pfill, p_ing = _eval_pressures(inst, s3.x_R, s3.x_S)
    psi_realized = 0.5 * (np.sum(p_net ** 2)
                          + np.sum(p_pfill ** 2)
                          + np.sum(p_ing ** 2))
    assert psi_realized <= s2b.psi_star + 1e-4
