"""Semantic tests for Stage 4 of the evacuation LP.

  1. hand-worked tiny instance  — closed-form V* on a 2-job, 1-dst instance
                                  with one forced-binding class and one slack
                                  class.
  2. dominates Stage 3          — V*(stage 4) <= max_q |r_q(s3) - r_bar(s3)|.
  3. respects prior stages      — sum(z) == Z*, p_i <= phi*, r_q <= H* + tol.
  4. 2b psi-ceiling honored     — realized 0.5 sum p_i^2 <= psi* when stage2b
                                  is supplied.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2b import solve_stage2b
from stage3 import recon_costs, solve_stage3
from stage4 import solve_stage4


def _eval_r_q(inst, x_R, x_S, z):
    c_R, c_S = recon_costs(inst)
    return ((c_R * x_R).sum(axis=1)
            + (c_S * x_S).sum(axis=1)
            + inst.d_miss * z) / inst.n


def _eval_pressures(inst, x_R, x_S):
    Q = inst.T.size
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
    """Two jobs, one model, one destination, n_q=1 each.

    Key trick: beta = eta = 1 makes the network load b_net_R = b_net_S, so
    L_net = beta*T_0 + beta*T_1 is INVARIANT under any (x_R, x_S) routing.
    With network the binding pressure index, Stage 2's phi* is achieved for
    every plan, leaving Stage 3 (and Stage 4) full routing freedom.

    Costs with lambda=10, mu=1, T=[2, 1.5], rho=[1, 3]:
        c_R[0] = beta*T_0/lambda + T_0/rho_0 = 0.2 + 2.0 = 2.2
        c_S[0] = eta*T_0/lambda + eta*T_0/mu  = 0.2 + 2.0 = 2.2   (job 0 binder)
        c_R[1] = 0.15 + 0.5 = 0.65
        c_S[1] = 0.15 + 1.5 = 1.65                                (job 1 slack)
    With H* = 2.2 (job 0), r_1 = 0.65 + x_S[1] in [0.65, 1.65]. Stage 4 wants
    to push r_1 toward r_bar:
        r_bar = (2.2 + r_1) / 2
        |r_0 - r_bar| = |r_1 - r_bar| = |1.55 - x_S[1]| / 2
    so V = |1.55 - x_S[1]| / 2. The H* ceiling allows x_S[1] up to 1.55, but
    the box bound x_S[1] <= n_q = 1 dominates, so V* = (1.55 - 1) / 2 = 0.275
    at x_S[1] = 1 (job 1 fully state-transferred), r_bar = 1.925.
    """
    inst = ProblemInstance(
        model_idx=np.array([0, 0]),
        T=np.array([2.0, 1.5]),
        beta=np.array([1.0, 1.0]),
        eta=np.array([1.0, 1.0]),
        rho=np.array([1.0, 3.0]),
        n=np.ones(2),
        lambda_bps=np.array([10.0]),
        W=np.array([[100.0]]),
        W_ing=np.array([[100.0]]),
        C_res=np.array([1e9]),
        mu_ing=1.0,
        D=100.0,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=200.0,
    )
    s1 = solve_stage1(inst)
    np.testing.assert_allclose(s1.Z_star, 0.0, atol=1e-6)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    np.testing.assert_allclose(s3.H_star, 2.2, atol=1e-3)
    s4 = solve_stage4(inst, s3)
    np.testing.assert_allclose(s4.V_star, 0.275, atol=1e-3)
    np.testing.assert_allclose(s4.r_q[1], 1.65, atol=1e-3)
    np.testing.assert_allclose(s4.r_bar, 1.925, atol=1e-3)


def test_dominates_stage3():
    """Stage 4 only adds constraints over Stage 3's feasible set; its V* must
    not exceed the V evaluated on Stage 3's plan."""
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    r_bar_s3 = float((inst.n * s3.r_q).sum() / inst.n.sum())
    V_pre = float(np.max(np.abs(s3.r_q - r_bar_s3)))
    assert s4.V_star <= V_pre + 1e-6


def test_respects_prior_stages():
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    np.testing.assert_allclose(s4.z.sum(), s3.Z_star, atol=1e-5)
    p_net, p_pfill, p_ing = _eval_pressures(inst, s4.x_R, s4.x_S)
    tol = 1e-6
    assert p_net.max() <= s3.phi_star + tol
    assert p_pfill.max() <= s3.phi_star + tol
    assert p_ing.max() <= s3.phi_star + tol
    r_q = _eval_r_q(inst, s4.x_R, s4.x_S, s4.z)
    assert r_q.max() <= s3.H_star + 1e-6


def test_2b_psi_ceiling_honored():
    inst = build_instance(D=300.0, seed=1)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s2b = solve_stage2b(inst, s1, stage2=s2)
    s3 = solve_stage3(inst, s2, stage2b=s2b)
    s4 = solve_stage4(inst, s3, stage2b=s2b)
    p_net, p_pfill, p_ing = _eval_pressures(inst, s4.x_R, s4.x_S)
    psi_realized = 0.5 * (np.sum(p_net ** 2)
                          + np.sum(p_pfill ** 2)
                          + np.sum(p_ing ** 2))
    assert psi_realized <= s2b.psi_star + 1e-4
