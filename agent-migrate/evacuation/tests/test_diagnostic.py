"""Semantic tests for Section 14's diagnostic overload LP.

  1. hand-worked tiny instance  — closed-form sigma* and per-index slacks
                                  on a 1-dst, 1-model, 1-job toy.
  2. forced evac is full        — x_R + x_S sums to n_q on a tight-D instance
                                  (no z leakage).
  3. feasible -> sigma_star = 0 — Z* = 0 implies sigma* = 0 and all slacks 0.
  4. infeasible -> sigma_star>0 — Z* > 0 implies sigma* > 0 and some s_i > 0.
"""

from __future__ import annotations

import numpy as np

from diagnostic import solve_diagnostic
from instance import ProblemInstance, build_instance
from stage1 import solve_stage1


def test_hand_worked_tiny():
    """One job, one dst, one model, T=beta=eta=rho=mu=lambda=W=1, D=0.5.

    Capacities (all 0.5): C_net = lambda*D = 0.5, C_pfill = W*D = 0.5,
    C_ing = W*mu*D = 0.5. Loads with x_R + x_S = 1:
        L_net = 1 (routing-invariant)         -> p_net = 2  -> s_net >= 1.
        L_pfill = x_R                          -> p_pfill = 2*x_R -> s_pfill >= max(0, 2*x_R - 1).
        L_ing   = x_S                          -> p_ing   = 2*x_S -> s_ing   >= max(0, 2*x_S - 1).

    Phase 1 (min sigma): sigma >= 1 (forced by s_net). x_R = x_S = 0.5 makes
    p_pfill = p_ing = 1, so s_pfill = s_ing = 0. sigma_star = 1.

    Phase 2 (sigma fixed at 1, min sum s): sum = s_net + s_pfill + s_ing.
    s_net = 1 forced. Minimizing s_pfill + s_ing under p_pfill <= 1+s_pfill,
    p_ing <= 1+s_ing, and x_R + x_S = 1: the symmetric split x_R = x_S = 0.5
    gives s_pfill = s_ing = 0. Total = 1.
    """
    inst = ProblemInstance(
        model_idx=np.array([0]),
        T=np.array([1.0]),
        beta=np.array([1.0]),
        eta=np.array([1.0]),
        rho=np.array([1.0]),
        n=np.ones(1),
        lambda_bps=np.array([1.0]),
        W=np.array([[1.0]]),
        mu_ing=1.0,
        D=0.5,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=1.0,
    )
    diag = solve_diagnostic(inst)
    np.testing.assert_allclose(diag.sigma_star, 1.0, atol=1e-4)
    np.testing.assert_allclose(diag.s_net[0], 1.0, atol=1e-4)
    np.testing.assert_allclose(diag.s_pfill[0, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(diag.s_ing[0, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(diag.x_R[0, 0], 0.5, atol=1e-3)
    np.testing.assert_allclose(diag.x_S[0, 0], 0.5, atol=1e-3)


def test_forced_evac_full():
    """At a tight D where Stage 1 has Z* > 0, the diagnostic must still
    satisfy sum_l (x_R + x_S) == n_q for every q (z is forced to 0)."""
    inst = build_instance(D=10.0, total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    assert s1.Z_star > 1.0  # confirm the regime
    diag = solve_diagnostic(inst)
    moved = diag.x_R.sum(axis=1) + diag.x_S.sum(axis=1)
    np.testing.assert_allclose(moved, inst.n, atol=1e-4)


def test_feasible_instance_zero_sigma():
    """When Stage 1 is feasible, sigma* = 0 and all per-index slacks are 0."""
    inst = build_instance(D=900.0, total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    np.testing.assert_allclose(s1.Z_star, 0.0, atol=1e-4)
    diag = solve_diagnostic(inst)
    assert diag.sigma_star <= 1e-6
    assert diag.s_net.max() <= 1e-6
    assert diag.s_pfill.max() <= 1e-6
    assert diag.s_ing.max() <= 1e-6


def test_infeasible_instance_positive_sigma():
    """When Stage 1 is infeasible, sigma* > 0 and at least one slack > 0."""
    inst = build_instance(D=10.0, total_jobs=500, seed=0)
    diag = solve_diagnostic(inst)
    assert diag.sigma_star > 1e-3
    worst = max(diag.s_net.max(), diag.s_pfill.max(), diag.s_ing.max())
    assert worst > 1e-3
