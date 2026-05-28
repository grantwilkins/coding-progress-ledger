"""Semantic tests for Stage 2b of the evacuation QP.

The five tests target believable errors in the QP wiring:
  1. conservation         — `sum(x_R[q]) + sum(x_S[q]) + z[q] == n_q`
  2. Stage 1 link         — `sum(z) == Z*`
  3. QP dominates Stage 2 — `Psi(stage2b) <= Psi(stage2)` (unconstrained min)
  4. monotonicity in D    — `psi*(D)` non-increasing
  5. Section 15 KKT       — per-class active-pair gradient equality on a
                            deliberately slack instance.
"""

from __future__ import annotations

import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2b import solve_stage2b


def _evaluate_loads(inst, x_R, x_S):
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
    C_ing = inst.W.T * inst.mu_ing * inst.D
    return L_net, L_pfill, L_ing, C_net, C_pfill, C_ing


def _psi(inst, x_R, x_S):
    L_net, L_pfill, L_ing, C_net, C_pfill, C_ing = _evaluate_loads(inst, x_R, x_S)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_net = np.where(C_net > 0, L_net / np.where(C_net > 0, C_net, 1.0), 0.0)
        p_pfill = np.where(C_pfill > 0, L_pfill / np.where(C_pfill > 0, C_pfill, 1.0), 0.0)
        p_ing = np.where(C_ing > 0, L_ing / np.where(C_ing > 0, C_ing, 1.0), 0.0)
    return 0.5 * (np.sum(p_net**2) + np.sum(p_pfill**2) + np.sum(p_ing**2))


def test_conservation():
    inst = build_instance(total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    res = solve_stage2b(inst, s1)
    total = res.x_R.sum(axis=1) + res.x_S.sum(axis=1) + res.z
    np.testing.assert_allclose(total, inst.n, atol=1e-4)


def test_stage1_link():
    inst = build_instance(total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    res = solve_stage2b(inst, s1)
    np.testing.assert_allclose(res.z.sum(), s1.Z_star, atol=1e-3)


def test_qp_dominates_stage2():
    inst = build_instance(total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s2b = solve_stage2b(inst, s1)
    psi_s2 = _psi(inst, s2.x_R, s2.x_S)
    assert s2b.psi_star <= psi_s2 + 1e-6


def test_psi_monotone_in_deadline():
    psi = []
    for D in (300.0, 600.0):
        inst = build_instance(D=D, total_jobs=500, seed=1)
        s1 = solve_stage1(inst)
        psi.append(solve_stage2b(inst, s1).psi_star)
    assert psi[0] >= psi[1] - 1e-4


def test_kkt_active_pair_equality():
    """Section 15 KKT on a slack-capacity instance.

    With D=3600s and 1000 jobs, Stage 1 evacuates everything (Z* = 0) and
    base capacities sit comfortably below their limits. KKT stationarity
    then reduces to: for each class q, all active (action, destination)
    pairs share the same per-unit gradient, and inactive pairs have larger
    gradient.
    """
    inst = build_instance(D=3600.0, total_jobs=1000, seed=42)
    s1 = solve_stage1(inst)
    np.testing.assert_allclose(s1.Z_star, 0.0, atol=1e-5)
    res = solve_stage2b(inst, s1)

    L_net, L_pfill, L_ing, C_net, C_pfill, C_ing = _evaluate_loads(inst, res.x_R, res.x_S)

    assert np.all(L_net <= C_net * (1 - 1e-3) + 1e-9)
    assert np.all(L_pfill <= C_pfill * (1 - 1e-3) + 1e-9)
    assert np.all(L_ing <= C_ing * (1 - 1e-3) + 1e-9)

    # KKT for the unconstrained QP (with slack base capacities): for each
    # class q, every active (action, destination) variable has the same
    # gradient = -lambda_q, and that is also the *minimum* gradient over the
    # class. So max(grad over actives) should equal min(grad over all) to
    # within solver tolerance.
    Q = inst.T.size
    L = inst.lambda_bps.size
    x_tol = 5e-2
    rel_tol = 5e-2
    classes_checked = 0

    inv_Cnet2 = np.where(C_net > 0, 1.0 / np.where(C_net > 0, C_net, 1.0) ** 2, 0.0)
    inv_Cpfill2 = np.where(C_pfill > 0, 1.0 / np.where(C_pfill > 0, C_pfill, 1.0) ** 2, 0.0)
    inv_Cing2 = np.where(C_ing > 0, 1.0 / np.where(C_ing > 0, C_ing, 1.0) ** 2, 0.0)

    for q in range(Q):
        m = int(inst.model_idx[q])
        Tq = inst.T[q]; bq = inst.beta[q]; eq = inst.eta[q]; rq = inst.rho[q]
        compat = inst.W[:, m] > 0
        all_grads = []
        active_grads = []
        for l in range(L):
            if not compat[l]:
                continue
            gR = (Tq / rq) * inv_Cpfill2[m, l] * L_pfill[m, l] + (bq * Tq) * inv_Cnet2[l] * L_net[l]
            gS = (eq * Tq) * inv_Cing2[m, l] * L_ing[m, l] + (eq * Tq) * inv_Cnet2[l] * L_net[l]
            all_grads.extend([gR, gS])
            if res.x_R[q, l] > x_tol:
                active_grads.append(gR)
            if res.x_S[q, l] > x_tol:
                active_grads.append(gS)
        if not active_grads:
            continue
        classes_checked += 1
        g_min = min(all_grads)
        g_active_max = max(active_grads)
        # Mixed abs+rel tolerance: solver tolerance is O(1e-8) absolute on
        # the objective, so gradient residuals up to ~1e-8 are noise.
        abs_err = g_active_max - g_min
        rel_err = abs_err / max(abs(g_min), 1e-30)
        assert abs_err < 1e-8 or rel_err < rel_tol, (
            f"class {q}: active max grad {g_active_max:.3e} > class min grad {g_min:.3e} "
            f"(abs {abs_err:.3e}, rel {rel_err*100:.2f}%) (KKT stationarity violated)"
        )

    assert classes_checked >= Q // 2, f"test underweight: only {classes_checked}/{Q} classes checked"
