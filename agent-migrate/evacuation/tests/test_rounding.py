"""Semantic tests for Section 17.1 rounding.

  1. hand-worked tiny instance   — largest-remainder wins; capacity-rejected
                                   highest pushes to next legal category.
  2. conservation                 — per-class sum = n_q after rounding.
  3. capacity respected           — integer-arithmetic aggregate loads <= caps
                                   (including the decode-HBM residency stock).
  4. residency rejection          — a unit that would exceed C_res falls to z.
  5. round-of-integer is identity — already-integer feasible plan is unchanged.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from loads import loads
from rounding import round_plan, round_plan_naive, evaluate_plan
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage3 import solve_stage3
from stage4 import solve_stage4


def _toy(Q=1, lambda_bps=10.0, W=100.0, mu_ing=1.0, D=100.0, C_res=1e9):
    return ProblemInstance(
        model_idx=np.zeros(Q, dtype=int),
        T=np.ones(Q),
        beta=np.ones(Q),
        eta=np.ones(Q),
        rho=np.ones(Q),
        n=np.ones(Q),
        lambda_bps=np.array([lambda_bps]),
        W=np.array([[W]]),
        W_ing=np.array([[W]]),
        C_res=np.array([C_res]),
        mu_ing=mu_ing,
        D=D,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=2.0 * D,
    )


def test_hand_worked_largest_remainder():
    """Largest fractional remainder wins when capacity is loose."""
    inst = _toy()
    xR_i, xS_i, z_i = round_plan(inst, np.array([[0.3]]), np.array([[0.4]]), np.array([0.3]))
    assert xR_i[0, 0] == 0
    assert xS_i[0, 0] == 1
    assert z_i[0] == 0


def test_hand_worked_capacity_rejection_routes_to_s():
    """Two jobs, both with R floors that together already exhaust C_pfill.
    Class 1's R remainder is largest but pushes L_pfill past C_pfill, so
    rounding must walk to the next category (S)."""
    inst = _toy(Q=2, lambda_bps=100.0, W=1.0, mu_ing=10.0, D=1.0)
    # Class 0 fully replays (consumes the 1 unit of prefill).
    # Class 1 has highest remainder on R but R is now saturated -> fall to S.
    x_R = np.array([[1.0], [0.5]])
    x_S = np.array([[0.0], [0.3]])
    z = np.array([0.0, 0.2])
    xR_i, xS_i, z_i = round_plan(inst, x_R, x_S, z)
    assert xR_i.tolist() == [[1], [0]]
    assert xS_i.tolist() == [[0], [1]]
    assert z_i.tolist() == [0, 0]


def test_residency_rejection_falls_to_z():
    """C_res fits exactly one job's KV; the second state unit must strand."""
    inst = _toy(Q=2, lambda_bps=100.0, W=1.0, mu_ing=10.0, D=1.0, C_res=1.0)
    x_S = np.array([[0.5], [0.5]])
    z = np.array([0.5, 0.5])
    xR_i, xS_i, z_i = round_plan(inst, np.zeros((2, 1)), x_S, z)
    assert int(xS_i.sum()) == 1 and int(z_i.sum()) == 1
    assert evaluate_plan(inst, xR_i, xS_i, z_i)[1] == 0.0


def test_conservation_on_stage4_plan():
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    xR_i, xS_i, z_i = round_plan(inst, s4.x_R, s4.x_S, s4.z)
    n_int = np.round(inst.n).astype(np.int64)
    np.testing.assert_array_equal(xR_i.sum(axis=1) + xS_i.sum(axis=1) + z_i, n_int)


def test_capacity_respected_on_stage4_plan():
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    xR_i, xS_i, z_i = round_plan(inst, s4.x_R, s4.x_S, s4.z)

    C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)
    L_net = b_net_R @ xR_i + b_net_S @ xS_i
    L_pfill = S_pfill @ xR_i.astype(float)
    L_ing = S_ing @ xS_i.astype(float)
    L_res = b_net_S @ (xR_i + xS_i).astype(float)
    assert np.all(L_net <= C_net + 1e-9)
    assert np.all(L_pfill <= C_pfill + 1e-9)
    assert np.all(L_ing <= C_ing + 1e-9)
    assert np.all(L_res <= C_res + 1e-9)


def test_naive_violates_where_repair_routes_around():
    """Same toy as the capacity-rejection test: class 1's largest remainder is R,
    but R prefill is already saturated. round_plan routes to S (no violation);
    round_plan_naive takes R and overloads C_pfill."""
    inst = _toy(Q=2, lambda_bps=100.0, W=1.0, mu_ing=10.0, D=1.0)
    x_R = np.array([[1.0], [0.5]])
    x_S = np.array([[0.0], [0.3]])
    z = np.array([0.0, 0.2])
    xR_n, xS_n, z_n = round_plan_naive(inst, x_R, x_S, z)
    assert xR_n.tolist() == [[1], [1]]                      # naive took R
    assert evaluate_plan(inst, xR_n, xS_n, z_n)[1] > 0.0    # overloads prefill
    xR_r, xS_r, z_r = round_plan(inst, x_R, x_S, z)
    assert evaluate_plan(inst, xR_r, xS_r, z_r)[1] == 0.0   # repair stays feasible
    # both conserve jobs per class
    for xR_, xS_, z_ in ((xR_n, xS_n, z_n), (xR_r, xS_r, z_r)):
        np.testing.assert_array_equal(xR_.sum(1) + xS_.sum(1) + z_, np.ones(2))


def test_evaluate_plan_reproduces_lp_pressure():
    """evaluate_plan on the fractional optimum recovers phi* and is feasible."""
    inst = build_instance(seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    phi, viol, z_tot = evaluate_plan(inst, s2.x_R, s2.x_S, s2.z)
    assert abs(phi - s2.phi_star) < 1e-6
    assert viol < 1e-9
    assert abs(z_tot - s2.Z_star) < 1e-6


def test_round_of_integer_is_identity():
    """An already-integer, capacity-respecting plan is unchanged."""
    inst = _toy()
    xR_i, xS_i, z_i = round_plan(inst, np.array([[1.0]]), np.array([[0.0]]), np.array([0.0]))
    assert xR_i[0, 0] == 1
    assert xS_i[0, 0] == 0
    assert z_i[0] == 0
