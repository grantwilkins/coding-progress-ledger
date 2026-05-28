"""Semantic tests for Section 17.1 rounding.

  1. hand-worked tiny instance   — largest-remainder wins; capacity-rejected
                                   highest pushes to next legal category.
  2. conservation                 — per-class sum = n_q after rounding.
  3. capacity respected           — integer-arithmetic aggregate loads <= caps.
  4. round-of-integer is identity — already-integer feasible plan is unchanged.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from rounding import round_plan
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage3 import solve_stage3
from stage4 import solve_stage4


def _make_toy(D: float = 100.0, W: float = 100.0):
    return ProblemInstance(
        model_idx=np.array([0]),
        T=np.array([1.0]),
        beta=np.array([1.0]),
        eta=np.array([1.0]),
        rho=np.array([1.0]),
        n=np.ones(1),
        lambda_bps=np.array([10.0]),
        W=np.array([[W]]),
        mu_ing=1.0,
        D=D,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=2.0 * D,
    )


def test_hand_worked_largest_remainder():
    """Largest fractional remainder wins when capacity is loose."""
    inst = _make_toy()
    x_R = np.array([[0.3]])
    x_S = np.array([[0.4]])
    z = np.array([0.3])
    xR_i, xS_i, z_i = round_plan(inst, x_R, x_S, z)
    assert xR_i[0, 0] == 0
    assert xS_i[0, 0] == 1
    assert z_i[0] == 0


def test_hand_worked_capacity_rejection_routes_to_s():
    """Two jobs, both with R floors that together already exhaust C_pfill.
    Class 1's R remainder is largest but pushes L_pfill past C_pfill, so
    rounding must walk to the next category (S)."""
    inst = ProblemInstance(
        model_idx=np.array([0, 0]),
        T=np.array([1.0, 1.0]),
        beta=np.array([1.0, 1.0]),
        eta=np.array([1.0, 1.0]),
        rho=np.array([1.0, 1.0]),
        n=np.ones(2),
        lambda_bps=np.array([100.0]),  # net loose
        W=np.array([[1.0]]),            # C_pfill = W*D = 1.0 exactly fits one R unit
        mu_ing=10.0,                    # ing loose
        D=1.0,
        M_names=("toy",),
        L_names=("toy_dst",),
        d_miss=2.0,
    )
    # Class 0 fully replays (consumes the 1 unit of prefill).
    # Class 1 has highest remainder on R but R is now saturated -> fall to S.
    x_R = np.array([[1.0], [0.5]])
    x_S = np.array([[0.0], [0.3]])
    z = np.array([0.0, 0.2])
    xR_i, xS_i, z_i = round_plan(inst, x_R, x_S, z)
    assert xR_i.tolist() == [[1], [0]]
    assert xS_i.tolist() == [[0], [1]]
    assert z_i.tolist() == [0, 0]


def test_conservation_on_stage4_plan():
    inst = build_instance(total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    xR_i, xS_i, z_i = round_plan(inst, s4.x_R, s4.x_S, s4.z)
    n_int = np.round(inst.n).astype(np.int64)
    np.testing.assert_array_equal(xR_i.sum(axis=1) + xS_i.sum(axis=1) + z_i, n_int)


def test_capacity_respected_on_stage4_plan():
    inst = build_instance(total_jobs=500, seed=0)
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    s4 = solve_stage4(inst, s3)
    xR_i, xS_i, z_i = round_plan(inst, s4.x_R, s4.x_S, s4.z)

    Q = inst.T.size
    M = len(inst.M_names)
    b_net_R = inst.beta * inst.T
    b_net_S = inst.eta * inst.T
    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]

    L_net = b_net_R @ xR_i + b_net_S @ xS_i
    L_pfill = S_pfill @ xR_i.astype(float)
    L_ing = S_ing @ xS_i.astype(float)
    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W.T * inst.mu_ing * inst.D
    assert np.all(L_net <= C_net + 1e-9)
    assert np.all(L_pfill <= C_pfill + 1e-9)
    assert np.all(L_ing <= C_ing + 1e-9)


def test_round_of_integer_is_identity():
    """An already-integer, capacity-respecting plan is unchanged."""
    inst = _make_toy()
    x_R = np.array([[1.0]])
    x_S = np.array([[0.0]])
    z = np.array([0.0])
    xR_i, xS_i, z_i = round_plan(inst, x_R, x_S, z)
    assert xR_i[0, 0] == 1
    assert xS_i[0, 0] == 0
    assert z_i[0] == 0
