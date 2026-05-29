"""CVXPY shadow prices of the Stage 2 pressure-ceiling constraints (Section 16).

Re-solves the Stage 2 LP with CLARABEL (reliable duals; SCIPY does not expose
them) and returns the duals of the three ceiling constraints L_i/C_i <= phi.
phi-stationarity forces sum_i pi_i = 1, so these are exactly the simplex prices
the dual decomposition (stage2_dual.py) converges to.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from stage1 import Stage1Result


def ceiling_duals(inst: ProblemInstance, stage1: Stage1Result) -> tuple[dict, float]:
    """Return ({("net", l) | ("pfill", l, m) | ("ing", l, m): pi_i}, phi*).

    Keys cover only active pressure indices (C_i > 0). Values are nonnegative
    and sum to 1.
    """
    Q, L, M = inst.T.size, inst.lambda_bps.size, len(inst.M_names)
    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)
    phi = cp.Variable(nonneg=True)

    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W.T * inst.mu_ing * inst.D

    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]

    L_net = (inst.beta * inst.T) @ x_R + (inst.eta * inst.T) @ x_S
    L_pfill = S_pfill @ x_R
    L_ing = S_ing @ x_S

    # Normalize every row to O(1) (loads/capacities span bytes..instance-seconds,
    # which wrecks the conditioning of an unscaled LP). Used only on default
    # all-positive-capacity instances, so require C > 0.
    assert (C_net > 0).all() and (C_pfill > 0).all() and (C_ing > 0).all()
    iCn, iCp, iCi = 1.0 / C_net, 1.0 / C_pfill, 1.0 / C_ing
    r_net = cp.multiply(iCn, L_net)
    r_pfill = cp.multiply(iCp, L_pfill)
    r_ing = cp.multiply(iCi, L_ing)
    c_net, c_pfill, c_ing = r_net <= phi, r_pfill <= phi, r_ing <= phi
    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        cp.sum(z) == stage1.Z_star,
        r_net <= 1, r_pfill <= 1, r_ing <= 1,
        c_net, c_pfill, c_ing,
    ]
    prob = cp.Problem(cp.Minimize(phi), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 2 dual LP returned {prob.status}")

    d_net = np.asarray(c_net.dual_value).reshape(L)
    d_pfill = np.asarray(c_pfill.dual_value).reshape(M, L)
    d_ing = np.asarray(c_ing.dual_value).reshape(M, L)

    pi = {}
    for l in range(L):
        if C_net[l] > 0:
            pi[("net", l)] = float(d_net[l])
        for m in range(M):
            if C_pfill[m, l] > 0:
                pi[("pfill", l, m)] = float(d_pfill[m, l])
            if C_ing[m, l] > 0:
                pi[("ing", l, m)] = float(d_ing[m, l])
    return pi, float(phi.value)
