"""Stage 2b of the staged evacuation program: minimize congestion potential.

QP form (Section 12, Stage 2b of `formulation.md`):

    min  Psi(x) = 1/2 * sum_i (L_i(x) / C_i)^2
    s.t. sum_l (x_R[q,l] + x_S[q,l]) + z_q = n_q          (conservation)
         sum_q z_q == Z*                                   (Stage 1 link)
         L_i(x) <= C_i                                      (base capacity)
         x_R, x_S, z >= 0

Formulation departure: the spec's optional `p_i(x) <= phi*` ceiling is
intentionally dropped, so this is the unconstrained QP minimizer rather than
a lex tie-breaker over Stage 2. The optimum satisfies the Section 15
endogenous-crossover KKT identity directly.

Implementation note: all coefficients are pre-normalized by capacity (so
each entry is O(1)) before being handed to CVXPY. Without this, CLARABEL
converges to a non-optimal stationary point because the raw coefficient
matrix spans ~20 orders of magnitude (network B/s vs prefill GPU-s) and the
KKT factorization breaks down.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from stage1 import Stage1Result


@dataclass(frozen=True)
class Stage2bResult:
    x_R: np.ndarray
    x_S: np.ndarray
    z: np.ndarray
    Z_star: float
    psi_star: float
    pressures: dict[str, float]
    status: str


def solve_stage2b(inst: ProblemInstance, stage1: Stage1Result) -> Stage2bResult:
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)

    C_net = inst.lambda_bps * inst.D                # (L,)
    C_pfill = inst.W.T * inst.D                     # (M, L)
    C_ing = inst.W.T * inst.mu_ing * inst.D         # (M, L)

    inv_C_net = 1.0 / C_net
    inv_C_pfill = np.where(C_pfill > 0, 1.0 / np.where(C_pfill > 0, C_pfill, 1.0), 0.0)
    inv_C_ing = np.where(C_ing > 0, 1.0 / np.where(C_ing > 0, C_ing, 1.0), 0.0)

    model_onehot = (inst.model_idx[None, :] == np.arange(M)[:, None]).astype(float)  # (M, Q)
    T_over_rho = inst.T / inst.rho                                                    # (Q,)
    eta_T = inst.eta * inst.T                                                          # (Q,)
    beta_T = inst.beta * inst.T                                                        # (Q,)

    a_net_R = beta_T[:, None] * inv_C_net[None, :]                                    # (Q, L)
    a_net_S = eta_T[:, None] * inv_C_net[None, :]                                     # (Q, L)
    a_pfill = (model_onehot * T_over_rho[None, :])[:, :, None] * inv_C_pfill[:, None, :]  # (M, Q, L)
    a_ing = (model_onehot * eta_T[None, :])[:, :, None] * inv_C_ing[:, None, :]       # (M, Q, L)

    p_net = cp.sum(cp.multiply(a_net_R, x_R), axis=0) + cp.sum(cp.multiply(a_net_S, x_S), axis=0)
    p_pfill = cp.vstack([cp.sum(cp.multiply(a_pfill[m], x_R), axis=0) for m in range(M)])
    p_ing = cp.vstack([cp.sum(cp.multiply(a_ing[m], x_S), axis=0) for m in range(M)])

    objective = 0.5 * (cp.sum_squares(p_net) + cp.sum_squares(p_pfill) + cp.sum_squares(p_ing))

    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        cp.sum(z) == stage1.Z_star,
        p_net <= 1,
        p_pfill <= 1,
        p_ing <= 1,
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 2b QP solver returned {prob.status}")

    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    z_val = np.maximum(np.asarray(z.value, dtype=float), 0.0)

    p_net_val = (a_net_R * x_R_val).sum(axis=0) + (a_net_S * x_S_val).sum(axis=0)
    p_pfill_val = np.array([(a_pfill[m] * x_R_val).sum(axis=0) for m in range(M)])
    p_ing_val = np.array([(a_ing[m] * x_S_val).sum(axis=0) for m in range(M)])

    pressures: dict[str, float] = {}
    for l, lname in enumerate(inst.L_names):
        if C_net[l] > 0:
            pressures[f"net|{lname}"] = float(p_net_val[l])
        for m, mname in enumerate(inst.M_names):
            if C_pfill[m, l] > 0:
                pressures[f"pfill|{lname}|{mname}"] = float(p_pfill_val[m, l])
            if C_ing[m, l] > 0:
                pressures[f"ing|{lname}|{mname}"] = float(p_ing_val[m, l])

    return Stage2bResult(
        x_R=x_R_val, x_S=x_S_val, z=z_val,
        Z_star=float(stage1.Z_star), psi_star=float(prob.value),
        pressures=pressures, status=prob.status,
    )
