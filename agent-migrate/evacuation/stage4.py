"""Stage 4 of the staged evacuation program: minimize variation across classes.

LP form (Section 12, Stage 4 of `formulation.md`):

    min  V
    s.t. sum_l (x_R[q,l] + x_S[q,l]) + z_q = n_q          (conservation)
         sum_q z_q == Z*                                    (Stage 1 link)
         L_i(x) <= C_i                                      (base capacity)
         L_i(x) <= phi* * C_i                               (Stage 2 ceiling)
         r_q(x, z) <= H*                                    (Stage 3 link)
         r_q(x, z) - r_bar(x, z) <= V                       (fairness, upper)
         r_bar(x, z) - r_q(x, z) <= V                       (fairness, lower)
         x_R, x_S, z >= 0; V >= 0

with r_bar = (1/N) * sum_q n_q * r_q and N = sum_q n_q.

When the optional `stage2b` is supplied, also enforce the Psi* ceiling, turning
the LP into a QCP; CLARABEL is used in that case to match `stage3.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from loads import inv_cap, loads, replay_infeasible
from stage2b import Stage2bResult
from stage3 import Stage3Result, recon_costs


@dataclass(frozen=True)
class Stage4Result:
    x_R: np.ndarray
    x_S: np.ndarray
    z: np.ndarray
    Z_star: float
    phi_star: float
    H_star: float
    V_star: float
    r_q: np.ndarray
    r_bar: float
    status: str


def solve_stage4(inst: ProblemInstance,
                 stage3: Stage3Result,
                 stage2b: Stage2bResult | None = None) -> Stage4Result:
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)
    V = cp.Variable(nonneg=True)
    r_bar = cp.Variable()  # aux: prevents dense expansion in |r_q - r_bar| <= V

    C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)

    L_net = b_net_R @ x_R + b_net_S @ x_S
    L_pfill = S_pfill @ x_R
    L_ing = S_ing @ x_S
    L_res = b_net_S @ (x_R + x_S)

    inv_C_net, inv_C_pfill, inv_C_ing = map(inv_cap, (C_net, C_pfill, C_ing))

    c_R, c_S = recon_costs(inst)
    r_expr = (cp.sum(cp.multiply(c_R, x_R), axis=1)
              + cp.sum(cp.multiply(c_S, x_S), axis=1)
              + inst.d_miss * z) / inst.n

    N_total = float(inst.n.sum())

    phi_ceil = stage3.phi_star + 1e-7
    H_ceil = stage3.H_star + 1e-7
    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        cp.sum(z) == stage3.Z_star,
        L_net <= C_net,
        L_pfill <= C_pfill,
        L_ing <= C_ing,
        L_res <= C_res,
        cp.multiply(inv_C_net, L_net) <= phi_ceil,
        cp.multiply(inv_C_pfill, L_pfill) <= phi_ceil,
        cp.multiply(inv_C_ing, L_ing) <= phi_ceil,
        r_expr <= H_ceil,
        inst.n @ r_expr == N_total * r_bar,
        r_expr - r_bar <= V,
        r_bar - r_expr <= V,
    ]
    bad = replay_infeasible(inst)
    if bad.any():
        constraints.append(x_R[bad, :] == 0)

    if stage2b is not None:
        p_net = cp.multiply(inv_C_net, L_net)
        p_pfill = cp.multiply(inv_C_pfill, L_pfill)
        p_ing = cp.multiply(inv_C_ing, L_ing)
        psi_expr = 0.5 * (cp.sum_squares(p_net)
                          + cp.sum_squares(p_pfill)
                          + cp.sum_squares(p_ing))
        constraints.append(psi_expr <= stage2b.psi_star + 1e-7)

    prob = cp.Problem(cp.Minimize(V), constraints)
    solver = cp.CLARABEL if stage2b is not None else cp.SCIPY
    prob.solve(solver=solver)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 4 solver returned {prob.status}")

    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    z_val = np.maximum(np.asarray(z.value, dtype=float), 0.0)

    r_q_val = ((c_R * x_R_val).sum(axis=1)
               + (c_S * x_S_val).sum(axis=1)
               + inst.d_miss * z_val) / inst.n
    r_bar_val = float(r_bar.value)

    return Stage4Result(
        x_R=x_R_val, x_S=x_S_val, z=z_val,
        Z_star=float(stage3.Z_star), phi_star=float(stage3.phi_star),
        H_star=float(stage3.H_star), V_star=float(V.value),
        r_q=r_q_val, r_bar=r_bar_val, status=prob.status,
    )
