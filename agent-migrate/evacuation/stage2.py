"""Stage 2 of the staged evacuation program: minimize peak normalized pressure.

LP form (Section 12, Stage 2 of `formulation.md`):

    min  phi
    s.t. sum_l (x_R[q,l] + x_S[q,l]) + z_q = n_q          (conservation)
         sum_q z_q == Z*                                   (Stage 1 link)
         L_net[l]      <= C_net[l]                         (base capacity)
         L_pfill[m,l]  <= C_pfill[m,l]
         L_ing[m,l]    <= C_ing[m,l]
         L_net[l]      <= phi * C_net[l]                   (pressure ceiling)
         L_pfill[m,l]  <= phi * C_pfill[m,l]
         L_ing[m,l]    <= phi * C_ing[m,l]
         x_R, x_S, z >= 0; phi >= 0

Zero-capacity (l, m) pairs (W[l,m] = 0) collapse to L <= 0 in both base and
pressure forms, matching the formulation's exclusion of those indices from I.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from stage1 import Stage1Result


@dataclass(frozen=True)
class Stage2Result:
    x_R: np.ndarray
    x_S: np.ndarray
    z: np.ndarray
    Z_star: float
    phi_star: float
    pressures: dict[str, float]
    status: str


def solve_stage2(inst: ProblemInstance, stage1: Stage1Result) -> Stage2Result:
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)
    phi = cp.Variable(nonneg=True)

    C_net = inst.lambda_bps * inst.D                # (L,)
    C_pfill = inst.W.T * inst.D                     # (M, L)
    C_ing = inst.W.T * inst.mu_ing * inst.D         # (M, L)

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

    # Normalized pressure form: L_i / C_i <= phi. Skip indices with C_i = 0
    # (W[l,m] = 0). Those loads are pinned to zero by base capacity already.
    inv_C_net = np.where(C_net > 0, 1.0 / np.where(C_net > 0, C_net, 1.0), 0.0)
    inv_C_pfill = np.where(C_pfill > 0, 1.0 / np.where(C_pfill > 0, C_pfill, 1.0), 0.0)
    inv_C_ing = np.where(C_ing > 0, 1.0 / np.where(C_ing > 0, C_ing, 1.0), 0.0)

    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        cp.sum(z) == stage1.Z_star,
        L_net <= C_net,
        L_pfill <= C_pfill,
        L_ing <= C_ing,
        cp.multiply(inv_C_net, L_net) <= phi,
        cp.multiply(inv_C_pfill, L_pfill) <= phi,
        cp.multiply(inv_C_ing, L_ing) <= phi,
    ]

    prob = cp.Problem(cp.Minimize(phi), constraints)
    prob.solve(solver=cp.SCIPY)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 2 LP solver returned {prob.status}")

    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    z_val = np.maximum(np.asarray(z.value, dtype=float), 0.0)

    L_net_val = b_net_R @ x_R_val + b_net_S @ x_S_val
    L_pfill_val = S_pfill @ x_R_val
    L_ing_val = S_ing @ x_S_val

    pressures: dict[str, float] = {}
    for l, lname in enumerate(inst.L_names):
        if C_net[l] > 0:
            pressures[f"net|{lname}"] = float(L_net_val[l] / C_net[l])
        for m, mname in enumerate(inst.M_names):
            if C_pfill[m, l] > 0:
                pressures[f"pfill|{lname}|{mname}"] = float(L_pfill_val[m, l] / C_pfill[m, l])
            if C_ing[m, l] > 0:
                pressures[f"ing|{lname}|{mname}"] = float(L_ing_val[m, l] / C_ing[m, l])

    return Stage2Result(
        x_R=x_R_val, x_S=x_S_val, z=z_val,
        Z_star=float(stage1.Z_star), phi_star=float(phi.value),
        pressures=pressures, status=prob.status,
    )
