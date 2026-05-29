"""Stage 2 of the staged evacuation program: minimize peak normalized pressure.

LP/CP form (Section 12, Stage 2 of `formulation.md`):

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

The Stage 1 optimum is preserved by objective-specific link constraints:
  throughput : sum z == Z*
  max_min    : sum z == Z*  and  z <= (1-alpha*) n
  prop_fair  : sum z == Z*  and  sum_q w_q log(eps + u_q) >= U* - delta

Zero-capacity (l, m) pairs (W[l,m] = 0) collapse to L <= 0 in both base and
pressure forms, matching the formulation's exclusion of those indices from I.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from loads import inv_cap, loads, norm_cap
from stage1 import CLARABEL_OPTS, Stage1Result


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
    x_R = cp.Variable(stage1.x_R.shape, nonneg=True)
    x_S = cp.Variable(stage1.x_S.shape, nonneg=True)
    z = cp.Variable(inst.T.size, nonneg=True)
    phi = cp.Variable(nonneg=True)

    C_net, C_pfill, C_ing, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)

    L_net = b_net_R @ x_R + b_net_S @ x_S
    L_pfill = S_pfill @ x_R
    L_ing = S_ing @ x_S

    # Pressure form: L_i / C_i <= phi. inv_cap is 0 where C_i = 0 (W[l,m] = 0),
    # making those ceilings vacuous; the base norm_cap rows pin those loads to 0.
    p_net = cp.multiply(inv_cap(C_net), L_net)
    p_pfill = cp.multiply(inv_cap(C_pfill), L_pfill)
    p_ing = cp.multiply(inv_cap(C_ing), L_ing)
    (a_net, r_net), (a_pf, r_pf), (a_in, r_in) = map(norm_cap, (C_net, C_pfill, C_ing))

    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        cp.sum(z) == stage1.Z_star,
        cp.multiply(a_net, L_net) <= r_net,
        cp.multiply(a_pf, L_pfill) <= r_pf,
        cp.multiply(a_in, L_ing) <= r_in,
        p_net <= phi, p_pfill <= phi, p_ing <= phi,
    ]

    solver = cp.SCIPY
    if stage1.objective == "max_min":
        constraints.append(z <= (1.0 - stage1.alpha_star) * inst.n + 1e-6 * inst.n)
    elif stage1.objective == "prop_fair":
        w = inst.n if stage1.utility_weights == "population" else np.ones_like(inst.n)
        u = 1.0 - cp.multiply(1.0 / inst.n, z)
        U_expr = cp.sum(cp.multiply(w, cp.log(stage1.utility_epsilon + u)))
        constraints += [u <= 1, U_expr >= stage1.U_star - stage1.utility_delta]
        solver = cp.CLARABEL

    prob = cp.Problem(cp.Minimize(phi), constraints)
    prob.solve(solver=solver, **(CLARABEL_OPTS if solver == cp.CLARABEL else {}))
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 2 solver returned {prob.status}")

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
