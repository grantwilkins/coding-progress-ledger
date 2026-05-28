"""Section 17.1 of `formulation.md`: convert a fractional plan to integer
job counts via floor + largest-fractional-remainder with capacity-aware
rejection. z_q participates as a category so the rounded plan cannot
evacuate more than the fractional Z* allowed.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance


def round_plan(inst: ProblemInstance,
               x_R: np.ndarray, x_S: np.ndarray, z: np.ndarray,
               tol: float = 1e-9
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer (x_R_int, x_S_int, z_int) matching the per-class
    deficit and respecting aggregate capacity at every resource index."""
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R_int = np.floor(x_R).astype(np.int64)
    x_S_int = np.floor(x_S).astype(np.int64)
    z_int = np.floor(z).astype(np.int64)

    r_R = x_R - x_R_int
    r_S = x_S - x_S_int
    r_z = z - z_int

    C_net = inst.lambda_bps * inst.D                # (L,)
    C_pfill = inst.W.T * inst.D                     # (M, L)
    C_ing = inst.W.T * inst.mu_ing * inst.D         # (M, L)

    # Aggregate loads from the floored assignment (running totals).
    b_net_R = inst.beta * inst.T
    b_net_S = inst.eta * inst.T
    L_net = b_net_R @ x_R_int + b_net_S @ x_S_int   # (L,) float64

    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]
    L_pfill = S_pfill @ x_R_int.astype(float)       # (M, L)
    L_ing = S_ing @ x_S_int.astype(float)

    n_int = np.round(inst.n).astype(np.int64)

    for q in range(Q):
        deficit = int(n_int[q] - x_R_int[q].sum() - x_S_int[q].sum() - z_int[q])
        if deficit <= 0:
            continue

        m = int(inst.model_idx[q])
        T_q = float(inst.T[q])
        beta_q = float(inst.beta[q])
        eta_q = float(inst.eta[q])
        rho_q = float(inst.rho[q])

        # Categories: ("R", l), ("S", l), ("z", None). Sort by remainder desc.
        cats: list[tuple[str, int | None, float]] = []
        for l in range(L):
            cats.append(("R", l, float(r_R[q, l])))
            cats.append(("S", l, float(r_S[q, l])))
        cats.append(("z", None, float(r_z[q])))
        cats.sort(key=lambda c: -c[2])

        assigned = 0
        for kind, l, _ in cats:
            if assigned >= deficit:
                break
            if kind == "z":
                z_int[q] += 1
                assigned += 1
                continue
            assert l is not None
            if kind == "R":
                d_net = beta_q * T_q
                d_pfill = T_q / rho_q
                d_ing = 0.0
            else:  # "S"
                d_net = eta_q * T_q
                d_pfill = 0.0
                d_ing = eta_q * T_q

            if L_net[l] + d_net > C_net[l] + tol:
                continue
            if d_pfill > 0:
                if C_pfill[m, l] == 0 or L_pfill[m, l] + d_pfill > C_pfill[m, l] + tol:
                    continue
            if d_ing > 0:
                if C_ing[m, l] == 0 or L_ing[m, l] + d_ing > C_ing[m, l] + tol:
                    continue

            if kind == "R":
                x_R_int[q, l] += 1
            else:
                x_S_int[q, l] += 1
            L_net[l] += d_net
            L_pfill[m, l] += d_pfill
            L_ing[m, l] += d_ing
            assigned += 1

        # Anything still unassigned falls back to z (no capacity).
        if assigned < deficit:
            z_int[q] += deficit - assigned

    return x_R_int, x_S_int, z_int
