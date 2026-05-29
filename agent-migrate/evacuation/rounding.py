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


def round_plan_naive(inst: ProblemInstance,
                     x_R: np.ndarray, x_S: np.ndarray, z: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Floor + largest-fractional-remainder, with NO capacity rejection.

    Each class's deficit goes to its highest-remainder categories regardless of
    resource ceilings, so the result can overload a resource (cf. round_plan)."""
    L = inst.lambda_bps.size
    x_R_int = np.floor(x_R).astype(np.int64)
    x_S_int = np.floor(x_S).astype(np.int64)
    z_int = np.floor(z).astype(np.int64)
    rem = np.concatenate([x_R - x_R_int, x_S - x_S_int, (z - z_int)[:, None]], axis=1)
    n_int = np.round(inst.n).astype(np.int64)

    for q in range(inst.T.size):
        deficit = int(n_int[q] - x_R_int[q].sum() - x_S_int[q].sum() - z_int[q])
        if deficit <= 0:
            continue
        for j in np.argsort(-rem[q])[:deficit]:
            if j < L:
                x_R_int[q, j] += 1
            elif j < 2 * L:
                x_S_int[q, j - L] += 1
            else:
                z_int[q] += 1
    return x_R_int, x_S_int, z_int


def evaluate_plan(inst: ProblemInstance,
                  x_R_int: np.ndarray, x_S_int: np.ndarray, z_int: np.ndarray
                  ) -> tuple[float, float, float]:
    """Return (phi, max_violation, z_total) for an integer plan.

    phi = peak normalized pressure; max_violation = max_i max(0, L_i/C_i - 1)
    over active indices (0 iff feasible); z_total = jobs left behind."""
    Q, L, M = inst.T.size, inst.lambda_bps.size, len(inst.M_names)
    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W.T * inst.mu_ing * inst.D

    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]

    xR, xS = x_R_int.astype(float), x_S_int.astype(float)
    L_net = (inst.beta * inst.T) @ xR + (inst.eta * inst.T) @ xS
    L_pfill = S_pfill @ xR
    L_ing = S_ing @ xS

    ratios = list(L_net[C_net > 0] / C_net[C_net > 0])
    ratios += list(L_pfill[C_pfill > 0] / C_pfill[C_pfill > 0])
    ratios += list(L_ing[C_ing > 0] / C_ing[C_ing > 0])
    r = np.array(ratios)
    return float(r.max()), float(np.maximum(r - 1.0, 0.0).max()), float(z_int.sum())
