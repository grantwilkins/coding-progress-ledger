"""Stage 1 of the staged evacuation program: maximize evacuated jobs.

LP form (Section 12, Stage 1 of `formulation.md`):

    min  sum_q z_q
    s.t. sum_l (x_R[q,l] + x_S[q,l]) + z_q = n_q          (conservation)
         L_net[l]      <= C_net[l]                         (per-destination network)
         L_pfill[m,l]  <= C_pfill[m,l]                     (per-(model, destination) prefill)
         L_ing[m,l]    <= C_ing[m,l]                       (per-(model, destination) state ingest)
         x_R, x_S, z >= 0; z <= n

Compatibility (W[l,m] = 0) is enforced implicitly: zero capacity at (m,l) forces
the corresponding x sums to zero, since loads are nonnegative linear combinations
of nonnegative variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance


@dataclass(frozen=True)
class Stage1Result:
    x_R: np.ndarray   # (Q, L) replay placement
    x_S: np.ndarray   # (Q, L) state-transfer placement
    z: np.ndarray     # (Q,)   unmoved
    Z_star: float
    status: str


def solve_stage1(inst: ProblemInstance) -> Stage1Result:
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)

    C_net = inst.lambda_bps * inst.D                # (L,)
    C_pfill = inst.W.T * inst.D                     # (M, L)
    C_ing = inst.W.T * inst.mu_ing * inst.D         # (M, L)

    # Per-model weighted assignment: rows of S_* select jobs of model m,
    # weighted by the per-job resource coefficient (prefill GPU-s, ingest bytes).
    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]

    b_net_R = inst.beta * inst.T   # (Q,) replay network bytes per job
    b_net_S = inst.eta * inst.T    # (Q,) state-transfer network bytes per job

    constraints = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        z <= inst.n,
        b_net_R @ x_R + b_net_S @ x_S <= C_net,
        S_pfill @ x_R <= C_pfill,
        S_ing @ x_S <= C_ing,
    ]

    prob = cp.Problem(cp.Minimize(cp.sum(z)), constraints)
    prob.solve(solver=cp.SCIPY)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 1 LP solver returned {prob.status}")

    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    z_val = np.maximum(np.asarray(z.value, dtype=float), 0.0)

    # Renormalize each class to enforce exact conservation despite solver slack.
    total = x_R_val.sum(axis=1) + x_S_val.sum(axis=1) + z_val
    scale = np.where(total > 0, inst.n / total, 1.0)
    x_R_val *= scale[:, None]
    x_S_val *= scale[:, None]
    z_val *= scale

    return Stage1Result(
        x_R=x_R_val, x_S=x_S_val, z=z_val,
        Z_star=float(z_val.sum()), status=prob.status,
    )
