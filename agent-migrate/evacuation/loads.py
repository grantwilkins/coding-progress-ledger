"""Shared load/capacity matrices for the evacuation LP (Stages 1 and 2).

Factors the per-instance coefficient construction so Stage 1 and Stage 2 build
identical loads. All quantities are linear in the placement variables x_R, x_S.
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance


def loads(inst: ProblemInstance):
    """Return (C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S).

    C_net   (L,)   network-byte budget per destination over the deadline
    C_pfill (M, L) prefill GPU-second budget per (model, destination)
    C_ing   (M, L) state-ingest byte budget per (model, destination)
    C_res   (L,)   decode-HBM residency budget per destination (bytes, a stock:
                   not deadline-scaled; evacuated KV must fit at the destination)
    S_pfill (M, Q) per-class prefill GPU-s coefficient, masked to its model row
    S_ing   (M, Q) per-class ingest bytes coefficient, masked to its model row
    b_net_R (Q,)   replay network bytes per job (context)
    b_net_S (Q,)   state-transfer network bytes per job (KV; also the per-job
                   residency footprint for either action)
    """
    Q = inst.T.size
    M = len(inst.M_names)

    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W_ing.T * inst.mu_ing * inst.D
    C_res = inst.C_res

    S_pfill = np.zeros((M, Q))
    S_ing = np.zeros((M, Q))
    for m in range(M):
        mask = inst.model_idx == m
        S_pfill[m, mask] = inst.T[mask] / inst.rho[mask]
        S_ing[m, mask] = inst.eta[mask] * inst.T[mask]

    b_net_R = inst.beta * inst.T
    b_net_S = inst.eta * inst.T
    return C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S


def replay_infeasible(inst: ProblemInstance) -> np.ndarray:
    """(Q,) bool: jobs whose solo prefill time T/rho exceeds the deadline.

    The fluid LP would otherwise spread such a job's prefill across the window
    even though no single node can finish it; these jobs must use state
    transfer or stay unmoved (Section 6 compatibility mask)."""
    return inst.T / inst.rho > inst.D


def inv_cap(C: np.ndarray) -> np.ndarray:
    """Safe reciprocal capacity (0 where C = 0, i.e. W[l,m] = 0)."""
    return np.where(C > 0, 1.0 / np.where(C > 0, C, 1.0), 0.0)


def norm_cap(C: np.ndarray):
    """Coefficient/RHS for a well-conditioned capacity constraint coef*L <= rhs.

    Where C > 0:  (1/C) L <= 1   (normalized, identical to L <= C).
    Where C = 0:    1  L <= 0   (pins the load to zero, W[l,m] = 0), with an
                                 O(1) coefficient that keeps the conic solve
                                 well-conditioned.
    """
    coef = np.where(C > 0, 1.0 / np.where(C > 0, C, 1.0), 1.0)
    rhs = (C > 0).astype(float)
    return coef, rhs
