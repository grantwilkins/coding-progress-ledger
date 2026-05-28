"""Section 14 of `formulation.md`: diagnostic overload problem.

Forces z = 0 (full evacuation by all jobs) and relaxes every capacity by a
normalized slack s_i >= 0 bounded by a minimax sigma:

    L_i(x) <= (1 + s_i) * C_i,   s_i <= sigma   for every i in I.

Phase 1 minimizes sigma (worst-resource overload factor for full evac).
Phase 2 fixes sigma = sigma_star and minimizes sum_i s_i to recover the
per-index breakdown (which resources are how short).

The two-phase split is the lex prescription in Section 14: Phase 1 picks
the worst-case bound; Phase 2 fairly distributes overload subject to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance


@dataclass(frozen=True)
class DiagnosticResult:
    x_R: np.ndarray
    x_S: np.ndarray
    sigma_star: float
    s_net: np.ndarray     # (L,)
    s_pfill: np.ndarray   # (M, L)
    s_ing: np.ndarray     # (M, L)
    overloads: dict[str, float]
    status_phase1: str
    status_phase2: str


def _build_loads_and_caps(inst: ProblemInstance, x_R, x_S):
    Q = inst.T.size
    M = len(inst.M_names)

    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D
    C_ing = inst.W.T * inst.mu_ing * inst.D

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
    return C_net, C_pfill, C_ing, L_net, L_pfill, L_ing


def _check_compat(inst: ProblemInstance) -> None:
    """Raise if any class has no compatible destination (no W>0 anywhere)."""
    for q in range(inst.T.size):
        m = inst.model_idx[q]
        if inst.W[:, m].sum() == 0:
            raise ValueError(
                f"class q={q} (model {inst.M_names[m]}) is structurally unroutable: "
                f"no destination has W>0 for this model"
            )


def solve_diagnostic(inst: ProblemInstance) -> DiagnosticResult:
    _check_compat(inst)

    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    s_net = cp.Variable(L, nonneg=True)
    s_pfill = cp.Variable((M, L), nonneg=True)
    s_ing = cp.Variable((M, L), nonneg=True)
    sigma = cp.Variable(nonneg=True)

    C_net, C_pfill, C_ing, L_net, L_pfill, L_ing = _build_loads_and_caps(inst, x_R, x_S)

    mask_pfill = (C_pfill > 0)
    mask_ing = (C_ing > 0)
    inv_C_net = 1.0 / C_net
    inv_C_pfill = np.where(mask_pfill, 1.0 / np.where(mask_pfill, C_pfill, 1.0), 0.0)
    inv_C_ing = np.where(mask_ing, 1.0 / np.where(mask_ing, C_ing, 1.0), 0.0)

    base = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) == inst.n,  # forced full evac (z=0)
        cp.multiply(inv_C_net, L_net) <= 1 + s_net,
        cp.multiply(inv_C_pfill, L_pfill) <= 1 + s_pfill,
        cp.multiply(inv_C_ing, L_ing) <= 1 + s_ing,
        # Compatibility: zero-capacity (l,m) cells force load to zero.
        cp.multiply((~mask_pfill).astype(float), L_pfill) == 0,
        cp.multiply((~mask_ing).astype(float), L_ing) == 0,
        s_net <= sigma,
        s_pfill <= sigma,
        s_ing <= sigma,
    ]

    # Phase 1: min sigma.
    prob1 = cp.Problem(cp.Minimize(sigma), base)
    prob1.solve(solver=cp.SCIPY)
    if prob1.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Diagnostic Phase 1 solver returned {prob1.status}")
    sigma_star = float(sigma.value)
    status_phase1 = prob1.status

    # Phase 2: fix sigma = sigma_star, min sum_i s_i.
    sigma2 = cp.Constant(sigma_star)
    base2 = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) == inst.n,
        cp.multiply(inv_C_net, L_net) <= 1 + s_net,
        cp.multiply(inv_C_pfill, L_pfill) <= 1 + s_pfill,
        cp.multiply(inv_C_ing, L_ing) <= 1 + s_ing,
        cp.multiply((~mask_pfill).astype(float), L_pfill) == 0,
        cp.multiply((~mask_ing).astype(float), L_ing) == 0,
        s_net <= sigma2,
        s_pfill <= sigma2,
        s_ing <= sigma2,
    ]
    obj2 = cp.Minimize(cp.sum(s_net) + cp.sum(s_pfill) + cp.sum(s_ing))
    prob2 = cp.Problem(obj2, base2)
    prob2.solve(solver=cp.SCIPY)
    if prob2.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Diagnostic Phase 2 solver returned {prob2.status}")
    status_phase2 = prob2.status

    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    s_net_val = np.maximum(np.asarray(s_net.value, dtype=float), 0.0)
    s_pfill_val = np.maximum(np.asarray(s_pfill.value, dtype=float), 0.0)
    s_ing_val = np.maximum(np.asarray(s_ing.value, dtype=float), 0.0)

    # Zero out slacks where capacity is zero (no resource to overload).
    s_pfill_val = np.where(C_pfill > 0, s_pfill_val, 0.0)
    s_ing_val = np.where(C_ing > 0, s_ing_val, 0.0)

    overloads: dict[str, float] = {}
    for l, lname in enumerate(inst.L_names):
        overloads[f"net|{lname}"] = float(s_net_val[l])
        for m, mname in enumerate(inst.M_names):
            if C_pfill[m, l] > 0:
                overloads[f"pfill|{lname}|{mname}"] = float(s_pfill_val[m, l])
            if C_ing[m, l] > 0:
                overloads[f"ing|{lname}|{mname}"] = float(s_ing_val[m, l])

    return DiagnosticResult(
        x_R=x_R_val, x_S=x_S_val,
        sigma_star=sigma_star,
        s_net=s_net_val, s_pfill=s_pfill_val, s_ing=s_ing_val,
        overloads=overloads,
        status_phase1=status_phase1, status_phase2=status_phase2,
    )
