"""Stage 1 of the staged evacuation program: swappable evacuation objective.

Base LP (Section 12, Stage 1 of `formulation.md`):

    sum_l (x_R[q,l] + x_S[q,l]) + z_q = n_q          (conservation)
    L_net[l]      <= C_net[l]                         (per-destination network)
    L_pfill[m,l]  <= C_pfill[m,l]                     (per-(model, destination) prefill)
    L_ing[m,l]    <= C_ing[m,l]                       (per-(model, destination) state ingest)
    L_res[l]      <= C_res[l]                         (per-destination decode-HBM residency)
    x_R[q,:] == 0 where T_q / rho_q > D               (replay compatibility mask)
    x_R, x_S, z >= 0; z <= n

Three objectives over the same base feasible set (u_q = 1 - z_q / n_q):
  throughput : min sum z                              (clear the most jobs)
  max_min    : max alpha s.t. u_q >= alpha, then min sum z s.t. u_q >= alpha*
  prop_fair  : max sum_q w_q log(eps + u_q), then min sum z s.t. U >= U* - delta

Compatibility (W[l,m] = 0) is enforced implicitly: zero capacity at (m,l) forces
the corresponding x sums to zero, since loads are nonnegative linear combinations
of nonnegative variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from loads import loads, norm_cap, replay_infeasible

EPS = 1e-3  # log-utility numerical-stability floor


@dataclass(frozen=True)
class Stage1Result:
    x_R: np.ndarray   # (Q, L) replay placement
    x_S: np.ndarray   # (Q, L) state-transfer placement
    z: np.ndarray     # (Q,)   unmoved
    Z_star: float
    status: str

    objective: str = "throughput"
    alpha_star: float | None = None
    U_star: float | None = None
    utility_epsilon: float | None = None
    utility_delta: float | None = None
    utility_weights: str | None = None


def _base(inst: ProblemInstance):
    """Variables + base constraints shared by every objective."""
    Q = inst.T.size
    L = inst.lambda_bps.size

    x_R = cp.Variable((Q, L), nonneg=True)
    x_S = cp.Variable((Q, L), nonneg=True)
    z = cp.Variable(Q, nonneg=True)

    C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)
    # Normalized capacity form (coef*L <= rhs): mathematically identical to
    # L_i <= C_i but well-conditioned for the conic (CLARABEL) prop_fair solves,
    # whose coefficients otherwise span ~1e15 (network bytes vs GPU-seconds).
    (a_net, r_net), (a_pf, r_pf), (a_in, r_in), (a_res, r_res) = map(
        norm_cap, (C_net, C_pfill, C_ing, C_res))
    cons = [
        cp.sum(x_R, axis=1) + cp.sum(x_S, axis=1) + z == inst.n,
        z <= inst.n,
        cp.multiply(a_net, b_net_R @ x_R + b_net_S @ x_S) <= r_net,
        cp.multiply(a_pf, S_pfill @ x_R) <= r_pf,
        cp.multiply(a_in, S_ing @ x_S) <= r_in,
        cp.multiply(a_res, b_net_S @ (x_R + x_S)) <= r_res,
    ]
    bad = replay_infeasible(inst)
    if bad.any():
        cons.append(x_R[bad, :] == 0)
    return x_R, x_S, z, cons


def _extract(inst, x_R, x_S, z):
    x_R_val = np.maximum(np.asarray(x_R.value, dtype=float), 0.0)
    x_S_val = np.maximum(np.asarray(x_S.value, dtype=float), 0.0)
    z_val = np.maximum(np.asarray(z.value, dtype=float), 0.0)
    # Renormalize each class to enforce exact conservation despite solver slack.
    total = x_R_val.sum(axis=1) + x_S_val.sum(axis=1) + z_val
    scale = np.where(total > 0, inst.n / total, 1.0)
    return x_R_val * scale[:, None], x_S_val * scale[:, None], z_val * scale


# Tight tolerances for the conic prop_fair solves: the log utility's gradient
# is ~1/eps near u=0, so loose feasibility lets renormalization drift below the
# U* floor and makes the Stage 2 link infeasible.
CLARABEL_OPTS = dict(tol_feas=1e-9, tol_gap_abs=1e-9, tol_gap_rel=1e-9)


def _solve(prob, solver, stage):
    opts = CLARABEL_OPTS if solver == cp.CLARABEL else {}
    prob.solve(solver=solver, **opts)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Stage 1 ({stage}) solver returned {prob.status}")


def solve_stage1(inst: ProblemInstance, objective: str = "throughput",
                 utility_weights: str = "population") -> Stage1Result:
    if objective == "throughput":
        x_R, x_S, z, cons = _base(inst)
        prob = cp.Problem(cp.Minimize(cp.sum(z)), cons)
        _solve(prob, cp.SCIPY, "throughput")
        x_R_v, x_S_v, z_v = _extract(inst, x_R, x_S, z)
        return Stage1Result(x_R=x_R_v, x_S=x_S_v, z=z_v, Z_star=float(z_v.sum()),
                            status=prob.status, objective=objective)

    if objective == "max_min":
        x_R, x_S, z, cons = _base(inst)
        alpha = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Maximize(alpha),
                          cons + [z + cp.multiply(alpha, inst.n) <= inst.n, alpha <= 1])
        _solve(prob, cp.SCIPY, "max_min/alpha")
        alpha_star = float(alpha.value)
        # Stage 1b tie-breaker: most jobs evacuated holding the floor. Tiny slack
        # keeps the exact-alpha* boundary feasible against solver tolerance.
        x_R, x_S, z, cons = _base(inst)
        prob = cp.Problem(cp.Minimize(cp.sum(z)),
                          cons + [z <= (1.0 - alpha_star) * inst.n + 1e-7 * inst.n])
        _solve(prob, cp.SCIPY, "max_min/Z")
        x_R_v, x_S_v, z_v = _extract(inst, x_R, x_S, z)
        return Stage1Result(x_R=x_R_v, x_S=x_S_v, z=z_v, Z_star=float(z_v.sum()),
                            status=prob.status, objective=objective,
                            alpha_star=alpha_star)

    if objective == "prop_fair":
        w = inst.n if utility_weights == "population" else np.ones_like(inst.n)
        x_R, x_S, z, cons = _base(inst)
        u = 1.0 - cp.multiply(1.0 / inst.n, z)
        U_expr = cp.sum(cp.multiply(w, cp.log(EPS + u)))
        prob = cp.Problem(cp.Maximize(U_expr), cons + [u <= 1])
        _solve(prob, cp.CLARABEL, "prop_fair/U")
        U_star = float(U_expr.value)
        # Robust link tolerance: covers renormalization + conic-solver drift in
        # the steep log utility while losing < 0.1% of utility.
        delta = 1e-3 * max(1.0, abs(U_star))
        # Stage 1b tie-breaker: most jobs evacuated holding the utility floor.
        x_R, x_S, z, cons = _base(inst)
        u = 1.0 - cp.multiply(1.0 / inst.n, z)
        U_expr = cp.sum(cp.multiply(w, cp.log(EPS + u)))
        prob = cp.Problem(cp.Minimize(cp.sum(z)),
                          cons + [u <= 1, U_expr >= U_star - delta])
        _solve(prob, cp.CLARABEL, "prop_fair/Z")
        x_R_v, x_S_v, z_v = _extract(inst, x_R, x_S, z)
        return Stage1Result(x_R=x_R_v, x_S=x_S_v, z=z_v, Z_star=float(z_v.sum()),
                            status=prob.status, objective=objective,
                            U_star=U_star, utility_epsilon=EPS,
                            utility_delta=delta, utility_weights=utility_weights)

    raise ValueError(f"unknown objective {objective!r}")
