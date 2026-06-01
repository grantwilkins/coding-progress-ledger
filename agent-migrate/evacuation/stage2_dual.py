"""Stage 2 first-order solvers via per-class dual decomposition (Section 16).

Dualize the pressure ceilings p_i(x) <= phi with pi_i >= 0; the phi-stationarity
condition forces sum_i pi_i = 1, so pi lives on the unit simplex of size |I|.
Dualize the Stage 1 link sum_q z_q = Z* with scalar mu. The Lagrangian inner
min then separates per class: each class q assigns its n_q jobs to the cheapest
option among {(l, R), (l, S)} for compatible l, or "stay" at cost mu.

  D(pi, mu) = sum_q n_q * min(min_(l,a) c_a(q, l; pi), mu) - mu Z*

Strong duality (LP) gives D* = phi*. dD/dpi_i = p_i(x*(pi, mu)) (realized
normalized load); dD/dmu = z_total - Z*.

Methods:
  subgradient    - projected subgradient ascent on (pi, mu)
  mirror_descent - exponentiated gradient on pi (simplex-natural); subgradient on mu
  admm           - sharing-problem ADMM (Boyd 7.3) on per-class load contributions;
                   per-class update is a small simplex QP, aggregate update is the
                   prox of max(.) solved by sorting. Specialized to Z* = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from instance import ProblemInstance
from stage1 import Stage1Result


@dataclass(frozen=True)
class Trajectory:
    iters: np.ndarray
    dual: np.ndarray      # D(pi_k, mu_k); lower bound on phi*
    primal: np.ndarray    # max pressure at averaged primal; upper bound on phi*
    prices: np.ndarray | None = None  # (iters, |I|) per-iteration pi (subgradient only)
    I_meta: list | None = None        # pressure-index metadata for `prices` columns


def build_dual_structure(inst: ProblemInstance):
    """Pack the LP coefficients into a single tensor.

    A[i, q, k] = (load coefficient of option k for class q at pressure i) / C_i.
    Options 0..L-1 are replay at destination k; L..2L-1 are state-transfer at
    destination (k - L). Infeasible (l, m(q)) pairs are masked in `feasible`.
    """
    Q = inst.T.size
    L = inst.lambda_bps.size
    M = len(inst.M_names)

    C_net = inst.lambda_bps * inst.D
    C_pfill = inst.W.T * inst.D                 # (M, L)
    C_ing = inst.W.T * inst.mu_ing * inst.D     # (M, L)

    I_meta = [("net", l, None) for l in range(L)]
    pfill_idx, ing_idx = {}, {}
    for l in range(L):
        for m in range(M):
            if inst.W[l, m] > 0:
                pfill_idx[(l, m)] = len(I_meta); I_meta.append(("pfill", l, m))
                ing_idx[(l, m)] = len(I_meta); I_meta.append(("ing", l, m))
    C = np.array([
        C_net[l] if k == "net" else (C_pfill[m, l] if k == "pfill" else C_ing[m, l])
        for (k, l, m) in I_meta
    ], dtype=float)

    K = 2 * L
    A = np.zeros((len(I_meta), Q, K))
    feasible = np.ones((Q, K), dtype=bool)

    beta_T = inst.beta * inst.T
    eta_T = inst.eta * inst.T
    tau = inst.T / inst.rho
    m_q = inst.model_idx

    for l in range(L):
        A[l, :, l] = beta_T / C_net[l]
        A[l, :, L + l] = eta_T / C_net[l]
        for m in range(M):
            mask = (m_q == m)
            if inst.W[l, m] == 0:
                feasible[mask, l] = False
                feasible[mask, L + l] = False
                continue
            A[pfill_idx[(l, m)], mask, l] = tau[mask] / C_pfill[m, l]
            A[ing_idx[(l, m)], mask, L + l] = eta_T[mask] / C_ing[m, l]

    return A, C, I_meta, feasible


def per_class_assign(A: np.ndarray, n: np.ndarray, pi: np.ndarray,
                     mu: float | None, feasible: np.ndarray):
    """Inner per-class min at prices (pi, mu).

    Returns (assign (Q,) with -1 = stay, p (|I|,) realized loads, z_total, cost_sum).
    """
    cost = np.einsum("i,iqk->qk", pi, A)
    cost = np.where(feasible, cost, np.inf)
    k_star = np.argmin(cost, axis=1)
    c_min = np.take_along_axis(cost, k_star[:, None], axis=1).ravel()

    if mu is None:
        assign = k_star
        c_chosen = c_min
        z_total = 0.0
    else:
        stay = c_min > mu
        assign = np.where(stay, -1, k_star)
        c_chosen = np.where(stay, mu, c_min)
        z_total = float(n[stay].sum())

    cost_sum = float(n @ c_chosen)

    active = assign >= 0
    if active.any():
        q_idx = np.where(active)[0]
        p = A[:, q_idx, k_star[q_idx]] @ n[q_idx]
    else:
        p = np.zeros(A.shape[0])
    return assign, p, z_total, cost_sum


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto {x >= 0, sum x = 1}. Duchi et al. 2008."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_arr = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0]
    rho = int(rho_arr[-1]) if rho_arr.size else 0
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def subgradient(inst: ProblemInstance, stage1: Stage1Result, phi_star: float,
                max_iter: int = 500) -> Trajectory:
    """Projected subgradient ascent on (pi, mu) with the Polyak step

        alpha_k = (phi* - D_k) / ||g_k||_2^2

    using the known LP optimum phi* as the target dual value.
    """
    A, C, I_meta, feasible = build_dual_structure(inst)
    n = inst.n
    Z_star = float(stage1.Z_star)
    use_mu = Z_star > 0

    pi = np.full(A.shape[0], 1.0 / A.shape[0])
    mu = 0.0 if use_mu else None

    # Ergodic primal recovery: weight iterate t by its step size alpha_t (Nedic-Ozdaglar).
    p_num = np.zeros(A.shape[0])
    alpha_sum = 0.0
    D_traj, phi_traj, pi_traj = [], [], []
    for k in range(1, max_iter + 1):
        pi_traj.append(pi.copy())
        _, p, z_tot, cost_sum = per_class_assign(A, n, pi, mu, feasible)
        D = cost_sum - (mu * Z_star if use_mu else 0.0)
        gap = max(phi_star - D, 0.0)
        grad_sq = float(p @ p) + ((z_tot - Z_star) ** 2 if use_mu else 0.0)
        if grad_sq < 1e-20:
            alpha = 0.0
        else:
            alpha = gap / grad_sq
        p_num += alpha * p
        alpha_sum += alpha
        p_avg = p_num / alpha_sum if alpha_sum > 0 else p

        D_traj.append(D); phi_traj.append(float(p_avg.max()))
        if gap < 1e-15 or alpha == 0.0:
            break

        pi = project_simplex(pi + alpha * p)
        if use_mu:
            mu = mu + alpha * (z_tot - Z_star)
    iters = np.arange(1, len(D_traj) + 1)
    return Trajectory(iters, np.array(D_traj), np.array(phi_traj),
                      prices=np.array(pi_traj), I_meta=I_meta)


def mirror_descent(inst: ProblemInstance, stage1: Stage1Result, phi_star: float,
                   max_iter: int = 500) -> Trajectory:
    """Entropic mirror descent on pi (simplex-natural) with Polyak step

        alpha_k = (phi* - D_k) / ||g_k||_inf^2

    matched to the dual norm of the KL mirror map (Beck-Teboulle 2003).
    """
    A, C, I_meta, feasible = build_dual_structure(inst)
    n = inst.n
    Z_star = float(stage1.Z_star)
    use_mu = Z_star > 0

    pi = np.full(A.shape[0], 1.0 / A.shape[0])
    mu = 0.0 if use_mu else None

    p_num = np.zeros(A.shape[0])
    alpha_sum = 0.0
    D_traj, phi_traj = [], []
    for k in range(1, max_iter + 1):
        _, p, z_tot, cost_sum = per_class_assign(A, n, pi, mu, feasible)
        D = cost_sum - (mu * Z_star if use_mu else 0.0)
        gap = max(phi_star - D, 0.0)
        g_inf = float(np.abs(p).max())
        if use_mu:
            g_inf = max(g_inf, abs(z_tot - Z_star))
        alpha = 0.0 if g_inf < 1e-20 else gap / (g_inf ** 2)
        p_num += alpha * p
        alpha_sum += alpha
        p_avg = p_num / alpha_sum if alpha_sum > 0 else p

        D_traj.append(D); phi_traj.append(float(p_avg.max()))
        if gap < 1e-15 or alpha == 0.0:
            break

        log_pi = np.log(pi + 1e-300) + alpha * p
        log_pi -= log_pi.max()
        pi_new = np.exp(log_pi)
        pi = pi_new / pi_new.sum()
        if use_mu:
            mu = mu + alpha * (z_tot - Z_star)
    iters = np.arange(1, len(D_traj) + 1)
    return Trajectory(iters, np.array(D_traj), np.array(phi_traj))


def prox_max(v: np.ndarray, lam: float) -> np.ndarray:
    """argmin_s max_i s_i + (1/(2 lam)) ||s - v||^2.

    Sort v in descending order. With active set of the top k components all
    pinned to a common phi: phi = (sum_top_k - lam) / k, and we want the
    smallest k such that phi >= v_(k+1). Then s_i = min(v_i, phi).
    """
    n = v.size
    order = np.argsort(v)[::-1]
    vs = v[order]
    csum = np.cumsum(vs)
    for k in range(1, n + 1):
        phi = (csum[k - 1] - lam) / k
        if k == n or phi >= vs[k]:
            return np.minimum(v, phi)
    raise RuntimeError("prox_max: no valid active set")


def admm(inst: ProblemInstance, stage1: Stage1Result,
         max_iter: int = 300, rho: float = 5.0) -> Trajectory:
    """Sharing-problem ADMM on per-class normalized load contributions.

    Variables x_q in R^|I| = n_q A_q y_q (each class's contribution to p).
    Aggregate consensus z := bar(x); s = Q * z = sum_q x_q (aggregate pressure).
    f_q(x_q) = I_{S_q}, where S_q = {n_q A_q y_q : y_q in simplex, y_q[k] = 0 if !feas}.
    g(s) = max_i s_i.

    Iterates (Boyd 7.3):
      x_q  <- argmin_{x_q in S_q} || x_q - (x_q - bar(x) + bar(z) - u) ||^2  (per class)
      bar(z) <- prox_{(1/(rho Q)) g · Q}( bar(x) + u )
              equivalently s <- prox_{(Q/rho) max}( sum_q x_q + Q u )
      u    <- u + bar(x) - bar(z)

    Requires Z* = 0 (no stay option).
    """
    if stage1.Z_star > 1e-6:
        raise NotImplementedError("ADMM specialized to Z* = 0")

    A, C, I_meta, feasible = build_dual_structure(inst)
    n_I, Q, K = A.shape
    n = inst.n

    B = A * n[None, :, None]                    # (|I|, Q, K)

    # Per-class simplex-QP via CVXPY with Parameters (K is small).
    # Use sum_squares form to stay DCP without declaring G as PSD.
    y_var = cp.Variable(K, nonneg=True)
    B_par = cp.Parameter((n_I, K))
    t_par = cp.Parameter(n_I)
    feas_par = cp.Parameter(K)
    qp = cp.Problem(
        cp.Minimize(cp.sum_squares(B_par @ y_var - t_par)),
        [cp.sum(y_var) == 1, y_var <= feas_par],
    )

    # init y uniform over feasible
    y = feasible.astype(float)
    y /= y.sum(axis=1, keepdims=True)
    x = np.einsum("iqk,qk->qi", B, y)           # (Q, |I|)
    s = x.sum(axis=0)                            # aggregate
    u = np.zeros(n_I)                            # scaled dual

    D_traj, phi_traj = [], []
    for it in range(1, max_iter + 1):
        x_bar = x.mean(axis=0)
        target = x - x_bar[None, :] + (s / Q)[None, :] - u[None, :]   # (Q, |I|)

        for q in range(Q):
            B_par.value = B[:, q, :]
            t_par.value = target[q]
            feas_par.value = feasible[q].astype(float)
            qp.solve(solver=cp.CLARABEL, warm_start=True)
            y[q] = np.clip(y_var.value, 0.0, 1.0)
            x[q] = B[:, q, :] @ y[q]

        s = prox_max(x.sum(axis=0) + Q * u, lam=Q / rho)
        u = u + x.mean(axis=0) - s / Q

        # Primal gap uses the actual aggregated primal sum_q x_q (always feasible
        # under per-class simplex); the auxiliary s is just the prox clip target.
        # Dual price recovery: rho * u is the unscaled dual; project onto the
        # simplex to evaluate D(pi).
        pi_eval = project_simplex(rho * u)
        _, _, _, cost_sum = per_class_assign(A, n, pi_eval, None, feasible)
        D_traj.append(cost_sum)
        phi_traj.append(float(x.sum(axis=0).max()))

    return Trajectory(np.arange(1, max_iter + 1), np.array(D_traj), np.array(phi_traj))


def _project_perclass(Y: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Project each row of Y onto its per-class simplex over feasible options."""
    out = np.zeros_like(Y)
    for q in range(Y.shape[0]):
        cols = np.nonzero(feasible[q])[0]
        out[q, cols] = project_simplex(Y[q, cols])
    return out


def pdhg(inst: ProblemInstance, stage1: Stage1Result, phi_star: float | None = None,
         max_iter: int = 800) -> Trajectory:
    """Chambolle-Pock on the min-peak-pressure saddle (Section 16).

    Stage 2 is the bilinear saddle  min_y max_{pi in simplex} <pi, B y>, where
    y_q is class q's distribution over (dest, action) options and B[i,q,k] =
    n_q * A[i,q,k] maps it to normalized pressure p_i. PDHG alternates a dual
    ascent (project pi onto the price simplex) with a primal descent (project
    each y_q onto its option simplex) plus over-relaxation, with steps
    tau = sigma = 1/||B||. The ergodic primal gives the converging upper bound
    on phi*. Specialized to Z* = 0 (no stay option), like ADMM.
    """
    if stage1.Z_star > 1e-6:
        raise NotImplementedError("PDHG specialized to Z* = 0")
    A, C, I_meta, feasible = build_dual_structure(inst)
    n = inst.n
    nI, Q, K = A.shape
    B = A * n[None, :, None]
    norm = np.linalg.norm((B * feasible[None, :, :]).reshape(nI, Q * K), 2)
    tau = sigma = 1.0 / norm if norm > 0 else 1.0

    y = feasible.astype(float)
    y /= y.sum(axis=1, keepdims=True)
    pi = np.full(nI, 1.0 / nI)
    ybar = y.copy()
    y_sum = np.zeros_like(y)

    D_traj, primal_traj = [], []
    for k in range(1, max_iter + 1):
        pi = project_simplex(pi + sigma * np.einsum("iqk,qk->i", B, ybar))
        y_prev = y
        grad = np.einsum("i,iqk->qk", pi, B)
        y = _project_perclass(y - tau * grad, feasible)
        ybar = 2.0 * y - y_prev

        y_sum += y
        primal = float(np.einsum("iqk,qk->i", B, y_sum / k).max())
        _, _, _, D = per_class_assign(A, n, pi, None, feasible)
        D_traj.append(D)
        primal_traj.append(primal)

    return Trajectory(np.arange(1, max_iter + 1), np.array(D_traj), np.array(primal_traj))


def bundle(inst: ProblemInstance, stage1: Stage1Result, phi_star: float | None = None,
           max_iter: int = 120, prox_c: float = 1.0, m_serious: float = 0.1,
           max_planes: int = 40) -> Trajectory:
    """Proximal bundle method on the concave dual D(pi) (Section 16).

    D(pi) = sum_q n_q min_k <pi, A[:,q,k]> is piecewise-linear concave with
    supergradient p (the realized loads at the per-class argmin). Each step
    maximizes the cutting-plane model under a proximal term centered at the
    incumbent:  max_{pi in simplex, v}  v - (1/2t)||pi - center||^2  s.t.
    v <= D_j + g_j.(pi - pi_j).  The plane multipliers give a convex primal whose
    peak pressure is the converging upper bound on phi*. prox weight scales with
    1/phi* since D ~ O(phi*). Z* = 0 only.
    """
    if stage1.Z_star > 1e-6:
        raise NotImplementedError("bundle specialized to Z* = 0")
    A, C, I_meta, feasible = build_dual_structure(inst)
    n = inst.n
    nI = A.shape[0]
    prox_t = prox_c / max(phi_star, 1e-12) if phi_star else 50.0

    pi = np.full(nI, 1.0 / nI)
    _, p, _, D = per_class_assign(A, n, pi, None, feasible)
    center, D_center = pi.copy(), D
    planes = [(pi.copy(), D, p.copy())]

    D_traj, primal_traj = [], []
    for k in range(1, max_iter + 1):
        var = cp.Variable(nI, nonneg=True)
        v = cp.Variable()
        plane_cons = [v <= Dj + gj @ (var - pj) for (pj, Dj, gj) in planes]
        prob = cp.Problem(cp.Maximize(v - (1.0 / (2 * prox_t)) * cp.sum_squares(var - center)),
                          [cp.sum(var) == 1] + plane_cons)
        prob.solve(solver=cp.CLARABEL)

        pi_new = np.maximum(np.asarray(var.value, dtype=float), 0.0)
        pi_new /= pi_new.sum()
        lam = np.array([max(float(c.dual_value), 0.0) for c in plane_cons])
        G = np.array([g for (_, _, g) in planes])
        primal = float(((lam / lam.sum()) @ G).max() if lam.sum() > 1e-12 else p.max())

        _, p, _, D_new = per_class_assign(A, n, pi_new, None, feasible)
        if D_new >= D_center + m_serious * (float(v.value) - D_center):
            center, D_center = pi_new.copy(), D_new
        planes.append((pi_new.copy(), D_new, p.copy()))
        if len(planes) > max_planes:
            planes = planes[-max_planes:]
        D_traj.append(D_center)
        primal_traj.append(primal)

    return Trajectory(np.arange(1, max_iter + 1), np.array(D_traj), np.array(primal_traj))
