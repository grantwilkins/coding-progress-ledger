"""Heuristic baseline allocators for the staged evacuation program.

Each baseline is a placement RULE: a ranked list of (destination, action)
preferences per class. One rule is evaluated two ways:
  hard=True  caps respected, remainder spills to z   (-> evacuation, fairness, H)
  hard=False move everything, z=0, measure pressure   (-> phi, overloaded resource)

Round-robin and least-loaded are gone: with a single destination they
degenerate to greedy's action choice and to replay-only. Replay-infeasible
jobs (T/rho > D) never take the replay action; both actions consume the
destination's decode-HBM residency budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from instance import ProblemInstance
from loads import loads, replay_infeasible
from stage1 import Stage1Result
from stage3 import recon_costs

R, S = 0, 1  # action indices
BASELINES = ("random", "greedy", "replay_only", "state_only")


def _coeff(inst):
    C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)
    q = np.arange(inst.T.size)
    pf = S_pfill[inst.model_idx, q]   # (Q,) prefill GPU-s per replay job (T/rho)
    ing = S_ing[inst.model_idx, q]    # (Q,) ingest bytes per state job (eta*T)
    net = np.stack([b_net_R, b_net_S])  # (2, Q) network bytes per job, by action
    return C_net, C_pfill, C_ing, C_res, net, pf, ing


def _compat(inst):
    """Per class, the destinations whose warm-instance count for its model is > 0."""
    return [np.where(inst.W[:, inst.model_idx[q]] > 0)[0] for q in range(inst.T.size)]


def _rank_action(inst, a):
    c = recon_costs(inst)[a]
    comp = _compat(inst)
    def rank(q):
        return [(l, a) for l in sorted(comp[q], key=lambda l: c[q, l])]
    return rank


def _rank_random(inst, seed):
    comp = _compat(inst)
    rng = np.random.default_rng(seed)
    def rank(q):
        cand = [(l, a) for l in comp[q] for a in (R, S)]
        rng.shuffle(cand)
        return cand
    return rank


def _rank_greedy(inst):
    c = recon_costs(inst)  # (c_R, c_S), each (Q, L); cheapest (dest, action) first
    comp = _compat(inst)
    def rank(q):
        return sorted([(l, a) for l in comp[q] for a in (R, S)], key=lambda la: c[la[1]][q, la[0]])
    return rank


def _build(inst, name, seed):
    if name == "random":        return _rank_random(inst, seed)
    if name == "greedy":        return _rank_greedy(inst)
    if name == "replay_only":   return _rank_action(inst, R)
    if name == "state_only":    return _rank_action(inst, S)
    raise ValueError(f"unknown baseline {name!r}")


def _run(inst, rank, hard):
    Q, L, M = inst.T.size, inst.lambda_bps.size, len(inst.M_names)
    C_net, C_pfill, C_ing, C_res, net, pf, ing = _coeff(inst)
    bad = replay_infeasible(inst)
    un, ur = np.zeros(L), np.zeros(L)                 # used network / residency
    up, ui = np.zeros((M, L)), np.zeros((M, L))       # used prefill / ingest
    xR, xS = np.zeros((Q, L)), np.zeros((Q, L))
    z = inst.n.astype(float).copy()
    for q in np.argsort(inst.n * inst.T):             # smallest demand first (rational greedy)
        m, left = inst.model_idx[q], float(inst.n[q])
        res_q = float(inst.eta[q] * inst.T[q])        # decode-HBM per job, either action
        for l, a in rank(q):
            if left <= 1e-12:
                break
            if a == R and bad[q]:
                continue
            r2, used2, cap2 = (pf[q], up, C_pfill) if a == R else (ing[q], ui, C_ing)
            if hard:
                sn = (C_net[l] - un[l]) / net[a][q] if net[a][q] > 0 else np.inf
                s2 = (cap2[m, l] - used2[m, l]) / r2 if r2 > 0 else np.inf
                sr = (C_res[l] - ur[l]) / res_q if res_q > 0 else np.inf
                take = max(0.0, min(left, sn, s2, sr))
                if take <= 0:
                    continue
            else:
                take = left
            un[l] += take * net[a][q]
            used2[m, l] += take * r2
            ur[l] += take * res_q
            (xR if a == R else xS)[q, l] += take
            left -= take
            if not hard:
                break                                  # soft: all on top preference
        z[q] = left
    return xR, xS, z


def allocate(inst: ProblemInstance, name: str, hard: bool = True, seed: int = 0) -> Stage1Result:
    xR, xS, z = _run(inst, _build(inst, name, seed), hard)
    return Stage1Result(x_R=xR, x_S=xS, z=z, Z_star=float(z.sum()),
                        status="baseline", objective=name)


def pressures(inst: ProblemInstance, x_R, x_S):
    """Pressure dict + phi for an allocation (mirrors stage2's load computation);
    residency utilization is reported separately, as in Stage 2."""
    C_net, C_pfill, C_ing, C_res, S_pfill, S_ing, b_net_R, b_net_S = loads(inst)
    Ln, Lp, Li = b_net_R @ x_R + b_net_S @ x_S, S_pfill @ x_R, S_ing @ x_S
    Lr = b_net_S @ (x_R + x_S)
    p = {}
    for l, ln in enumerate(inst.L_names):
        if C_net[l] > 0:
            p[f"net|{ln}"] = float(Ln[l] / C_net[l])
        for m, mn in enumerate(inst.M_names):
            if C_pfill[m, l] > 0:
                p[f"pfill|{ln}|{mn}"] = float(Lp[m, l] / C_pfill[m, l])
            if C_ing[m, l] > 0:
                p[f"ing|{ln}|{mn}"] = float(Li[m, l] / C_ing[m, l])
    return SimpleNamespace(pressures=p, phi_star=max(p.values(), default=0.0),
                           residency_utilization=float((Lr / C_res).max()))
