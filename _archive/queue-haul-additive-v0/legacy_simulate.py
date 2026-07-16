"""Serial-link reconstruction simulator for the archived additive model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch import DestFleet, Event, Plan, bind_dp, movement_columns
from impact import Impact, Movement
from instance import JobPopulation


@dataclass(frozen=True)
class SimResult:
    egress_start: np.ndarray
    egress_done: np.ndarray
    rebuild_start: np.ndarray
    rebuild_done: np.ndarray
    realized_shed: float
    reconstruction_shed: float
    reconstruction_success_count: int
    makespan: float
    analytic_lb: float
    analytic_ub: float
    realized_load: np.ndarray
    certified_load: np.ndarray
    load_cap: np.ndarray
    discipline: str
    mode: str


def _source_node(pop):
    if pop.source_node is None:
        raise ValueError("node_marginal_pd requires pop.source_node")
    node = np.asarray(pop.source_node, int)
    if len(node) != len(pop) or np.any(node < 0):
        raise ValueError("source_node must assign every job to a nonnegative node")
    return node


def _node_marginal_order(pop, pool, Yf, p1, mv, K):
    node = _source_node(pop)
    resid = np.bincount(node, weights=pop.ell, minlength=int(node.max()) + 1)
    todo, order = list(mv), []
    while todo:
        f = np.asarray(todo)
        j = f // K
        load = pop.ell[j] * Yf[f]
        gain = pool.node_power(resid[node[j]]) - pool.node_power(resid[node[j]] - load)
        k = int(np.argmax(gain / np.maximum(p1[f], 1e-300)))
        pick = int(todo.pop(k))
        order.append(pick)
        resid[node[pick // K]] -= pop.ell[pick // K] * Yf[pick]
    return np.asarray(order, int)


def _order(mv, p1, p2, dens, discipline, pop=None, pool=None, Yf=None, K=1):
    if discipline == "fifo":
        return mv
    if discipline == "lpt":
        return mv[np.argsort(-p1[mv])]
    if discipline in ("pd", "certified_pd"):
        return mv[np.argsort(-dens[mv])]
    if discipline == "node_marginal_pd":
        return _node_marginal_order(pop, pool, Yf, p1, mv, K)
    if discipline == "johnson":
        a, b = mv[p1[mv] <= p2[mv]], mv[p1[mv] > p2[mv]]
        return np.concatenate([a[np.argsort(p1[a])], b[np.argsort(-p2[b])]])
    raise ValueError(f"unknown discipline {discipline!r}")


def _stage_lb(off, p2s, w):
    return off + max(p2s.max(), p2s.sum() / w) if p2s.size and p2s.max() > 0 else 0.0


def simulate(pop: JobPopulation, pool, imp: Impact, plan: Plan, event: Event = Event(),
             move: Movement = Movement(), mode: str = "sf", discipline: str = "pd",
             fleet: DestFleet = None) -> SimResult:
    if mode not in ("sf", "cutthrough"):
        raise ValueError(f"unknown mode {mode!r}")
    if not 0 <= move.alpha_in < 1:
        raise ValueError(f"alpha_in must be in [0, 1), got {move.alpha_in}")
    n = len(pop)
    multidest = fleet is not None
    fleet = fleet or DestFleet.from_event(event, move, pool, pop)
    K, W = len(fleet), np.atleast_1d(fleet.W)
    YR, YS = (plan.Y_R, plan.Y_S) if multidest else (plan.y_R[:, None], plan.y_S[:, None])
    Y, dp = YR + YS, bind_dp(imp)
    cols = movement_columns(pop, pool, imp, fleet, move)
    p1 = (YR * cols["R"]["egress"] + YS * cols["S"]["egress"]) / move.lambda_src
    p2R = YR * cols["R"]["prefill"]
    p2S = YS * cols["S"]["ingest"] / move.mu_in
    p1f, p2Rf, p2Sf, YRf, YSf, Yf = (a.ravel() for a in (p1, p2R, p2S, YR, YS, Y))
    shedf = (dp[:, None] * Y).ravel()
    densf = shedf / np.maximum(p1f, 1e-300)
    mv = np.flatnonzero(Y.ravel() > 1e-9)

    order = _order(mv, p1f, p2Rf + p2Sf, densf, discipline, pop, pool, Yf, K)
    es, ed = np.full(n * K, np.nan), np.full(n * K, np.nan)
    t = event.tau_src
    for f in order:
        es[f], ed[f] = t, t + p1f[f]
        t = ed[f]

    rs, rd = np.full(n * K, np.nan), np.full(n * K, np.nan)
    pf = [np.full(int(W[dest]), event.tau_pre) for dest in range(K)]
    ig = [np.full(int(W[dest]), event.tau_in) for dest in range(K)]
    floor = es if mode == "cutthrough" else ed
    for f in order:
        dest, starts, done = f % K, [], ed[f]
        for srv, w, work, drag in ((pf[dest], YRf[f], p2Rf[f], move.alpha_in),
                                   (ig[dest], YSf[f], p2Sf[f], 0.0)):
            if w <= 1e-9:
                continue
            k = int(np.argmin(srv))
            st = max(floor[f], srv[k])
            if drag:
                work /= 1.0 - drag * (ig[dest] > max(st, event.tau_in)).mean()
            srv[k] = max(ed[f], st + work)
            starts.append(st)
            done = max(done, srv[k])
        rs[f], rd[f] = (min(starts) if starts else ed[f]), done

    e_ok = np.where(np.isfinite(ed), ed, np.inf) <= event.D
    r_ok = np.where(np.isfinite(rd), rd, np.inf) <= event.D
    ellY = pop.ell[:, None] * Y
    resident = (np.where(np.isfinite(rd), rd, np.inf) <= event.D).reshape(n, K)
    realized_load = (ellY * resident).sum(0)
    certified_load = ellY.sum(0)
    load_cap = np.asarray(fleet.spare, float) * pool.rho_star
    if mv.size:
        lb = max(event.tau_src + p1f[mv].sum(),
                 max(_stage_lb(event.tau_pre, p2R[:, dest], int(W[dest]))
                     for dest in range(K)),
                 max(_stage_lb(event.tau_in, p2S[:, dest], int(W[dest]))
                     for dest in range(K)))
        ub = max(event.tau_src + p1f[mv].sum(), event.tau_pre, event.tau_in) \
            + p2Rf[mv].sum() / (1 - move.alpha_in) + p2Sf[mv].sum()
        makespan = float(np.nanmax(rd))
    else:
        lb = ub = makespan = 0.0
    sq = (lambda a: a.reshape(n, K)[:, 0]) if K == 1 else (lambda a: a.reshape(n, K))
    return SimResult(sq(es), sq(ed), sq(rs), sq(rd), float(shedf[e_ok].sum()),
                     float(shedf[r_ok].sum()), int(r_ok.sum()), makespan, float(lb), float(ub),
                     realized_load, certified_load, load_cap, discipline, mode)
