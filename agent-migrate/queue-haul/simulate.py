"""Reconstruction DES (formulation.md §10.2 validation; replays a solved dispatch Plan).

Deterministic flow-shop checker: a solved Plan moves jobs over one shared egress link
(λ_src, serial) then W parallel rebuild servers — prefill (replay, T/ρ_dest) or ingest
(transfer, η·T/μ_in, W channels). No job selection; replay the plan and report where
realized shed (egress done by D) and reconstruction (rebuild done by D) fall short of
what the LP certified. Stage-2 uses BARE rates — finite servers model the contention the
LP folded into c_*'s (1+φ) factor, so feeding c_* in would double-count the queue wait.
A split job (LP fractional y_R>0 AND y_S>0) rebuilds BOTH pieces, each on its own resource.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch import Event, Plan, bind_dp
from impact import Impact, Movement
from instance import JobPopulation
from power import rho_dest


@dataclass(frozen=True)
class SimResult:
    egress_start: np.ndarray  # per-job, NaN if not moved
    egress_done: np.ndarray
    rebuild_start: np.ndarray
    rebuild_done: np.ndarray
    realized_shed: float  # Σ dp·y over jobs whose egress completes by D (grid relief)
    reconstruction_shed: float  # Σ dp·y over jobs whose rebuild completes by D
    reconstruction_success_count: int
    makespan: float
    analytic_lb: float
    analytic_ub: float
    discipline: str
    mode: str


def _order(mv, p1, p2, dens, discipline):
    """Order moving jobs at the shared egress link."""
    if discipline == "fifo":
        return mv
    if discipline == "lpt":
        return mv[np.argsort(-p1[mv])]
    if discipline == "pd":  # power-density greedy: strong (not optimal) for realized shed
        return mv[np.argsort(-dens[mv])]
    if discipline == "johnson":  # 2-machine makespan-optimal (exact only at W=1, single-action)
        a, b = mv[p1[mv] <= p2[mv]], mv[p1[mv] > p2[mv]]
        return np.concatenate([a[np.argsort(p1[a])], b[np.argsort(-p2[b])]])
    raise ValueError(f"unknown discipline {discipline!r}")


def _stage_lb(off, p2s, w):
    """P||Cmax lower bound for a parallel-server stage: off + max(longest job, total/W)."""
    return off + max(p2s.max(), p2s.sum() / w) if p2s.size and p2s.max() > 0 else 0.0


def simulate(pop: JobPopulation, pool, imp: Impact, plan: Plan, event: Event = Event(),
             move: Movement = Movement(), mode: str = "sf", discipline: str = "pd") -> SimResult:
    n = len(pop)
    mv = np.flatnonzero(plan.y > 1e-9)
    dp = bind_dp(imp)
    rho = rho_dest(pop.T, pop.mfu)
    p1 = (plan.y_R * imp.b_replay + plan.y_S * imp.b_transfer) / move.lambda_src  # egress secs
    p2R = plan.y_R * pop.T / rho  # prefill work, 0 if not replayed
    p2S = plan.y_S * imp.b_transfer / move.mu_in  # ingest work, 0 if not transferred
    # value density = dp·y/p1; for a mover p1=y·bytes/λ (≥4 B ⇒ p1≥4e-6s), so this reduces to
    # dp·λ/bytes — finite, y cancels. The floor only guards the 0/0 of non-movers (never sorted).
    dens = dp * plan.y / np.maximum(p1, 1e-300)

    order = _order(mv, p1, p2R + p2S, dens, discipline)
    es, ed = np.full(n, np.nan), np.full(n, np.nan)
    t = event.tau_src  # link available once at τ_src, then continuous
    for j in order:  # serial link; egress_done is monotone along `order`
        es[j], ed[j] = t, t + p1[j]
        t = ed[j]

    rs, rd = np.full(n, np.nan), np.full(n, np.nan)
    pf, ig = np.full(event.W, event.tau_pre), np.full(event.W, event.tau_in)
    # cut-through = optimistic earliest-overlap bound: rebuild may run from egress_start, but
    # still completes no sooner than full byte arrival ed (the outer max below). sf waits for ed.
    floor = es if mode == "cutthrough" else ed
    for j in order:  # split job rebuilds both pieces on their own resources, in egress order
        starts, done = [], ed[j]
        for srv, w, work in ((pf, plan.y_R[j], p2R[j]), (ig, plan.y_S[j], p2S[j])):
            if w <= 1e-9:
                continue
            k = int(np.argmin(srv))
            st = max(floor[j], srv[k])
            srv[k] = max(ed[j], st + work)  # outer max = cut-through byte-arrival cap
            starts.append(st); done = max(done, srv[k])
        rs[j], rd[j] = (min(starts) if starts else ed[j]), done

    shed = dp * plan.y
    e_ok = np.where(np.isfinite(ed), ed, np.inf) <= event.D
    r_ok = np.where(np.isfinite(rd), rd, np.inf) <= event.D
    if mv.size:
        lb = max(event.tau_src + p1[mv].sum(),
                 _stage_lb(event.tau_pre, p2R[mv], event.W),
                 _stage_lb(event.tau_in, p2S[mv], event.W))
        ub = event.tau_src + p1[mv].sum() + p2R[mv].sum() + p2S[mv].sum()
        makespan = float(np.nanmax(rd))
    else:
        lb = ub = makespan = 0.0
    return SimResult(es, ed, rs, rd, float(shed[e_ok].sum()), float(shed[r_ok].sum()),
                     int(r_ok.sum()), makespan, float(lb), float(ub), discipline, mode)
