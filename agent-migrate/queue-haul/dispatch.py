"""Dispatch solver (formulation.md §Dispatch program; §5 pools & event).

Pick which jobs to move and how (replay vs KV-transfer) to hit a grid shed target
S* by deadline D at least total downtime, subject to source egress, destination
rebuild, and destination headroom. Each job gets two variables — a replay var y_R
and a transfer var y_S with y_R+y_S≤1 — so the action choice needs no separate
indicator. The same program is solved as a fractional LP (y∈[0,1]) and an integer
MILP (y∈{0,1}); the gap is the granularity cost. Two solves, not a branch: if the
primary is infeasible, a second solve maximizes shed and reports the shortfall.
A resource-blind greedy (sort by cost per watt) is the T5 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from impact import Impact, Movement
from instance import JobPopulation
from power import PoolPower, rho_dest


@dataclass(frozen=True)
class Event:
    """§5 pools & event + §6 timing the solver reads. Center defaults, all swept."""

    D: float = 300.0  # deadline (s)
    W: int = 8  # rebuild-capable nodes: gate prefill compute AND transfer ingest
    dest_nodes: int = 32  # whole destination pool: gates load/held headroom (≠ W)
    spare_frac: float = 0.40  # spare per dest node, below its knee
    tau_src: float = 2.0  # egress connection ramp
    tau_pre: float = 5.0  # prefill batch-form
    tau_in: float = 3.0  # ingest pipeline-fill
    pinned: tuple = ()  # class names forced to y=0 (service floor)

    def l_dest(self, pool: PoolPower) -> float:
        return self.spare_frac * self.dest_nodes * pool.rho_star

    def s_dest(self, pool: PoolPower) -> float:
        return self.spare_frac * self.dest_nodes * pool.s_node


@dataclass(frozen=True)
class Plan:
    y_R: np.ndarray  # per-job replay fraction
    y_S: np.ndarray  # per-job transfer fraction
    shed_guaranteed: float  # Σ y·dp_bind — the committed floor (=S* for a feasible LP)
    shed_expected: float  # Σ y·dp_expected — optimistic upside, reported not bound
    cost: float  # Σ y_R·c_replay + y_S·c_transfer (total downtime, s)
    feasible: bool
    shortfall: float  # max(0, S* − shed_guaranteed), >0 only when infeasible
    regime: str
    method: str  # 'lp' | 'milp' | 'greedy'

    @property
    def y(self) -> np.ndarray:
        return self.y_R + self.y_S

    @property
    def action(self) -> np.ndarray:
        moved = self.y > 1e-9
        return np.where(~moved, "", np.where(self.y_R >= self.y_S, "R", "S"))


def bind_dp(imp: Impact) -> np.ndarray:
    """Committed floor for the ≥S* constraint: dp_memory when memory binds (the
    node holds idle sessions, sits at idle, so μ is realized), else dp_guaranteed
    (plateau slope, realized even if no node drains). NEVER dp_expected — that
    optimistic upside is reported on the plan, never bound, or we'd promise the
    grid watts contingent on the autoscaler draining nodes. Pre-move regime
    (static snapshot); tracking the regime as the pool drains is the deferred
    receding-horizon re-solve, not an oversight here."""
    return imp.dp_memory if imp.regime == "memory" else imp.dp_guaranteed


def _plan(x, n, dp, imp, method, regime, s_star, feasible) -> Plan:
    y_R, y_S = x[:n], x[n:]
    y = y_R + y_S
    shed_g = float(dp @ y)
    return Plan(y_R, y_S, shed_g, float(imp.dp_expected @ y),
                float(imp.c_replay @ y_R + imp.c_transfer @ y_S), feasible,
                0.0 if feasible else max(0.0, s_star - shed_g), regime, method)


def _resource_constraints(pop, pool, imp, event, move):
    """Pairing (y_R+y_S≤1) and the five movement constraints (everything but shed)."""
    n = len(pop)
    z = np.zeros(n)
    reb = pop.T / rho_dest(pop.T, pop.mfu)  # prefill node-seconds (replay)
    pair = LinearConstraint(np.hstack([np.eye(n), np.eye(n)]), 0, 1)
    rows = np.array([
        np.concatenate([imp.b_replay, imp.b_transfer]),  # egress bytes
        np.concatenate([reb, z]),  # prefill node-seconds (replay only)
        np.concatenate([z, imp.b_transfer]),  # ingest bytes (transfer only)
        np.concatenate([pop.ell, pop.ell]),  # destination load
        np.concatenate([np.ones(n), np.ones(n)]),  # destination held sessions
    ])
    ub = np.array([
        move.lambda_src * (event.D - event.tau_src),
        event.W * (event.D - event.tau_pre),
        event.W * move.mu_in * (event.D - event.tau_in),
        event.l_dest(pool),
        event.s_dest(pool),
    ])
    hi = np.ones(2 * n)
    if event.pinned:
        pin = np.isin(pop.job_type, event.pinned)
        hi[np.concatenate([pin, pin])] = 0.0
    return n, pair, LinearConstraint(rows, -np.inf, ub), Bounds(np.zeros(2 * n), hi)


def solve(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
          event: Event = Event(), move: Movement = Movement(), integer: bool = False) -> Plan:
    """Primary solve; on infeasibility re-solve to max shed and report shortfall."""
    n, pair, res, bounds = _resource_constraints(pop, pool, imp, event, move)
    dp = bind_dp(imp)
    dp2 = np.concatenate([dp, dp])
    intg = np.full(2 * n, int(integer))
    method = "milp" if integer else "lp"

    shed = LinearConstraint(dp2, s_star, np.inf)
    obj = np.concatenate([imp.c_replay, imp.c_transfer])
    r = milp(c=obj, constraints=[pair, res, shed], integrality=intg, bounds=bounds)
    if r.success:
        return _plan(r.x, n, dp, imp, method, imp.regime, s_star, feasible=True)

    r = milp(c=-dp2, constraints=[pair, res], integrality=intg, bounds=bounds)
    if not r.success:
        raise RuntimeError(f"max-shed re-solve failed: status={r.status} ({r.message})")
    return _plan(r.x, n, dp, imp, method, imp.regime, s_star, feasible=False)


def greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
           event: Event = Event(), move: Movement = Movement()) -> Plan:
    """Decentralized first-fit baseline: each job self-selects its cheaper action
    and the best-deal jobs (lowest downtime per watt) move first — but every move
    draws down the SHARED movement budgets the deadline window allows (egress
    bytes, rebuild node-seconds and ingest bytes, destination load and held), so it
    cannot ship more than the links carry. A job whose cheaper action no longer
    fits falls back to its other action, else waits. One myopic pass in priority
    order, no global repacking — that repacking is exactly what the LP buys."""
    n = len(pop)
    dp = bind_dp(imp)
    reb = pop.T / rho_dest(pop.T, pop.mfu)
    budget = {  # same RHS as the solve() constraints; consumed as jobs are accepted
        "egress": move.lambda_src * (event.D - event.tau_src),
        "prefill": event.W * (event.D - event.tau_pre),
        "ingest": event.W * move.mu_in * (event.D - event.tau_in),
        "load": event.l_dest(pool),
        "held": event.s_dest(pool),
    }
    draw = {  # per-job resource a unit move consumes (same coefficients as the LP rows)
        "R": [("egress", imp.b_replay), ("prefill", reb), ("load", pop.ell), ("held", np.ones(n))],
        "S": [("egress", imp.b_transfer), ("ingest", imp.b_transfer), ("load", pop.ell), ("held", np.ones(n))],
    }
    cheaper = np.where(imp.c_replay <= imp.c_transfer, "R", "S")
    movable = dp > 0
    if event.pinned:
        movable &= ~np.isin(pop.job_type, event.pinned)
    ratio = np.where(movable, np.minimum(imp.c_replay, imp.c_transfer) / np.where(movable, dp, 1.0), np.inf)

    yR, yS = np.zeros(n), np.zeros(n)
    cum = 0.0
    for j in np.argsort(ratio):
        if not movable[j] or cum >= s_star:
            break
        for a in (cheaper[j], "RS"[cheaper[j] == "R"]):  # try cheaper action, then fall back
            fit = min([budget[r] / col[j] for r, col in draw[a] if col[j] > 0] + [1.0])
            f = min(1.0, fit, (s_star - cum) / dp[j])
            if f > 1e-12:
                for r, col in draw[a]:
                    budget[r] -= f * col[j]
                (yR if a == "R" else yS)[j] = f
                cum += f * dp[j]
                break
    return _plan(np.concatenate([yR, yS]), n, dp, imp, "greedy", imp.regime, s_star,
                 feasible=cum >= s_star - 1e-6 * max(s_star, 1.0))
