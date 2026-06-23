"""Dispatch solver (formulation.md §Dispatch program; §5 pools & event).

Pick which jobs to move and how (replay vs KV-transfer) to hit a grid shed target
S* by deadline D at least total downtime, subject to source egress, destination
rebuild, and destination headroom. Each job gets two variables — a replay var y_R
and a transfer var y_S with y_R+y_S≤1 — so the action choice needs no separate
indicator. The same program is solved as a fractional LP (y∈[0,1]) and an integer
MILP (y∈{0,1}); the gap is the granularity cost. Two solves, not a branch: if the
primary is infeasible, a second solve maximizes shed and reports the shortfall.
A budget-respecting greedy (sort by cost per watt) is the T5 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from impact import Impact, Movement, move_costs
from instance import JobPopulation
from power import PoolPower, rho_replay


@dataclass(frozen=True)
class Event:
    """§5 pools & event + §6 timing the solver reads. Center defaults, all swept."""

    D: float = 300.0  # deadline (s)
    W: int = 8  # dedicated rebuild servers; not the serving spare counted by dest_nodes
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
class DestFleet:
    """§4 destination index: K sites the source sheds to, coupled only by the shared
    source uplink. Each field a length-K parallel array. spare_ℓ is a spare node-count →
    load headroom L̄_ℓ = spare_ℓ·ρ*, held headroom S̄_ℓ = spare_ℓ·s_node."""

    W: np.ndarray  # rebuild servers per dest (prefill compute + ingest channels)
    spare: np.ndarray  # spare node-count per dest
    mfu: np.ndarray  # destination MFU → ρ_ℓ(T); decoupled from the source's pop.mfu
    prefill_util: np.ndarray  # destination prefill load → φ_pre,ℓ

    def __len__(self) -> int:
        return len(np.atleast_1d(self.W))

    @classmethod
    def from_event(
        cls, event: "Event", move: Movement, pool: PoolPower, pop: JobPopulation
    ) -> "DestFleet":
        """The K=1 fleet that reproduces the single-dest Event/Movement headroom exactly."""
        return cls(
            np.array([event.W]),
            np.array([event.spare_frac * event.dest_nodes]),
            np.array([pop.mfu]),
            np.array([move.dest_prefill_util]),
        )


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
    Y_R: np.ndarray = (
        None  # (n,K) replay routing by destination; None for greedy/random
    )
    Y_S: np.ndarray = (
        None  # (n,K) transfer routing; y_R/y_S are its Σ-over-ℓ aggregates
    )
    theta_egress: float = None  # shared-uplink dual (max-shed LP); None for milp/greedy
    theta_admit: np.ndarray = None  # per-ℓ admission dual (load+held)

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
    return Plan(
        y_R,
        y_S,
        shed_g,
        float(imp.dp_expected @ y),
        float(imp.c_replay @ y_R + imp.c_transfer @ y_S),
        feasible,
        0.0 if feasible else max(0.0, s_star - shed_g),
        regime,
        method,
    )


def _plan2(YR, YS, dp, imp, cost, method, s_star, feasible, duals) -> Plan:
    """Build a Plan from the (n,K) routing: y_R/y_S are the Σ-over-ℓ aggregates."""
    y_R, y_S = YR.sum(1), YS.sum(1)
    y = y_R + y_S
    shed_g = float(dp @ y)
    te, ta = duals if duals else (None, None)
    return Plan(
        y_R,
        y_S,
        shed_g,
        float(imp.dp_expected @ y),
        cost,
        feasible,
        0.0 if feasible else max(0.0, s_star - shed_g),
        imp.regime,
        method,
        YR,
        YS,
        te,
        ta,
    )


def _build(pop, pool, imp, fleet, event, move, integer):
    """Variables Y_R, Y_S (n,K) + pairing, the ONE shared egress row, and the per-ℓ movement
    blocks (everything but shed). Returns the dual-carrying constraint handles too."""
    if np.any(pop.ell > pool.rho_star): raise ValueError("job ell exceeds rho_star; split the job or lower its offered load")
    n, K = len(pop), len(fleet)
    kw = {"boolean": True} if integer else {"nonneg": True}
    YR, YS = cp.Variable((n, K), **kw), cp.Variable((n, K), **kw)
    Y = YR + YS
    reb = move_costs(pop, fleet, move)[2]  # prefill node-seconds at each dest's ρ_ℓ
    bR, bS = imp.b_replay[:, None], imp.b_transfer[:, None]
    W, spare = np.asarray(fleet.W), np.asarray(fleet.spare)
    egress = cp.sum(cp.multiply(bR, YR) + cp.multiply(bS, YS)) <= move.lambda_src * (
        event.D - event.tau_src
    )
    # TODO(dest-load): recompute ell per destination when fleet hardware/precision differs.
    load = cp.sum(cp.multiply(pop.ell[:, None], Y), axis=0) <= spare * pool.rho_star
    held_w = (pop.T / pool.mean_context_tokens) * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)
    held = (
        cp.sum(cp.multiply(held_w[:, None], Y), axis=0)
        <= spare * pool.s_node
    )
    cons = [
        cp.sum(Y, axis=1)
        <= 1,  # pairing: each job moves at most once, across all destinations
        egress,  # the single shared source uplink — the entire multi-dest coupling
        # TODO(background-util): if W is shared with serving, reduce this RHS by destination load.
        cp.sum(cp.multiply(reb, YR), axis=0)
        <= W * (event.D - event.tau_pre),  # per-ℓ prefill
        cp.sum(cp.multiply(bS, YS), axis=0)
        <= W * move.mu_in * (event.D - event.tau_in),  # per-ℓ ingest
        load,
        held,  # per-ℓ destination headroom
    ]
    if event.pinned:
        pin = np.isin(pop.job_type, event.pinned)
        if pin.any():
            cons += [YR[pin] == 0, YS[pin] == 0]
    return YR, YS, cons, egress, load, held


def _run(obj, cons, solver):
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        return "solver_error", False
    return prob.status, prob.status in ("optimal", "optimal_inaccurate")


def solve(
    pop: JobPopulation,
    pool: PoolPower,
    imp: Impact,
    s_star: float,
    event: Event = Event(),
    move: Movement = Movement(),
    integer: bool = False,
    fleet: DestFleet = None,
) -> Plan:
    """Primary solve; on infeasibility re-solve to max shed and report shortfall. fleet=None
    ⇒ a single destination from Event/Movement using imp's frozen costs (the original program);
    an explicit fleet recomputes per-ℓ costs (each dest's ρ_ℓ/φ_pre,ℓ)."""
    multidest = fleet is not None
    fleet = fleet or DestFleet.from_event(event, move, pool, pop)
    YR, YS, cons, egress, load, held = _build(
        pop, pool, imp, fleet, event, move, integer
    )
    dp = bind_dp(imp)
    method, solver = (
        "milp" if integer else "lp"
    ), cp.SCIPY  # HiGHS via cvxpy (scale-robust; CLARABEL is not)
    if multidest:
        cR, cS, _ = move_costs(pop, fleet, move)
        cost = cp.sum(cp.multiply(cR, YR) + cp.multiply(cS, YS))
    else:
        cost = imp.c_replay @ cp.sum(YR, axis=1) + imp.c_transfer @ cp.sum(YS, axis=1)
    total = cp.sum(YR + YS, axis=1)

    status, feasible = _run(cp.Minimize(cost + 1e-9 * (dp @ total)), cons + [dp @ total >= s_star], solver)
    if not feasible:
        status, ok = _run(
            cp.Maximize(dp @ total), cons, solver
        )  # S*-independent capacity prices
        if not ok:
            raise RuntimeError(f"max-shed re-solve failed: status={status}")
    duals = (
        None
        if integer
        else (
            float(egress.dual_value),
            np.asarray(load.dual_value) + np.asarray(held.dual_value),
        )
    )
    return _plan2(
        YR.value, YS.value, dp, imp, float(cost.value), method, s_star, feasible, duals
    )


def _movable(pop, dp, event) -> np.ndarray:
    m = dp > 0  # idle jobs free no power: never worth moving
    if event.pinned:
        m &= ~np.isin(pop.job_type, event.pinned)
    return m


def _first_fit(pop, pool, imp, s_star, event, move, order, prefer, method) -> Plan:
    """Myopic single pass: accept jobs in `order`, each via prefer[j] (then its
    fallback action), drawing down the SHARED movement budgets the deadline window
    allows until S* is met. No global repacking. Engine for the greedy and random
    baselines; `order` already excludes idle/pinned jobs so dp[j] > 0."""
    if np.any(pop.ell > pool.rho_star): raise ValueError("job ell exceeds rho_star; split the job or lower its offered load")
    n = len(pop)
    dp = bind_dp(imp)
    reb = pop.T / rho_replay(pop.T, pop.mfu)
    held_w = (pop.T / pool.mean_context_tokens) * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)
    budget = {  # same RHS as the solve() constraints; consumed as jobs are accepted
        "egress": move.lambda_src * (event.D - event.tau_src),
        "prefill": event.W * (event.D - event.tau_pre),
        "ingest": event.W * move.mu_in * (event.D - event.tau_in),
        "load": event.l_dest(pool),
        "held": event.s_dest(pool),
    }
    draw = {  # per-job resource a unit move consumes (same coefficients as the LP rows)
        "R": [
            ("egress", imp.b_replay),
            ("prefill", reb),
            ("load", pop.ell),
            ("held", held_w),
        ],
        "S": [
            ("egress", imp.b_transfer),
            ("ingest", imp.b_transfer),
            ("load", pop.ell),
            ("held", held_w),
        ],
    }
    yR, yS = np.zeros(n), np.zeros(n)
    cum = 0.0
    for j in order:
        if cum >= s_star:
            break
        acts = (prefer[j], "RS"[prefer[j] == "R"])
        fs = []
        for a in acts:
            fit = min([budget[r] / col[j] for r, col in draw[a] if col[j] > 0] + [1.0])
            fs.append(min(1.0, fit, (s_star - cum) / dp[j]))
        a, f = (acts[1], fs[1]) if fs[1] > fs[0] else (acts[0], fs[0])
        if f > 1e-12:
            for r, col in draw[a]:
                budget[r] -= f * col[j]
            (yR if a == "R" else yS)[j] = f
            cum += f * dp[j]
    return _plan(
        np.concatenate([yR, yS]),
        n,
        dp,
        imp,
        method,
        imp.regime,
        s_star,
        feasible=cum >= s_star - 1e-6 * max(s_star, 1.0),
    )


def greedy(
    pop: JobPopulation,
    pool: PoolPower,
    imp: Impact,
    s_star: float,
    event: Event = Event(),
    move: Movement = Movement(),
) -> Plan:
    """Decentralized first-fit: each job self-selects its cheaper action and the
    best-deal jobs (lowest downtime per watt) move first, drawing down the shared
    budgets in one myopic pass — no global repacking (that is what the LP buys)."""
    dp = bind_dp(imp)
    movable = _movable(pop, dp, event)
    ratio = np.where(
        movable,
        np.minimum(imp.c_replay, imp.c_transfer) / np.where(movable, dp, 1.0),
        np.inf,
    )
    order = np.argsort(ratio)
    order = order[movable[order]]  # movable jobs, best deal first
    prefer = np.where(imp.c_replay <= imp.c_transfer, "R", "S")
    return _first_fit(pop, pool, imp, s_star, event, move, order, prefer, "greedy")


def random_dispatch(
    pop: JobPopulation,
    pool: PoolPower,
    imp: Impact,
    s_star: float,
    event: Event = Event(),
    move: Movement = Movement(),
    seed: int = 0,
) -> Plan:
    """Random baseline: shuffle the movable jobs and pick each action by coin flip,
    same budget-respecting first-fit. The floor any real policy should beat."""
    dp = bind_dp(imp)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.flatnonzero(_movable(pop, dp, event)))
    prefer = np.where(rng.random(len(pop)) < 0.5, "R", "S")
    return _first_fit(pop, pool, imp, s_star, event, move, order, prefer, "random")
