"""Dispatch solver (formulation.md §Dispatch program; §5 pools & event).

Pick which jobs to move and how (replay vs KV-transfer) to hit a grid shed target
S* by deadline D at least total downtime, subject to source egress, destination
rebuild, and destination headroom. Each job gets two variables — a replay var y_R
and a transfer var y_S with y_R+y_S≤1 — so the action choice needs no separate
indicator. The same program is solved as a fractional LP (y∈[0,1]) and an integer
MILP (y∈{0,1}); the gap is the granularity cost. Two solves, not a branch: if the
primary is infeasible, a second solve maximizes shed and reports the shortfall.
A budget-respecting integer greedy (sort by cost per watt) is the T5 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from impact import Impact, Movement, move_costs
from instance import JobPopulation
from power import PoolPower


@dataclass(frozen=True)
class Event:
    """§5 pools & event + §6 timing the solver reads. Center defaults, all swept."""

    D: float = 300.0  # deadline (s)
    dest_nodes: int = 32  # whole destination pool: spare_frac of it is the shared spare pool
    spare_frac: float = 0.40  # spare per dest node; rebuild runs on floor(spare) of these nodes
    tau_src: float = 2.0  # egress connection ramp
    tau_pre: float = 5.0  # prefill batch-form
    tau_in: float = 3.0  # ingest pipeline-fill
    pinned: tuple = ()  # class names forced to y=0 (service floor)
    dest_load_budget_ell: float | None = None

    def __post_init__(self):
        if self.dest_load_budget_ell is not None and self.dest_load_budget_ell <= 0:
            raise ValueError("dest_load_budget_ell must be positive")

    @property
    def spare(self) -> float:
        return self.spare_frac * self.dest_nodes

    def l_dest(self, pool: PoolPower) -> float:
        return self.spare * pool.rho_star if self.dest_load_budget_ell is None else self.dest_load_budget_ell

    def s_dest(self, pool: PoolPower) -> float:
        return self.spare * pool.s_node


@dataclass(frozen=True)
class DestFleet:
    """§4 destination index: K sites the source sheds to, coupled only by the shared
    source uplink. Each field a length-K parallel array. spare_ℓ is a spare node-count →
    load headroom L̄_ℓ = spare_ℓ·ρ*, held headroom S̄_ℓ = spare_ℓ·s_node, and rebuild
    (prefill + ingest) runs on its ⌊spare_ℓ⌋ whole nodes — no dedicated rebuild hardware."""

    spare: np.ndarray  # spare node-count per dest
    mfu: np.ndarray  # destination MFU → ρ_ℓ(T); decoupled from the source's pop.mfu
    prefill_util: np.ndarray  # destination prefill load → φ_pre,ℓ

    def __post_init__(self):
        if np.any(self.W < 1):
            raise ValueError("floor(spare) < 1: destination has no whole spare node for rebuild")

    @property
    def W(self) -> np.ndarray:  # rebuild servers = whole spare nodes, shared with serving headroom
        return np.floor(np.atleast_1d(self.spare) + 1e-9).astype(int)  # +1e-9 absorbs float wobble

    def __len__(self) -> int:
        return len(np.atleast_1d(self.spare))

    @classmethod
    def from_event(
        cls, event: "Event", move: Movement, pool: PoolPower, pop: JobPopulation
    ) -> "DestFleet":
        """The K=1 fleet that reproduces the single-dest Event/Movement headroom exactly."""
        return cls(
            np.array([event.spare]),
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
    resource_duals: dict = None  # row duals: egress/prefill/ingest/load/held

    @property
    def y(self) -> np.ndarray:
        return self.y_R + self.y_S

    @property
    def action(self) -> np.ndarray:
        moved = self.y > 1e-9
        return np.where(~moved, "", np.where(self.y_R >= self.y_S, "R", "S"))


def bind_dp(imp: Impact) -> np.ndarray:
    """Certified average source-power change from moving active work.

    KV memory remains a capacity constraint, but held bytes do not create certified
    watts unless a later node-drain model proves whole nodes can turn off by D.
    """
    return imp.dp_certified


def held_weight(pop: JobPopulation, pool: PoolPower) -> np.ndarray:
    return (pop.T / pool.mean_context_tokens) * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)


def movement_budgets(pool: PoolPower, event: Event, move: Movement, fleet: DestFleet = None,
                     kappa: float = 1.0) -> dict:
    """kappa < 1 is the planner-side rebuild cushion (prefill/ingest RHS only); the DES
    never sees it — pass kappa only from planner solves, never from feasibility audits."""
    if not 0 < kappa <= 1:
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")
    spare = np.array([event.spare]) if fleet is None else np.asarray(fleet.spare)
    if event.dest_load_budget_ell is not None and len(spare) != 1:
        raise ValueError("dest_load_budget_ell requires one aggregate destination")
    W = np.floor(spare + 1e-9)  # whole spare nodes; matches the DES server count exactly
    return {
        "egress": move.lambda_src * max(0.0, event.D - event.tau_src),
        "prefill": kappa * W * max(0.0, event.D - event.tau_pre),
        "ingest": kappa * W * move.mu_in * max(0.0, event.D - event.tau_in),
        "load": spare * pool.rho_star if event.dest_load_budget_ell is None else np.array([event.dest_load_budget_ell]),
        "held": spare * pool.s_node,
    }


def single_movement_budgets(pool: PoolPower, event: Event, move: Movement) -> dict:
    return {k: float(np.asarray(v, float).reshape(-1)[0]) for k, v in movement_budgets(pool, event, move).items()}


def movement_columns(pop: JobPopulation, pool: PoolPower, imp: Impact, fleet: DestFleet,
                     move: Movement = Movement()) -> dict:
    cR, cS, reb = move_costs(pop, fleet, move)
    hw = held_weight(pop, pool)[:, None]
    return {
        "cost_R": cR,
        "cost_S": cS,
        "R": {"egress": imp.b_replay[:, None], "prefill": reb, "load": pop.ell[:, None], "held": hw},
        "S": {"egress": imp.b_transfer[:, None], "ingest": imp.b_transfer[:, None],
              "load": pop.ell[:, None], "held": hw},
    }


def movement_draws(pop: JobPopulation, pool: PoolPower, imp: Impact, event: Event = Event(),
                   move: Movement = Movement()) -> dict:
    fleet = DestFleet.from_event(event, move, pool, pop)
    cols = movement_columns(pop, pool, imp, fleet, move)
    return {
        action: {r: np.asarray(v, float)[:, 0] for r, v in cols[action].items()}
        for action in ("R", "S")
    }


def deadline_infeasible(pop, imp, fleet, event, move, mode="sf"):
    """(n,K) per-action bans: the session misses D even with the link and a rebuild server
    entirely to itself. Mirrors the single-session DES timeline (sf: rebuild after full byte
    arrival; cutthrough: overlapped, capped by arrival). Whole-job (y=1) basis, so it also
    prunes fractional moves the LP could otherwise split — a deliberate tightening."""
    if mode not in ("sf", "cutthrough"):
        raise ValueError(f"unknown mode {mode!r}")
    edR = event.tau_src + imp.b_replay[:, None] / move.lambda_src
    edS = event.tau_src + imp.b_transfer[:, None] / move.lambda_src
    reb = move_costs(pop, fleet, move)[2]
    ing = np.broadcast_to(imp.b_transfer[:, None] / move.mu_in, reb.shape)
    if mode == "cutthrough":
        doneR = np.maximum(edR, max(event.tau_src, event.tau_pre) + reb)
        doneS = np.maximum(edS, max(event.tau_src, event.tau_in) + ing)
    else:
        doneR = np.maximum(edR, event.tau_pre) + reb
        doneS = np.maximum(edS, event.tau_in) + ing
    return doneR > event.D, doneS > event.D


def movement_draws_filtered(pop, pool, imp, event, move, mode="sf"):
    """Baseline-facing draws with deadline-banned actions priced infinite, so every
    budget check (first-fit, node-knee bundles, exact oracle) rejects them for free.
    Feasibility audits/diagnostics keep the raw movement_draws — the ban is planner
    hygiene, not a physical budget."""
    draws = movement_draws(pop, pool, imp, event, move)
    fleet = DestFleet.from_event(event, move, pool, pop)
    badR, badS = deadline_infeasible(pop, imp, fleet, event, move, mode)
    draws["R"]["egress"] = np.where(badR[:, 0], np.inf, draws["R"]["egress"])
    draws["S"]["egress"] = np.where(badS[:, 0], np.inf, draws["S"]["egress"])
    return draws


def movement_used(draws: dict, y_R, y_S) -> dict:
    y_R, y_S = np.asarray(y_R, float), np.asarray(y_S, float)
    resources = set(draws["R"]) | set(draws["S"])
    return {
        r: float(
            (draws["R"][r] @ y_R if r in draws["R"] else 0.0)
            + (draws["S"][r] @ y_S if r in draws["S"] else 0.0)
        )
        for r in resources
    }


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
    te = float(np.max(np.atleast_1d(duals["egress"]))) if duals else None
    ta = np.atleast_1d(duals["load"]) + np.atleast_1d(duals["held"]) if duals else None
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
        duals,
    )


def _build(pop, pool, imp, fleet, event, move, integer, kappa=1.0, mode="sf"):
    """Variables Y_R, Y_S (n,K) + pairing, the ONE shared egress row, and the per-ℓ movement
    blocks (everything but shed). Returns the dual-carrying constraint handles too."""
    if np.any(pop.ell > pool.rho_star): raise ValueError("job ell exceeds rho_star; split the job or lower its offered load")
    n, K = len(pop), len(fleet)
    kw = {"boolean": True} if integer else {"nonneg": True}
    YR, YS = cp.Variable((n, K), **kw), cp.Variable((n, K), **kw)
    Y = YR + YS
    cols = movement_columns(pop, pool, imp, fleet, move)
    budgets = movement_budgets(pool, event, move, fleet, kappa)
    egress = cp.sum(cp.multiply(cols["R"]["egress"], YR) + cp.multiply(cols["S"]["egress"], YS)) <= budgets["egress"]
    # TODO(dest-load): recompute ell per destination when fleet hardware/precision differs.
    load = cp.sum(cp.multiply(cols["R"]["load"], Y), axis=0) <= budgets["load"]
    held = cp.sum(cp.multiply(cols["R"]["held"], Y), axis=0) <= budgets["held"]
    prefill = cp.sum(cp.multiply(cols["R"]["prefill"], YR), axis=0) <= budgets["prefill"]
    ingest = cp.sum(cp.multiply(cols["S"]["ingest"], YS), axis=0) <= budgets["ingest"]
    cons = [
        cp.sum(Y, axis=1)
        <= 1,  # pairing: each job moves at most once, across all destinations
        egress,  # the single shared source uplink — the entire multi-dest coupling
        prefill,  # per-ℓ prefill
        ingest,  # per-ℓ ingest
        load,
        held,  # per-ℓ destination headroom
    ]
    if event.pinned:
        pin = np.isin(pop.job_type, event.pinned)
        if pin.any():
            cons += [YR[pin] == 0, YS[pin] == 0]
    badR, badS = deadline_infeasible(pop, imp, fleet, event, move, mode)
    for bad, V in ((badR, YR), (badS, YS)):  # nonneg/boolean vars ⇒ zero-sum pins elementwise
        if bad.any():
            cons.append(cp.sum(V[np.nonzero(bad)]) == 0)
    return YR, YS, cons, {
        "egress": egress,
        "prefill": prefill,
        "ingest": ingest,
        "load": load,
        "held": held,
    }


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
    kappa: float = 1.0,
    mode: str = "sf",
) -> Plan:
    """Primary solve; on infeasibility re-solve to max shed and report shortfall. fleet=None
    ⇒ a single destination from Event/Movement using imp's frozen costs (the original program);
    an explicit fleet recomputes per-ℓ costs (each dest's ρ_ℓ/φ_pre,ℓ). mode sets the
    deadline pre-filter's timeline (sf bans a superset of cutthrough)."""
    multidest = fleet is not None
    fleet = fleet or DestFleet.from_event(event, move, pool, pop)
    YR, YS, cons, handles = _build(
        pop, pool, imp, fleet, event, move, integer, kappa, mode
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
    duals = None if integer else {
        k: np.atleast_1d(np.asarray(v.dual_value, dtype=float))
        for k, v in handles.items()
    }
    return _plan2(
        YR.value, YS.value, dp, imp, float(cost.value), method, s_star, feasible, duals
    )


def _movable(pop, dp, event) -> np.ndarray:
    m = dp > 0  # idle jobs free no power: never worth moving
    if event.pinned:
        m &= ~np.isin(pop.job_type, event.pinned)
    return m


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    out = np.empty(len(a), float)
    vals = a[order]
    cuts = np.r_[0, np.flatnonzero(vals[1:] != vals[:-1]) + 1, len(a)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        out[order[lo:hi]] = 0.5 * (lo + hi - 1)
    return out


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
        return np.nan
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1])


def dispatch_diagnostics(pop, pool, imp, plan, s_star, event=Event(), move=Movement()) -> dict:
    """Row-level audit numbers for dispatch plots; all ratios use LP row coefficients."""
    dp = bind_dp(imp)
    budgets = single_movement_budgets(pool, event, move)
    draws = movement_draws(pop, pool, imp, event, move)
    used = movement_used(draws, plan.y_R, plan.y_S)
    max_draw = {
        "egress": float(np.maximum(draws["R"]["egress"], draws["S"]["egress"]).max()),
        "prefill": float(draws["R"]["prefill"].max()),
        "ingest": float(draws["S"]["ingest"].max()),
        "load": float(draws["R"]["load"].max()),
        "held": float(draws["R"]["held"].max()),
    }
    util = {k: used[k] / budgets[k] for k in budgets}
    duals = {
        k: float(np.nan if plan.resource_duals is None else np.max(np.atleast_1d(plan.resource_duals[k])))
        for k in budgets
    }
    vals = np.concatenate([plan.Y_R.ravel(), plan.Y_S.ravel()]) if plan.Y_R is not None else np.concatenate([plan.y_R, plan.y_S])
    return {
        "active_constraints": tuple(k for k, v in util.items() if v >= 1 - 1e-6),
        "utilization": util,
        "duals": duals,
        "fractional_variables": int(((vals > 1e-9) & (vals < 1 - 1e-9)).sum()),
        "max_dp_over_s": float(dp.max() / s_star),
        "max_resource_draw_over_budget": {k: max_draw[k] / budgets[k] for k in budgets},
        "spearman": {
            "cost": _spearman(dp, np.minimum(imp.c_replay, imp.c_transfer)),
            "egress": _spearman(dp, np.maximum(imp.b_replay, imp.b_transfer)),
            "prefill": _spearman(dp, draws["R"]["prefill"]),
            "held": _spearman(dp, draws["R"]["held"]),
            "load": _spearman(dp, pop.ell),
        },
    }


def _first_fit(pop, pool, imp, s_star, event, move, order, prefer, method) -> Plan:
    """Myopic single pass: accept jobs in `order`, each via prefer[j] (then its
    fallback action), drawing down whole-job SHARED movement budgets the deadline window
    allows until S* is met or no more jobs fit. No global repacking. Engine for the greedy and random
    baselines; `order` already excludes idle/pinned jobs so dp[j] > 0."""
    if np.any(pop.ell > pool.rho_star): raise ValueError("job ell exceeds rho_star; split the job or lower its offered load")
    n = len(pop)
    dp = bind_dp(imp)
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    draw = {a: list(draws[a].items()) for a in ("R", "S")}
    yR, yS = np.zeros(n), np.zeros(n)
    cum = 0.0
    for j in order:
        if cum >= s_star:
            break
        acts = (prefer[j], "RS"[prefer[j] == "R"])
        a = next((a for a in acts if all(col[j] <= budget[r] + 1e-12 for r, col in draw[a])), None)
        if a is None:
            continue
        for r, col in draw[a]:
            budget[r] -= col[j]
        (yR if a == "R" else yS)[j] = 1.0
        cum += dp[j]
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
    """Decentralized integer first-fit: each job self-selects its cheaper action and the
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
    same integer budget-respecting first-fit. The floor any real policy should beat."""
    dp = bind_dp(imp)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.flatnonzero(_movable(pop, dp, event)))
    prefer = np.where(rng.random(len(pop)) < 0.5, "R", "S")
    return _first_fit(pop, pool, imp, s_star, event, move, order, prefer, "random")
