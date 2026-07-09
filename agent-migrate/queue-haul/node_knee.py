"""Node-knee exploration: source placement, expected node shed, and tiny solvers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

import cvxpy as cp
import numpy as np

from dispatch import (
    DestFleet,
    Event,
    _build,
    _run,
    held_weight,
    movement_draws,
    movement_draws_filtered,
    movement_used,
    single_movement_budgets,
    solve,
)
from impact import Impact, Movement
from instance import JobPopulation
from power import PoolPower

MAX_EXHAUSTIVE_ACTIVE_NODES = 8


def place_source_nodes(pop: JobPopulation, pool: PoolPower, n_nodes: int, policy: str = "memory") -> np.ndarray:
    """Deterministic source placement; source overload is allowed for stress fixtures."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    if policy == "balanced":
        return np.arange(len(pop)) % n_nodes
    metric = {"memory": held_weight(pop, pool), "load": pop.ell}.get(policy)
    if metric is None:
        raise ValueError(f"unknown placement policy {policy!r}")
    node, used = np.zeros(len(pop), int), np.zeros(n_nodes)
    for j in np.argsort(-metric, kind="mergesort"):
        i = int(np.argmin(used))
        node[j], used[i] = i, used[i] + metric[j]
    return node


def with_source_nodes(pop: JobPopulation, source_node) -> JobPopulation:
    return replace(pop, source_node=np.asarray(source_node, int))


def _source_node(pop: JobPopulation) -> np.ndarray:
    if pop.source_node is None:
        raise ValueError("node-knee methods require pop.source_node")
    node = np.asarray(pop.source_node, int)
    if len(node) != len(pop) or np.any(node < 0):
        raise ValueError("source_node must assign every job to a nonnegative node")
    return node


def node_loads(pop: JobPopulation) -> np.ndarray:
    node = _source_node(pop)
    return np.bincount(node, weights=pop.ell, minlength=int(node.max()) + 1)


def removed_loads(pop: JobPopulation, y) -> np.ndarray:
    node = _source_node(pop)
    return np.bincount(node, weights=pop.ell * np.asarray(y, float), minlength=int(node.max()) + 1)


def evaluate_node_expected_w(pop: JobPopulation, pool: PoolPower, y) -> float:
    load = node_loads(pop)
    removed = removed_loads(pop, y)
    return float(np.sum(pool.node_power(load) - pool.node_power(load - removed)))


def evaluate_active_floor_w(imp: Impact, y) -> float:
    return float(imp.dp_guaranteed @ np.asarray(y, float))


def _plan_y(plan) -> np.ndarray:
    return ((plan.y_R + plan.y_S)[:, None] if plan.Y_R is None or plan.Y_S is None
            else plan.Y_R + plan.Y_S)


def execution_realization_metrics(pop: JobPopulation, pool: PoolPower, imp: Impact,
                                  plan, sim, D: float) -> dict:
    """Selected vs deadline-realized node power for a replayed plan."""
    Y = np.asarray(_plan_y(plan), float)
    ed, rd = np.asarray(sim.egress_done), np.asarray(sim.rebuild_done)
    ed = ed[:, None] if ed.ndim == 1 else ed
    rd = rd[:, None] if rd.ndim == 1 else rd
    if ed.shape != Y.shape or rd.shape != Y.shape:
        raise ValueError("simulation completion arrays do not match plan routing")
    masks = {
        "selected": np.ones_like(Y, bool),
        "egress_realized": np.isfinite(ed) & (ed <= D),
        "rebuild_realized": np.isfinite(rd) & (rd <= D),
    }
    out = {}
    for name, mask in masks.items():
        y = (Y * mask).sum(1)
        node = evaluate_node_expected_w(pop, pool, y)
        active = evaluate_active_floor_w(imp, y)
        out[f"{name}_node_expected_w"] = node
        out[f"{name}_active_floor_w"] = active
        out[f"{name}_node_s_per_kw"] = plan.cost / (node / 1e3) if node > 0 else np.nan
        out[f"{name}_active_s_per_kw"] = plan.cost / (active / 1e3) if active > 0 else np.nan
    return out


@dataclass(frozen=True)
class NodeKneeResult:
    y_R: np.ndarray
    y_S: np.ndarray
    cost: float
    method: str
    feasible: bool
    movement_feasible: bool
    method_target_feasible: bool
    true_expected_feasible: bool
    active_floor_w: float
    node_expected_w: float
    expected_shortfall_w: float
    floor_shortfall_w: float
    order: tuple[int, ...] = ()

    @property
    def y(self) -> np.ndarray:
        return self.y_R + self.y_S


def _movement_feasible(pop, pool, imp, y_R, y_S, event, move) -> bool:
    y_R, y_S = np.asarray(y_R, float), np.asarray(y_S, float)
    y = y_R + y_S
    tol = 1e-6
    if np.any(y_R < -tol) or np.any(y_S < -tol) or np.any(y > 1 + tol):
        return False
    if event.pinned and np.any(y[np.isin(pop.job_type, event.pinned)] > tol):
        return False
    budgets = single_movement_budgets(pool, event, move)
    used = movement_used(movement_draws(pop, pool, imp, event, move), y_R, y_S)
    return all(used[r] <= budgets[r] + tol * max(1.0, abs(budgets[r])) for r in budgets)


def _result(pop, pool, imp, y_R, y_S, cost, method, s_star, event=Event(), move=Movement(),
            method_target_feasible=True, order=()) -> NodeKneeResult:
    y = y_R + y_S
    active = evaluate_active_floor_w(imp, y)
    expected = evaluate_node_expected_w(pop, pool, y)
    true = expected >= s_star - 1e-6 * max(s_star, 1.0)
    movement = _movement_feasible(pop, pool, imp, y_R, y_S, event, move)
    return NodeKneeResult(
        y_R, y_S, float(cost), method, movement and true, movement, method_target_feasible, true,
        active, expected, max(0.0, s_star - expected), max(0.0, s_star - active), tuple(order)
    )


def _r_expr(pop, total):
    node = _source_node(pop)
    return [cp.sum(cp.multiply(pop.ell[node == i], total[node == i])) for i in range(node.max() + 1)]


def _tangent(pop: JobPopulation, pool: PoolPower, r0):
    load = node_loads(pop)
    g = pool.node_power_slope(load - r0)
    f0 = pool.node_power(load) - pool.node_power(load - r0)
    node = _source_node(pop)
    return g[node] * pop.ell, float(np.sum(f0 - g * r0))


def _region_affine(pop: JobPopulation, pool: PoolPower, active_nodes):
    load = node_loads(pop)
    active = set(active_nodes)
    slope = np.full(len(load), pool.ramp_slope)
    intercept = 0.0
    for i, L in enumerate(load):
        if L > pool.power_knee and i not in active:
            slope[i] = pool.s_plat
        elif L > pool.power_knee:
            gap = L - pool.power_knee
            intercept += float(pool.node_power(L) - pool.node_power(pool.power_knee) - pool.ramp_slope * gap)
    return slope[_source_node(pop)] * pop.ell, intercept


def _lp(pop, pool, imp, s_star, weights, intercept=0.0, r0=None, event=Event(),
        move=Movement(), active_alpha=0.0, active_nodes=(), method="node_lp",
        region_consistent=False, integer=False, kappa=1.0) -> NodeKneeResult:
    fleet = DestFleet.from_event(event, move, pool, pop)
    YR, YS, cons, _ = _build(pop, pool, imp, fleet, event, move, integer, kappa)
    total = cp.sum(YR + YS, axis=1)
    cost = imp.c_replay @ cp.sum(YR, axis=1) + imp.c_transfer @ cp.sum(YS, axis=1)
    surrogate = intercept + weights @ total
    rexpr = _r_expr(pop, total)
    load = node_loads(pop)
    active = set(active_nodes)
    cons += [load[i] - rexpr[i] <= pool.power_knee for i in active if load[i] > pool.power_knee]
    if region_consistent:
        cons += [load[i] - rexpr[i] >= pool.power_knee for i in range(len(load))
                 if i not in active and load[i] > pool.power_knee]
    if active_alpha:
        cons_target = cons + [imp.dp_certified @ total >= active_alpha * s_star, surrogate >= s_star]
    else:
        cons_target = cons + [surrogate >= s_star]
    status, ok = _run(cp.Minimize(cost), cons_target, cp.SCIPY)
    method_target_feasible = ok
    if not ok:
        status, ok = _run(cp.Maximize(surrogate), cons, cp.SCIPY)
        if not ok:
            raise RuntimeError(f"node-knee LP failed: status={status}")
    return _result(
        pop, pool, imp, YR.value.sum(1), YS.value.sum(1), cost.value,
        method, s_star, event, move, method_target_feasible
    )


def solve_tangent_lp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                     event: Event = Event(), move: Movement = Movement(),
                     max_iter: int = 5, active_alpha: float = 0.0, kappa: float = 1.0,
                     method: str = "tangent_lp") -> NodeKneeResult:
    r0 = np.zeros_like(node_loads(pop))
    best = None
    for _ in range(max_iter):
        w, b = _tangent(pop, pool, r0)
        res = _lp(pop, pool, imp, s_star, w, b, r0, event, move, active_alpha, method=method,
                  kappa=kappa)
        if best is None or (res.true_expected_feasible and res.cost < best.cost) or (
            not best.true_expected_feasible and res.node_expected_w > best.node_expected_w
        ):
            best = res
        r = removed_loads(pop, res.y)
        if np.allclose(r, r0, atol=1e-8):
            break
        r0 = r
    return best


def solve_power_function_lp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                            event: Event = Event(), move: Movement = Movement(),
                            active_alpha: float = 0.0, kappa: float = 1.0) -> NodeKneeResult:
    return solve_tangent_lp(pop, pool, imp, s_star, event, move, active_alpha=active_alpha,
                            kappa=kappa, method="power_function_lp_relaxation")


def _job_cost(imp: Impact) -> np.ndarray:
    return np.minimum(imp.c_replay, imp.c_transfer)


def _node_movable(pop: JobPopulation, event: Event) -> np.ndarray:
    m = pop.ell > 0
    if event.pinned:
        m &= ~np.isin(pop.job_type, event.pinned)
    return m


def _knee_candidates(pop: JobPopulation, pool: PoolPower, imp: Impact, event: Event = Event(),
                     k: int = 4) -> list[tuple[int, ...]]:
    active = tuple(np.flatnonzero(node_loads(pop) > pool.power_knee))
    if len(active) > MAX_EXHAUSTIVE_ACTIVE_NODES:
        raise ValueError(f"active-knee exhaustive solve supports at most {MAX_EXHAUSTIVE_ACTIVE_NODES} active source nodes")
    return [tuple(c) for m in range(len(active) + 1) for c in combinations(active, m)]


def _solve_active_knee(pop, pool, imp, s_star, event, move, active_alpha, integer, method, kappa=1.0):
    best = None
    for active in _knee_candidates(pop, pool, imp, event):
        w, b = _region_affine(pop, pool, active)
        try:
            res = _lp(pop, pool, imp, s_star, w, b, None, event, move, active_alpha,
                      active, method, region_consistent=True, integer=integer, kappa=kappa)
        except RuntimeError:
            continue
        if best is None or (res.true_expected_feasible and res.cost < best.cost) or (
            not best.true_expected_feasible and res.node_expected_w > best.node_expected_w
        ):
            best = res
    if best is None:
        if integer:
            raise RuntimeError("active-knee MILP failed for every candidate")
        return solve_tangent_lp(pop, pool, imp, s_star, event, move, active_alpha=active_alpha,
                                kappa=kappa)
    return best


def solve_active_knee_lp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                         event: Event = Event(), move: Movement = Movement(),
                         active_alpha: float = 0.0, kappa: float = 1.0) -> NodeKneeResult:
    return _solve_active_knee(pop, pool, imp, s_star, event, move, active_alpha, False,
                              "active_knee_lp_relaxation", kappa)


def solve_active_knee_milp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                           event: Event = Event(), move: Movement = Movement(),
                           active_alpha: float = 0.0, kappa: float = 1.0) -> NodeKneeResult:
    return _solve_active_knee(pop, pool, imp, s_star, event, move, active_alpha, True,
                              "active_knee_milp", kappa)


def _fits(budget, draws, action, j) -> bool:
    return all(draws[action][r][j] <= budget[r] + 1e-12 for r in draws[action])


def _take(budget, draws, action, j):
    for r, col in draws[action].items():
        budget[r] -= col[j]


def _action(budget, draws, imp, j, pref=None):
    pref = pref or ("R" if imp.c_replay[j] <= imp.c_transfer[j] else "S")
    return next((a for a in (pref, "RS"[pref == "R"]) if _fits(budget, draws, a, j)), None)


def _best_feasible_action(budget, draws, imp, j):
    options = [(a, imp.c_replay[j] if a == "R" else imp.c_transfer[j])
               for a in "RS" if _fits(budget, draws, a, j)]
    return min(options, key=lambda x: x[1]) if options else (None, np.inf)


def _move(budget, draws, yR, yS, action, j):
    _take(budget, draws, action, j)
    (yR if action == "R" else yS)[j] = 1.0


def _knee_bundle(pop, pool, node, load, cost, movable, i):
    gap = max(0.0, load[i] - pool.power_knee)
    if gap <= 0:
        return []
    js = np.flatnonzero((node == i) & movable)
    order = js[np.argsort(cost[js] / np.maximum(pop.ell[js], 1e-12), kind="mergesort")]
    return order[: np.searchsorted(np.cumsum(pop.ell[order]), gap, side="left") + 1]


def _assign_candidate(budget, draws, imp, jobs):
    tmp, picks, cost, egress = budget.copy(), [], 0.0, 0.0
    for j in jobs:
        act, c = _best_feasible_action(tmp, draws, imp, int(j))
        if act is None:
            return None
        _take(tmp, draws, act, int(j))
        picks.append((int(j), act))
        cost += float(c)
        egress += float(draws[act]["egress"][j]) if "egress" in draws[act] else 0.0
    return picks, tmp, cost, egress


def _bundle_jobs(pop, imp, budget, draws, node, movable, used, i):
    jobs = [int(j) for j in np.flatnonzero((node == i) & movable) if int(j) not in used]
    priced = []
    for j in jobs:
        act, c = _best_feasible_action(budget, draws, imp, j)
        if act is None:
            return []
        priced.append((c / max(float(pop.ell[j]), 1e-12), j))
    return [j for _, j in sorted(priced)]


def _candidate_gain(pop, pool, resid, node, jobs):
    i = int(node[jobs[0]])
    removed = float(pop.ell[list(jobs)].sum())
    return float(pool.node_power(resid[i]) - pool.node_power(max(0.0, resid[i] - removed)))


def _node_aware_candidates(pop, pool, imp, budget, draws, y, used, movable):
    node, resid = _source_node(pop), node_loads(pop) - removed_loads(pop, y)
    for j in np.flatnonzero(movable):
        if int(j) not in used:
            yield "single", int(node[j]), (int(j),)
    for i in range(len(resid)):
        jobs = _bundle_jobs(pop, imp, budget, draws, node, movable, used, i)
        if not jobs:
            continue
        if resid[i] > pool.power_knee:
            need = resid[i] - pool.power_knee
            csum = np.cumsum(pop.ell[jobs])
            if csum[-1] >= need - 1e-12:
                yield "knee", i, tuple(jobs[: int(np.searchsorted(csum, need, side="left")) + 1])
        yield "drain", i, tuple(jobs)


def solve_node_aware_greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                            event: Event = Event(), move: Movement = Movement()) -> NodeKneeResult:
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    movable = _node_movable(pop, event)
    node = _source_node(pop)
    yR, yS, used, order = np.zeros(len(pop)), np.zeros(len(pop)), set(), []
    while evaluate_node_expected_w(pop, pool, yR + yS) < s_star:
        y = yR + yS
        resid = node_loads(pop) - removed_loads(pop, y)
        remaining = s_star - evaluate_node_expected_w(pop, pool, y)
        candidates = []
        for kind, i, jobs in _node_aware_candidates(pop, pool, imp, budget, draws, y, used, movable):
            assigned = _assign_candidate(budget, draws, imp, jobs)
            if assigned is None:
                continue
            picks, tmp, cost, egress = assigned
            gain = _candidate_gain(pop, pool, resid, node, jobs)
            useful = min(gain, remaining)
            if useful <= 0:
                continue
            candidates.append((cost / useful, -gain, egress, cost, kind, i, jobs, picks, tmp))
        if not candidates:
            break
        *_score, jobs, picks, budget = sorted(candidates)[0]
        for j, act in picks:
            (yR if act == "R" else yS)[j] = 1.0
            used.add(j)
        order.extend(jobs)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                   "node_aware_greedy", s_star, event, move, order=order)


def _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used, movable):
    node = _source_node(pop)
    resid = node_loads(pop) - removed_loads(pop, yR + yS)
    left = set(np.flatnonzero(movable)) - set(used)
    while left and evaluate_node_expected_w(pop, pool, yR + yS) < s_star:
        scores = []
        for j in tuple(left):
            act, action_cost = _best_feasible_action(budget, draws, imp, j)
            if act is None:
                left.remove(j)
                continue
            val = float(pool.node_power(resid[node[j]]) - pool.node_power(resid[node[j]] - pop.ell[j]))
            scores.append((-(val / max(action_cost, 1e-12)), j, act))
        if not scores:
            break
        _, j, act = sorted(scores)[0]
        _move(budget, draws, yR, yS, act, j)
        resid[node[j]] -= pop.ell[j]
        left.remove(j)
    return yR, yS


def solve_live_greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                      event: Event = Event(), move: Movement = Movement()) -> NodeKneeResult:
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    movable = _node_movable(pop, event)
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, np.zeros(len(pop)), np.zeros(len(pop)), set(), movable)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                   "live_greedy", s_star, event, move)


def solve_random_jobs(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                      event: Event = Event(), move: Movement = Movement(),
                      seed: int = 0) -> NodeKneeResult:
    rng = np.random.default_rng(seed)
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    yR, yS = np.zeros(len(pop)), np.zeros(len(pop))
    for j in rng.permutation(np.flatnonzero(_node_movable(pop, event))):
        if evaluate_node_expected_w(pop, pool, yR + yS) >= s_star:
            break
        act = _action(budget, draws, imp, int(j), "R" if rng.random() < 0.5 else "S")
        if act:
            _move(budget, draws, yR, yS, act, int(j))
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                   "random_jobs", s_star, event, move)


def solve_random_nodes(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                       event: Event = Event(), move: Movement = Movement(),
                       seed: int = 0) -> NodeKneeResult:
    node, load, cost = _source_node(pop), node_loads(pop), _job_cost(imp)
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    movable = _node_movable(pop, event)
    yR, yS, used = np.zeros(len(pop)), np.zeros(len(pop)), set()
    for i in np.random.default_rng(seed).permutation(len(load)):
        if evaluate_node_expected_w(pop, pool, yR + yS) >= s_star:
            break
        picks, tmp = [], budget.copy()
        for j in _knee_bundle(pop, pool, node, load, cost, movable, int(i)):
            act = _action(tmp, draws, imp, int(j))
            if act is None:
                picks = []
                break
            _take(tmp, draws, act, int(j))
            picks.append((int(j), act))
        for j, act in picks:
            (yR if act == "R" else yS)[j] = 1.0
            used.add(j)
        if picks:
            budget = tmp
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used, movable)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                   "random_nodes", s_star, event, move)


def solve_node_drain_greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                            event: Event = Event(), move: Movement = Movement()) -> NodeKneeResult:
    node, load, cost = _source_node(pop), node_loads(pop), _job_cost(imp)
    budget = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    movable = _node_movable(pop, event)
    yR, yS, used = np.zeros(len(pop)), np.zeros(len(pop)), set()
    bundles = []
    for i in range(len(load)):
        take = _knee_bundle(pop, pool, node, load, cost, movable, i)
        if len(take):
            val = pool.node_power(load[i]) - pool.node_power(load[i] - pop.ell[take].sum())
            bundles.append((-(val / max(cost[take].sum(), 1e-12)), take))
    for _, jobs in sorted(bundles):
        if evaluate_node_expected_w(pop, pool, yR + yS) >= s_star:
            break
        picks, tmp = [], budget.copy()
        for j in jobs:
            act = _action(tmp, draws, imp, j)
            if act is None or j in used:
                picks = []
                break
            _take(tmp, draws, act, j)
            picks.append((j, act))
        for j, act in picks:
            (yR if act == "R" else yS)[j] = 1.0
            used.add(j)
        if picks:
            budget = tmp
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used, movable)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                   "node_drain_greedy", s_star, event, move)


def solve_exact_oracle(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                       event: Event = Event(), move: Movement = Movement(),
                       max_jobs: int = 14) -> NodeKneeResult:
    if len(pop) > max_jobs:
        raise ValueError(f"exact oracle is limited to {max_jobs} jobs")
    budget0 = single_movement_budgets(pool, event, move)
    draws = movement_draws_filtered(pop, pool, imp, event, move)
    choices = ["NRS" if m else "N" for m in _node_movable(pop, event)]
    best, best_short = None, None
    for acts in product(*choices):
        yR = np.array([a == "R" for a in acts], float)
        yS = np.array([a == "S" for a in acts], float)
        budget, ok = budget0.copy(), True
        for j, a in enumerate(acts):
            if a != "N":
                ok = _fits(budget, draws, a, j)
                if not ok:
                    break
                _take(budget, draws, a, j)
        if not ok:
            continue
        res = _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS),
                      "exact_oracle", s_star, event, move)
        if res.true_expected_feasible:
            if best is None or res.cost < best.cost:
                best = res
        elif best is None and (best_short is None or res.node_expected_w > best_short.node_expected_w):
            best_short = res
    return best or best_short


def comparison_rows(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                    event: Event = Event(), move: Movement = Movement()) -> list[dict]:
    base = solve(pop, pool, imp, s_star, event, move)
    results = [
        _result(pop, pool, imp, base.y_R, base.y_S, base.cost, "additive_lp", s_star,
                event, move, base.feasible),
        solve_tangent_lp(pop, pool, imp, s_star, event, move),
        solve_active_knee_lp(pop, pool, imp, s_star, event, move),
        solve_active_knee_milp(pop, pool, imp, s_star, event, move),
        solve_node_aware_greedy(pop, pool, imp, s_star, event, move),
        solve_live_greedy(pop, pool, imp, s_star, event, move),
        solve_node_drain_greedy(pop, pool, imp, s_star, event, move),
        solve_random_jobs(pop, pool, imp, s_star, event, move),
        solve_random_nodes(pop, pool, imp, s_star, event, move),
    ]
    if len(pop) <= 14:
        results.append(solve_exact_oracle(pop, pool, imp, s_star, event, move))
    fields = (
        "method", "cost", "feasible", "movement_feasible", "method_target_feasible",
        "true_expected_feasible", "active_floor_w", "node_expected_w",
        "expected_shortfall_w", "floor_shortfall_w",
    )
    return [{k: getattr(r, k) for k in fields} for r in results]


def placement_sensitivity_rows(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                               n_nodes: int, policies=("memory", "load", "balanced")) -> list[dict]:
    rows = []
    for policy in policies:
        placed = with_source_nodes(pop, place_source_nodes(pop, pool, n_nodes, policy))
        for row in comparison_rows(placed, pool, imp, s_star):
            rows.append({"placement": policy, **row})
    return rows
