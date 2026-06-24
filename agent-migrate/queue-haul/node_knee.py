"""Node-knee exploration: source placement, expected node shed, and tiny solvers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import cvxpy as cp
import numpy as np

from dispatch import DestFleet, Event, _build, _run, solve
from impact import Impact, Movement
from instance import JobPopulation
from power import PoolPower, rho_replay


def held_weight(pop: JobPopulation, pool: PoolPower) -> np.ndarray:
    return (pop.T / pool.mean_context_tokens) * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)


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
    return float(imp.dp_certified @ np.asarray(y, float))


@dataclass(frozen=True)
class NodeKneeResult:
    y_R: np.ndarray
    y_S: np.ndarray
    cost: float
    method: str
    feasible: bool
    movement_feasible: bool
    surrogate_feasible: bool
    true_expected_feasible: bool
    active_floor_w: float
    node_expected_w: float
    expected_shortfall_w: float
    floor_shortfall_w: float

    @property
    def y(self) -> np.ndarray:
        return self.y_R + self.y_S


def _result(pop, pool, imp, y_R, y_S, cost, method, s_star, surrogate_feasible=True) -> NodeKneeResult:
    y = y_R + y_S
    active = evaluate_active_floor_w(imp, y)
    expected = evaluate_node_expected_w(pop, pool, y)
    true = expected >= s_star - 1e-6 * max(s_star, 1.0)
    return NodeKneeResult(
        y_R, y_S, float(cost), method, true, True, surrogate_feasible, true,
        active, expected, max(0.0, s_star - expected), max(0.0, s_star - active)
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


def _lp(pop, pool, imp, s_star, weights, intercept=0.0, r0=None, event=Event(),
        move=Movement(), active_alpha=0.0, active_nodes=(), method="node_lp") -> NodeKneeResult:
    fleet = DestFleet.from_event(event, move, pool, pop)
    YR, YS, cons, _ = _build(pop, pool, imp, fleet, event, move, False)
    total = cp.sum(YR + YS, axis=1)
    cost = imp.c_replay @ cp.sum(YR, axis=1) + imp.c_transfer @ cp.sum(YS, axis=1)
    surrogate = intercept + weights @ total
    rexpr = _r_expr(pop, total)
    load = node_loads(pop)
    cons += [load[i] - rexpr[i] <= pool.power_knee for i in active_nodes if load[i] > pool.power_knee]
    if active_alpha:
        cons_target = cons + [imp.dp_certified @ total >= active_alpha * s_star, surrogate >= s_star]
    else:
        cons_target = cons + [surrogate >= s_star]
    status, ok = _run(cp.Minimize(cost), cons_target, cp.SCIPY)
    surrogate_feasible = ok
    if not ok:
        status, ok = _run(cp.Maximize(surrogate), cons, cp.SCIPY)
        if not ok:
            raise RuntimeError(f"node-knee LP failed: status={status}")
    return _result(
        pop, pool, imp, YR.value.sum(1), YS.value.sum(1), cost.value,
        method, s_star, surrogate_feasible
    )


def solve_tangent_lp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                     event: Event = Event(), move: Movement = Movement(),
                     max_iter: int = 5, active_alpha: float = 0.0) -> NodeKneeResult:
    r0 = np.zeros_like(node_loads(pop))
    best = None
    for _ in range(max_iter):
        w, b = _tangent(pop, pool, r0)
        res = _lp(pop, pool, imp, s_star, w, b, r0, event, move, active_alpha, method="tangent_lp")
        if best is None or (res.true_expected_feasible and res.cost < best.cost) or (
            not best.true_expected_feasible and res.node_expected_w > best.node_expected_w
        ):
            best = res
        r = removed_loads(pop, res.y)
        if np.allclose(r, r0, atol=1e-8):
            break
        r0 = r
    return best


def _job_cost(imp: Impact) -> np.ndarray:
    return np.minimum(imp.c_replay, imp.c_transfer)


def _knee_candidates(pop: JobPopulation, pool: PoolPower, imp: Impact, k: int = 2) -> list[tuple[int, ...]]:
    load, node, cost = node_loads(pop), _source_node(pop), _job_cost(imp)
    gap = np.maximum(0.0, load - pool.power_knee)
    rows = []
    for i, L in enumerate(load):
        js = np.flatnonzero(node == i)
        order = js[np.argsort(cost[js] / np.maximum(pop.ell[js], 1e-12), kind="mergesort")]
        acc = np.cumsum(pop.ell[order])
        take = order[: np.searchsorted(acc, gap[i], side="left") + 1] if gap[i] > 0 and order.size else []
        c = float(cost[take].sum()) if len(take) and acc[min(len(take) - 1, len(acc) - 1)] >= gap[i] else np.inf
        rmax = float(pop.ell[js].sum())
        bonus = float(pool.node_power(L) - pool.node_power(L - rmax) - pool.s_plat * rmax)
        rows.append((i, L, c, bonus, gap[i] / max(c, 1e-12), bonus / max(c, 1e-12)))
    cand = set()
    for col, rev in ((1, True), (2, False), (3, True), (4, False), (5, True)):
        order = [r[0] for r in sorted(rows, key=lambda x: x[col], reverse=rev) if np.isfinite(r[2])]
        for m in range(1, min(k, len(order)) + 1):
            cand.add(tuple(sorted(order[:m])))
    return sorted(cand)


def solve_active_knee_lp(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                         event: Event = Event(), move: Movement = Movement(),
                         active_alpha: float = 0.0) -> NodeKneeResult:
    load = node_loads(pop)
    best = None
    for active in _knee_candidates(pop, pool, imp):
        r0 = np.zeros_like(load)
        r0[list(active)] = np.maximum(0.0, load[list(active)] - pool.power_knee)
        w, b = _tangent(pop, pool, r0)
        try:
            res = _lp(pop, pool, imp, s_star, w, b, r0, event, move, active_alpha, active, "active_knee_lp")
        except RuntimeError:
            continue
        if best is None or (res.true_expected_feasible and res.cost < best.cost) or (
            not best.true_expected_feasible and res.node_expected_w > best.node_expected_w
        ):
            best = res
    return best or solve_tangent_lp(pop, pool, imp, s_star, event, move, active_alpha=active_alpha)


def _budgets(pop, pool, event, move):
    return {
        "egress": move.lambda_src * (event.D - event.tau_src),
        "prefill": event.W * (event.D - event.tau_pre),
        "ingest": event.W * move.mu_in * (event.D - event.tau_in),
        "load": event.l_dest(pool),
        "held": event.s_dest(pool),
    }


def _draws(pop, pool, imp):
    reb = pop.T / rho_replay(pop.T, pop.mfu)
    hw = held_weight(pop, pool)
    return {
        "R": {"egress": imp.b_replay, "prefill": reb, "load": pop.ell, "held": hw},
        "S": {"egress": imp.b_transfer, "ingest": imp.b_transfer, "load": pop.ell, "held": hw},
    }


def _fits(budget, draws, action, j) -> bool:
    return all(draws[action][r][j] <= budget[r] + 1e-12 for r in draws[action])


def _take(budget, draws, action, j):
    for r, col in draws[action].items():
        budget[r] -= col[j]


def _action(budget, draws, imp, j, pref=None):
    pref = pref or ("R" if imp.c_replay[j] <= imp.c_transfer[j] else "S")
    return next((a for a in (pref, "RS"[pref == "R"]) if _fits(budget, draws, a, j)), None)


def _move(budget, draws, yR, yS, action, j):
    _take(budget, draws, action, j)
    (yR if action == "R" else yS)[j] = 1.0


def _knee_bundle(pop, pool, node, load, cost, i):
    gap = max(0.0, load[i] - pool.power_knee)
    if gap <= 0:
        return []
    js = np.flatnonzero(node == i)
    order = js[np.argsort(cost[js] / np.maximum(pop.ell[js], 1e-12), kind="mergesort")]
    return order[: np.searchsorted(np.cumsum(pop.ell[order]), gap, side="left") + 1]


def _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used):
    node = _source_node(pop)
    resid = node_loads(pop) - removed_loads(pop, yR + yS)
    cost = _job_cost(imp)
    left = set(range(len(pop))) - set(used)
    while left and evaluate_node_expected_w(pop, pool, yR + yS) < s_star:
        scores = []
        for j in left:
            val = float(pool.node_power(resid[node[j]]) - pool.node_power(resid[node[j]] - pop.ell[j]))
            scores.append((-(val / max(cost[j], 1e-12)), j))
        moved = False
        for _, j in sorted(scores):
            act = _action(budget, draws, imp, j)
            if act:
                _move(budget, draws, yR, yS, act, j)
                resid[node[j]] -= pop.ell[j]
                left.remove(j)
                moved = True
                break
            left.remove(j)
        if not moved:
            break
    return yR, yS


def solve_live_greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                      event: Event = Event(), move: Movement = Movement()) -> NodeKneeResult:
    budget, draws = _budgets(pop, pool, event, move), _draws(pop, pool, imp)
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, np.zeros(len(pop)), np.zeros(len(pop)), set())
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS), "live_greedy", s_star)


def solve_random_jobs(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                      event: Event = Event(), move: Movement = Movement(),
                      seed: int = 0) -> NodeKneeResult:
    rng = np.random.default_rng(seed)
    budget, draws = _budgets(pop, pool, event, move), _draws(pop, pool, imp)
    yR, yS = np.zeros(len(pop)), np.zeros(len(pop))
    for j in rng.permutation(np.flatnonzero(pop.ell > 0)):
        if evaluate_node_expected_w(pop, pool, yR + yS) >= s_star:
            break
        act = _action(budget, draws, imp, int(j), "R" if rng.random() < 0.5 else "S")
        if act:
            _move(budget, draws, yR, yS, act, int(j))
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS), "random_jobs", s_star)


def solve_random_nodes(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                       event: Event = Event(), move: Movement = Movement(),
                       seed: int = 0) -> NodeKneeResult:
    node, load, cost = _source_node(pop), node_loads(pop), _job_cost(imp)
    budget, draws = _budgets(pop, pool, event, move), _draws(pop, pool, imp)
    yR, yS, used = np.zeros(len(pop)), np.zeros(len(pop)), set()
    for i in np.random.default_rng(seed).permutation(len(load)):
        if evaluate_node_expected_w(pop, pool, yR + yS) >= s_star:
            break
        picks, tmp = [], budget.copy()
        for j in _knee_bundle(pop, pool, node, load, cost, int(i)):
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
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS), "random_nodes", s_star)


def solve_node_drain_greedy(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                            event: Event = Event(), move: Movement = Movement()) -> NodeKneeResult:
    node, load, cost = _source_node(pop), node_loads(pop), _job_cost(imp)
    budget, draws = _budgets(pop, pool, event, move), _draws(pop, pool, imp)
    yR, yS, used = np.zeros(len(pop)), np.zeros(len(pop)), set()
    bundles = []
    for i in range(len(load)):
        take = _knee_bundle(pop, pool, node, load, cost, i)
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
    yR, yS = _finish_live(pop, pool, imp, s_star, budget, draws, yR, yS, used)
    return _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS), "node_drain_greedy", s_star)


def solve_exact_oracle(pop: JobPopulation, pool: PoolPower, imp: Impact, s_star: float,
                       event: Event = Event(), move: Movement = Movement(),
                       max_jobs: int = 14) -> NodeKneeResult:
    if len(pop) > max_jobs:
        raise ValueError(f"exact oracle is limited to {max_jobs} jobs")
    budget0, draws = _budgets(pop, pool, event, move), _draws(pop, pool, imp)
    best, best_short = None, None
    for acts in product("NRS", repeat=len(pop)):
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
        res = _result(pop, pool, imp, yR, yS, float(imp.c_replay @ yR + imp.c_transfer @ yS), "exact_oracle", s_star)
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
        _result(pop, pool, imp, base.y_R, base.y_S, base.cost, "additive_lp", s_star),
        solve_tangent_lp(pop, pool, imp, s_star, event, move),
        solve_active_knee_lp(pop, pool, imp, s_star, event, move),
        solve_live_greedy(pop, pool, imp, s_star, event, move),
        solve_node_drain_greedy(pop, pool, imp, s_star, event, move),
        solve_random_jobs(pop, pool, imp, s_star, event, move),
        solve_random_nodes(pop, pool, imp, s_star, event, move),
    ]
    if len(pop) <= 14:
        results.append(solve_exact_oracle(pop, pool, imp, s_star, event, move))
    fields = (
        "method", "cost", "feasible", "movement_feasible", "surrogate_feasible",
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
