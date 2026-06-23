"""Claim:
Dispatch constraints account for movement resources and destination admission in
the same units as the LP rows.

Plausible wrong implementations:
- Charge held capacity as one session regardless of context size.
- Forget cold-session discount in held capacity.
- Use final-context marginal prefill time for full replay.
- Check only aggregate load, compare against 1.0, or silently clamp an oversized job.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import Event, Plan, bind_dp, greedy, random_dispatch, solve
from impact import Movement, compute
from instance import Workload, generate
from power import PoolPower, rho_replay

POOL = PoolPower()
SLACK_E = Event(D=1e9, W=10**7, dest_nodes=10**7)  # no movement limit binds
SLACK_M = replace(Movement(), lambda_src=1e18, mu_in=1e18)


def _pop(n_nodes=8):
    pop = generate(POOL, n_nodes=n_nodes)
    return pop, compute(pop, POOL)


def _violations(plan: Plan, pop, imp, event, move, pool=POOL):
    """Recompute every movement constraint from the plan; return the slack list (all ≥ −tol)."""
    yR, yS, y = plan.y_R, plan.y_S, plan.y
    reb = pop.T / rho_replay(pop.T, pop.mfu)
    held_w = (pop.T / pool.mean_context_tokens) * np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)
    return [
        1.0 - (yR + yS).max(),  # pairing y_R+y_S ≤ 1
        move.lambda_src * (event.D - event.tau_src) - (imp.b_replay @ yR + imp.b_transfer @ yS),
        event.W * (event.D - event.tau_pre) - reb @ yR,
        event.W * move.mu_in * (event.D - event.tau_in) - imp.b_transfer @ yS,
        event.l_dest(pool) - pop.ell @ y,
        event.s_dest(pool) - held_w @ y,
    ]


def test_lp_feasible_sheds_exactly():
    pop, imp = _pop()
    S = 0.3 * bind_dp(imp).sum()
    lp = solve(pop, POOL, imp, S, SLACK_E, SLACK_M)
    assert lp.feasible and lp.shortfall == 0.0
    assert lp.shed_guaranteed == pytest.approx(S, rel=1e-6)  # LP splits the last job


def test_single_job_above_setpoint_hard_fails():
    pop, imp = _pop(n_nodes=4)
    pop.ell_pre[0], pop.ell_dec[0] = POOL.rho_star, 0.0
    solve(pop, POOL, compute(pop, POOL), 1.0, SLACK_E, SLACK_M)
    pop.ell_pre[0] = POOL.rho_star + 1e-6
    imp = compute(pop, POOL)
    with pytest.raises(ValueError, match="rho_star"):
        solve(pop, POOL, imp, 1.0, SLACK_E, SLACK_M)
    with pytest.raises(ValueError, match="rho_star"):
        greedy(pop, POOL, imp, 1.0, SLACK_E, SLACK_M)


def test_milp_feasible_sheds_at_least_with_bounded_overshoot():
    pop, imp = _pop(n_nodes=4)  # smaller pop keeps the MIP fast
    dp = bind_dp(imp)
    S = 0.3 * dp.sum()
    mi = solve(pop, POOL, imp, S, SLACK_E, SLACK_M, integer=True)
    assert mi.feasible
    assert np.allclose(mi.y, np.round(mi.y))  # y ∈ {0,1}
    assert mi.shed_guaranteed >= S - 1e-6  # never under
    assert mi.shed_guaranteed - S <= dp[mi.y > 0.5].max() + 1e-6  # overshoot ≤ one job


@pytest.mark.parametrize("integer", [False, True])
def test_every_constraint_satisfied(integer):
    pop, imp = _pop(n_nodes=4)
    # default (tight) limits + a service-floor pin; S* high enough to push on them
    event = replace(Event(), pinned=("chat",))
    plan = solve(pop, POOL, imp, 0.10e6, event, Movement(), integer=integer)
    assert min(_violations(plan, pop, imp, event, Movement())) >= -1e-6
    assert np.all(plan.y[pop.job_type == "chat"] == 0.0)  # pinned never moved


def test_infeasible_returns_max_shed_and_shortfall():
    pop, imp = _pop()
    S = 0.30e6  # above the memory ceiling under default headroom
    plan = solve(pop, POOL, imp, S, Event(), Movement())
    assert not plan.feasible
    assert plan.shortfall == pytest.approx(S - plan.shed_guaranteed)
    assert plan.shortfall > 0
    assert min(_violations(plan, pop, imp, Event(), Movement())) >= -1e-6  # still legal


def test_greedy_equals_lp_off_boundary():
    # No resource constraint binds ⇒ both pick the cheapest-per-watt set and split
    # the last job, so they coincide (the T5 claim; greedy-vs-LP, not MILP).
    pop, imp = _pop()
    S = 0.3 * bind_dp(imp).sum()
    lp = solve(pop, POOL, imp, S, SLACK_E, SLACK_M)
    g = greedy(pop, POOL, imp, S, SLACK_E, SLACK_M)
    assert g.feasible
    assert g.shed_guaranteed == pytest.approx(lp.shed_guaranteed, rel=1e-6)
    assert g.cost == pytest.approx(lp.cost, rel=1e-4)


def test_greedy_respects_movement_budgets():
    # The decentralized greedy draws down the SHARED budgets as it accepts jobs, so
    # its plan must satisfy every movement constraint — it can never ship more than
    # the links carry (the bug this fixes: a resource-blind greedy shipped >link).
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8, W=4), Movement()
    plan = greedy(pop, POOL, imp, 0.30e6, event, move)  # S* beyond reach ⇒ budgets bind
    assert not plan.feasible and plan.shortfall > 0  # links cap the shed
    assert min(_violations(plan, pop, imp, event, move)) >= -1e-6  # but never over-subscribed


def test_greedy_ceiling_at_most_lp_ceiling():
    # Coordination value: the LP repacks to fit more shed under the same budgets, so
    # the myopic greedy can never out-shed it.
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8, W=4), Movement()
    g = greedy(pop, POOL, imp, 1e12, event, move)
    lp = solve(pop, POOL, imp, 2 * bind_dp(imp).sum(), event, move)
    assert g.shed_guaranteed <= lp.shed_guaranteed + 1e-6


def test_random_respects_budgets_and_bounded_by_lp():
    # The random floor uses the same budget-respecting engine: it can never
    # over-subscribe a link, and the LP max-shed bounds it (as it bounds any policy).
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8, W=4), Movement()
    r = random_dispatch(pop, POOL, imp, 0.30e6, event, move, seed=0)
    assert min(_violations(r, pop, imp, event, move)) >= -1e-6
    lp = solve(pop, POOL, imp, 2 * bind_dp(imp).sum(), event, move)
    assert r.shed_guaranteed <= lp.shed_guaranteed + 1e-6
    # deterministic for a fixed seed
    r2 = random_dispatch(pop, POOL, imp, 0.30e6, event, move, seed=0)
    assert np.array_equal(r.y, r2.y)


def test_lp_lower_bounds_milp():
    pop, imp = _pop(n_nodes=4)
    S = 0.3 * bind_dp(imp).sum()
    lp = solve(pop, POOL, imp, S, SLACK_E, SLACK_M)
    mi = solve(pop, POOL, imp, S, SLACK_E, SLACK_M, integer=True)
    assert lp.cost <= mi.cost + 1e-6  # relaxation lower-bounds the integer optimum


def test_bind_dp_picks_floor_by_regime():
    pop, imp = _pop()  # default population is memory-bound
    assert imp.regime == "memory"
    assert np.array_equal(bind_dp(imp), imp.dp_memory)

    sp = replace(POOL, mean_context_tokens=3378)
    short = generate(sp, replace(Workload(), t_mix=((1.0, 8.0, 0.5),), rate_agentic=0.075, rate_chat=0.01))
    impL = compute(short, sp)
    assert impL.regime == "load"
    assert np.array_equal(bind_dp(impL), impL.dp_guaranteed)
    # expected (amortized) ≥ guaranteed (plateau) on the same shed plan
    plan = solve(short, sp, impL, 0.3 * bind_dp(impL).sum(), SLACK_E, SLACK_M)
    assert plan.shed_expected >= plan.shed_guaranteed - 1e-6
