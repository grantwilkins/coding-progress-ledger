"""Claim:
Dispatch constraints account for movement resources and destination admission in
the same units as the LP rows.

Plausible wrong implementations:
- Charge held capacity as one session regardless of context size.
- Forget cold-session discount in held capacity.
- Use final-context marginal prefill time for full replay.
- Check only aggregate load, compare against 1.0, or silently clamp an oversized job.
- Treat an unreachable tight-deadline target as a hard solver failure instead of
  returning the legal max-shed plan.
- Split the marginal greedy job fractionally instead of taking whole jobs.
- Report diagnostic ratios from consumed resources instead of LP row budgets.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import (DestFleet, Event, Plan, bind_dp, deadline_infeasible, dispatch_diagnostics,
                      greedy, movement_draws, movement_used, random_dispatch,
                      single_movement_budgets, solve)
from simulate import simulate
from impact import Impact, Movement, compute
from instance import JobPopulation, Workload, _mean_T, class_workload, generate
from power import PoolPower, rho_replay

POOL = PoolPower()
SLACK_E = Event(D=1e9, dest_nodes=10**7)  # no movement limit binds
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
        np.floor(event.spare_frac * event.dest_nodes) * (event.D - event.tau_pre) - reb @ yR,
        np.floor(event.spare_frac * event.dest_nodes) * move.mu_in * (event.D - event.tau_in) - imp.b_transfer @ yS,
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


def test_tight_deadline_returns_max_shed_instead_of_crashing():
    wl = class_workload("ordinary_chat", state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(POOL, mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=2)
    imp = compute(pop, pool)
    S = 2 * bind_dp(imp).sum()
    tight = solve(pop, pool, imp, S, Event(D=8, dest_nodes=48), Movement())
    loose = solve(pop, pool, imp, S, Event(D=30, dest_nodes=48), Movement())
    assert not tight.feasible and tight.shed_guaranteed > 0
    assert tight.shed_guaranteed <= loose.shed_guaranteed + 1e-6


def test_integer_greedy_overshoots_off_boundary_and_lp_lower_bounds():
    # No resource constraint binds: the LP can split the last job exactly, but the
    # deployable greedy baseline must take whole jobs and can overshoot by at most one job.
    pop, imp = _pop()
    S = 0.3 * bind_dp(imp).sum()
    lp = solve(pop, POOL, imp, S, SLACK_E, SLACK_M)
    g = greedy(pop, POOL, imp, S, SLACK_E, SLACK_M)
    dp = bind_dp(imp)
    assert g.feasible
    assert np.allclose(g.y, np.round(g.y))
    assert g.shed_guaranteed >= S - 1e-6
    assert g.shed_guaranteed - S <= dp[g.y > 0.5].max() + 1e-6
    assert lp.cost <= g.cost + 1e-6


def test_greedy_respects_movement_budgets():
    # The decentralized greedy draws down the SHARED budgets as it accepts jobs, so
    # its plan must satisfy every movement constraint — it can never ship more than
    # the links carry.
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8), Movement()
    plan = greedy(pop, POOL, imp, 0.30e6, event, move)  # S* beyond reach ⇒ budgets bind
    assert not plan.feasible and plan.shortfall > 0  # links cap the shed
    assert min(_violations(plan, pop, imp, event, move)) >= -1e-6  # but never over-subscribed


def test_greedy_falls_back_from_partial_preferred_action():
    pop = JobPopulation(
        np.array(["agentic", "agentic"]),
        np.array(["agentic_tool_loop", "agentic_tool_loop"]),
        np.array(["active", "active"]),
        np.array([False, False]),
        np.array([1.0, 1.0]),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.array([True, True]),
        np.array([0.01, 0.01]),
        np.zeros(2),
        np.ones(2),
        "bf16",
        0.35,
    )
    imp = Impact(
        np.ones(2), np.ones(2), np.ones(2), np.ones(2), np.ones(2),
        np.full(2, 2.0), np.ones(2), np.ones(2), np.full(2, 100.0), "load"
    )
    event = Event(D=12, dest_nodes=1000, spare_frac=1.0)
    move = replace(Movement(), lambda_src=0.2)
    g = greedy(pop, POOL, imp, 2.0, event, move)
    mi = solve(pop, POOL, imp, 2.0, event, move, integer=True)
    assert np.array_equal(g.y_R, np.ones(2))
    assert np.array_equal(g.y_S, np.zeros(2))
    assert g.shed_guaranteed == pytest.approx(mi.shed_guaranteed)


def test_greedy_ceiling_at_most_lp_ceiling():
    # LP max-shed is an upper bound on any budget-respecting first-fit policy.
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8), Movement()
    g = greedy(pop, POOL, imp, 1e12, event, move)
    lp = solve(pop, POOL, imp, 2 * bind_dp(imp).sum(), event, move)
    assert g.shed_guaranteed <= lp.shed_guaranteed + 1e-6


def test_dispatch_diagnostics_report_row_level_quantities():
    n = 3
    pop = JobPopulation(
        np.array(["chat"] * n),
        np.array(["ordinary_chat"] * n),
        np.array(["active"] * n),
        np.zeros(n, bool),
        np.array([1.0, 2.0, 3.0]),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n, bool),
        np.array([0.1, 0.2, 0.3]),
        np.zeros(n),
        np.ones(n),
        "bf16",
        0.35,
    )
    imp = Impact(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([10.0, 20.0, 30.0]),
        "load",
    )
    plan = Plan(
        np.array([1.0, 0.5, 0.0]),
        np.zeros(n),
        2.0,
        2.0,
        4.0,
        True,
        0.0,
        "load",
        "lp",
        resource_duals={k: np.array([v]) for k, v in {
            "egress": 7.0, "prefill": 0.0, "ingest": 0.0, "load": 2.0, "held": 3.0
        }.items()},
    )
    event = Event(D=12, dest_nodes=1000, spare_frac=1.0)
    move = replace(Movement(), lambda_src=2.0)
    d = dispatch_diagnostics(pop, POOL, imp, plan, 6.0, event, move)
    assert d["active_constraints"] == ("egress",)
    assert d["fractional_variables"] == 1
    assert d["max_dp_over_s"] == pytest.approx(0.5)
    assert d["max_resource_draw_over_budget"]["egress"] == pytest.approx(1.5)
    assert d["duals"]["egress"] == pytest.approx(7.0)
    assert d["spearman"]["cost"] == pytest.approx(-1.0)
    assert d["spearman"]["egress"] == pytest.approx(1.0)
    assert d["spearman"]["load"] == pytest.approx(1.0)


def test_random_respects_budgets_and_bounded_by_lp():
    # The random floor uses the same budget-respecting engine: it can never
    # over-subscribe a link, and the LP max-shed bounds it (as it bounds any policy).
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=8), Movement()
    r = random_dispatch(pop, POOL, imp, 0.30e6, event, move, seed=0)
    assert min(_violations(r, pop, imp, event, move)) >= -1e-3
    lp = solve(pop, POOL, imp, 2 * bind_dp(imp).sum(), event, move)
    assert r.shed_guaranteed <= lp.shed_guaranteed + 1e-6
    # deterministic for a fixed seed
    r2 = random_dispatch(pop, POOL, imp, 0.30e6, event, move, seed=0)
    assert np.array_equal(r.y, r2.y)


def test_kappa_derates_planner_rebuild_rows_only():
    # Pure-transfer, ingest-bound max-shed instance: at kappa=1 the plan fills the physical
    # ingest budget; at kappa=0.5 it must stay inside half of it, and max shed can only drop.
    # D is wide enough that single sessions pass the deadline filter; the AGGREGATE row binds.
    pop, imp = _pop(n_nodes=4)
    event = Event(D=40, dest_nodes=8, tau_pre=40.0)  # prefill window 0 ⇒ transfers only
    move = replace(Movement(), lambda_src=1e18, mu_in=1e9)  # link slack, ingest binds
    S = 1e15
    budgets = single_movement_budgets(POOL, event, move)  # physical (kappa=1) budgets
    full = solve(pop, POOL, imp, S, event, move)
    cut = solve(pop, POOL, imp, S, event, move, kappa=0.5)
    draws = movement_draws(pop, POOL, imp, event, move)
    tol = 1e-6 * budgets["ingest"]
    assert movement_used(draws, full.y_R, full.y_S)["ingest"] > 0.5 * budgets["ingest"] + tol
    assert movement_used(draws, cut.y_R, cut.y_S)["ingest"] <= 0.5 * budgets["ingest"] + tol
    assert cut.shed_guaranteed <= full.shed_guaranteed + 1e-6  # tighter RHS never gains shed
    with pytest.raises(ValueError, match="kappa"):
        solve(pop, POOL, imp, S, event, move, kappa=0.0)


@pytest.mark.parametrize("mode", ["sf", "cutthrough"])
def test_deadline_filter_matches_single_session_des(mode):
    # The filter's defining property, both directions: a session that passes completes by D
    # when played ALONE through the DES in that mode; a banned session played alone misses D.
    pop, imp = _pop(n_nodes=4)
    event = Event(D=18, dest_nodes=8)
    move = replace(Movement(), mu_in=1e9)  # slow ingest ⇒ long-context transfers get banned
    badR, badS = deadline_infeasible(pop, imp, DestFleet.from_event(event, move, POOL, pop),
                                     event, move, mode)
    assert badR[:, 0].any() and badS[:, 0].any() and (~badS[:, 0]).any()  # non-vacuous both ways
    z = np.zeros(len(pop))
    for j in range(len(pop)):
        for bad, (yR, yS) in ((badR[j, 0], (1.0, 0.0)), (badS[j, 0], (0.0, 1.0))):
            one = z.copy(); one[j] = 1.0
            plan = Plan(yR * one, yS * one, 0.0, 0.0, 0.0, True, 0.0, "load", "test")
            s = simulate(pop, POOL, imp, plan, event, move, mode=mode, discipline="fifo")
            assert (s.rebuild_done[j] > event.D) == bad


def test_planner_and_greedy_never_pick_deadline_banned_sessions():
    pop, imp = _pop(n_nodes=4)
    event = Event(D=18, dest_nodes=8)
    move = replace(Movement(), mu_in=1e9)
    badR, badS = deadline_infeasible(pop, imp, DestFleet.from_event(event, move, POOL, pop),
                                     event, move)
    for plan in (solve(pop, POOL, imp, 1e15, event, move),
                 greedy(pop, POOL, imp, 1e15, event, move),
                 random_dispatch(pop, POOL, imp, 1e15, event, move)):
        assert np.all(plan.y_R[badR[:, 0]] < 1e-9)
        assert np.all(plan.y_S[badS[:, 0]] < 1e-9)


def test_lp_lower_bounds_milp():
    pop, imp = _pop(n_nodes=4)
    S = 0.3 * bind_dp(imp).sum()
    lp = solve(pop, POOL, imp, S, SLACK_E, SLACK_M)
    mi = solve(pop, POOL, imp, S, SLACK_E, SLACK_M, integer=True)
    assert lp.cost <= mi.cost + 1e-6  # relaxation lower-bounds the integer optimum


def test_bind_dp_uses_active_power_not_memory_regime():
    pop, imp = _pop()  # default population is memory-bound
    assert imp.regime == "memory"
    assert np.array_equal(bind_dp(imp), imp.dp_certified)

    sp = replace(POOL, mean_context_tokens=3378)
    short = generate(sp, replace(Workload(), t_mix=((1.0, 8.0, 0.5),), rate_means=(0.01, 0.01, 0.01, 0.075)))
    impL = compute(short, sp)
    assert impL.regime == "load"
    assert np.array_equal(bind_dp(impL), impL.dp_certified)
    cold = generate(POOL, replace(Workload(), state_mix=(0.0, 0.0, 1.0)))
    assert bind_dp(compute(cold, POOL)).sum() == pytest.approx(0.0)
    # expected (amortized) ≥ guaranteed (plateau) on the same shed plan
    plan = solve(short, sp, impL, 0.3 * bind_dp(impL).sum(), SLACK_E, SLACK_M)
    assert plan.shed_expected >= plan.shed_guaranteed - 1e-6
