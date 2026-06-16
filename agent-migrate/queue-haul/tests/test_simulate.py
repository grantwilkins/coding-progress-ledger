import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import Event, Plan, bind_dp, solve
from impact import Impact, Movement, compute
from instance import JobPopulation, generate
from power import PoolPower, rho_dest

POOL = PoolPower()
SLACK_L = replace(Movement(), lambda_src=1e18)  # egress instantaneous
SLACK_R = replace(Movement(), mu_in=1e18)  # rebuild (ingest) instantaneous

import simulate as sim_mod
from simulate import simulate


def _pop(n_nodes=4):
    pop = generate(POOL, n_nodes=n_nodes)
    return pop, compute(pop, POOL)


def _plan(yR, yS):
    yR, yS = np.asarray(yR, float), np.asarray(yS, float)
    return Plan(yR, yS, 0.0, 0.0, 0.0, True, 0.0, "load", "test")


def _recurrence(order, p1, p2, t1, t2):
    """Textbook 2-machine flow-shop makespan (single stage-2 server) — independent oracle."""
    for j in order:
        t1 += p1[j]
        t2 = max(t2, t1) + p2[j]
    return t2


# ---- Layer 1: single-isolated-resource plans reproduce the LP budget at equality ----

def test_egress_isolation_matches_lp_budget():
    pop, imp = _pop()
    pick = np.arange(5)
    yS = np.zeros(len(pop)); yS[pick] = 1.0
    plan = _plan(np.zeros(len(pop)), yS)
    move = SLACK_R  # only the λ_src=1e9 link binds
    s = simulate(pop, POOL, imp, plan, Event(W=10**6), move, discipline="fifo")
    expected = Event().tau_src + (imp.b_transfer[pick] / move.lambda_src).sum()
    assert np.nanmax(s.egress_done) == pytest.approx(expected, rel=1e-9)


def test_prefill_isolation_matches_lp_budget():
    pop, imp = _pop()
    pick = np.arange(5)
    yR = np.zeros(len(pop)); yR[pick] = 1.0
    plan = _plan(yR, np.zeros(len(pop)))
    event = Event(W=1)  # equality regime: single serial prefill server
    s = simulate(pop, POOL, imp, plan, event, SLACK_L, discipline="fifo")
    reb = pop.T / rho_dest(pop.T, pop.mfu)
    assert s.makespan == pytest.approx(event.tau_pre + reb[pick].sum(), rel=1e-9)


def test_ingest_isolation_matches_lp_budget():
    pop, imp = _pop()
    pick = np.arange(5)
    yS = np.zeros(len(pop)); yS[pick] = 1.0
    plan = _plan(np.zeros(len(pop)), yS)
    event = Event(W=1)  # equality regime: single serial ingest channel
    s = simulate(pop, POOL, imp, plan, event, SLACK_L, discipline="fifo")
    expected = event.tau_in + (imp.b_transfer[pick] / Movement().mu_in).sum()
    assert s.makespan == pytest.approx(expected, rel=1e-9)


# ---- Layer 2: hand-computed precedence cases (synthetic, round numbers) ----

def _toy(n):
    z = np.zeros(n)
    pop = JobPopulation(np.array(["chat"] * n), np.array(["active"] * n),
                        np.zeros(n, bool), np.full(n, 1e4), z, z, z, "bf16", 0.35)
    return pop


def _toy_imp(b_transfer):
    o = np.ones(len(b_transfer))
    z = np.zeros(len(b_transfer))
    return Impact(o, o, o, o, z, z, z, np.asarray(b_transfer, float), "load")


def test_hand_computed_two_job_flow_shop():
    # transfers, λ=μ=1e9 ⇒ p1=[2,1], p2=[2,1]; τ=0, W=1, FIFO order [0,1]
    pop = _toy(2)
    imp = _toy_imp([2e9, 1e9])
    plan = _plan([0.0, 0.0], [1.0, 1.0])
    event = Event(W=1, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    move = replace(Movement(), lambda_src=1e9, mu_in=1e9)
    s = simulate(pop, POOL, imp, plan, event, move, discipline="fifo")
    # link: ed=[2,3]; ingest W=1: rd0=max(2,2)+2... = max(2,max(2,0)+2)=4; rd1=max(3,max(3,4)+1)=5
    assert s.egress_done.tolist() == [2.0, 3.0]
    assert s.rebuild_done.tolist() == [4.0, 5.0]
    assert s.makespan == 5.0


def test_cutthrough_overlaps_egress():
    # same toy, cut-through: rebuild may start at egress_start, capped by egress_done
    pop = _toy(2)
    imp = _toy_imp([2e9, 1e9])
    plan = _plan([0.0, 0.0], [1.0, 1.0])
    event = Event(W=1, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    move = replace(Movement(), lambda_src=1e9, mu_in=1e9)
    s = simulate(pop, POOL, imp, plan, event, move, mode="cutthrough", discipline="fifo")
    # rd0=max(ed0=2, rs0=0 +p2=2)=2; rd1=max(ed1=3, max(es1=2,rd0=2)+1=3)=3
    assert s.rebuild_done.tolist() == [2.0, 3.0]


# ---- Layer 3: conservation invariants + Johnson==DES at W=1 single-action ----

def test_w1_single_action_equals_recurrence_and_johnson_optimal():
    pop, imp = _pop()
    yR = np.ones(len(pop))
    plan = _plan(yR, np.zeros(len(pop)))
    event, move = Event(W=1), Movement()
    p1 = yR * imp.b_replay / move.lambda_src
    p2 = pop.T / rho_dest(pop.T, pop.mfu)
    mv = np.flatnonzero(plan.y > 1e-9)
    ms = {}
    for disc in ("fifo", "lpt", "johnson", "pd"):
        s = simulate(pop, POOL, imp, plan, event, move, discipline=disc)
        order = mv[np.argsort(s.egress_start[mv])]
        assert s.makespan == pytest.approx(
            _recurrence(order, p1, p2, event.tau_src, event.tau_pre), rel=1e-9)
        ms[disc] = s.makespan
    assert ms["johnson"] <= min(ms["fifo"], ms["lpt"]) + 1e-9  # makespan-optimal


@pytest.mark.parametrize("mode", ["sf", "cutthrough"])
@pytest.mark.parametrize("disc", ["fifo", "lpt", "johnson", "pd"])
def test_conservation(disc, mode):
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.3 * bind_dp(imp).sum(), event, move)
    s = simulate(pop, POOL, imp, plan, event, move, mode=mode, discipline=disc)
    mv = plan.y > 1e-9
    assert np.all(s.rebuild_done[mv] >= s.egress_done[mv] - 1e-9)
    assert s.analytic_lb <= s.makespan + 1e-6
    assert s.makespan <= s.analytic_ub + 1e-6
    assert s.realized_shed <= plan.shed_guaranteed + 1e-6
    assert s.reconstruction_shed <= s.realized_shed + 1e-6  # rebuild_ok ⊆ egress_ok
    o = np.argsort(s.egress_start[mv])  # serial link never overlaps
    es, ed = s.egress_start[mv][o], s.egress_done[mv][o]
    assert np.all(es[1:] >= ed[:-1] - 1e-9)


def test_cutthrough_never_slower_than_store_and_forward():
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.3 * bind_dp(imp).sum(), event, move)
    mv = plan.y > 1e-9
    for disc in ("fifo", "lpt", "johnson", "pd"):
        sf = simulate(pop, POOL, imp, plan, event, move, "sf", disc)
        ct = simulate(pop, POOL, imp, plan, event, move, "cutthrough", disc)
        assert np.all(ct.rebuild_done[mv] <= sf.rebuild_done[mv] + 1e-9)


def test_power_density_banks_most_realized_shed():
    # Tight deadline ⇒ only a prefix of the serial link clears; PD (value-density first)
    # banks more watts by D than the makespan-oriented disciplines (order-blindness cost).
    pop, imp = _pop(n_nodes=8)
    base, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.5 * bind_dp(imp).sum(), base, move)
    egress = (plan.y_R @ imp.b_replay + plan.y_S @ imp.b_transfer) / move.lambda_src
    event = replace(base, D=base.tau_src + 0.5 * egress)  # cut the link mid-stream
    pd = simulate(pop, POOL, imp, plan, event, move, discipline="pd").realized_shed
    for disc in ("fifo", "lpt", "johnson"):
        other = simulate(pop, POOL, imp, plan, event, move, discipline=disc).realized_shed
        assert pd >= other - 1e-6
