"""Claim:
The DES replays the solved plan with the same resource equations as dispatch.

Plausible wrong implementations:
- Use final-context marginal prefill rate instead of average full-replay rate.
- Drop one side of a split replay/transfer shipment.
- Let rebuild complete before full egress arrival.
- Keep stale source-node marginal values in node-aware ordering.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import DestFleet, Event, Plan, bind_dp, movement_columns, solve
from impact import Impact, Movement, compute
from instance import JobPopulation, generate
from power import PoolPower, rho_replay

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
    reb = pop.T / rho_replay(pop.T, pop.mfu)
    assert s.makespan == pytest.approx(event.tau_pre + reb[pick].sum(), rel=1e-9)


def test_replay_stage_uses_dispatch_prefill_column():
    pop, imp = _pop()
    j = 0
    yR = np.zeros(len(pop)); yR[j] = 1.0
    fleet = DestFleet(np.array([1]), np.array([32.0]), np.array([0.47]), np.array([0.0]))
    event = Event(W=1, tau_src=0.0, tau_pre=2.0, tau_in=0.0)
    plan = Plan(yR, np.zeros(len(pop)), 0.0, 0.0, 0.0, True, 0.0, "load", "test",
                yR[:, None], np.zeros((len(pop), 1)))
    s = simulate(pop, POOL, imp, plan, event, SLACK_L, discipline="fifo", fleet=fleet)
    expected = event.tau_pre + movement_columns(pop, POOL, imp, fleet, SLACK_L)["R"]["prefill"][j, 0]
    assert s.makespan == pytest.approx(expected, rel=1e-9)


def test_ingest_isolation_matches_lp_budget():
    pop, imp = _pop()
    pick = np.arange(5)
    yS = np.zeros(len(pop)); yS[pick] = 1.0
    plan = _plan(np.zeros(len(pop)), yS)
    event = Event(W=1)  # equality regime: single serial ingest channel
    move = replace(Movement(), lambda_src=1e18, mu_in=1e9)  # ingest work dominates τ_in
    s = simulate(pop, POOL, imp, plan, event, move, discipline="fifo")
    expected = event.tau_in + (imp.b_transfer[pick] / move.mu_in).sum()
    assert expected - event.tau_in > 10 * event.tau_in  # the test isn't just measuring τ_in
    assert s.makespan == pytest.approx(expected, rel=1e-9)


def test_split_job_charges_both_fractions():
    # The LP can split a marginal job (y_R>0 AND y_S>0). The DES must rebuild BOTH pieces —
    # prefill for y_R, ingest for y_S — not silently drop the minority action's work.
    pop, imp = _pop()
    j = int(np.argmax(pop.T))  # big context ⇒ prefill piece ≫ ingest piece
    yR, yS = np.zeros(len(pop)), np.zeros(len(pop))
    yR[j], yS[j] = 0.4, 0.6  # action() picks "S", yet the prefill (y_R) piece is far larger
    event, move = Event(W=1), Movement()
    s = simulate(pop, POOL, imp, _plan(yR, yS), event, move, discipline="fifo")
    ed = event.tau_src + (yR[j] * imp.b_replay[j] + yS[j] * imp.b_transfer[j]) / move.lambda_src
    p2R = yR[j] * pop.T[j] / rho_replay(pop.T[j], pop.mfu)
    p2S = yS[j] * imp.b_transfer[j] / move.mu_in
    assert p2R > p2S  # the dropped-fraction bug would have kept only the smaller ingest piece
    assert s.makespan == pytest.approx(ed + max(p2R, p2S), rel=1e-9)  # both pieces from ed (W=1)


# ---- Layer 2: hand-computed precedence cases (synthetic, round numbers) ----

def _toy(n):
    z = np.zeros(n)
    pop = JobPopulation(np.array(["chat"] * n), np.array(["ordinary_chat"] * n),
                        np.array(["active"] * n), np.zeros(n, bool), np.full(n, 1e4),
                        z, z, z, z, z, np.ones(n, bool), z, z, z, "bf16", 0.35)
    return pop


def _node_value_toy():
    z = np.zeros(3)
    ell = np.array([0.10, 0.05, 0.05])
    return JobPopulation(np.array(["chat"] * 3), np.array(["ordinary_chat"] * 3),
                         np.array(["active"] * 3), np.zeros(3, bool), np.full(3, 1e4),
                         z, z, z, z, z, np.ones(3, bool), ell, z, z, "bf16", 0.35,
                         np.array([0, 0, 1]))


def _toy_imp(b_transfer):
    o = np.ones(len(b_transfer))
    z = np.zeros(len(b_transfer))
    return Impact(o, o, o, o, o, z, z, z, np.asarray(b_transfer, float), "load")


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


def test_node_marginal_pd_updates_residual_source_node_value():
    # Node 0 starts above the power knee. Moving job 0 crosses into the high-slope ramp,
    # so job 1's value jumps and should beat the equal-sized job on node 1 by stable tie-break.
    pop = _node_value_toy()
    imp = _toy_imp([1e9, 1e9, 1e9])
    plan = _plan([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    event = Event(D=10.0, W=10, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    move = replace(Movement(), lambda_src=1e9, mu_in=1e18)
    s = simulate(pop, POOL, imp, plan, event, move, discipline="node_marginal_pd")
    assert np.argsort(s.egress_start).tolist() == [0, 1, 2]


def test_node_marginal_pd_requires_source_nodes():
    pop = _toy(1)
    imp = _toy_imp([1e9])
    event = Event(D=10.0, W=10, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    move = replace(Movement(), lambda_src=1e9)
    with pytest.raises(ValueError, match="source_node"):
        simulate(pop, POOL, imp, _plan([0.0], [1.0]), event, move, discipline="node_marginal_pd")


# ---- Layer 3: conservation invariants + Johnson==DES at W=1 single-action ----

def test_w1_single_action_equals_recurrence_and_johnson_optimal():
    pop, imp = _pop()
    yR = np.ones(len(pop))
    plan = _plan(yR, np.zeros(len(pop)))
    event, move = Event(W=1), Movement()
    p1 = yR * imp.b_replay / move.lambda_src
    p2 = pop.T / rho_replay(pop.T, pop.mfu)
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


@pytest.mark.parametrize("mode", ["sf", "cutthrough"])
def test_mode_switch_pins_rebuild_start(mode):
    # sf: rebuild can't start before egress completes; cut-through: not before egress starts.
    # Pins the floor switch (would catch a "zeroth vs full chunk" regression).
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.4 * bind_dp(imp).sum(), event, move)
    s = simulate(pop, POOL, imp, plan, event, move, mode=mode, discipline="fifo")
    mv = plan.y > 1e-9
    floor = s.egress_start if mode == "cutthrough" else s.egress_done
    assert np.all(s.rebuild_start[mv] >= floor[mv] - 1e-9)
    assert np.all(s.rebuild_done[mv] >= s.egress_done[mv] - 1e-9)  # precedence: rebuild ≥ egress done


def test_unknown_mode_raises():
    pop, imp = _pop()
    with pytest.raises(ValueError, match="unknown mode"):
        simulate(pop, POOL, imp, _plan(np.zeros(len(pop)), np.zeros(len(pop))), mode="sd")


def test_cutthrough_never_slower_than_store_and_forward():
    pop, imp = _pop(n_nodes=8)
    event, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.3 * bind_dp(imp).sum(), event, move)
    mv = plan.y > 1e-9
    for disc in ("fifo", "lpt", "johnson", "pd"):
        sf = simulate(pop, POOL, imp, plan, event, move, "sf", disc)
        ct = simulate(pop, POOL, imp, plan, event, move, "cutthrough", disc)
        assert np.all(ct.rebuild_done[mv] <= sf.rebuild_done[mv] + 1e-9)


def test_discipline_sensitivity_only_when_link_binds():
    # Realized shed is a knapsack-prefix problem on the serial link: order is irrelevant
    # when D is slack (everything clears) and order-sensitive when D binds. No discipline is
    # universally optimal — PD is a heuristic, not a dominant strategy.
    pop, imp = _pop(n_nodes=8)
    base, move = Event(dest_nodes=48, W=16), Movement()
    plan = solve(pop, POOL, imp, 0.5 * bind_dp(imp).sum(), base, move)
    eg = (plan.y_R @ imp.b_replay + plan.y_S @ imp.b_transfer) / move.lambda_src
    disc = ("fifo", "lpt", "johnson", "pd")
    slack = [simulate(pop, POOL, imp, plan, replace(base, D=base.tau_src + 2 * eg), move, discipline=d).realized_shed for d in disc]
    tight = [simulate(pop, POOL, imp, plan, replace(base, D=base.tau_src + 0.3 * eg), move, discipline=d).realized_shed for d in disc]
    assert max(slack) - min(slack) < 1e-6  # slack link ⇒ order-invariant
    assert max(tight) - min(tight) > 1e-6  # binding link ⇒ order matters
