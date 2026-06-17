"""T17 — multi-destination tests (formulation.md §4 ℓ-index; T13 dispatch + T15 per-ℓ DES).

The shared egress row is the whole multi-dest coupling; everything else is K independent
blocks. These pin: K=1 collapses to the single-dest program, the routing is shed-/cost-
indifferent across homogeneous destinations, θ_egress is the uplink-binding price, and the
per-ℓ DES respects admission and reproduces the certified routing when the plan is slack.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import DestFleet, Event, Movement, bind_dp, solve
from impact import compute, move_costs
from instance import Workload, generate
from power import PoolPower
from simulate import simulate

POOL = PoolPower()


def _setup(n_nodes=8):
    pop = generate(POOL, n_nodes=n_nodes)
    return pop, compute(pop, POOL)


def _homog_fleet(K, ev, mv, pop, spare_each):
    return DestFleet(np.full(K, ev.W), np.full(K, spare_each), np.full(K, pop.mfu),
                     np.full(K, mv.dest_prefill_util))


# (1) K=1 reduces to the single-dest solve at rel=1e-9 ----------------------------------------

def test_k1_reduces_to_single_dest():
    pop, imp = _setup()
    ev, mv = Event(dest_nodes=48, W=16), Movement()
    sstar = 0.4 * bind_dp(imp).sum()
    base = solve(pop, POOL, imp, sstar, ev, mv)  # fleet=None ⇒ frozen imp costs
    k1 = solve(pop, POOL, imp, sstar, ev, mv, fleet=DestFleet.from_event(ev, mv, POOL, pop))
    assert k1.Y_R.shape[1] == 1
    assert base.shed_guaranteed == pytest.approx(k1.shed_guaranteed, rel=1e-9)
    assert base.cost == pytest.approx(k1.cost, rel=1e-9)
    assert np.allclose(base.y_R, k1.Y_R[:, 0], rtol=1e-7, atol=1e-9)
    assert np.allclose(base.y_S, k1.Y_S[:, 0], rtol=1e-7, atol=1e-9)


# (2) homogeneous destinations ⇒ routing is cost-indifferent (any feasible split optimal) ------

def test_homogeneous_split_is_cost_indifferent():
    pop, imp = _setup()
    ev, mv = Event(dest_nodes=48, W=16), Movement()
    K = 3
    fl = _homog_fleet(K, ev, mv, pop, spare_each=0.4 * ev.dest_nodes / K)  # same aggregate cap as single-dest
    plan = solve(pop, POOL, imp, 0.4 * bind_dp(imp).sum(), ev, mv, fleet=fl)
    assert plan.feasible
    cR, cS, _ = move_costs(pop, fl, mv)
    assert np.allclose(cR, cR[:, :1]) and np.allclose(cS, cS[:, :1])  # identical columns ⇒ dest-blind cost
    for perm in ([2, 0, 1], [1, 2, 0]):  # any column permutation is an equally-optimal routing
        cost_perm = float((cR * plan.Y_R[:, perm] + cS * plan.Y_S[:, perm]).sum())
        assert cost_perm == pytest.approx(plan.cost, rel=1e-9)


# (3) shed is invariant to routing (Σ dp·y unchanged across destination permutations) ----------

def test_shed_invariant_to_routing_permutation():
    pop, imp = _setup()
    ev, mv = Event(dest_nodes=48, W=16), Movement()
    K = 4
    fl = DestFleet(np.full(K, ev.W), np.linspace(6, 14, K), np.linspace(0.3, 0.5, K),
                   np.full(K, mv.dest_prefill_util))
    plan = solve(pop, POOL, imp, 0.4 * bind_dp(imp).sum(), ev, mv, fleet=fl)
    dp, Y = bind_dp(imp), plan.Y_R + plan.Y_S
    for perm in ([3, 2, 1, 0], [1, 0, 3, 2]):
        assert dp @ Y[:, perm].sum(1) == pytest.approx(plan.shed_guaranteed, rel=1e-9)


# (4) θ_egress = 0 when the uplink is slack, > 0 when it binds (max-shed dual) ------------------

def test_theta_egress_zero_slack_positive_binding():
    pool = replace(POOL, mean_context_tokens=163000)  # long context ⇒ big η·T ⇒ uplink can bind
    pop = generate(pool, Workload(state_mix=(1.0, 0.0, 0.0), t_mix=((1.0, 12.0, 0.2),)), n_nodes=8)
    ev, mv = Event(D=300, tau_pre=300, W=8), Movement(lambda_src=1e9, mu_in=1e13)  # pure transfer
    imp = compute(pop, pool, mv)
    spare_bar = mv.lambda_src * (ev.D - ev.tau_src) / imp.b_transfer.mean() / (5 * pool.s_node)
    slack = solve(pop, pool, imp, 1e15, ev, mv, fleet=_homog_fleet(2, ev, mv, pop, 0.5 * spare_bar))
    binding = solve(pop, pool, imp, 1e15, ev, mv, fleet=_homog_fleet(12, ev, mv, pop, spare_bar))
    assert slack.theta_egress == pytest.approx(0.0, abs=1e-12)  # admission binds, uplink slack
    assert binding.theta_egress > 1e-12                          # Σspare exceeds uplink feed


# (5) the per-ℓ DES never admits more load than L̄_dest,ℓ ---------------------------------------

def test_des_never_exceeds_load_cap():
    pop, imp = _setup(n_nodes=8)
    ev, mv = Event(dest_nodes=48, W=16), Movement()
    K = 4
    fl = DestFleet(np.full(K, ev.W), np.linspace(8, 20, K), np.linspace(0.3, 0.5, K),
                   np.full(K, mv.dest_prefill_util))
    plan = solve(pop, POOL, imp, 0.5 * bind_dp(imp).sum(), ev, mv, fleet=fl)
    for D in (ev.D, 0.5 * ev.D, 0.3 * ev.D):  # realized ⊆ certified ⊆ cap at every deadline
        r = simulate(pop, POOL, imp, plan, replace(ev, D=D), mv, fleet=fl)
        assert np.all(r.realized_load <= r.load_cap + 1e-6)
        assert np.all(r.realized_load <= r.certified_load + 1e-6)


# (6) on a slack plan, realized routing == certified within by-D jobs --------------------------

def test_slack_plan_realized_routing_matches_certified():
    pop, _ = _setup(n_nodes=8)
    ev = Event(dest_nodes=48, W=64, D=1e6)
    mv = Movement(lambda_src=1e12, mu_in=1e14)  # links so fast everything rebuilds well before D
    imp = compute(pop, POOL, mv)
    K = 4
    fl = DestFleet(np.full(K, ev.W), np.linspace(8, 20, K), np.linspace(0.3, 0.5, K),
                   np.full(K, mv.dest_prefill_util))
    plan = solve(pop, POOL, imp, 0.5 * bind_dp(imp).sum(), ev, mv, fleet=fl)
    r = simulate(pop, POOL, imp, plan, ev, mv, fleet=fl)
    assert r.makespan < ev.D
    assert np.allclose(r.realized_load, r.certified_load, rtol=1e-9, atol=1e-12)
