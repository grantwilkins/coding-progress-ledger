"""T9 — the T8 regime-boundary results (load↔memory)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import solve
from impact import compute
from instance import Workload, _draw, _mean_T, generate
from power import PoolPower

WL = Workload()


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def _shift_tmix(target):
    c = np.log(target / _mean_T(WL))
    return tuple((w, mu + c, s) for w, mu, s in WL.t_mix)


def _R(pool, pop):
    return (len(pop) / pool.s_node) / (pop.ell.sum() / pool.rho_star)


def test_rankings_uncorrelated_at_center():
    # T2 draws T independent of Δ/Y, so the load ranking (dp_expected) and the memory
    # ranking (dp_memory) are near-uncorrelated — the substantive thing the flip exploits.
    pop = _draw(np.random.default_rng(0), 40000, WL, "bf16")
    imp = compute(pop, PoolPower())
    act = pop.state == "active"
    assert abs(_spearman(imp.dp_expected[act], imp.dp_memory[act])) < 0.2


def test_regime_flip_reorders_shed_set():
    # Forcing the load vs memory ranking on one near-boundary population sheds a
    # near-disjoint job set — the regime flip materially reorders the dispatch.
    pool = replace(PoolPower(), mean_context_tokens=13000.0)
    pop = generate(pool, replace(WL, t_mix=_shift_tmix(13000.0)), n_nodes=8, seed=0)
    imp = compute(pop, pool)
    yL = solve(pop, pool, replace(imp, regime="load"), 0.3 * imp.dp_guaranteed.sum()).y > 0.5
    yM = solve(pop, pool, replace(imp, regime="memory"), 0.3 * imp.dp_memory.sum()).y > 0.5
    assert (yL & yM).sum() / (yL | yM).sum() < 0.3


def test_crossover_at_memory_bound():
    # Both walks straddle R=1 and the measured regime flips exactly there (= the
    # N=max(L/ρ*, S_held/S_node) crossover): memory ⟺ R>1, for either knob — they agree.
    def walk_a(act):
        rest = 1 - act
        wl = replace(WL, state_mix=(act, 0.25 / 0.70 * rest, 0.45 / 0.70 * rest), t_mix=_shift_tmix(13000.0))
        pool = replace(PoolPower(), mean_context_tokens=13000.0)
        return pool, generate(pool, wl, n_nodes=4, seed=0)

    def walk_b(et):
        wl = replace(WL, t_mix=_shift_tmix(et))
        pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
        return pool, generate(pool, wl, n_nodes=4, seed=0)

    pts = [walk_a(0.05), walk_a(0.6), walk_b(5e3), walk_b(4e4)]
    Rs = [_R(p, pop) for p, pop in pts]
    assert min(Rs) < 1 < max(Rs)  # the walks bracket the crossover
    assert _R(*walk_a(0.05)) > 1 > _R(*walk_a(0.6))  # walk (a) straddles
    assert _R(*walk_b(5e3)) < 1 < _R(*walk_b(4e4))  # walk (b) straddles
    for pool, pop in pts:
        assert (compute(pop, pool).regime == "memory") == (_R(pool, pop) > 1)
