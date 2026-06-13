import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impact import Movement, compute
from instance import JobPopulation, Workload, _draw, generate
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_dest


def _pop(T, ell_pre=0.0, ell_dec=0.0, klass="agentic", state="active", reasoning=False, mfu=0.35):
    T = np.atleast_1d(np.asarray(T, float))
    n = len(T)
    col = lambda v: v if isinstance(v, np.ndarray) else np.full(n, v)
    return JobPopulation(col(klass), col(state), col(reasoning), T, col(ell_pre), col(ell_dec),
                         ETA_BYTES_PER_TOK * T, "bf16", mfu)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def _rebuild_per_tok(T):
    pop = _pop(T)
    ship = BETA_BYTES_PER_TOK * pop.T / Movement().lambda_src
    return (compute(pop, PoolPower()).c_replay - ship) / pop.T


def test_rebuild_flat_then_one_over_rho():
    # ρ_dest(T) is a function: rebuild/tok ≈ flat below T*≈29k, steepens toward ∝T above.
    pt = _rebuild_per_tok(np.array([1e4, 2e4, 1e5, 2e5]))
    short, long = pt[1] / pt[0], pt[3] / pt[2]  # per-tok response to doubling T
    assert short < 1.35  # short regime: rate ≈ flat, cost barely moves
    assert long > 1.7  # long regime: rate decays ~1/T, doubling ~doubles cost
    assert long > 1.35 * short  # the regime contrast (steepening, not constant-F)
    assert pt[2] > 2 * pt[0]  # not constant: 100k far above 10k


def test_two_price_opposite_sign_by_class():
    pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
    imp = compute(pop, PoolPower())
    act = pop.state == "active"
    ag = act & (pop.klass == "agentic") & ~pop.is_reasoning  # prefill-skewed
    ch = act & (pop.klass == "chat")  # decode-skewed
    assert imp.dp_expected[ag].mean() < imp.dp_expected_single[ag].mean()  # prefill discount
    assert imp.dp_expected[ch].mean() > imp.dp_expected_single[ch].mean()  # decode premium


def test_two_price_closure_for_balanced_phase():
    # ℓ_pre == ℓ_dec ⇒ two-price == single, verifying T1's (p̄_pre+p̄_dec)/2 = p̄ split.
    imp = compute(_pop([5e4], ell_pre=0.1, ell_dec=0.1), PoolPower())
    assert imp.dp_expected[0] == pytest.approx(imp.dp_expected_single[0])


def test_memory_score():
    pool = PoolPower()
    assert compute(_pop([pool.mean_context_tokens]), pool).dp_memory[0] == pytest.approx(pool.mu)
    Ts = np.array([1e3, 9e4, 3e4, 2e5, 5e3])
    assert np.array_equal(np.argsort(compute(_pop(Ts), pool).dp_memory), np.argsort(Ts))
    dm = compute(_draw(np.random.default_rng(0), 40000, Workload(), "bf16"), pool).dp_memory
    assert dm.std() / dm.mean() > 0.5  # tail-heavy T → wide spread around μ


def test_congestion_only_on_rebuild():
    pool, move = PoolPower(), Movement()
    pop = _pop(np.array([1e3, 5e4, 2e5]))
    ship = BETA_BYTES_PER_TOK * pop.T / move.lambda_src
    bare = pop.T / rho_dest(pop.T, pop.mfu)
    reb = compute(pop, pool, move).c_replay - ship  # ship term carries no multiplier
    assert np.allclose(reb, (1 + congestion(move.dest_prefill_util)) * bare)


def test_congestion_asymmetry_is_independent():
    pool, pop = PoolPower(), _pop(np.array([5e4, 1e5]))
    base = compute(pop, pool, Movement())
    ship_s = ETA_BYTES_PER_TOK * pop.T / Movement().lambda_src
    ingest = ETA_BYTES_PER_TOK * pop.T / Movement().mu_in
    assert np.allclose(base.c_transfer - ship_s, ingest)  # φ_in=0: no ingest congestion at default
    hi_in = compute(pop, pool, replace(Movement(), dest_ingest_util=0.5))
    assert np.all(hi_in.c_transfer > base.c_transfer) and np.allclose(hi_in.c_replay, base.c_replay)
    hi_pre = compute(pop, pool, replace(Movement(), dest_prefill_util=0.8))
    assert np.all(hi_pre.c_replay > base.c_replay) and np.allclose(hi_pre.c_transfer, base.c_transfer)


def test_egress_bytes():
    imp = compute(_pop(np.array([1e3, 1e5])), PoolPower())
    assert np.allclose(imp.b_replay, BETA_BYTES_PER_TOK * np.array([1e3, 1e5]))
    assert np.allclose(imp.b_transfer, ETA_BYTES_PER_TOK * np.array([1e3, 1e5]))
    assert ETA_BYTES_PER_TOK / BETA_BYTES_PER_TOK == pytest.approx(48128)  # transfer ships ~48k× more


def test_regime_flag_matches_pool():
    pool = PoolPower()
    assert compute(generate(pool), pool).regime == "memory"
    short_pool = replace(pool, mean_context_tokens=3378)
    short = generate(short_pool, replace(Workload(), t_mix=((1.0, 8.0, 0.5),)))
    assert compute(short, short_pool).regime == "load"


def test_ranking_diagnostic_is_finite_not_signed():
    # Measured, not asserted: T2 draws T independent of Δ/Y, so load- vs memory-ranking
    # may genuinely disagree — a near-zero ρ is the finding, surfaced in T8, not a failure.
    pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
    imp = compute(pop, PoolPower())
    act = pop.state == "active"
    rho = _spearman(imp.dp_expected[act], imp.dp_memory[act])
    assert np.isfinite(rho) and -1 <= rho <= 1


def test_units_seconds_and_watts():
    imp = compute(generate(PoolPower()), PoolPower())
    for c in (imp.c_replay, imp.c_transfer):
        assert np.all(np.isfinite(c)) and np.all(c > 0)  # seconds
    for d in (imp.dp_guaranteed, imp.dp_expected, imp.dp_expected_single, imp.dp_memory):
        assert np.all(np.isfinite(d)) and np.all(d >= 0)  # watts
