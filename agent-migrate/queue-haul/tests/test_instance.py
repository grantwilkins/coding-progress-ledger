import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instance import JobPopulation, Workload, _draw, generate
from power import PoolPower


def test_marginals():
    wl = Workload()
    pop = _draw(np.random.default_rng(0), 40000, wl, "bf16")
    assert pop.T.mean() == pytest.approx(65800, rel=0.10)
    for s, p in zip(("active", "idle", "cold"), wl.state_mix):
        assert (pop.state == s).mean() == pytest.approx(p, abs=0.02)
    assert (pop.job_type == "agentic").mean() == pytest.approx(0.5, abs=0.02)
    agentic = pop.job_type == "agentic"
    assert pop.is_reasoning[agentic].mean() == pytest.approx(0.3, abs=0.03)
    assert not pop.is_reasoning[~agentic].any()


def test_within_class_load_heterogeneity():
    # Rate σ (plus Δ, Y spread) gives a continuous ℓ within one class, so greedy can
    # diverge from LP at constraint boundaries (T5/T7) instead of collapsing to a point.
    pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
    sel = (pop.state == "active") & (pop.job_type == "agentic") & ~pop.is_reasoning
    ell = pop.ell[sel]
    assert ell.std() / ell.mean() > 0.2  # meaningful spread, not a point mass


def test_cold_idle_carry_no_load_but_keep_kv():
    pop = generate(PoolPower())
    assert np.all(pop.ell[pop.state != "active"] == 0)
    assert np.all(pop.ell[pop.state == "active"] > 0)
    assert np.all(pop.m > 0)


def test_loads_returned_unsummed():
    pop = generate(PoolPower())
    assert np.allclose(pop.ell, pop.ell_pre + pop.ell_dec)
    assert "ell" not in {f.name for f in fields(JobPopulation)}


def test_population_size_is_an_input():
    pool = PoolPower()
    wl = Workload()
    assert len(generate(pool, wl)) == round(wl.alpha * 32 * pool.s_node)
    doubled = generate(pool, replace(wl, alpha=2 * wl.alpha))
    assert len(doubled) == pytest.approx(2 * len(generate(pool, wl)), rel=0.01)


def test_regime_is_measured_not_imposed():
    pool = PoolPower()  # center: E[T]=65800 matches Workload center
    pop = generate(pool)
    assert pool.memory_bound(pop.ell.sum(), len(pop))  # memory binds at center
    short_pool = replace(pool, mean_context_tokens=3378)
    short_wl = replace(Workload(), t_mix=((1.0, 8.0, 0.5),))  # E[T] ≈ 3378
    short = generate(short_pool, short_wl)
    assert not short_pool.memory_bound(short.ell.sum(), len(short))  # load binds


def test_generate_hard_fails():
    pool = PoolPower()
    with pytest.raises(ValueError):  # cap matches neither precision → G undefined
        generate(replace(pool, cap_gb=200.0))
    with pytest.raises(ValueError):  # pool context scale ≠ workload E[T]
        generate(replace(pool, mean_context_tokens=3000.0))


def test_precision_toggle_shifts_only_decode():
    wl = Workload()
    bf16 = _draw(np.random.default_rng(1), 5000, wl, "bf16")
    fp8 = _draw(np.random.default_rng(1), 5000, wl, "fp8")
    assert np.array_equal(bf16.T, fp8.T)
    assert np.allclose(bf16.ell_pre, fp8.ell_pre)
    assert np.array_equal(bf16.m, fp8.m)
    active = bf16.state == "active"
    assert np.allclose(fp8.ell_dec[active], bf16.ell_dec[active] * wl.g_bf16 / wl.g_fp8)
