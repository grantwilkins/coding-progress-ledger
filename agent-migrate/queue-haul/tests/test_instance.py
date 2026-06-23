"""Claim:
The generator samples session classes and raw turn quantities, then derives load
from the stated equations in turns-correction.md.

Plausible wrong implementations:
- Draw every class from one shared context distribution.
- Store raw turn fields but compute ell from different hidden quantities.
- Treat cache misses as Delta-only prefill work.
- Let idle/cold sessions keep nonzero current turn rate or load.
- Let precision change prefill work instead of decode normalization only.
- Let rare long turns make one session consume more than its occupation cap.
"""

import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instance import SESSION_CLASSES, JobPopulation, Workload, _draw, class_workload, generate
from power import PoolPower, rho_dest


def test_marginals():
    wl = Workload()
    pop = _draw(np.random.default_rng(0), 40000, wl, "bf16")
    assert pop.T.mean() == pytest.approx(65800, rel=0.10)
    for s, p in zip(("active", "idle", "cold"), wl.state_mix):
        assert (pop.state == s).mean() == pytest.approx(p, abs=0.02)
    assert set(pop.session_class) == {"agentic_tool_loop"}
    assert np.all(pop.job_type == "agentic")
    assert not pop.is_reasoning.any()


def test_class_mix_and_context_are_class_specific():
    wl = replace(Workload(), class_mix=(0.25, 0.25, 0.25, 0.25))
    pop = _draw(np.random.default_rng(0), 40000, wl, "bf16")
    for cls in SESSION_CLASSES:
        assert (pop.session_class == cls).mean() == pytest.approx(0.25, abs=0.02)
    assert np.all(pop.job_type[pop.session_class == "agentic_tool_loop"] == "agentic")
    assert np.all(pop.job_type[pop.session_class != "agentic_tool_loop"] == "chat")
    assert np.array_equal(pop.is_reasoning, pop.session_class == "reasoning_chat")

    chat = _draw(np.random.default_rng(1), 20000, class_workload("ordinary_chat"), "bf16")
    agent = _draw(np.random.default_rng(1), 20000, class_workload("agentic_tool_loop"), "bf16")
    assert chat.T.mean() < 5000
    assert agent.T.mean() > 50000
    assert agent.T.mean() / chat.T.mean() > 10


def test_within_class_load_heterogeneity():
    # Rate σ (plus Δ, Y spread) gives a continuous ℓ within one class, so greedy can
    # diverge from LP at constraint boundaries (T5/T7) instead of collapsing to a point.
    pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
    sel = (pop.state == "active") & (pop.session_class == "agentic_tool_loop")
    ell = pop.ell[sel]
    assert ell.std() / ell.mean() > 0.2  # meaningful spread, not a point mass


def test_raw_turn_accounting_and_cache_misses():
    hit = _draw(
        np.random.default_rng(0), 1000,
        class_workload("ordinary_chat", state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0)),
        "bf16",
    )
    assert np.all(hit.cache_hit)
    assert np.allclose(hit.f, hit.turn_rate * hit.Delta)
    assert np.allclose(hit.g, hit.turn_rate * hit.Y)
    assert np.allclose(hit.ell_pre, hit.f / rho_dest(hit.T, hit.mfu))

    miss = _draw(
        np.random.default_rng(0), 1000,
        class_workload("ordinary_chat", state_mix=(1.0, 0.0, 0.0), cache_hit=(0.0, 0.0, 0.0, 0.0)),
        "bf16",
    )
    assert not miss.cache_hit.any()
    assert np.allclose(miss.f, miss.turn_rate * miss.T)


def test_long_turns_cap_effective_turn_rate():
    wl = class_workload("agentic_tool_loop", state_mix=(1.0, 0.0, 0.0), max_ell=0.25)
    pop = _draw(np.random.default_rng(0), 20000, wl, "bf16")
    assert pop.ell.max() <= wl.max_ell + 1e-9
    assert (pop.turn_rate < wl.rate_means[-1]).any()


def test_cold_idle_carry_no_load_but_keep_kv():
    pop = generate(PoolPower())
    assert np.all(pop.turn_rate[pop.state != "active"] == 0)
    assert np.all(pop.f[pop.state != "active"] == 0)
    assert np.all(pop.g[pop.state != "active"] == 0)
    assert np.all(pop.ell[pop.state != "active"] == 0)
    assert np.all(pop.ell[pop.state == "active"] > 0)
    assert np.all(pop.m > 0)


def test_loads_returned_unsummed():
    pop = generate(PoolPower())
    assert np.allclose(pop.ell, pop.ell_pre + pop.ell_dec)
    assert "ell" not in {f.name for f in fields(JobPopulation)}
    for raw in ("turn_rate", "Delta", "Y", "f", "g", "session_class"):
        assert raw in {f.name for f in fields(JobPopulation)}


def test_population_size_is_an_input():
    pool = PoolPower()
    wl = Workload()
    assert len(generate(pool, wl)) == round(wl.occupancy * 32 * pool.s_node)
    doubled = generate(pool, replace(wl, occupancy=2 * wl.occupancy))
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
    wl = replace(Workload(), max_ell=1e9)
    bf16 = _draw(np.random.default_rng(1), 5000, wl, "bf16")
    fp8 = _draw(np.random.default_rng(1), 5000, wl, "fp8")
    assert np.array_equal(bf16.T, fp8.T)
    assert np.array_equal(bf16.session_class, fp8.session_class)
    assert np.allclose(bf16.f, fp8.f)
    assert np.allclose(bf16.g, fp8.g)
    assert np.allclose(bf16.ell_pre, fp8.ell_pre)
    assert np.array_equal(bf16.m, fp8.m)
    active = bf16.state == "active"
    assert np.allclose(fp8.ell_dec[active], bf16.ell_dec[active] * wl.g_bf16 / wl.g_fp8)
