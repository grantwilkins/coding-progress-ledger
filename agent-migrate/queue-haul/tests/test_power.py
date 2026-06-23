import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impact import compute
from instance import Workload, generate
from power import CAP_BF16_GB, CAP_FP8_GB, C_ATTN, N_ACT, PoolPower, rho_dest


def test_center_prices():
    p = PoolPower()
    assert p.p_bar == pytest.approx(10500.0)
    assert p.s_plat == pytest.approx(350.0)
    assert p.base_w_per_load == pytest.approx(4000.0)
    assert p.c_prefill_j_per_tok == pytest.approx(0.148)
    assert p.c_decode_j_per_tok == pytest.approx(1.76)
    assert p.c_decode_j_per_tok > 5 * p.c_prefill_j_per_tok


def test_memory_price_by_precision():
    bf16 = PoolPower()
    fp8 = replace(bf16, cap_gb=CAP_FP8_GB)
    assert bf16.mu == pytest.approx(208.0, rel=0.01)
    assert fp8.mu == pytest.approx(74.0, rel=0.01)
    assert fp8.s_node / bf16.s_node == pytest.approx(365 / 130)


def test_ranking_invariant_under_p_bar_scaling():
    ell = np.array([0.03, 0.5, 0.001, 0.2, 0.07])
    p = PoolPower()
    scaled = replace(p, p_busy_w=7 * p.p_busy_w)
    assert np.array_equal(np.argsort(p.p_bar * ell), np.argsort(scaled.p_bar * ell))


def test_regime_crossover():
    p = PoolPower()
    load = 4.0
    s_held_cross = (load / p.rho_star) * p.s_node
    assert not p.memory_bound(load, 0.99 * s_held_cross)
    assert p.memory_bound(load, 1.01 * s_held_cross)
    assert p.node_count(load, 0.5 * s_held_cross) == pytest.approx(load / p.rho_star)
    assert p.node_count(load, 2 * s_held_cross) == pytest.approx(2 * load / p.rho_star)


def test_precision_shifts_memory_threshold_not_load_regime():
    # #6 consolidated: FP8 scales S_node by the cap ratio (365/130 ≈ 2.8×, NOT a literal
    # 2×), which scales the held-session crossover by the same factor; a load-bound config
    # stays load-bound either way (precision moves the memory threshold, not the load story).
    bf16, fp8 = PoolPower(), replace(PoolPower(), cap_gb=CAP_FP8_GB)
    assert fp8.s_node / bf16.s_node == pytest.approx(CAP_FP8_GB / CAP_BF16_GB)
    load = 4.0
    cross = lambda p: (load / p.rho_star) * p.s_node  # held sessions where memory binds
    assert cross(fp8) / cross(bf16) == pytest.approx(CAP_FP8_GB / CAP_BF16_GB)
    s_held = 0.5 * cross(bf16)  # below the smaller (bf16) crossover ⇒ load-bound for both
    assert not bf16.memory_bound(load, s_held) and not fp8.memory_bound(load, s_held)
    # end-to-end: a short-context (load-bound) population stays load-bound under FP8
    swl = replace(Workload(), t_mix=((1.0, 8.0, 0.5),))  # E[T] ≈ 3378
    sp = replace(PoolPower(), mean_context_tokens=3378.0)
    assert compute(generate(sp, swl), sp).regime == "load"
    assert compute(generate(replace(sp, cap_gb=CAP_FP8_GB), swl), replace(sp, cap_gb=CAP_FP8_GB)).regime == "load"


def test_rho_dest_landmarks():
    t_star = 2 * N_ACT / C_ATTN
    assert t_star == pytest.approx(29000, rel=0.02)  # query heads, not KV heads
    assert 2 * N_ACT / (2 * 94 * 4 * 128) > 400_000  # H_kv would mislocate T* ~16x out
    assert rho_dest(0.0) == pytest.approx(63000, rel=0.02)
    assert rho_dest(t_star) == pytest.approx(rho_dest(0.0) / 2)  # half rate at T*
    assert rho_dest(100_000) == pytest.approx(14000, rel=0.05)
