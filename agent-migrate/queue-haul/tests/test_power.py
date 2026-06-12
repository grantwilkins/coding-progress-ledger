import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power import CAP_FP8_GB, PoolPower


def test_center_prices():
    p = PoolPower()
    assert p.p_bar == pytest.approx(10500.0)
    assert p.s_plat == pytest.approx(350.0)
    assert p.p_pre == pytest.approx(3500.0)
    assert p.p_dec == pytest.approx(17500.0)
    assert (p.p_pre + p.p_dec) / 2 == pytest.approx(p.p_bar)


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
