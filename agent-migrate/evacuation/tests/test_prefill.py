"""
Claim:
  prefill.rho(name, T) = EFF / (2 N_active + C T), with C = 2 * attn_layers *
  q_heads * head_dim, is the per-instance (8x H100, BF16, MFU=0.35) prefill rate.
  ffn_ceiling = EFF/(2 N_active); attn_crossover T* = 2 N_active / C.

Plausible wrong implementations:
- C uses KV heads instead of Q heads (GQA confusion) -> wrong T^2 term.
- drops a factor of 2 in the FFN (2 N_act) or attention (causal) term.
- uses total params for MoE instead of active params.
- Qwen3-Next counts all 48 layers in the T^2 term instead of its 12 full-attn
  layers -> kills its long-context advantage.
- ceiling / crossover use the wrong term (e.g. attention coef in the ceiling).
"""

from __future__ import annotations

import numpy as np

import prefill

# Published 8x H100 prefill rates (tok/s) at 1k/10k/100k/1M, the model's
# calibration anchors for the two models whose rates were quoted in the suite.
PUBLISHED = {
    "Qwen3-235B-A22B":    [60_800, 46_600, 14_000, 1_700],
    "Qwen3-Next-80B-A3B": [454_300, 396_800, 175_000, 26_600],
}


def test_reproduces_published_anchors():
    # Tolerance matches the table's 2-3 significant-figure rounding (e.g. the 1M
    # anchor 1749 -> 1700). The model otherwise lands within ~0.1%.
    for name, vals in PUBLISHED.items():
        np.testing.assert_allclose(prefill.anchors(name), vals, rtol=3e-2)


def test_short_context_hits_ffn_ceiling():
    # At T << T*, attention is negligible and rho -> EFF/(2 N_active).
    for name in prefill.ARCH:
        rate = prefill.rho(name, 1.0)
        np.testing.assert_allclose(rate, prefill.ffn_ceiling(name), rtol=1e-3)


def test_long_context_decays_as_inverse_T():
    # At T >> T*, FFN is negligible and rho*T -> EFF/C (constant).
    for name in prefill.ARCH:
        C = prefill.attn_coef(prefill.ARCH[name])
        big = prefill.rho(name, 1e12) * 1e12
        np.testing.assert_allclose(big, prefill.EFF / C, rtol=1e-2)


def test_crossover_equates_ffn_and_attention_terms():
    for name in prefill.ARCH:
        a = prefill.ARCH[name]
        Tstar = prefill.attn_crossover(name)
        # FFN term == attention term at T*  =>  rho(T*) is exactly half the ceiling.
        assert np.isclose(2 * a.n_active, prefill.attn_coef(a) * Tstar)
        assert np.isclose(prefill.rho(name, Tstar), 0.5 * prefill.ffn_ceiling(name))


def test_rho_strictly_decreasing_in_context():
    T = np.array([1e3, 1e4, 1e5, 1e6])
    for name in prefill.ARCH:
        r = prefill.rho(name, T)
        assert np.all(np.diff(r) < 0)


def test_next_beats_dense_same_active_at_long_context():
    # Qwen3-Next and Qwen3-30B-A3B both have 3B active and equal H_q*d_head
    # (16*256 == 32*128), but Next carries the T^2 term on only 12 of 48 layers,
    # so at long context it prefills ~4x faster (C ratio = 48/12).
    ratio = prefill.rho("Qwen3-Next-80B-A3B", 1e6) / prefill.rho("Qwen3-30B-A3B", 1e6)
    assert ratio > 3.0
