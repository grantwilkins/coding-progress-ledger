"""
Claim:
migration_ratio computes the six-model H100/BF16 replay-to-KV-transfer ratio.
Prefill is 2*A*T plus 2*H_q*(d_qk+d_v) times the causal (query, key) pairs
implied by each layer group's compression m, top-k cap k, and sliding window w.
Dense: T^2/(2m) per layer. Once the compressed pool outgrows k (T > k*m) each
query is capped at k entries, so the group goes linear in T.

Plausible wrong implementations:
- capping at T > k instead of T > k*m, moving the seam by a factor of m
- dropping the -k^2*m/2 term, so pairs jump discontinuously at the seam
- losing the causal 1/2, or applying compression to the window branch
- letting the refactor change the four dense-attention models
- reversing bytes/bits or Gbps in transfer time
- rounding DeepSeek compressed entries down instead of up
"""

import math

import migration_ratio as mr

# Layers holding per-token KV, from each released config.
EXPECTED_LAYERS = {
    "DeepSeek V4 Pro": 61,
    "Qwen3 Next 80B": 12,
    "Qwen3.5 397B": 15,
    "Kimi K2.6": 61,
    "GLM 5": 78,
    "Qwen3 235B": 94,
}


def attn_flops(model, tokens):
    return mr.prefill_flops(model, tokens) - 2.0 * model.active_b * 1e9 * tokens


def test_restored_catalogue_is_the_pre_inkling_six():
    assert [model.label for model in mr.MODELS] == list(EXPECTED_LAYERS)


def test_attention_groups_cover_the_configs_kv_layers():
    for model in mr.MODELS:
        assert sum(g.layers for g in model.attn) == EXPECTED_LAYERS[model.label]


def test_dense_pairs_are_causal_and_scale_with_layers():
    assert mr.Attn(2).pairs(10) == 100  # 2 layers * 10^2/2


def test_compression_divides_the_pair_count():
    assert mr.Attn(1, compress=4).pairs(100) == 1250  # 100^2/(2*4)


def test_sliding_window_is_linear_and_per_layer():
    assert mr.Attn(3, window=5).pairs(10) == 300  # 3 * (10^2/2 + 5*10)


def test_topk_seam_sits_at_k_times_m_and_is_continuous():
    k, m = 1_024, 4
    group, seam = mr.Attn(1, compress=m, topk=k), 1_024 * 4
    # Both branches agree at T = k*m; dropping -k^2*m/2 would double this.
    assert group.pairs(seam) == seam**2 / (2 * m) == k * seam - k**2 * m / 2
    assert math.isclose(group.pairs(seam + 1), group.pairs(seam), rel_tol=1e-3)


def test_topk_threshold_accounts_for_compression():
    # Pool is 20/4 = 5 entries, below k=8, so this is still dense.
    assert mr.Attn(1, compress=4, topk=8).pairs(20) == 50  # not 8*20 - 64*2


def test_topk_makes_attention_linear_above_the_seam():
    group = mr.Attn(1, topk=100)
    assert group.pairs(200_000) / group.pairs(100_000) < 2.1  # dense would be ~4


def test_dense_models_keep_the_uncompressed_quadratic_form():
    for model in mr.MODELS:
        if any(g.compress > 1 or g.topk or g.window for g in model.attn):
            continue
        layers = sum(g.layers for g in model.attn)
        expected = layers * model.query_heads * (model.qk_dim + model.v_dim) * 1e8
        assert math.isclose(attn_flops(model, 10_000), expected)


def test_dsa_model_replay_grows_linearly_but_dense_model_does_not():
    glm, qwen = mr.model("GLM 5"), mr.model("Qwen3 235B")
    assert math.isclose(
        mr.t_replay(glm, 1_000_000) / mr.t_replay(glm, 100_000), 10, rel_tol=0.02
    )
    assert mr.t_replay(qwen, 1_000_000) / mr.t_replay(qwen, 100_000) > 50


def test_gqa_and_mla_count_the_declared_bf16_state():
    assert mr.gqa_kv(2, 3, 4)(5) == 2 * mr.BPE * 2 * 3 * 4 * 5
    assert mr.mla_kv(2, 5, 7)(3) == 2 * (5 + 7) * 2 * 3


def test_deepseek_cache_rounds_compressed_entries_up():
    assert mr.dsv4_kv(5) > mr.dsv4_kv(4)
    assert mr.dsv4_kv(6) == mr.dsv4_kv(5)


def test_transfer_uses_bits_and_decimal_gbps():
    model = mr.Model("hand", 1, (mr.Attn(1),), 1, 1, 1, lambda _: 1e9)
    assert mr.t_transfer(model, 1, 8) == 1


def test_ratio_is_one_at_the_derived_crossover():
    model, tokens = mr.model("DeepSeek V4 Pro"), 1_000
    replay, state = mr.t_replay(model, tokens), model.kv_bytes(tokens)
    crossover = state * 8 / replay / 1e9
    assert math.isclose(replay / mr.t_transfer(model, tokens, crossover), 1)


def test_context_frame_follows_the_requested_model():
    for label in ("GLM 5", mr.CONTEXT_MODEL):
        m = mr.model(label)
        frame = mr.context_ratio_frame(label, 5, [1_000])
        assert frame.iloc[0]["ratio"] == mr.t_replay(m, 1_000) / mr.t_transfer(
            m, 1_000, 5
        )
    assert mr.context_ratio_frame("GLM 5", 5, [1_000]).iloc[0]["ratio"] != (
        mr.context_ratio_frame(mr.CONTEXT_MODEL, 5, [1_000]).iloc[0]["ratio"]
    )


def test_context_panel_model_spans_a_wide_ratio_range():
    """The panel is only informative if the decision flips inside the sweep."""
    m = mr.model(mr.CONTEXT_MODEL)
    lo, hi = mr.CONTEXT_TOKENS.min(), mr.CONTEXT_TOKENS.max()
    slow, fast = mr.CONTEXT_BANDWIDTHS_GBPS.min(), mr.CONTEXT_BANDWIDTHS_GBPS.max()
    assert mr.t_replay(m, int(lo)) / mr.t_transfer(m, int(lo), slow) < 1
    assert mr.t_replay(m, int(hi)) / mr.t_transfer(m, int(hi), fast) > 10
