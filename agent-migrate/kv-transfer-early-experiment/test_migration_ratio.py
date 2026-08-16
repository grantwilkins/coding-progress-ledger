"""
Claim:
migration_ratio computes the six-model H100/BF16 replay-to-KV-transfer ratio.
Prefill is 2*A*T plus 2*H_q*(d_qk+d_v) times the causal (query, key) pairs
implied by each layer group's compression m, top-k cap k, and sliding window w.
Dense: T^2/(2m) per layer. Once the compressed pool outgrows k (T > k*m) each
query is capped at k entries, so the group goes linear in T.

Plausible wrong implementations:
- retaining a model explicitly excluded from the rendered catalogue
- treating hybrid local or linear-attention layers as full quadratic attention
- using query-head counts instead of KV-head counts for migration bytes
- applying Gemma's 512-wide global heads to its 256-wide local layers
- capping at T > k instead of T > k*m, moving the seam by a factor of m
- dropping the -k^2*m/2 term, so pairs jump discontinuously at the seam
- losing the causal 1/2, or applying compression to the window branch
- reversing bytes/bits or Gbps in transfer time
- rounding DeepSeek compressed entries down instead of up
- choosing curve colors that disappear into either shaded decision region
"""

import math

import migration_ratio as mr
from matplotlib.colors import to_rgb

# Attention groups represented by the cost model; Qwen3.8 omits fixed state.
EXPECTED_LAYERS = {
    "DeepSeek V4 Pro": 61,
    "Gemma 4 26B-A4B": 30,
    "gpt-oss-20b": 24,
    "Qwen3.8 27B": 16,
    "Kimi K2.6": 61,
    "GLM 5": 78,
}


def attn_flops(model, tokens):
    return mr.prefill_flops(model, tokens) - 2.0 * model.active_b * 1e9 * tokens


def test_catalogue_matches_the_requested_six_models():
    assert [model.label for model in mr.MODELS] == list(EXPECTED_LAYERS)


def test_model_colors_contrast_with_both_decision_regions():
    def luminance(color):
        rgb = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in color
        ]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    backgrounds = [
        tuple(mr.REGION_ALPHA * v + 1 - mr.REGION_ALPHA for v in to_rgb(color))
        for color in (mr.KV_REGION_COLOR, mr.CONTEXT_REGION_COLOR)
    ]
    for model in mr.MODELS:
        for background in backgrounds:
            light, dark = sorted(
                (luminance(to_rgb(model.color)), luminance(background)), reverse=True
            )
            assert (light + 0.05) / (dark + 0.05) >= 4.5


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


def test_new_models_follow_their_hybrid_attention_layouts():
    T = 2_048
    gpt_pairs = 12 * T**2 / 2 + 12 * (128 * T - 128**2 / 2)
    qwen_pairs = 16 * T**2 / 2
    gemma_local = 25 * (1024 * T - 1024**2 / 2)
    gemma_global = 5 * T**2 / 2
    assert math.isclose(
        attn_flops(mr.model("gpt-oss-20b"), T), 2 * 64 * (64 + 64) * gpt_pairs
    )
    assert math.isclose(
        attn_flops(mr.model("Qwen3.8 27B"), T),
        2 * 24 * (256 + 256) * qwen_pairs,
    )
    assert math.isclose(
        attn_flops(mr.model("Gemma 4 26B-A4B"), T),
        2 * 16 * (256 + 256) * gemma_local
        + 2 * 16 * (512 + 512) * gemma_global,
    )


def test_new_models_count_only_live_bf16_migration_state():
    T = 2_048
    assert mr.gpt_oss_kv(T) == 2 * mr.BPE * 8 * 64 * (12 * T + 12 * 128)
    assert mr.model("Qwen3.8 27B").kv_bytes(T) == 2 * mr.BPE * 16 * 4 * 256 * T
    assert mr.gemma4_kv(T) == mr.BPE * (
        5 * 2 * 512 * T + 2 * 25 * 8 * 256 * 1024
    )


def test_dsa_model_replay_grows_linearly_but_dense_model_does_not():
    glm, dense = mr.model("GLM 5"), mr.model("Kimi K2.6")
    assert math.isclose(
        mr.t_replay(glm, 1_000_000) / mr.t_replay(glm, 100_000), 10, rel_tol=0.02
    )
    assert mr.t_replay(dense, 1_000_000) / mr.t_replay(dense, 100_000) > 50


def test_instance_size_follows_the_weight_footprint():
    # 640 GB/node, FP8 weights + 100 GB headroom: 1.6T needs 3, 744B needs 2,
    # and everything at or under 540B fits on one.
    got = {m.label: mr.nodes(m) for m in mr.MODELS}
    assert got == {
        "DeepSeek V4 Pro": 3,
        "Gemma 4 26B-A4B": 1,
        "gpt-oss-20b": 1,
        "Qwen3.8 27B": 1,
        "Kimi K2.6": 2,
        "GLM 5": 2,
    }


def test_crossover_scales_with_the_instance_count():
    """Nodes only buy prefill FLOPs; KV bytes are per-token and do not move."""
    m = mr.model("GLM 5")
    one_node = mr.prefill_flops(m, 100_000) / mr.NODE_EFF_FLOPS
    assert math.isclose(one_node / mr.t_replay(m, 100_000), mr.nodes(m))
    assert mr.t_transfer(m, 100_000, 10) == m.kv_bytes(100_000) * 8 / 10e9


def test_gqa_and_mla_count_the_declared_bf16_state():
    assert mr.gqa_kv(2, 3, 4)(5) == 2 * mr.BPE * 2 * 3 * 4 * 5
    assert mr.mla_kv(2, 5, 7)(3) == 2 * (5 + 7) * 2 * 3


def test_deepseek_cache_rounds_compressed_entries_up():
    assert mr.dsv4_kv(5) > mr.dsv4_kv(4)
    assert mr.dsv4_kv(6) == mr.dsv4_kv(5)


def test_transfer_uses_bits_and_decimal_gbps():
    model = mr.Model("hand", 1, 1, (mr.Attn(1),), 1, 1, 1, lambda _: 1e9)
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
