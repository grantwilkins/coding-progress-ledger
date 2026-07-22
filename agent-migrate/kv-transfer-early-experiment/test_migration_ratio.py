"""
Claim:
migration_ratio computes the pre-Inkling six-model H100/BF16 replay-to-KV-transfer
ratio from the declared GQA, MLA, and DeepSeek compressed-cache layouts.

Plausible wrong implementations:
- retaining the newer Inkling catalogue instead of the requested old six points
- omitting K or V, or using bits where the cache formula requires bytes
- rounding DeepSeek compressed entries down instead of up
- reversing bytes/bits or Gbps in transfer time
- plotting Qwen3-235B in the GLM-labeled context surface
"""

import math

import migration_ratio as mr


def test_restored_catalogue_is_the_pre_inkling_six():
    assert [model.label for model in mr.MODELS] == [
        "DeepSeek V4 Pro",
        "Qwen3 Next 80B",
        "Qwen3.5 397B",
        "Kimi K2.6",
        "GLM 5",
        "Qwen3 235B",
    ]


def test_gqa_and_mla_count_the_declared_bf16_state():
    assert mr.gqa_kv(2, 3, 4)(5) == 2 * mr.BPE * 2 * 3 * 4 * 5
    assert mr.mla_kv(2, 5, 7)(3) == 2 * (5 + 7) * 2 * 3


def test_deepseek_cache_rounds_compressed_entries_up():
    assert mr.dsv4_kv(5) > mr.dsv4_kv(4)
    assert mr.dsv4_kv(6) == mr.dsv4_kv(5)


def test_prefill_contains_linear_ffn_and_quadratic_attention():
    model = mr.Model("hand", 1, 1, 1, 2, 3, lambda _: 0)
    assert mr.prefill_flops(model, 2) == 4e9 + 20


def test_transfer_uses_bits_and_decimal_gbps():
    model = mr.Model("hand", 1, 1, 1, 1, 1, lambda _: 1e9)
    assert mr.t_transfer(model, 1, 8) == 1


def test_ratio_is_one_at_the_derived_crossover():
    model, tokens = mr.model("DeepSeek V4 Pro"), 1_000
    replay, state = mr.t_replay(model, tokens), model.kv_bytes(tokens)
    crossover = state * 8 / replay / 1e9
    assert math.isclose(replay / mr.t_transfer(model, tokens, crossover), 1)


def test_glm_context_frame_uses_glm_values():
    frame = mr.context_ratio_frame("GLM 5", 5, [1_000])
    glm = mr.model("GLM 5")
    expected = mr.t_replay(glm, 1_000) / mr.t_transfer(glm, 1_000, 5)
    assert frame.iloc[0]["ratio"] == expected
