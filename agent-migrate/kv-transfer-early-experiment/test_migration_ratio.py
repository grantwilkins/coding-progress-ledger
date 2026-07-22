"""
Claim:
migration_ratio computes runnable-state bytes and replay/transfer timing from the
published hybrid-attention layouts and cache precisions.

Plausible wrong implementations:
- treating FP4 weights as an FP4 KV cache
- charging sliding-window KV for the full context instead of the retained window
- omitting Nemotron's fixed recurrent state
- reversing bytes/bits or Gbps in transfer time
- plotting undisclosed closed/preview architectures using invented dimensions
"""

import math

import migration_ratio as mr


def test_gqa_state_counts_k_and_v_at_the_declared_precision():
    assert mr.gqa_state(2, 3, 4, 2)(5) == 2 * 3 * 4 * 2 * 2 * 5


def test_published_cache_geometries_are_encoded_exactly():
    assert mr.model("GLM-5.2").state_bytes(1) == 78 * (512 + 64)
    assert mr.inkling_state(1) == 2 * 2 * 128 * (11 * 8 + 55 * 16)
    assert mr.deepseek_v4_state(1) == 30 * (129 * 512 + 64) + 31 * 129 * 512


def test_inkling_local_cache_stops_growing_after_512_tokens():
    global_per_token = 2 * 11 * 8 * 128 * 2
    assert mr.inkling_state(513) - mr.inkling_state(512) == global_per_token


def test_deepseek_cache_rounds_compressed_entries_up():
    assert mr.deepseek_v4_state(5) > mr.deepseek_v4_state(4)
    assert mr.deepseek_v4_state(6) == mr.deepseek_v4_state(5)


def test_nemotron_includes_fixed_mamba_state():
    kv_per_token = 2 * 12 * 2 * 128
    assert mr.nemotron_state(2) - mr.nemotron_state(1) == kv_per_token
    assert mr.nemotron_state(1) > kv_per_token


def test_transfer_uses_bits_and_decimal_gbps():
    item = mr.Model("hand", 1, 1, lambda _: 1e9, lambda _: 0, "", "", "k")
    assert mr.transfer_time(item, 1, 8) == 1


def test_ratio_is_one_at_the_derived_crossover():
    item = mr.model("GLM-5.2")
    tokens = 1_000
    crossover = item.state_bytes(tokens) * 8 / mr.replay_time(item, tokens) / 1e9
    assert math.isclose(mr.ratio(item, tokens, crossover), 1)


def test_undisclosed_models_are_not_plotted():
    assert {m.label for m in mr.modeled_models()}.isdisjoint({"Kimi K3", "Qwen3.7-Max"})
