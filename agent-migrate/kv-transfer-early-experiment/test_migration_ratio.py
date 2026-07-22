"""
Claim:
migration_ratio compares architecture-derived runnable state with published,
fresh-context H200 TTFT only for non-Llama models released since mid-2025.

Plausible wrong implementations:
- substituting peak-FLOP/MFU estimates for public TTFT
- omitting Qwen's fixed recurrent state or counting it once per token
- treating full-precision KV as FP8 because the weights are FP8
- reversing bytes/bits or Gbps in transfer time
- silently plotting old, closed, cached, or unbenchmarked models
"""

import math

import migration_ratio as mr


def test_published_h200_ttft_points_are_literal_inputs():
    qwen, kimi = mr.BENCHMARKS
    assert qwen.ttft_seconds == ((1024, 0.077), (8192, 0.2), (32768, 0.6), (65536, 1.6), (98304, 2.7), (131072, 4.2), (262144, 12.4))
    assert kimi.ttft_seconds == ((1024, 0.112),)


def test_qwen_state_has_growing_gqa_and_fixed_gated_deltanet_parts():
    kv_per_token = 10 * 2 * 2 * 256 * 2
    fixed = 30 * 32 * 128 * 128 * 4 + 30 * (2 * 16 * 128 + 32 * 128) * 4 * 2
    assert mr.qwen35_state(1) == kv_per_token + fixed
    assert mr.qwen35_state(2) - mr.qwen35_state(1) == kv_per_token


def test_kimi_mla_state_counts_latent_and_rope_key_in_bf16():
    assert mr.kimi_k25_state(1) == 61 * (512 + 64) * 2


def test_transfer_uses_bits_and_decimal_gbps():
    assert mr.transfer_time(1_000_000_000, 8) == 1


def test_ratio_is_one_at_the_derived_crossover():
    state, ttft = mr.qwen35_state(98304), 2.7
    assert math.isclose(mr.ratio(ttft, state, mr.crossover_gbps(ttft, state)), 1)


def test_quantitative_models_obey_release_and_no_llama_constraints():
    assert all(benchmark.released >= "2025-07" for benchmark in mr.BENCHMARKS)
    assert all("llama" not in benchmark.label.lower() for benchmark in mr.BENCHMARKS)


def test_requested_models_without_comparable_ttft_are_not_plotted():
    plotted = {benchmark.label for benchmark in mr.BENCHMARKS}
    assert plotted.isdisjoint({label for label, _, _ in mr.UNMODELED})
