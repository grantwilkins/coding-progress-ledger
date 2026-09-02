"""The quick SLO sweep keeps its rate grid and uncertainty honest."""

import numpy as np
import pytest

import quick_slo_sweep as sweep


def test_plan_is_shared_across_hardware_and_targets_the_boundary():
    h100 = sweep.make_plan(7, "h100")
    a100 = sweep.make_plan(7, "a100")

    assert h100["rates_rps"] == [
        .5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24,
    ]
    assert h100["requests_per_point"] == 50
    assert h100["slo"] == {"p90_ttft_s": 1., "p90_tpot_s": .05}
    assert h100["statistics"]["request_block_lengths"] == [5, 10]
    assert h100["statistics"]["scope"] == \
        "pointwise_conditional_on_each_rate_episode"
    assert h100["semantics"]["engine_reuse"] == \
        "one_warmed_launch_until_failure_or_resume"
    assert h100["runtime"]["attention_backend"] == "TRITON_ATTN"
    assert not h100["runtime"]["async_scheduling"]
    assert "one_warmed_engine" not in h100["semantics"]
    assert h100["comparison_sha256"] == a100["comparison_sha256"]
    assert h100["rate_order_rps"] == a100["rate_order_rps"]

    changed = {**h100, "requests_per_point": 49}
    with pytest.raises(ValueError, match="quick SLO plan"):
        sweep.validate_plan(changed)


def test_circular_blocks_always_resample_one_episode():
    first = sweep.moving_block_counts(50, 5, 20, np.random.default_rng(3))
    second = sweep.moving_block_counts(50, 5, 20, np.random.default_rng(3))

    assert first.shape == (20, 50)
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == 50)


def test_weighted_p90_matches_expanded_request_clusters():
    clusters = [np.array([1., 2.]), np.array([10., 20.])]
    counts = np.array([[1, 1], [2, 0], [0, 2]])

    observed = sweep.weighted_p90(clusters, counts)
    expected = [
        np.quantile([1., 2., 10., 20.], .9),
        np.quantile([1., 1., 2., 2.], .9),
        np.quantile([10., 10., 20., 20.], .9),
    ]

    assert observed == pytest.approx(expected)


def test_bootstrap_keeps_each_requests_token_intervals_together():
    requests = [{
        "scheduled_ns": index,
        "ttft_s": index / 100,
        "token_itls_s": [index / 1000, (index + 1) / 1000],
        "exact_token_timestamps": True,
        "status": 200,
        "done": True,
        "finish_reason": "length",
        "output_tokens": 2,
        "planned_output_tokens": 2,
        "recorded_output_tokens": 2,
    } for index in range(50)]

    intervals = sweep.bootstrap_intervals(
        requests, 11, draws=200, block_lengths=(5, 10),
    )

    assert intervals["p90_ttft_s"]["point"] == pytest.approx(.441)
    assert intervals["p90_tpot_s"]["point"] == pytest.approx(.045)
    for interval in intervals.values():
        assert interval["low"] <= interval["point"] <= interval["high"]
        assert set(interval["by_block_length"]) == {"5", "10"}
