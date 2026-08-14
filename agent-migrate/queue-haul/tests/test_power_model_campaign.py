from __future__ import annotations

import pytest

import power_model_campaign as campaign


def test_grid_separates_discovery_and_unseen_confirmation_cells():
    rows = campaign.cells(4)
    discovery = [row for row in rows if row.stage == "discovery"]
    held = [row for row in rows if row.stage == "confirmation"]

    assert len(discovery) == 90
    assert len(held) == 18
    assert rows[0].stage == rows[91].stage == rows[-1].stage == "idle"
    assert {row.concurrency for row in discovery} == {1, 2, 4, 8, 16}
    assert {row.concurrency for row in held} == {3, 6, 12}
    assert (604, 64) in {(row.prompt_tokens, row.output_tokens) for row in held}


def test_parse_metrics_sums_engines_and_requires_realized_counters():
    text = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
vllm:prompt_tokens_total{engine="0"} 10
vllm:prompt_tokens_total{engine="1"} 20
vllm:generation_tokens_total{engine="0"} 7
vllm:prompt_tokens_cached_total{engine="0"} 0
"""
    assert campaign.parse_metrics(text) == {
        campaign.PROMPT_COUNTER: 30,
        campaign.DECODE_COUNTER: 7,
        campaign.CACHED_COUNTER: 0,
    }
    with pytest.raises(RuntimeError, match="missing vLLM counters"):
        campaign.parse_metrics("vllm:prompt_tokens_total 1")


def synthetic_rows(alpha=1 / 1000, beta=1 / 500):
    rows = []
    for replicate in range(2):
        for family, f, g in (("prefill", 1000, 1), ("decode", 100, 500),
                             ("campaign", 500, 250)):
            ell = alpha * f + beta * g
            rows.append({"stage": "discovery", "family": family,
                         "replicate": replicate, "prompt_tokens": 1,
                         "output_tokens": 1, "concurrency": 1,
                         "realized_prefill_tps": f,
                         "realized_decode_tps": g,
                         "power_mean_w": 80 + 220 * (1 - __import__("math").exp(-ell / .8)),
                         "cached_prompt_tokens": 0})
    return rows * 15 + [{"stage": "idle", "power_mean_w": 80}] * 3


def test_fit_uses_realized_rates_and_recovers_saturating_model():
    fit = campaign.saturating_fit(synthetic_rows())

    assert fit["alpha_s_per_prefill_token"] == pytest.approx(1 / 1000)
    assert fit["beta_s_per_decode_token"] == pytest.approx(1 / 500, rel=.03)
    assert fit["power_idle_w"] == pytest.approx(80)
    assert fit["power_max_w"] == pytest.approx(300, rel=.03)
