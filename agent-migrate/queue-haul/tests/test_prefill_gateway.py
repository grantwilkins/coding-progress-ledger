"""Claim: the live gateway measures and caps uncached prefill work."""

import pytest

from prefill_gateway import PrefillCompletionLimiter, _usage


def test_gateway_extracts_uncached_prompt_usage_from_stream():
    payload = (
        b'data: {"usage":{"prompt_tokens":10,'
        b'"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
        b"data: [DONE]\n\n"
    )

    assert _usage(payload) == (10, 4)


def test_gateway_live_control_reserves_aggregate_completion_time(monkeypatch):
    clock = iter((0.0, 0.0, 0.0, 0.0, 0.0))
    slept = []
    monkeypatch.setattr("prefill_gateway.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("prefill_gateway.time.sleep", slept.append)
    limiter = PrefillCompletionLimiter()

    assert limiter.update(100) == {"tokens_per_s": 100}
    assert limiter.wait(10) == pytest.approx(.1)
    assert limiter.wait(10) == pytest.approx(.2)
    assert slept == pytest.approx([.1, .2])

    assert limiter.update(None) == {"tokens_per_s": None}
    assert limiter.wait(10) == 0
