"""
Claim:
Destination load uses scheduled work, exact session identity, and conservative
normal/emergency/stability classifications.

Plausible wrong implementations:
- Use achieved completions instead of offered tokens.
- Reuse one prefix across nominally distinct sessions.
- Treat an SLO boundary as infeasible or a just-outside value as feasible.
- Declare a growing destination queue stable.
"""

import pytest

import destination_runner as runner


def request(ttft=.1, tpot=.01, error=""):
    return {"status": 200, "error": error, "output_tokens": 2,
            "planned_output_tokens": 2, "ttft_s": ttft, "mean_tpot_s": tpot,
            "input_tokens": 10}


def metrics(slope=0):
    return [{"monotonic_ns": i * 10**9,
             "vllm:num_requests_waiting": 1 + slope * i} for i in range(100)]


def test_schedule_and_session_tokens_are_deterministic_but_isolated():
    assert runner.poisson_schedule(2, 4, 7) == runner.poisson_schedule(2, 4, 7)
    a = runner.Session("a", 4, 2, 3, 100, 7)
    b = runner.Session("b", 4, 2, 3, 100, 7)
    first, forced = a.prompt(0)
    a.commit(first, forced)
    assert a.prompt(1)[0][:4] == first[:4]
    assert b.prompt(0)[0][:4] != first[:4]


def test_offered_work_does_not_depend_on_completion():
    rows = [request(), request(error="failed")]
    assert runner.offered_work(rows, 2) == (10, 2)


def test_slo_boundary_is_inclusive_and_queue_growth_is_not_stable():
    slos = {"normal": {"p90_ttft_s": 2, "p90_mean_tpot_s": .1},
            "emergency": {"p90_ttft_s": 10, "p90_mean_tpot_s": .25}}
    exact = runner.classify([request(2, .1)], metrics(), True, slos)
    outside = runner.classify([request(2.001, .1)], metrics(), True, slos)
    growing = runner.classify([request()], metrics(.1), True, slos)
    assert exact == {"normal": True, "emergency": True, "stable": True}
    assert not outside["normal"] and outside["emergency"]
    assert not growing["stable"]


def test_queue_drift_requires_real_samples():
    with pytest.raises(ValueError, match="sampled"):
        runner.queue_drift_upper(metrics()[:1])


def test_anchor_drift_gate_is_inclusive_at_fifteen_percent():
    expected = {("prefill", 4096): 100, ("decode", 4096): 50}
    runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 85},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 57.5},
    ], expected)
    with pytest.raises(ValueError, match="drift"):
        runner.anchor_gate([
            {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 84.9},
            {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 50},
        ], expected)


def test_adaptive_search_brackets_each_nested_boundary():
    boundaries = {"normal": 1, "emergency": 2, "stable": 3}
    found = runner.find_boundaries(
        lambda radius: {mode: radius <= bound for mode, bound in boundaries.items()}
    )
    for mode, bound in boundaries.items():
        assert found[mode][0] <= bound <= found[mode][1]
        assert found[mode][1] - found[mode][0] <= .05 * found[mode][1]
