from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_resident_queue import simulate, source_turns


C = dict(prefill_step_s=.1, prefill_token_s=.1, prefill_attention_s=0., decode_step_s=.1, decode_attention_s=0., endpoint_s=.2)


def request(name, arrival=0., prompt=1, output=1, history=None, gpu="a", cached=0):
    return dict(request_id=name, gpu=gpu, history=history or name, arrival_s=arrival,
                prompt_tokens=prompt, cached_tokens=cached, output_tokens=output)


def test_local_affinity_and_causal_history():
    rows = [request("replay", prompt=8), request("resident", arrival=.1),
            request("other_gpu", arrival=.1, gpu="b"),
            request("next_turn", arrival=.2, history="resident")]
    result = {r["request_id"]: r for r in simulate(rows, C, 10, token_budget=2)["requests"]}
    assert result["resident"]["ttft_s"] > result["other_gpu"]["ttft_s"]
    assert result["next_turn"]["admitted_s"] >= result["resident"]["end_s"]
    with pytest.raises(ValueError, match="change GPU"):
        simulate([request("x", history="resident"), request("y", history="resident", gpu="b")], C, 10)


def test_chunking_overlaps_decode_and_keeps_running_insertion_order():
    result = simulate([request("a", output=3), request("b", prompt=4)], C, 10, token_budget=2)
    a, b = result["requests"]
    assert a["first_s"] == pytest.approx(.5)
    assert a["end_s"] == pytest.approx(.9)
    assert b["first_s"] == pytest.approx(1.1)
    assert a["end_s"] < b["end_s"]
    assert result["gpus"]["a"]["busy_s"] == pytest.approx(.9)
    assert b["computed_prompt_tokens"] == 4


def test_censoring_cache_and_endpoint_overhead_are_separate_from_gpu_work():
    result = simulate([request("a", prompt=5, cached=4, output=3), request("b", prompt=4)], C, .65, token_budget=2)
    a, b = result["requests"]
    assert a["first_s"] == pytest.approx(.5)
    assert not a["done"] and a["end_s"] is None and a["mean_tpot_s"] is None
    assert not b["done"] and b["first_s"] is None
    assert a["computed_prompt_tokens"] == 1
    assert result["gpus"]["a"]["busy_s"] == pytest.approx(.65)


def test_scheduler_limits_and_known_input_validation():
    result = simulate([request("a", output=3), request("b")], C, 10, max_sequences=1)
    a, b = result["requests"]
    assert b["admitted_s"] == pytest.approx(a["server_end_s"])
    assert b["admitted_s"] < a["end_s"]
    for rows in ([request("x"), request("x")], [request("x", cached=1)], [request("x", output=0)]):
        with pytest.raises(ValueError):
            simulate(rows, C, 10)


def test_endpoint_placement_does_not_charge_gpu_time():
    for fraction in (0., 1.):
        result = simulate([request("a")], C, 10, endpoint_before_fraction=fraction)
        assert result["requests"][0]["end_s"] == pytest.approx(.4)
        assert result["gpus"]["a"]["busy_s"] == pytest.approx(.2)


def test_equal_arrivals_keep_input_order_and_attention_costs_use_retained_context():
    rows = [request("z_turn_0", history="h", prompt=3, cached=1, output=2), request("a_turn_1", history="h")]
    c = {**C, "prefill_attention_s": .01, "decode_attention_s": .01}
    result = simulate(rows, c, 10, token_budget=1)["requests"]
    first, following = result
    assert first["request_id"] == "z_turn_0"
    assert first["first_s"] == pytest.approx(.68)  # .2+.03, then .2+.05, plus endpoint .2.
    assert first["mean_tpot_s"] == pytest.approx(.14)  # Decode context is prompt+first output.
    assert following["admitted_s"] == pytest.approx(first["end_s"])


def test_arrivals_during_final_censored_iteration_remain_visible_in_queue():
    rows = [request("a", prompt=8), request("b", arrival=1.)]
    result = simulate(rows, {**C, "prefill_token_s": 1.}, 2.)["requests"]
    assert result[1]["eligible_s"] == 1.
    assert result[1]["admitted_s"] is None and not result[1]["done"]


def source_fleet(durations, **metadata):
    return SimpleNamespace(count=np.ones(len(durations)), metadata={
        'turn_sequences': [[{} for _ in row] for row in durations],
        'turn_duration_s': durations, 'source_session_rps': 1 / 22, **metadata})


def test_long_source_generation_queues_later_offers_and_finite_trace_ends():
    fleet = source_fleet([[45., 2., 3.], []])
    for now, expected in ((44., (1, 0, 45.)), (45., (2, 1, 47.)),
                          (47., (3, 2, 50.)), (200., (3, 3, 50.))):
        started, completed, finish = source_turns(fleet, now)
        assert (started[0], completed[0], finish[0]) == expected
        assert started[1] == completed[1] == 0 and finish[1] == -np.inf


def test_source_timeline_cache_preserves_phases_offsets_and_arbitrary_query_order():
    fleet = source_fleet([[45., 2., 3.]], sequence_cycle=True, turn_offset=[1], source_phase_s=[1.])
    cache = {}
    for now, expected in ((88., (4, 3, 90.)), (-2., (0, 0, -np.inf)), (0., (1, 0, 1.)),
                          (1., (1, 1, 1.)), (10., (1, 1, 1.)), (66., (3, 2, 88.))):
        cached = source_turns(fleet, now, cache)
        assert tuple(v[0] for v in cached) == expected
        for a, b in zip(cached, source_turns(fleet, now)):
            np.testing.assert_array_equal(a, b)
    other = source_fleet([[1.]], sequence_cycle=True)
    assert tuple(v[0] for v in source_turns(other, 0., cache)) == (1, 0, 1.)


def test_source_timeline_requires_valid_duration_and_offered_pacing():
    for duration in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='positive durations'):
            source_turns(source_fleet([[duration]]), 0.)
    with pytest.raises(ValueError, match='within one offered period'):
        source_turns(source_fleet([[1.]], source_phase_s=[22.]), 0.)
    with pytest.raises(ValueError, match='invalid source trace pacing'):
        source_turns(source_fleet([[1.]], source_session_rps=0.), 0.)
    with pytest.raises(ValueError, match='time must be finite'):
        source_turns(source_fleet([[1.]]), np.inf)
