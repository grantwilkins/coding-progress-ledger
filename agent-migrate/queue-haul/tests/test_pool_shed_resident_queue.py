import pytest

from pool_shed_resident_queue import simulate


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
