from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_ttft import check_requests, resident_templates


C = dict(prefill_step_s=.1, prefill_token_s=.1, prefill_attention_s=0.,
         decode_step_s=.1, decode_attention_s=0., endpoint_s=.2)


def fleet():
    return SimpleNamespace(count=np.array([8]), metadata=dict(reference_rps=8., sequence_cycle=True,
        turn_offset=[0], turn_sequences=[[dict(context=32, prompt=2, output=2, reset=False),
            dict(context=34, prompt=4, output=0, reset=False),
            dict(context=37, prompt=3, output=1, reset=True)]]))


def request(name, arrival=0., output=1, release=0., gpu="destination"):
    return dict(request_id=name, history=name, cohort="resident", gpu=gpu,
                arrival_s=arrival, release_s=release, prompt_tokens=1, cached_tokens=0, output_tokens=output)


def test_templates_freeze_offers_and_only_reuse_completed_self_history():
    source = fleet()
    a = resident_templates(source, 1., 7101, 4., baseline_s=1.)
    assert a == resident_templates(source, 1., 7101, 4., baseline_s=1.)
    assert a["requests"] == resident_templates(source, 1., 7101, 4., baseline_s=2.)["requests"]
    assert a["metadata"]["request_count"] == 32
    assert a["metadata"]["offered_rps"] == a["metadata"]["observed_calendar_rps"] == 8.
    assert a["metadata"]["baseline_request_count"] == 8
    rows = [r for r in a["requests"] if r["history"] == "resident-7101-0"]
    assert [r["cached_tokens"] for r in rows] == [32, 32, 0, 0]
    assert [r["cache_basis"] for r in rows] == ["initial_warm_prefix", "completed_self_history", "reset_cold", "reset_cold"]
    assert rows[1]["output_tokens"] == 0
    assert np.diff([r["arrival_s"] for r in rows]) == pytest.approx([1., 1., 1.])
    changed_seed = resident_templates(source, 1., 7102, 4., baseline_s=1.)
    assert changed_seed["metadata"]["phase_s"] != a["metadata"]["phase_s"]
    assert changed_seed["metadata"]["input_sha256"] == a["metadata"]["input_sha256"]
    assert resident_templates(source, 0., 7101, 4., baseline_s=1.)["requests"] == []
    source.metadata["turn_sequences"][0][1].update(observed_dispatch=999., observed_cached_tokens=999)
    assert resident_templates(source, 1., 7101, 4., baseline_s=1.) == a


def test_template_validation_and_finite_history_preserve_recorded_inputs():
    source = fleet()
    source.metadata.update(sequence_cycle=False, turn_offset=[1])
    rows = resident_templates(source, 1., 7101, 8., baseline_s=1.)["requests"]
    assert len(rows) == 16 and {r["recorded_turn"] for r in rows} == {1, 2}
    source.metadata["turn_sequences"][0][2].update(context=1, prompt=1, reset=False)
    with pytest.raises(ValueError, match="retained history"):
        resident_templates(source, 1., 7101, 8., baseline_s=1.)
    for load in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            resident_templates(fleet(), load, 7101, 4., baseline_s=1.)


def test_template_selection_respects_source_mass_weights():
    source = fleet()
    source.count = np.array([0., 8.])
    source.metadata["turn_sequences"] *= 2
    source.metadata["turn_offset"] = [0, 1]
    template = resident_templates(source, 1., 7101, 4., baseline_s=1.)
    assert template["metadata"]["selected_source_cohorts"] == [1] * 8
    assert template["metadata"]["selected_turn_offsets"] == [1] * 8
    assert {r["source_cohort"] for r in template["requests"]} == {1}


def test_checker_retains_migration_wait_censoring_and_zero_output_service():
    rows = [request("delayed", arrival=.1, release=4.), request("prefill_only", arrival=.2, output=0),
            request("fast", output=2, gpu="other"), request("late", arrival=4.5, release=9., output=2)]
    original = deepcopy(rows)
    result = check_requests(rows, C, dict(baseline=(0., 1.), post=(1., 5.), all=(0., 5.)), 5.)
    assert rows == original
    by_id = {r["request_id"]: r for r in result["requests"]}
    assert by_id["delayed"]["ttft_s"] == pytest.approx(4.3)
    assert by_id["delayed"]["eligible_s"] == 4.
    assert by_id["prefill_only"]["first_s"] is None and by_id["prefill_only"]["done"]
    baseline, post, all_rows = [result["windows"][name]["all"] for name in ("baseline", "post", "all")]
    assert baseline["arrivals"] == 3 and baseline["completed"] == 2
    assert baseline["first_tokens_observed"] == baseline["ttft_right_censored"] == baseline["ttft_unresolved"] == 1
    assert post["arrivals"] == post["unfinished"] == post["not_yet_eligible"] == 1
    assert all_rows["zero_output_requests"] == 1 and all_rows["output_requests"] == 3
    assert all_rows["known_ttft_violations"] == all_rows["ttft_unresolved"] == 1
    assert all_rows["ttft_violation_fraction_lower"] == pytest.approx(1 / 3)
    assert all_rows["ttft_violation_fraction_upper"] == pytest.approx(2 / 3)
    assert all_rows["tpot_completed_requests"] == all_rows["tpot_right_censored"] == 1
    assert result["resident_latency_validated"] is False


def test_checker_counts_missing_first_tokens_past_target_as_known_violations():
    result = check_requests([request("waiting", release=100.)], C, {"all": (0., 5.)}, 5.)
    summary = result["windows"]["all"]["all"]
    assert summary["known_ttft_violations"] == summary["ttft_right_censored"] == 1
    assert summary["ttft_violation_fraction_lower"] == summary["ttft_violation_fraction_upper"] == 1.
    assert summary["ttft_s"]["p90"] is None and summary["ttft_lower_bound_s"]["p90"] == 5.
    for windows in ({}, {"outside": (0., 6.)}, {"empty": (1., 1.)}):
        with pytest.raises(ValueError, match="windows"):
            check_requests([], C, windows, 5.)


def test_arrival_window_can_observe_later_completions_without_selecting_later_arrivals():
    rows = [request("delayed", arrival=.1, release=4.), request("later", arrival=2.)]
    result = check_requests(rows, C, {"early": (0., 1.), "early_followed": (0., 1., 5.)}, 5.)
    early, followed = [result["windows"][name]["all"] for name in ("early", "early_followed")]
    assert early["arrivals"] == followed["arrivals"] == 1
    assert early["ttft_right_censored"] == early["unfinished"] == 1
    assert followed["completed"] == followed["known_ttft_violations"] == 1
    assert followed["ttft_s"]["p90"] == pytest.approx(4.3)
    for bad in ((0., 1., .5), (0., 1., 6.), (0., 1., 5., 5.)):
        with pytest.raises(ValueError, match="windows"):
            check_requests(rows, C, {"bad": bad}, 5.)
