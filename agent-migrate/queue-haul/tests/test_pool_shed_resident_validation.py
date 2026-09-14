import pytest

from pool_shed_resident_validation import checks, comparison, server_conditioned, server_summary, window


def test_window_counts_censored_arrivals_and_only_complete_tpot():
    rows = [dict(scheduled_s=60., first_s=62., done=False, end_s=300., mean_tpot_s=None),
            dict(scheduled_s=65., first_s=None, done=False, end_s=None, mean_tpot_s=None),
            dict(scheduled_s=89.9, first_s=90.1, done=True, end_s=91., mean_tpot_s=.02)]
    result = window(rows, 60, 90)
    assert result["arrivals"] == result["unfinished"] == 3
    assert result["known_ttft_violations"] == 2
    assert result["ttft_p90_s"] == 2
    assert result["tpot_p90_s"] is None and result["tpot_samples"] == 0
    assert not comparison(2, None, .2)["pass"]
    assert comparison(None, .2, .2)["pass"] is None
    assert not checks([])["gate_pass"]


def server_fixture(cohort='resident'):
    row = dict(row_id='r', request_id='external', cohort=cohort, phase='service', session=1, turn=1,
        serving_role='destination', scheduled_s=42., client_dispatch_s=44., submitted_prompt_tokens=10,
        planned_prompt_tokens=10, planned_output_tokens=2, output_tokens=2, done=True,
        effective_cached_tokens=9, duration_s=60., episode='heldout', start_s=44., end_s=45.)
    event = dict(registration_ns=10**15, scheduled_ns=10**15, ready_ns=[10**15 + 200000000, 10**15 + 300000000],
        local_tokens_complete=True, client_prefix_joined=True, worker_only_tokens=0,
        domain=('destination', 'boot', 'namespace', 'offsets'))
    coefficients = dict(prefill_step_s=.2, prefill_token_s=0., prefill_attention_s=0.,
        decode_step_s=.1, decode_attention_s=0., endpoint_s=7.)
    return row, event, coefficients


def test_server_check_uses_common_server_intervals_and_excludes_endpoint():
    row, event, coefficients = server_fixture()
    result = server_conditioned([row], {'external': event}, coefficients)
    request = result['requests'][0]
    assert request['registration_to_ready_s']['predicted'] == pytest.approx(.2)
    assert request['server_ready_tpot_s']['predicted'] == pytest.approx(.1)
    assert request['first_schedule_wait_s']['predicted'] == 0
    shifted = {k: [n + 987654321 for n in v] if k == 'ready_ns' else v + 987654321 if k.endswith('_ns') else v
               for k, v in event.items()}
    assert server_conditioned([row], {'external': shifted}, coefficients) == result
    summary = server_summary(result)
    assert summary['residents']['registration_to_ready_s']['within_band'] == 1
    assert summary['resident_windows'][0]['metrics']['registration_to_ready_s']['p90']['pass']
    with pytest.raises(ValueError, match='clock domain'):
        server_conditioned([row], {'external': {**event, 'domain': None}}, coefficients)


def test_worker_only_migration_tokens_are_bracketed_without_certifying_delivery():
    row, event, coefficients = server_fixture('migration')
    row.update(output_tokens=1, planned_output_tokens=1)
    event.update(local_tokens_complete=False, worker_only_tokens=1)
    included = server_conditioned([row], {'external': event}, coefficients)['requests'][0]
    excluded = server_conditioned([row], {'external': event}, coefficients, False)['requests'][0]
    assert included['modeled_output_tokens'] == 2 and not included['local_tokens_complete']
    assert included['predicted_generation_span_s'] == pytest.approx(.1)
    assert excluded['modeled_output_tokens'] == 1 and excluded['predicted_generation_span_s'] == 0
    with pytest.raises(ValueError, match='joined token prefix'):
        server_conditioned([row], {'external': {**event, 'client_prefix_joined': False}}, coefficients)
    with pytest.raises(ValueError, match='service request'):
        server_conditioned([{**row, 'cohort': 'resident'}], {'external': event}, coefficients)


def test_missing_registration_is_only_excluded_after_residents_finish():
    row, event, coefficients = server_fixture()
    tail = {**row, 'row_id': 'tail', 'request_id': None, 'cohort': 'incoming', 'session': 2,
            'done': False, 'start_s': 50., 'effective_cached_tokens': None}
    result = server_conditioned([row, tail], {'external': event}, coefficients)
    assert result['excluded_unregistered_tail'] == ['tail']
    with pytest.raises(ValueError, match='overlaps resident evidence'):
        server_conditioned([row, {**tail, 'start_s': 45.}], {'external': event}, coefficients)
    with pytest.raises(ValueError, match='completed destination demand'):
        server_conditioned([row, {**tail, 'done': True}], {'external': event}, coefficients)
