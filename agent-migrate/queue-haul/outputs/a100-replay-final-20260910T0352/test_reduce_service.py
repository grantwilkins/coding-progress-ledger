import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('service_reduction',Path(__file__).with_name('reduce-service.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_original_arrival_timing_and_cleanup_censoring():
    rows=[dict(scheduled_ns=61_000_000_000,start_ns=64_000_000_000,first_ns=65_000_000_000,end_ns=end,
        status=200,done=True,exact_token_timestamps=True,mean_tpot_s=.02,prompt_tokens=8192,cached_tokens=None)
        for end in (66_000_000_000,310_000_000_000)]
    trace=[{'offset_s':61},{'offset_s':61}]
    result=module.window(rows,trace,0,60,300)
    assert result['p90_original_arrival_ttft_s']==4
    assert result['p90_request_mean_tpot_s']==.02
    assert result['exact_timing_coverage_of_completed']==1
    assert result['exact_completed_fraction_of_arrivals']==.5
    assert result['outstanding_all_prior_arrivals']==1
    assert result['cached_tokens'] is None
    assert not result['completed_request_latency_screen']
    assert not result['tail_guarantee']


def test_missing_engine_telemetry_is_unknown():
    result=module.engine_window([{'monotonic_ns':1}],0,0,1)
    assert result['gauges']['kv_cache_usage_perc']['max'] is None
    assert result['counter_deltas']['num_preemptions_total'] is None
    assert result['max_sample_gap_s'] is None


def test_censored_stream_keeps_observed_ttft_without_claiming_full_tpot():
    import json
    row=dict(episode='e',cohort='resident',session=2,turn=7,status='censored',done=False,scheduled_ns=70_000_000_000)
    events=[dict(episode='e',cohort='resident',session=2,turn=7,monotonic_ns=t,
        data=json.dumps({'id':'r','choices':[{'token_ids':[1]}]})) for t in (70_200_000_000,90_000_000_000)]
    partial=module.recover_partial([row],events)
    summary=module.window([row],[{'offset_s':70}],0,30,90)
    assert partial[0]['observed_token_ids']==2 and partial[0]['mean_request_tpot_s'] is None
    assert summary['p90_original_arrival_ttft_s']==.2 and summary['ttft_observed_requests']==1
    assert summary['outstanding_all_prior_arrivals']==1 and summary['tpot_requests']==0
    assert not summary['completed_request_latency_screen']


def test_pause_after_last_token_is_not_verified_execution_overlap():
    request=dict(request_id='r',session=0,scheduled_ns=0,start_ns=1,cohort='incoming',serving_role='source',exact_token_timestamps=True,first_ns=10,last_token_ns=20,end_ns=30)
    events=[dict(kind='pause',session=0,monotonic_ns=25,in_flight_request={'request_id':'r','first_token_ns':10}),
        dict(kind='source_idle',session=0,monotonic_ns=31,last_source_request={'request_id':'r'})]
    assert not module.quiescence(events,[request])[0]['client_token_stream_overlap_verified']
    events[0]['monotonic_ns']=15
    result=module.quiescence(events,[request])[0]
    assert result['client_token_stream_overlap_verified'] and not result['server_execution_timestamps_available']
