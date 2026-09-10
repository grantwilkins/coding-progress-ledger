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
