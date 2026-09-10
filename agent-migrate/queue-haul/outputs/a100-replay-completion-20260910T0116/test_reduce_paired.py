import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('paired_reduction', Path(__file__).with_name('reduce-paired.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wire_counts_each_record_once_and_reports_unique_payload():
    rows = [dict(command='GET',key_hashes=key,start_ns='10',end_ns='20',payload_bytes=str(size),request_wire_bytes='8',response_wire_bytes=str(size+10)) for key,size in [('a',100),('a',100),('b',200)]]
    proxy = [dict(monotonic_ns='5',interval_ns='20',route='kv',direction='target_to_client',bytes='430')]
    result = module.wire_summary(rows,proxy,12,18)
    assert result['unique_get_payload_bytes'] == 300
    assert result['actual_get_payload_bytes_including_repeated_keys'] == 400
    assert result['get_response_protocol_bytes'] == 430
    assert result['whole_proxy_bucket_bytes_by_direction'] == {'kv/target_to_client':430}


def test_timeout_never_passes_and_partial_events_remain_explicit(tmp_path):
    plan = {'scenarios':[dict(scenario_id='a',seed=7101,method='replay',context_size=8192,activity_tokens=32,sessions=[{'session_id':'s'}])]}
    (tmp_path/'paired-plan.json').write_text(json.dumps(plan))
    (tmp_path/'paired-attempts.json').write_text(json.dumps([{'scenario':'a','status':'timeout'}]))
    folder=tmp_path/'a'; folder.mkdir()
    (folder/'result.json').write_text(json.dumps({'status':'complete','deadline_met':True,'migrations':[{'move':{'session_id':'s'},'error':None}]}))
    event={'event':'request_start','session_id':'s','route_port':1,'context_hash':'h','monotonic_ns':1,'request_id':'source_warm'}
    (folder/'events.jsonl').write_text(json.dumps(event)+'\n{"unfinished":')
    result=module.reduce(tmp_path)
    assert not result['scenarios'][0]['scenario_valid']
    assert not result['scenarios'][0]['latency_screen_usable']
    assert result['scenarios'][0]['unfinished_request_count']==1
    assert result['partial_raw_lines'][0]['unfinished_final_line_bytes']==14
    assert result['lanes'][0]['catch_up_cached_tokens'] is None
    assert result['scenarios'][0]['engine']['source']['counter_deltas']['prefix_cache_hits_total'] is None
