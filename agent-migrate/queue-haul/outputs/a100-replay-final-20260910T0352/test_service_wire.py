import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('service_wire',Path(__file__).with_name('service-wire.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def test_source_exports_are_not_get_payload_and_phase_needs_known_session_window():
    key=next(s.keyed.keys([1]*256,'model'))
    result={'spec':{'episode':'e','arm':'kv_transfer','workload':'coding','seed':7101},'epoch_ns':0,'boundary_ns':100,
        'migration_events':[{'kind':'initial_start','session':0,'monotonic_ns':1},{'kind':'initial_end','session':0,'monotonic_ns':10}]}
    requests=[{'episode':'e','cohort':'migration','session':0,'phase':'initial','full_prompt_token_ids':[1]*256}]
    row={'command':'GET','key_hashes':key,'payload_bytes':'100','request_wire_bytes':'30','response_wire_bytes':'110','request_body_bytes':'20','start_ns':'2','end_ns':'3'}
    value=s.episode(result,requests,[row,row,{**row,'command':'SET','payload_bytes':'0','request_body_bytes':'2000'}, {**row,'key_hashes':'unknown','start_ns':'20','end_ns':'21'}],'model')
    assert value['scenario']['total_payload_bytes']==300 and value['scenario']['unique_payload_bytes']==200
    assert value['scenario']['retransferred_payload_bytes']==100 and value['scenario']['protocol_bytes']==120
    assert value['set_request_body_bytes']==2000 and value['physical_wan_get_payload_bytes'] is None
    assert value['ownership_unmapped_records']==1 and not value['phase_attribution_complete']
    initial=next(r for r in value['lanes'] if r['phase']=='initial')
    assert initial['total_payload_bytes']==200 and initial['initial_full_rounded_prefix_transfer']


def test_origin_split_counts_zero_payload_protocol_and_excludes_local_reads():
    key=next(s.keyed.keys([1]*256,'model'))
    result={'spec':{'episode':'e','arm':'kv_transfer','workload':'coding','seed':7101},'epoch_ns':0,'boundary_ns':100,'migration_events':[]}
    requests=[{'episode':'e','cohort':'migration','session':0,'phase':'initial','full_prompt_token_ids':[1]*256}]
    row={'command':'GET','connection_id':'destination','key_hashes':key,'payload_bytes':'100','request_wire_bytes':'30','response_wire_bytes':'110','request_body_bytes':'20','start_ns':'2','end_ns':'3'}
    rows=[row,row,{**row,'connection_id':'source'},{**row,'key_hashes':'missing','payload_bytes':'0','response_wire_bytes':'5'}]
    value=s.episode(result,requests,rows,'model',{'destination_connection_ids':['destination'],'source_connection_ids':['source']})
    assert value['physical_wan_get_payload_bytes']==200 and value['source_local_get']['total_payload_bytes']==100
    dest=value['destination_get']
    assert dest['unique_payload_bytes']==100 and dest['retransferred_payload_bytes']==100
    assert dest['all_get_records']==3 and dest['zero_payload_get_records']==1
    assert dest['protocol_bytes']==115 and dest['wire_bytes']==315
    assert dest['request_protocol_bytes']==90 and dest['response_protocol_bytes']==25
    assert value['scenario']['total_payload_bytes']==dest['total_payload_bytes']+value['source_local_get']['total_payload_bytes']
