import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('keyed_wire',Path(__file__).with_name('keyed-wire.py'))
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)


def test_keys_preserve_prefix_and_separate_changed_history_and_salt():
    base=list(k.keys([1]*512,'model'))
    assert list(k.keys([1]*512+[2]*256,'model'))[:2]==base
    assert not set(base)&set(k.keys([3]*512,'model'))
    assert not set(base)&set(k.keys([1]*512,'model','salt'))


def test_unique_lane_mapping_preserves_retransfers_and_unknown_keys():
    key=next(k.keys([1]*256,'model'))
    events=[{'event':'rendered_request','token_ids':[1]*256,'session_id':'a','request_label':'kv_transfer_initial'},
        {'event':'copy_start','phase':'initial','session_id':'a','monotonic_ns':0},
        {'event':'copy_end','phase':'initial','session_id':'a','monotonic_ns':5}]
    row={'command':'GET','key_hashes':key,'payload_bytes':'100','request_wire_bytes':'30','response_wire_bytes':'110','start_ns':'1','end_ns':'2'}
    result=k.attribute(events,[row,row,{**row,'key_hashes':'unknown'}],'model')
    assert result['scenario']['total_payload_bytes']==300 and result['scenario']['unique_payload_bytes']==200
    assert result['scenario']['retransferred_payload_bytes']==100 and result['scenario']['protocol_bytes']==120
    assert result['ownership_unmapped_records']==1 and not result['phase_attribution_complete']
    initial=next(r for r in result['lanes'] if r['phase']=='initial')
    assert initial['get_records']==2 and initial['initial_full_rounded_prefix_transfer']


def test_shared_key_is_never_assigned_arbitrarily_and_size_mismatch_fails():
    key=next(k.keys([1]*256,'model'))
    events=[{'event':'rendered_request','token_ids':[1]*256,'session_id':s} for s in ('a','b')]
    row={'command':'GET','key_hashes':key,'payload_bytes':'100','request_wire_bytes':'30','response_wire_bytes':'110','start_ns':'1','end_ns':'2'}
    result=k.attribute(events,[row],'model')
    assert result['ownership_multiple_session_records']==1 and not result['lanes']
    with pytest.raises(ValueError,match='inconsistent'):k.payload([row,{**row,'payload_bytes':'101'}])
