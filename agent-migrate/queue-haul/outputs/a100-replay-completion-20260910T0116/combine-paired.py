"""Summarize final bounded paired attempts without upgrading partial KV evidence."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

out=Path(__file__).resolve().parent
reports={name:json.loads((out/name/'paired-analysis.json').read_text()) for name in ('paired','paired-corrected')}
valid=[]
for block,r in reports.items():
    for scenario in r['scenarios']:
        if not scenario['scenario_valid']:continue
        lanes=[x for x in r['lanes'] if x['scenario']==scenario['scenario']]
        row={k:scenario[k] for k in ('scenario','seed','method','target_context','target_append','requests_started','exact_timing_requests','state_valid_requests')}
        row.update(block=block,initial_span_s=(max(x['initial_end_ns'] for x in lanes)-min(x['initial_start_ns'] for x in lanes))/1e9,
            catchup_span_s=(max(x['catch_up_end_ns'] for x in lanes)-min(x['catch_up_start_ns'] for x in lanes))/1e9,
            complete_switch_span_s=(max(x['switch_end_ns'] for x in lanes)-min(x['initial_start_ns'] for x in lanes))/1e9,
            updated_context_range=[min(x['catch_up_prompt_tokens'] for x in lanes),max(x['catch_up_prompt_tokens'] for x in lanes)],
            initial_native_cached_tokens=sorted({x['initial_cached_tokens'] for x in lanes}),catchup_cached_tokens=sorted({x['catch_up_cached_tokens'] for x in lanes}),
            maximum_pause_to_idle_ms=max(x['idle_ns']-x['pause_start_ns'] for x in lanes)/1e6,
            source_requests_overlapping_initial=sum(a['overlapped_initial_copy'] for x in lanes for a in x['source_activities']))
        valid.append(row)
folder=out/'paired-corrected/paired-7101-8192-kv_transfer'
header='connection_id,command,key_hashes,start_ns,end_ns,request_wire_bytes,response_wire_bytes,request_body_bytes,payload_bytes\n'
transfers=list(csv.DictReader(io.StringIO(header+(folder/'attached-resp_transfers.csv.raw').read_text())))
events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
corrected=reports['paired-corrected']['scenarios'][0]
gets=[r for r in transfers if r['command']=='GET' and int(r['payload_bytes'])>0]
sets=[r for r in transfers if r['command']=='SET']
assert len(valid)==3 and all(r['method']=='replay' for r in valid)
assert not any(s['scenario_valid'] for s in reports['paired-corrected']['scenarios'])
assert not any(e['event']=='route_switch' for e in events)
result={'scope':'Supplemental controlled-source append checks. Not original recorded agentic shared-resident episodes. All original/corrective failures preserved; no simulator timing coefficients fitted.',
    'counts':{name:{'scenario_statuses':dict(collections.Counter(s['attempt_status'] for s in r['scenarios'])),
        'requests_started':sum(s['requests_started'] for s in r['scenarios']),'http_complete_exact_requests':sum(s['exact_timing_requests'] for s in r['scenarios']),
        'state_valid_requests':sum(s['state_valid_requests'] for s in r['scenarios'])} for name,r in reports.items()},
    'valid_replay_conditions':valid,
    'corrected_kv':{'status':'timeout_at_global_cutoff; invalid', 'second_seed':'unmeasured_after_timeout; budget omission',
        'requests_started':corrected['requests_started'],'source_http_complete_exact_state_valid_requests':corrected['state_valid_requests'],
        'usable_exact_timing_fraction_of_started':corrected['exact_timing_fraction_of_started'],
        'destination_requests_started':sum(e['event']=='request_start' and e['route_port']==18400 for e in events),
        'destination_requests_completed':sum(e['event']=='request_end' and e['route_port']==18400 for e in events),
        'destination_stream_events_before_cutoff':sum(e['event']=='response_event' and e['request_label']=='kv_transfer_initial' for e in events),
        'source_unique_set_keys':len({r['key_hashes'] for r in sets}), 'source_set_request_body_sizes':dict(collections.Counter(r['request_body_bytes'] for r in sets)),
        'unique_get_keys':len({r['key_hashes'] for r in gets}),'get_payload_sizes':dict(collections.Counter(r['payload_bytes'] for r in gets)),
        'wire':corrected['wire']['scenario'],'destination_engine':corrected['engine']['destination'],
        'verified':'Destination log reports Retrieved8192tokens in0.019s, matching +8192external cache-hit tokens and zero native hits. Source produced264unique SET keys after source-export fix. This demonstrates actual external retrieval on one unfinished request.',
        'unvalidated':'No completed KV initial request, pause/catchup/commit or successful KV handoff. Partial RESP bytes do not establish a full32768-token wire anchor.16/17exact completed requests falls below99%coverage.',
        'byte_coverage':'RESP transfer records log after client drain; proxy buckets flush separately. At cutoff45complete GETpayload records exceed copied proxy bucket bytes. Preserve both amounts; do not force equality or call proxy bucket coverage complete.'},
    'modeling_status':{'warm_full_source_evolved_catchup':'verified for24lanes across3valid replay conditions; full messages and original512token probe retained',
        'source_activity_overlap':'24valid source requests overlap initial replay; maximum measured pause-to-idle<0.55ms because source work finished before pause. Long in-flight quiescence remains unmeasured.',
        'source_error_safety':'Data exposed stale-state switch after invalid source output; two-line wait_idle guard plus regression applied after original block. Invalid original switches remain failures.',
        'kv_path':'Source-export alignment corrected from observation; partial corrected run demonstrates8192external retrievedtokens, not a successful migration.',
        'wire_anchor':'Observed GETchunk12,582,912B per256tokens equals49,152B/token native geometry. Extrapolation to32768=1,610,612,736B is native geometry, not measured full-context effective wire.0.80GB remains separate and unvalidated.',
        'fleet_readiness':False},
    'reproduction':['python3 outputs/a100-replay-completion-20260910T0116/reduce-paired.py outputs/a100-replay-completion-20260910T0116/'+name for name in reports]+['python3 outputs/a100-replay-completion-20260910T0116/combine-paired.py'],
    'input_sha256':{str(out/name/'paired-analysis.json'):hashlib.sha256((out/name/'paired-analysis.json').read_bytes()).hexdigest() for name in reports}}
(out/'paired-final-facts.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
