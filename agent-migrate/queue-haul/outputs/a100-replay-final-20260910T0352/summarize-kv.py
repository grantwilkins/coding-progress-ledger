"""Summarize the four clean conditions and preserve failed original attempts."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
rows=[];hashes={}
for name in ('paired','paired-l2measured'):
    folder=root/name
    paths=[folder/'paired-analysis.json',folder/'keyed-wire-analysis.json']
    report,keyed=(json.loads(p.read_text()) for p in paths)
    hashes.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    for spec in report['scenarios']:
        lanes=[r for r in report['lanes'] if r['scenario']==spec['scenario']]
        wire=next(r for r in keyed['scenarios'] if r['scenario_id']==spec['scenario'])
        result_path=folder/spec['scenario']/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        row={'group':name,**{k:spec[k] for k in ('scenario','seed','target_context','target_append','width','attempt_status','scenario_valid','requests_started','request_end_records','exact_timing_fraction_of_started')},
            'migration_through_valid_continuations_s':result.get('elapsed_s'),
            'validated_destination_continuations':len(result.get('continuations',[])) if spec['scenario_valid'] else 0,
            **wire['scenario'],'wire_phase_attribution_complete':wire['phase_attribution_complete'],
            'all_proxy_byte_scope':'Original scenario totals include both source-local and destination GET responses; origin-filtered fields follow.',
            **{'destination_get_'+k:v for k,v in wire['destination_get'].items()},
            **{'source_local_get_'+k:v for k,v in wire['source_local_get'].items()},
            'connection_origin_proof':'attribution-proof.json',
            'pauses_while_source_request_inflight':sum(r['pause_during_source_request_verified'] for r in lanes)}
        for offset in (30,120):
            row[f'ownership_commits_by_{offset}s']=sum(m['switch_end_ns'] <= result['started_ns']+offset*1e9 for m in result.get('migrations',[]) if m.get('switch_end_ns')) if result else None
            row[f'valid_continuation_first_token_by_{offset}s']=sum(c['first_byte_ns'] <= result['started_ns']+offset*1e9 for c in result.get('continuations',[])) if spec['scenario_valid'] else None
        for phase in ('initial','catch_up'):
            starts=[r['event_'+phase+'_start_ns'] for r in lanes if r['event_'+phase+'_start_ns'] is not None]
            ends=[r['event_'+phase+'_end_ns'] for r in lanes if r['event_'+phase+'_end_ns'] is not None]
            row[phase+'_batch_elapsed_s']=(max(ends)-min(starts))/1e9 if len(starts)==len(ends)==spec['width'] else None
            elapsed=[r[phase+'_elapsed_s'] for r in lanes if r[phase+'_elapsed_s'] is not None]
            row[phase+'_lane_elapsed_min_s']=min(elapsed,default=None);row[phase+'_lane_elapsed_max_s']=max(elapsed,default=None)
            row[phase+'_request_cached_tokens']=[r[phase+'_cached_tokens'] for r in lanes]
            row[phase+'_actual_prompt_tokens']=[r[phase+'_prompt_tokens'] for r in lanes]
        rows.append(row)
clean=[r for r in rows if r['scenario_valid']]
if len(clean)!=4 or {(r['seed'],r['target_context']) for r in clean}!={(s,c) for s in (7101,7102) for c in (8192,30000)}:raise ValueError('four requested clean KV conditions are not complete')
value={'conditions':rows,'clean_conditions':len(clean),'clean_destination_continuations':sum(r['validated_destination_continuations'] for r in clean),
    'exact_completed_requests_clean_conditions':sum(r['request_end_records'] for r in clean),'input_sha256':hashes,
    'wire_interpretation':'Measured native256-token payload chunks12582912B =49152B/token, or1610612736B per32768serialized tokens before protocol/retransfers. This path does not reproduce the separate800000000B effective-wire anchor; no private-KV discount applied. Unique payload and repeated GET payload are distinct; protocol is RESP application framing/commands/keys, not TCP/IP retransmitted packets, HTTP/SSE or SSH overhead. Destination GET subsets use archived persistent-pool connection-origin proof; unfiltered original proxy totals also include source-local retrieval. The hardware proxy is1000Mbit/s aggregate, separate from the archived simulator1000Gbit/s scenario.',
    'timing_interpretation':'Elapsed initial/catch-up batch intervals overlap across lanes and must not be added. End-to-end includes completed ownership transitions and valid destination continuations. No resident traffic in these controlled conditions; no loaded latency or fleet SLO claim.',
    'counterexamples':'Both original30K attempts remain failed despite completed HTTP requests, because an unsupportedL1-only guard prevented complete continuation evidence. Only the guard changed; corrected attempts retain realL2refetch latency.','simulator_fitting':'stopped_pending_user_review'}
(root/'kv-observations.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
with (root/'kv-observations.csv').open('w') as handle:
    writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'clean_conditions':len(clean),'clean_continuations':value['clean_destination_continuations'],'retained_failed_conditions':len(rows)-len(clean)}))
