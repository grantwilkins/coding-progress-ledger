"""Reduce controlled paired source diagnostics; never infer fleet SLO feasibility."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

PROXY = 'monotonic_ns,wall_ns,interval_ns,connection_id,route,direction,bytes,billed'
RESP = 'connection_id,command,key_hashes,start_ns,end_ns,request_wire_bytes,response_wire_bytes,request_body_bytes,payload_bytes'
METRICS = ('prefix_cache_queries_total', 'prefix_cache_hits_total', 'external_prefix_cache_hits_total', 'request_queue_time_seconds_sum', 'request_prefill_time_seconds_sum', 'request_decode_time_seconds_sum', 'request_prefill_kv_computed_tokens_sum', 'num_preemptions_total')


def wire_summary(rows, proxy, start=None, end=None):
    selected = [r for r in rows if r['command'] == 'GET' and int(r['payload_bytes']) > 0 and (start is None or int(r['start_ns']) < end and int(r['end_ns']) > start)]
    unique = {}
    for r in selected:
        value = int(r['payload_bytes'])
        if r['key_hashes'] in unique and unique[r['key_hashes']] != value:
            raise ValueError('same KV key has inconsistent payload size')
        unique[r['key_hashes']] = value
    buckets = [r for r in proxy if start is None or int(r['monotonic_ns']) < end and int(r['monotonic_ns']) + int(r['interval_ns']) > start]
    directions = {}
    for r in buckets:
        key = r['route'] + '/' + r['direction']
        directions[key] = directions.get(key, 0) + int(r['bytes'])
    return {'unique_get_keys': len(unique), 'unique_get_payload_bytes': sum(unique.values()),
        'get_transfers': len(selected), 'actual_get_payload_bytes_including_repeated_keys': sum(int(r['payload_bytes']) for r in selected),
        'retransferred_get_payload_bytes': sum(int(r['payload_bytes']) for r in selected)-sum(unique.values()),
        'get_request_wire_bytes': sum(int(r['request_wire_bytes']) for r in selected),
        'get_response_wire_bytes': sum(int(r['response_wire_bytes']) for r in selected),
        'get_request_protocol_bytes': sum(int(r['request_wire_bytes']) for r in selected),
        'get_request_resp_framing_bytes': sum(int(r['request_wire_bytes'])-int(r['request_body_bytes']) for r in selected),
        'get_request_command_key_bytes': sum(int(r['request_body_bytes']) for r in selected),
        'get_response_protocol_bytes': sum(int(r['response_wire_bytes'])-int(r['payload_bytes']) for r in selected),
        'get_first_start_ns': min((int(r['start_ns']) for r in selected), default=None),
        'get_last_end_ns': max((int(r['end_ns']) for r in selected), default=None),
        'whole_proxy_bucket_bytes_by_direction': directions,
        'window_ns': [start, end], 'scope': 'Whole intersecting transfer records and proxy buckets, never prorated; phase windows may overlap. Scenario totals count each record once. Unique payload does not count retransfers; actual GET payload does.'}


def reduce(root):
    hashes, gaps = {}, []
    def raw(path):
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data
    def read(path, default=None):
        return json.loads(raw(path)) if path.exists() else default
    def lines(path):
        if not path.exists(): return []
        data = raw(path)
        if data and not data.endswith(b'\n'):
            tail = data[data.rfind(b'\n')+1:]
            gaps.append({'path': str(path), 'unfinished_final_line_bytes': len(tail), 'sha256': hashlib.sha256(tail).hexdigest()})
            data = data[:data.rfind(b'\n')+1]
        return [json.loads(line) for line in data.splitlines() if line]
    def table(folder, name, header=None):
        path = folder/name
        if path.exists(): return list(csv.DictReader(io.StringIO(raw(path).decode())))
        path = folder/('attached-' + name + '.raw')
        return list(csv.DictReader(io.StringIO(header + '\n' + raw(path).decode()))) if header and path.exists() else []
    plan = read(root/'paired-plan.json')
    attempts = {r['scenario']: r for r in read(root/'paired-attempts.json', [])}
    scenarios, lanes, requests = [], [], []
    for spec in plan['scenarios']:
        folder = root/spec['scenario_id']
        spec = read(folder/'scenario.json', spec)
        attempt = attempts.get(spec['scenario_id'], read(folder/'attempt.json', {'status': 'unmeasured'}))
        result = read(folder/'result.json', {})
        events = lines(folder/'events.jsonl')
        state = {e['request_id']: e['valid'] for e in events if e['event'] == 'state_validation'}
        starts = [e for e in events if e['event'] == 'request_start']
        ends = [e for e in events if e['event'] == 'request_end']
        exact = sum(bool(e['result']['exact_token_timestamps']) for e in ends)
        success = attempt['status'] == 'complete' and result.get('status') == 'complete' and not any(m['error'] for m in result.get('migrations', [])) and len(result.get('migrations', [])) == len(spec['sessions'])
        continuations = result.get('continuations', [])
        success = success and len(continuations) == len(spec['sessions']) and all(
            c.get('status_code') == 200 and state.get(c.get('request_id')) is True
            and c.get('context_hash') == c.get('committed_context_hash')
            and c.get('context_hash') is not None for c in continuations)
        base = {'scenario': spec['scenario_id'], 'seed': spec['seed'], 'method': spec['method'], 'target_context': spec['context_size'], 'target_append': spec['activity_tokens'], 'width': len(spec['sessions'])}
        for event in ends:
            r = event['result']
            matching = [e for e in starts if e['session_id'] == event['session_id'] and e['route_port'] == event['route_port'] and e['context_hash'] == event['context_hash'] and e['monotonic_ns'] <= event['monotonic_ns']]
            label = max(matching, key=lambda e: e['monotonic_ns'])['request_id'] if matching else None
            requests.append(base | {'session': event['session_id'], 'phase_label': label, 'route_port': event['route_port'], 'state_valid': state.get(r['request_id']), **{k:r.get(k) for k in ('request_id','status_code','start_ns','first_byte_ns','last_token_ns','end_ns','prompt_tokens','output_tokens','cached_tokens','exact_token_timestamps','context_hash')},
                'request_mean_tpot_s': (r['last_token_ns']-r['first_byte_ns'])/1e9/(r['output_tokens']-1) if r.get('exact_token_timestamps') and r['output_tokens'] > 1 else None})
        for session in spec['sessions']:
            sid = session['session_id']
            ev = [e for e in events if e.get('session_id') == sid]
            move = next((m for m in result.get('migrations', []) if m['move']['session_id'] == sid), {})
            row = base | {'session': sid, 'attempt_status': attempt['status'], 'scenario_valid': success, 'error': move.get('error')}
            row.update({k:move.get(k) for k in ('queued_ns','initial_start_ns','initial_end_ns','pause_start_ns','idle_ns','catch_up_start_ns','catch_up_end_ns','switch_start_ns','switch_end_ns')})
            for phase in ('initial', 'catch_up'):
                phaseevents = [e for e in ev if e.get('phase') == phase]
                for kind, suffix in (('copy_start','start_ns'),('copy_end','end_ns')):
                    observed = next((e['monotonic_ns'] for e in phaseevents if e['event'] == kind), None)
                    row['event_' + phase + '_' + suffix] = observed
                request = move.get(phase) or {}
                row.update({phase + '_' + k:request.get(k) for k in ('prompt_tokens','cached_tokens','processed_tokens','processed_tokens_basis','exact_token_timestamps','context_hash')})
            row['actual_full_context_growth_tokens'] = row['catch_up_prompt_tokens']-row['initial_prompt_tokens'] if row['catch_up_prompt_tokens'] is not None and row['initial_prompt_tokens'] is not None else None
            row['source_activities'] = [a for a in result.get('activities', []) if a['session_id'] == sid]
            a, z = row['event_initial_start_ns'], row['event_initial_end_ns']
            row['source_activity_requests'] = [r | {'overlapped_initial_copy_from_client_events': r['start_ns'] < z and r['end_ns'] > a if a is not None and z is not None else None} for r in requests if r['scenario'] == spec['scenario_id'] and r['session'] == sid and (r['phase_label'] or '').startswith('controlled_turn_')]
            row.update({'event_' + key + '_ns': next((e['monotonic_ns'] for e in ev if e['event'] == key), None) for key in ('pause','idle','route_switch')})
            row['snapshot_events'] = [e for e in ev if e['event'] in ('snapshot','pause','idle','route_switch','resume_source')]
            pause, idle = row['event_pause_ns'], row['event_idle_ns']
            row['pause_to_idle_s'] = (idle-pause)/1e9 if pause is not None and idle is not None else None
            row['pause_during_source_request_verified'] = any(
                r['start_ns'] < pause < r['end_ns'] for r in row['source_activity_requests']) if pause is not None else False
            for phase in ('initial', 'catch_up'):
                a, z = row['event_'+phase+'_start_ns'], row['event_'+phase+'_end_ns']
                row[phase+'_elapsed_s'] = (z-a)/1e9 if a is not None and z is not None else None
            row['continuations'] = [c for c in result.get('continuations', []) if c['session_id'] == sid]
            lanes.append(row)
        proxy, resp = table(folder,'proxy_bytes.csv',PROXY), table(folder,'resp_transfers.csv',RESP)
        wire = {'scenario':wire_summary(resp,proxy)}
        for phase in ('initial','catch_up'):
            active = [e for e in events if e.get('phase') == phase and e['event'] in ('copy_start','copy_end')]
            begin = [e['monotonic_ns'] for e in active if e['event'] == 'copy_start']
            finish = [e['monotonic_ns'] for e in active if e['event'] == 'copy_end']
            if begin and finish: wire[phase] = wire_summary(resp,proxy,min(begin),max(finish))
        engine = {}
        for role in ('source','destination'):
            samples = table(folder,'engine-' + role + '.csv')
            engine[role] = {'samples':len(samples),'interval_ns':[samples[0]['monotonic_ns'],samples[-1]['monotonic_ns']] if samples else None,
                'max_sample_gap_s':max(((int(z['monotonic_ns'])-int(a['monotonic_ns']))/1e9 for a,z in zip(samples,samples[1:])),default=None),
                'gauges':{key:{'min':min(values,default=None),'max':max(values,default=None)} for key in ('num_requests_running','num_requests_waiting','kv_cache_usage_perc') for values in ([float(r['vllm:'+key]) for r in samples if r.get('vllm:'+key)],)},
                'counter_deltas':{k:float(samples[-1]['vllm:'+k])-float(samples[0]['vllm:'+k]) if len(samples)>1 and samples[0].get('vllm:'+k) and samples[-1].get('vllm:'+k) else None for k in METRICS}}
        scenarios.append(base | {'attempt_status':attempt['status'], 'scenario_valid':success,'deadline_met':result.get('deadline_met'),
            'requests_started':len(starts),'request_end_records':len(ends),'unfinished_request_count':len(starts)-len(ends),'exact_timing_requests':exact,
            'exact_timing_fraction_of_started':exact/len(starts) if starts else None,'state_valid_requests':sum(state.values()),
            'latency_screen_usable':success and bool(starts) and exact/len(starts)>=.99 and len(state)==len(starts) and all(state.values()),
            'wire':wire,'engine':engine})
    return {'scope':'Controlled one-append source migration diagnostic; no continuing resident arrival process or fleet SLO validation.',
        'campaign_ready':False,'model_changes':[], 'scenarios':scenarios,'lanes':lanes,'requests':requests,'input_sha256':hashes,'partial_raw_lines':gaps,
        'limitations':['Native prefix hits and external retrieval remain separate engine counters; request cached_tokens may combine them. Missing fields stay null.',
            'Engine queue/prefill/decode counter sums overlap request execution and are not additive GPU time. Per-lane logical bytes are not summed because source key sets can overlap.',
            'Phase wire intervals include entire overlapping proxy buckets and RESP transfers; phase totals are not disjoint. Failed/timeout attempts remain invalid even if some requests finish.',
            'Actual activity overlap is preserved per lane; an at_s=0 scheduled turn does not guarantee overlap with initial migration.',
            'Raw token IDs/events/full prompts remain in source events.jsonl; reduction stores hashes and request-level timing, not reconstructed server token timing.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    value = reduce(args.root)
    value['command'] = 'python3 ' + str(Path(__file__)) + ' ' + str(args.root)
    value['reducer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root/'paired-analysis.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('scenario','attempt_status','scenario_valid','requests_started','exact_timing_fraction_of_started')} for r in value['scenarios']],indent=2))
