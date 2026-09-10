"""Compact the frozen two-GPU acquisition without inventing server timing."""
import argparse
import csv
import hashlib
import json
import shlex
import statistics
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path('outputs/a100-replay-final-20260910T0352')
OUTPUT = Path('outputs/a100-resident-queues-20260910')
SERVICE = ('resident', 'incoming')
ENGINE_FIELDS = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                 'num_preemptions_total', 'prefix_cache_queries_total', 'prefix_cache_hits_total',
                 'external_prefix_cache_hits_total', 'request_prefill_kv_computed_tokens_sum',
                 'request_queue_time_seconds_sum', 'request_prefill_time_seconds_sum',
                 'request_decode_time_seconds_sum')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def checked_lines(stream, expected):
    checksum, size = hashlib.sha256(), 0
    for line in stream:
        checksum.update(line); size += len(line)
        yield line
    if (size, checksum.hexdigest()) != (expected['bytes'], expected['sha256']):
        raise ValueError(f"archive member hash mismatch: {expected['path']}")


def token_observation(observed, event, boundary_ns, usage=None):
    if not event.get('data') or event['data'] == '[DONE]':
        return
    payload = json.loads(event['data'])
    if usage is not None and payload.get('usage') and not payload.get('choices'):
        usage[payload['id']] = payload['usage']
    if event.get('cohort') not in SERVICE:
        return
    for choice in payload.get('choices', []):
        tokens = choice.get('token_ids', [])
        if not tokens:
            continue
        key = tuple(event[k] for k in ('episode', 'cohort', 'session', 'turn'))
        row = observed.setdefault(key, {'first_ns': event['monotonic_ns'], 'output_tokens': 0,
                                      'tokens_before_boundary': 0, 'exact': True, 'request_id': payload['id'], 'gaps_s': []})
        if row['request_id'] != payload['id']:
            raise ValueError(f'multiple requests for service arrival: {key}')
        if row['output_tokens']:
            gap = (event['monotonic_ns']-row['last_token_ns'])/1e9
            if gap < 0:
                raise ValueError(f'token observations reversed client time: {key}')
            row['gaps_s'].append(gap)
        row['last_token_ns'] = event['monotonic_ns']
        row['output_tokens'] += len(tokens)
        row['tokens_before_boundary'] += len(tokens) * (event['monotonic_ns'] <= boundary_ns)
        row['exact'] &= len(tokens) == 1


def compact_request(raw, result, workload, observed, usage=None):
    episode, epoch, boundary = result['spec']['episode'], result['epoch_ns'], result['boundary_ns']
    service = raw['cohort'] in SERVICE
    key = (episode, raw['cohort'], raw['session'], raw.get('turn'))
    stream = observed.get(key, {})
    fields = ('cohort', 'session', 'turn', 'phase', 'serving_role', 'session_id', 'request_id', 'status', 'done',
              'prompt_tokens', 'planned_prompt_tokens', 'recorded_append_tokens', 'planned_output_tokens',
              'output_tokens', 'cached_tokens', 'derived_prompt_minus_cache_tokens', 'reset', 'recorded_turn',
              'actual_new_tokens_excluding_retained_output', 'context_hash', 'retained_context_hash',
              'exact_token_timestamps', 'mean_tpot_s', 'external_cache_bypassed')
    row = {k: result['spec'].get(k) for k in ('episode', 'workload', 'arm', 'seed', 'rate')}
    row.update({k: raw.get(k) for k in fields})
    row['row_id'] = '/'.join(map(str, key)) if service else raw['request_id']
    row['predecessor_id'] = '/'.join(map(str, (*key[:3], raw['turn']-1))) if service and raw['turn'] else None
    row['owner_role'] = 'destination' if raw['cohort'] == 'resident' else raw.get('serving_role', raw.get('ownership'))
    row['dispatched'] = raw.get('client_dispatch_ns') is not None
    row['duration_s'] = (boundary-epoch)/1e9
    row['completed_within_observation'] = bool(raw['done'] and raw['status'] == 200 and raw['end_ns'] <= boundary)
    row['submitted_prompt_tokens'] = len(raw['full_prompt_token_ids']) if 'full_prompt_token_ids' in raw else None
    row['observed_output_tokens'] = stream.get('output_tokens', raw.get('recorded_output_tokens'))
    row['observed_tokens_before_boundary'] = stream.get('tokens_before_boundary')
    row['partial_exact_token_timestamps'] = stream.get('exact') if not raw['done'] else None
    gaps = stream['gaps_s'] if stream.get('exact') else None
    row.update(token_gaps_exact_single_token_frames=stream.get('exact'),
               token_gap_count=len(gaps) if gaps is not None else None,
               submillisecond_token_gaps=sum(g < .001 for g in gaps) if gaps is not None else None,
               median_token_gap_s=statistics.median(gaps) if gaps else None,
               max_token_gap_s=max(gaps) if gaps else None)
    final_usage = (usage or {}).get(raw.get('request_id'))
    if usage is not None and raw['done'] and raw['status'] == 200 and final_usage is None:
        raise ValueError(f'completed request has no raw final usage: {key}')
    row['final_usage_observed'] = final_usage is not None
    row['effective_cached_tokens'] = raw.get('cached_tokens')
    row['cache_usage_basis'] = 'reported' if row['effective_cached_tokens'] is not None else 'unknown'
    if final_usage is not None and raw['done'] and raw['status'] == 200:
        if (final_usage['prompt_tokens'], final_usage['completion_tokens']) != (raw['prompt_tokens'], raw['output_tokens']):
            raise ValueError(f'final usage differs from completed request: {key}')
        reported = (final_usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if reported != raw.get('cached_tokens'):
            raise ValueError(f'cached usage differs from completed request: {key}')
        if reported is None:
            row.update(effective_cached_tokens=0, cache_usage_basis='vllm_0_22_omitted_zero')
    if service:
        sequence, offset = workload['turn_sequences'][raw['session']], workload['turn_offset'][raw['session']]
        index = (offset+raw['turn']) % len(sequence)
        shape = sequence[index]
        if raw.get('recorded_turn') is not None and raw['recorded_turn'] != index:
            raise ValueError(f'recorded trajectory order differs from offered order: {key}')
        row['planned_prompt_tokens'] = shape['context']+shape['prompt']
        row['planned_output_tokens'] = shape['output']
        row['planned_append_tokens'] = shape['prompt']
        row['planned_context_tokens'] = shape['context']
        row['planned_reset'] = raw['turn'] == 0 or index == 0 or shape['reset']
    for field in ('scheduled', 'client_wakeup', 'client_dispatch', 'start', 'first', 'last_token', 'end'):
        value = raw.get(field+'_ns')
        if value is None and field in ('first', 'last_token'):
            value = stream.get(field+'_ns')
        row[field+'_s'] = (value-epoch)/1e9 if value is not None else None
    row['arrival_ttft_s'] = row['first_s']-row['scheduled_s'] if row['first_s'] is not None else None
    row['ttft_s'] = row['first_s']-row['start_s'] if row['first_s'] is not None else None
    row['history_hash_after'] = (hashlib.sha256(json.dumps(raw['full_prompt_token_ids']+raw['token_ids'],
        separators=(',', ':')).encode()).hexdigest() if service and raw['done'] else None)
    if service and raw['done']:
        if stream.get('output_tokens') != raw['output_tokens'] or stream.get('request_id') != raw['request_id']:
            raise ValueError(f'completed request differs from raw token events: {key}')
        if raw['output_tokens'] != row['planned_output_tokens'] or raw['prompt_tokens'] != row['planned_prompt_tokens']:
            raise ValueError(f'completed request differs from planned trajectory: {key}')
    return row


def audit(rows, results, traces):
    offered = {(episode, r['cohort'], r['session'], r['turn']): r for episode, trace in traces.items() for r in trace}
    service = [r for r in rows if r['cohort'] in SERVICE]
    keys = [(r['episode'], r['cohort'], r['session'], r['turn']) for r in service]
    if len(keys) != len(set(keys)) or set(keys) != set(offered):
        raise ValueError('raw service rows do not match every unique offered arrival')
    histories = defaultdict(list)
    for row in service:
        key = row['episode'], row['cohort'], row['session'], row['turn']
        if abs(row['scheduled_s']-offered[key]['offset_s']) > 1e-8:
            raise ValueError(f'original scheduled arrival changed: {key}')
        if row['dispatched'] and row['client_dispatch_s'] < row['scheduled_s']:
            raise ValueError(f'dispatch preceded arrival: {key}')
        histories[key[:3]].append(row)
    links = transitions = 0
    for group in histories.values():
        group.sort(key=lambda r: r['turn'])
        for previous, row in zip(group, group[1:]):
            if row['dispatched'] and (not previous['done'] or previous['end_s'] > row['client_dispatch_s']):
                raise ValueError(f'session dependency violated: {row["row_id"]}')
            if row['dispatched'] and not row['reset']:
                if row['retained_context_hash'] != previous['history_hash_after']:
                    raise ValueError(f'retained history violated: {row["row_id"]}')
                links += 1
            transitions += previous['serving_role'] == 'source' and row['serving_role'] == 'destination'
        switches = [e for e in results[group[0]['episode']].get('migration_events', [])
                    if e['kind'] == 'route_switch' and e['session'] == group[0]['session']]
        for row in group:
            if row['cohort'] == 'resident' and row['dispatched'] and row['serving_role'] != 'destination':
                raise ValueError('resident request changed GPU')
            if row['cohort'] == 'incoming' and row['dispatched'] and switches:
                epoch = results[row['episode']]['epoch_ns']
                switch = (switches[0]['monotonic_ns']-epoch)/1e9
                if row['serving_role'] != ('source' if row['client_dispatch_s'] < switch else 'destination'):
                    raise ValueError('incoming dispatch disagrees with ownership switch')
                if row['serving_role'] == 'source' and row['end_s'] > switch:
                    raise ValueError('source request outlived ownership switch')
    return {'offered_service_requests': len(offered), 'service_requests': len(service),
            'service_status_counts': dict(Counter(str(r['status']) for r in service)),
            'undispatched_service_requests': sum(not r['dispatched'] for r in service),
            'incomplete_with_observed_tokens': sum(not r['done'] and r['first_s'] is not None for r in service),
            'verified_nonreset_history_links': links, 'source_to_destination_service_links': transitions}


def extract(source=SOURCE):
    hashes, results, traces, workloads = {}, {}, {}, {}
    def read(name):
        data = (source/name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    reference = read('service-analysis.json')
    for path in sorted(source.glob('*/result.json')):
        result = read(str(path.relative_to(source)))
        if 'spec' not in result:
            continue
        episode = result['spec']['episode']
        results[episode] = result
        traces[episode] = read(episode+'/offered-trace.json')
        workloads[episode] = read(episode+'/physical-workload.json')
    for name in hashes.keys() & reference['input_sha256'].keys():
        if hashes[name] != reference['input_sha256'][name]:
            raise ValueError(f'frozen reducer input changed: {name}')
    manifest = read('raw-telemetry-archive.json')
    source_command = next(r['command'] for r in read('source-launches.json') if r['name'] == 'source')
    commands = [shlex.split(source_command[-1]), read('stack/remote-commands.json')['sink']]
    patch_hash = digest(source.parent/'a100-replay-live-20260909T1920/output_processor.patched.py')
    for role, command in zip(('source', 'destination'), commands):
        runtime = read(role+'-runtime.json')
        recorded = read(role+'-runtime-sha256.json')
        if runtime['vllm'] != '0.22.0' or '--enable-prompt-tokens-details' not in command:
            raise ValueError('omitted cache usage requires the verified vLLM 0.22 details-enabled contract')
        if patch_hash != next(v for k, v in recorded.items() if k.endswith('/output_processor.py')):
            raise ValueError('frozen output processor differs from recorded runtime')
    if digest(source/manifest['archive']) != manifest['sha256']:
        raise ValueError('raw telemetry archive hash mismatch')
    required = {'requests.jsonl', 'request-events.jsonl'} | {episode+'/'+name+'.csv'
        for episode, result in results.items() for name in (('engine',) if result['spec']['arm'] == 'resident' else ('engine', 'engine-source'))}
    expected = {r['path']: r for r in manifest['members'] if r['path'] in required}
    if expected.keys() != required:
        raise ValueError('required raw inputs absent from archive manifest')
    observed, usage, rows, engine, found = {}, {}, [], [], set()
    process = subprocess.Popen(['zstd', '--long=27', '-dc', str(source/manifest['archive'])], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if member.name not in required:
                    continue
                if member.name in found:
                    raise ValueError(f'duplicate archive member: {member.name}')
                found.add(member.name)
                lines = checked_lines(archive.extractfile(member), expected[member.name])
                if member.name == 'requests.jsonl':
                    if 'request-events.jsonl' not in found:
                        raise ValueError('frozen archive must place request events before request summaries')
                    for line in lines:
                        row = json.loads(line)
                        if row.get('episode') in results:
                            rows.append(compact_request(row, results[row['episode']], workloads[row['episode']], observed, usage))
                elif member.name == 'request-events.jsonl':
                    for line in lines:
                        row = json.loads(line)
                        if row.get('episode') in results:
                            token_observation(observed, row, results[row['episode']]['boundary_ns'], usage)
                else:
                    episode = member.name.split('/')[0]
                    for row in csv.DictReader(line.decode() for line in lines):
                        engine.append({'episode': episode, 'serving_role': 'source' if member.name.endswith('engine-source.csv') else 'destination',
                            'time_s': (int(row['monotonic_ns'])-results[episode]['epoch_ns'])/1e9,
                            **{k: float(row['vllm:'+k]) if row.get('vllm:'+k) else None for k in ENGINE_FIELDS}})
                hashes[member.name] = expected[member.name]['sha256']
    except BaseException:
        process.kill()
        raise
    finally:
        process.stdout.close()
        code = process.wait()
    if code or found != required:
        raise ValueError(f'raw telemetry stream incomplete: exit={code}, missing={sorted(required-found)}')
    if any(hashes[k] != reference['input_sha256'][k] for k in hashes.keys() & reference['input_sha256'].keys()):
        raise ValueError('raw archive inputs differ from frozen service reduction')
    rows.sort(key=lambda r: (r['episode'], r['scheduled_s'], r['row_id']))
    events = [{k: v for k, v in {**result['spec'], **event, 'time_s': (event['monotonic_ns']-result['epoch_ns'])/1e9}.items()
               if not isinstance(v, (dict, list)) and k != 'monotonic_ns'}
              for result in results.values() for event in result.get('migration_events', [])]
    report = {'source': str(source), 'archive_sha256': manifest['sha256'], 'input_sha256': hashes,
        'episodes': [{**r['spec'], 'duration_s': (r['boundary_ns']-r['epoch_ns'])/1e9,
                      'placement': r['placement']} for r in results.values()],
        'request_rows': len(rows), 'request_cohort_counts': dict(Counter(r['cohort'] for r in rows)),
        'cache_usage_basis_counts': dict(Counter(r['cache_usage_basis'] for r in rows)),
        'engine_rows': len(engine), 'migration_event_rows': len(events), **audit(rows, results, traces),
        'scope': 'One source and one destination GPU per episode; residents retain their destination GPU and per-session causal history. Materialization and control probes are retained separately from service arrivals.',
        'timing_scope': 'All times are client/controller monotonic seconds relative to result.json epoch. TTFT includes server waiting and execution; no per-request server queue, prefill or decode scheduling timestamps are available.',
        'token_gap_scope': 'Token-gap statistics use all observed client-monotonic intervals, including observations after the episode boundary; compare last_token_s with duration_s. Statistics remain unknown unless every observed token frame contains exactly one token. Single-token framing does not exclude buffered delivery and does not establish physical GPU decode timing.',
        'cache_scope': 'Raw cached_tokens remains unchanged. effective_cached_tokens decodes omitted zero only for successful requests with raw final usage under the verified vLLM 0.22 details-enabled runtime; incomplete usage stays unknown. Cache counts are conditioned measurements, not an ex-ante cache prediction or synthetic replay discount.',
        'cache_serialization_source': 'https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/entrypoints/openai/completion/serving.py#L409-L424',
        'cache_serialization_verification': 'The zero-omission rule is inferred from the recorded vLLM version, enabled-details launch flags, matching recorded output-processor bytes and raw final usage. The completion API serializer itself was not separately hashed on the GPU nodes.',
        'verified_output_processor_sha256': patch_hash,
        'censoring_scope': 'Raw observations after the boundary remain recorded; completed_within_observation and observed_tokens_before_boundary retain the observation cutoff. Unknown completed output and cached tokens for unfinished requests remain missing.',
        'planned_scope': 'Planned service shapes follow the frozen offered per-session trajectory. A failed or unfinished predecessor prevents dependent execution; planned shapes for undispatched requests are potential demand, not completed work.',
        'engine_scope': 'Running/waiting counts cover all populations on the specified GPU. Histogram sums overlap across requests and are completion-accounted; never sum them as exclusive GPU service time.'}
    return rows, events, engine, report


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def load_requests(path=OUTPUT/'requests.csv'):
    strings = {'episode', 'workload', 'arm', 'cohort', 'phase', 'serving_role', 'session_id', 'request_id',
               'row_id', 'predecessor_id', 'owner_role', 'context_hash', 'retained_context_hash', 'history_hash_after', 'cache_usage_basis'}
    with Path(path).open() as stream:
        return [{k: None if v == '' else v if k in strings or k == 'status' and v != '200'
                 else {'True': True, 'False': False}[v] if v in ('True', 'False') else json.loads(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, events, engine, report = extract(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (('requests', rows), ('migration-events', events), ('engine', engine)):
        write_csv(args.out/(name+'.csv'), values)
    report['output_sha256'] = {name+'.csv': digest(args.out/(name+'.csv')) for name in ('requests', 'migration-events', 'engine')}
    report['reducer_sha256'] = digest(Path(__file__))
    (args.out/'data-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('request_rows', 'engine_rows', 'offered_service_requests', 'service_status_counts')}))


if __name__ == '__main__':
    main()
