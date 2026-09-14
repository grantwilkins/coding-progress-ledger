"""Reduce raw server timing without fitting or equating GPU and host clocks."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as file:
        if '.jsonl' in str(path):
            for line in file:
                yield json.loads(line)
        else:
            yield json.load(file)


def client_rows(value):
    if isinstance(value, dict):
        if 'done' in value and 'status' in value:
            yield value
        else:
            for nested in value.values():
                yield from client_rows(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from client_rows(nested)


def domain(identity):
    keys = ('host', 'boot_id', 'time_namespace', 'time_namespace_offsets')
    return tuple(identity[key] for key in keys) if all(key in identity for key in keys) else None


def distribution(values):
    return dict(count=len(values), min=min(values), median=statistics.median(values), max=max(values)) if values else dict(count=0)


def reduce(telemetry_paths, clients, client_identity=None):
    errors, processes, schedules, workers, registrations = [], {}, {}, {}, {}
    ingress, http_maps = {}, {}
    boundaries = {kind: {} for kind in ('frontend_receipt', 'collector_put', 'collector_pop')}
    request_iterations = defaultdict(list)
    gpu_durations = defaultdict(list)
    ready_times = defaultdict(list)
    id_maps = defaultdict(set)
    def error(kind, **detail):
        errors.append(dict(kind=kind, **detail))
    def token_insert(target, request_id, ordinal, token, record, process):
        key = (request_id, ordinal)
        if key in target:
            error('duplicate_token', request_id=request_id, ordinal=ordinal, boundary=record['kind'])
        target[key] = (token, record, process)
    for path in telemetry_paths:
        rows = iter(read_json(path))
        identity = next(rows, None)
        if identity is None or identity.get('kind') != 'process_start':
            error('missing_process_start', path=str(path))
            continue
        process = str(path)
        processes[process] = identity
        if domain(identity) is None:
            error('missing_clock_domain', path=process)
        previous = identity['sequence']
        if previous != 1:
            error('invalid_initial_sequence', path=process)
        last = identity
        modules = []
        for row in rows:
            if row.get('sequence') != previous + 1:
                error('record_sequence_gap', path=process, previous=previous, current=row.get('sequence'))
            previous = row.get('sequence', previous)
            last = row
            kind = row['kind']
            if kind == 'module':
                modules.append(dict(path=row['path'], sha256=row['sha256']))
            elif kind == 'schedule':
                iteration = row['iteration']
                if iteration in schedules:
                    error('duplicate_iteration', iteration=iteration)
                schedules[iteration] = (row, process)
                q = sum(request['scheduled_tokens'] for request in row['requests'])
                if q > 8192 or q < 0:
                    error('scheduled_token_budget', iteration=iteration, total=q)
                for request in row['requests']:
                    if request['scheduled_tokens']:
                        request_iterations[request['request_id']].append((iteration, request))
            elif kind == 'worker_output':
                if row['iteration'] in workers:
                    error('duplicate_worker_iteration', iteration=row['iteration'])
                workers[row['iteration']] = (row, process)
                for phase in ('forward', 'logits', 'sample'):
                    value = row[phase + '_stream_ms']
                    if value < 0:
                        error('negative_cuda_elapsed', iteration=row['iteration'], phase=phase)
                    gpu_durations[phase].append(value)
                for request in row['requests']:
                    if request['token_ids']:
                        ready_times[(process, request['request_id'])].append(row['output_ready_ns'])
            elif kind == 'frontend_registration':
                registrations[row['request_id']] = (row, process)
                id_maps[row['external_request_id']].add(row['request_id'])
            elif kind == 'http_ingress':
                ingress[row['ingress_id']] = row
            elif kind == 'http_request_mapping':
                http_maps[row['request_id']] = row['ingress_id']
            elif kind in boundaries:
                for offset, token in enumerate(row['token_ids']):
                    token_insert(boundaries[kind], row['request_id'], row['ordinal_start'] + offset, token, row, process)
        if last['kind'] != 'process_final' or last.get('dropped_records') != 0 or last.get('records_before_final') != previous - 1:
            error('missing_or_invalid_process_final', path=process, last_sequence=previous)
        identity['imported_modules'] = modules
        if not modules:
            error('missing_imported_module_hashes', path=process)
    generated = {}
    for iteration, (row, process) in workers.items():
        if iteration not in schedules:
            error('worker_without_scheduler', iteration=iteration)
            continue
        scheduled_requests = {r['request_id']: r for r in schedules[iteration][0]['requests'] if r['scheduled_tokens']}
        for request in row['requests']:
            if request['request_id'] not in scheduled_requests:
                error('worker_request_not_scheduled', iteration=iteration, request_id=request['request_id'])
            for offset, token in enumerate(request['token_ids']):
                token_insert(generated, request['request_id'], request['ordinal_start'] + offset, token, row, process)
    for iteration, (row, _) in schedules.items():
        if any(r['scheduled_tokens'] for r in row['requests']) and iteration not in workers:
            error('scheduled_iteration_without_output', iteration=iteration)
    lags = defaultdict(list)
    for external, ingress_id in http_maps.items():
        if ingress_id not in ingress:
            error('http_mapping_without_ingress', request_id=external)
        if external not in id_maps and external + '-0' not in id_maps:
            error('http_request_without_frontend_registration', request_id=external)
    for ingress_id in ingress.keys() - set(http_maps.values()):
        error('http_ingress_without_request_mapping', ingress_id=ingress_id)
    for internal, (registration, process) in registrations.items():
        external = registration['external_request_id']
        if external not in http_maps and external.removesuffix('-0') not in http_maps:
            error('frontend_registration_without_http_mapping', request_id=internal)
        candidates = request_iterations.get(internal, [])
        if candidates:
            first, scheduler_process = min((schedules[iteration] for iteration, _ in candidates), key=lambda pair: pair[0]['start_ns'])
            if domain(processes[process]) is not None and domain(processes[process]) == domain(processes[scheduler_process]):
                delta = (first['start_ns'] - registration['mono_ns']) / 1e6
                if delta < 0:
                    error('negative_host_interval', request_id=internal, boundary='first_schedule', milliseconds=delta)
                lags['registration_to_first_schedule_ms'].append(delta)
    for kind, records in boundaries.items():
        previous_boundary = generated if kind == 'frontend_receipt' else boundaries['frontend_receipt' if kind == 'collector_put' else 'collector_put']
        for key, (token, row, process) in records.items():
            if key not in previous_boundary:
                error('unjoined_token', request_id=key[0], ordinal=key[1], boundary=kind)
                continue
            before_token, before_row, before_process = previous_boundary[key]
            if token != before_token:
                error('token_mismatch', request_id=key[0], ordinal=key[1], boundary=kind)
            before_domain, after_domain = domain(processes[before_process]), domain(processes[process])
            if before_domain is not None and before_domain == after_domain:
                before_ns = before_row.get('output_ready_ns', before_row['mono_ns'])
                delta = (row['mono_ns'] - before_ns) / 1e6
                if delta < 0:
                    error('negative_host_interval', request_id=key[0], boundary=kind, milliseconds=delta)
                lags[kind + '_from_previous_ms'].append(delta)
        missing = previous_boundary.keys() - records.keys()
        if missing:
            error('upstream_tokens_without_downstream', boundary=kind, count=len(missing))
    for kind, records in dict(worker=generated, **boundaries).items():
        ordinals = defaultdict(list)
        for request_id, ordinal in records:
            ordinals[request_id].append(ordinal)
        for request_id, values in ordinals.items():
            if sorted(values) != list(range(len(values))):
                error('noncontiguous_token_ordinals', boundary=kind, request_id=request_id)
    counts = Counter()
    delivered_counts = Counter(key[0] for key in boundaries['collector_pop'])
    joined_clients = set()
    missing_client_ids = []
    client_durations, causal_queue = [], []
    for client in clients:
        if client.get('timing_mode') == 'off' or client.get('telemetry_expected') is False:
            counts['uninstrumented_client_requests'] += 1
            continue
        counts['client_requests'] += 1
        complete = client.get('done') and client.get('status') == 200
        counts['completed' if complete else 'failed_or_censored'] += 1
        if 'start_ns' not in client and 'scheduled_ns' in client and client.get('status') in ('censored', 'dependency_failed'):
            counts['undispatched_offered_requests'] += 1
            continue
        external = client.get('request_id')
        internal_ids = id_maps.get(external + '-0', set()) if external else set()
        internal_ids = internal_ids or id_maps.get(external, set())
        if len(internal_ids) != 1:
            missing_client_ids.append(external)
            continue
        internal = next(iter(internal_ids))
        joined_clients.add(internal)
        tokens = client.get('token_ids', [token for event in client.get('token_events', []) for token in event['token_ids']])
        if complete and len(tokens) != client['output_tokens']:
            error('client_usage_mismatch', request_id=external, observed=len(tokens), usage=client['output_tokens'])
        for ordinal, token in enumerate(tokens):
            key = (internal, ordinal)
            if key not in boundaries['collector_pop'] or boundaries['collector_pop'][key][0] != token:
                error('client_token_not_joined', request_id=external, ordinal=ordinal)
        if complete:
            delivered = delivered_counts[internal]
            if delivered != len(tokens):
                error('completed_delivery_count_mismatch', request_id=external, server_tokens=delivered, client_tokens=len(tokens))
            client_durations.append((client['end_ns'] - client['start_ns']) / 1e9)
        if client.get('scheduled_ns') is not None:
            causal_queue.append((client['start_ns'] - client['scheduled_ns']) / 1e9)
        if domain(client_identity or {}) is not None:
            events = client.get('token_events', [])
            ordinal = 0
            for event in events:
                for token in event['token_ids']:
                    entry = boundaries['collector_pop'].get((internal, ordinal))
                    if entry and domain(processes[entry[2]]) == domain(client_identity):
                        delta = (event['monotonic_ns'] - entry[1]['mono_ns']) / 1e6
                        if delta < 0:
                            error('negative_host_interval', request_id=external, boundary='client_receive', milliseconds=delta)
                        lags['client_receive_from_collector_pop_ms'].append(delta)
                    ordinal += 1
    if missing_client_ids:
        error('client_request_mapping_missing', count=len(missing_client_ids), request_ids=missing_client_ids)
    for internal in registrations.keys() - joined_clients:
        error('frontend_request_without_client_record', request_id=internal)
    undelivered = [key for key in generated if key not in boundaries['frontend_receipt']]
    if undelivered:
        error('generated_tokens_without_frontend_receipt', count=len(undelivered), explanation='May include in-flight tokens discarded after EOS/abort; unresolved until engine discard evidence is joined.')
    ready_intervals = [(b - a) / 1e6 for values in ready_times.values() for a, b in zip(sorted(values), sorted(values)[1:])]
    return dict(schema='qh-server-timing-quality-v1', accepted=not errors and bool(workers) and bool(counts['client_requests']),
                errors=errors, process_identity=processes, counts=dict(counts, scheduler_iterations=len(schedules),
                worker_iterations=len(workers), generated_tokens=len(generated), **{kind + '_tokens': len(rows) for kind, rows in boundaries.items()}),
                gpu_stream_elapsed_ms={phase: distribution(values) for phase, values in gpu_durations.items()},
                server_output_ready_intervals_ms=distribution(ready_intervals),
                same_clock_host_intervals_ms={kind: distribution(values) for kind, values in lags.items()},
                client_completion_s=distribution(client_durations), causal_client_queue_s=distribution(causal_queue),
                limitations=['CUDA events are relative stream elapsed time and can include waits, not exclusive kernel busy time.',
                             'Host intervals require matching host, boot ID, time namespace and offsets; missing domains are never subtracted.',
                             'Collector pop is not wire flush; cross-host delivery is not derived from monotonic subtraction.',
                             'Registration-to-first-schedule is the observed server admission delay; client causal queue is dispatch minus offered time.',
                             'This reducer does not certify offered traces, frozen warm cache conditions or calibration fits.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', type=Path, nargs='+', required=True)
    parser.add_argument('--clients', type=Path, nargs='+', required=True)
    parser.add_argument('--client-identity', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    clients = [row for path in args.clients for value in read_json(path) for row in client_rows(value)]
    result = reduce(args.telemetry, clients, json.loads(args.client_identity.read_text()) if args.client_identity else None)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(accepted=result['accepted'], counts=result['counts'], errors=len(result['errors']))))
    raise SystemExit(0 if result['accepted'] else 1)


if __name__ == '__main__':
    main()
