"""Bounded, conditional GPU queue validation; never launches the fleet campaign."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import shlex

import numpy as np

from pool_shed_resident_data import OUTPUT, ENGINE_FIELDS, audit, checked_lines, compact_request, digest, load_requests, token_observation, write_csv
from pool_shed_resident_fit import UNLOADED, calibrate, calibrate_server_decode, prefill_seconds
from pool_shed_resident_queue import simulate
from pool_replay_server_reduce import domain


SERVER_OUTPUT = Path('outputs/a100-replay-queue-resolution-20260910')


def server_bundle(source):
    """Verify and compact the four East episodes, retaining failed global integrity."""
    source = Path(source)
    manifest = json.loads((source / 'sha256-manifest.json').read_text())
    hashes = {'sha256-manifest.json': digest(source / 'sha256-manifest.json')}
    def checked(name):
        path = source / name
        hashes[name] = digest(path)
        if hashes[name] != manifest[name]:
            raise ValueError('server evidence hash mismatch: ' + name)
        return path
    def read(name):
        return json.loads(checked(name).read_text())
    compressed = {r['original_path']: r for r in read('compression-manifest.json')}
    def lines(name):
        path = checked(compressed[name]['gzip_path'] if name in compressed else name)
        with (gzip.open(path, 'rb') if name in compressed else path.open('rb')) as stream:
            records = checked_lines(stream, {'path': name, 'bytes': compressed[name]['original_bytes'],
                                    'sha256': compressed[name]['original_sha256']}) if name in compressed else stream
            yield from (json.loads(line) for line in records)
    plan, report = read('plan.json'), read('report.json')
    certificate = read('critic-warm-certificate.json')
    if len(plan['agentic_contract']['episodes']) != 4 or report['episodes_completed'] != 4:
        raise ValueError('four frozen East episodes required')
    for role, prefix, name in (('source', 'stack', 'source'), ('destination', 'destination', 'sink')):
        launch = next(r for r in lines(prefix + '/launches.jsonl') if r['name'] == name)
        if read(role + '/runtime.json')['vllm'] != '0.22.0' or '--enable-prompt-tokens-details' not in shlex.split(launch['argv'][-1]):
            raise ValueError('native cache decoding requires the captured details-enabled v0.22 runtime')
        directory = ('destination/' if role == 'destination' else '') + 'instrumentation/'
        modules = read(directory + 'manifest.json')
        for module in ('v1/engine/output_processor.py', 'entrypoints/openai/completion/serving.py'):
            if digest(checked(directory + 'vllm/' + module)) != modules[module]['patched_sha256']:
                raise ValueError('captured cache serializer differs from instrumented runtime')
    results, workloads, traces = {}, {}, {}
    for episode in plan['agentic_contract']['episodes']:
        name = episode['spec']['episode']
        results[name] = read(name + '/result.json')
        workloads[name] = read(name + '/physical-workload.json')
        traces[name] = read(name + '/offered-trace.json')
        if (results[name]['spec'] != episode['spec']
                or hashes[name + '/physical-workload.json'] != episode['physical_workload_sha256']
                or hashes[name + '/offered-trace.json'] != episode['offered_trace_sha256']):
            raise ValueError('East episode changed the frozen offered workload')
    observed, usage, rows, engine, expected_tokens = {}, {}, [], [], {}
    for event in lines('request-events.jsonl'):
        if event.get('episode') in results:
            token_observation(observed, event, results[event['episode']]['boundary_ns'], usage)
    for raw in lines('requests.jsonl'):
        if raw.get('episode') in results:
            rows.append(compact_request(raw, results[raw['episode']], workloads[raw['episode']], observed, usage))
            if raw.get('request_id') and raw.get('serving_role') == 'destination':
                expected_tokens[raw['request_id']] = raw.get('token_ids', [])
    audit_result = audit(rows, results, traces)
    for name, result in results.items():
        with checked(name + '/engine.csv').open() as stream:
            engine.extend({'episode': name, 'serving_role': 'destination',
                'time_s': (int(r['monotonic_ns']) - result['epoch_ns']) / 1e9,
                **{k: float(r['vllm:' + k]) if r.get('vllm:' + k) else None for k in ENGINE_FIELDS}}
                for r in csv.DictReader(stream))
    server, internal = {}, {}
    frontend = next(p for p in certificate['telemetry_files'] if '/destination/timing/' in p and '-7195.' in p)
    core = next(p for p in certificate['telemetry_files'] if '/destination/timing/' in p and '-7625.' in p)
    for path in (frontend, core):
        name = path.split('/destination/', 1)[1]
        name = 'destination/' + name.removesuffix('.gz')
        identity = None
        previous = 0
        for event in lines(name):
            if event['sequence'] != previous + 1:
                raise ValueError('local server record sequence gap')
            previous = event['sequence']
            if event['kind'] == 'process_start':
                identity = event
            elif event['kind'] == 'frontend_registration':
                external = event['external_request_id'].removesuffix('-0')
                if external in expected_tokens:
                    if external in server:
                        raise ValueError('duplicate destination registration')
                    server[external] = {'registration_ns': event['mono_ns'], 'domain': domain(identity), 'tokens': [], 'ready_ns': []}
                    internal[event['request_id']] = external
            elif event['kind'] == 'schedule':
                for request in event['requests']:
                    if request['request_id'] in internal and request['scheduled_tokens']:
                        record = server[internal[request['request_id']]]
                        if record['domain'] is None or record['domain'] != domain(identity):
                            raise ValueError('server timestamps lack a verified common clock domain')
                        record.setdefault('scheduled_ns', event['start_ns'])
                        record.setdefault('first_schedule_iteration', event['iteration'])
                        record.setdefault('first_scheduled_tokens', request['scheduled_tokens'])
            elif event['kind'] == 'worker_output':
                for request in event['requests']:
                    if request['request_id'] in internal and request['token_ids']:
                        record = server[internal[request['request_id']]]
                        if record['domain'] is None or record['domain'] != domain(identity):
                            raise ValueError('worker timestamps lack a verified common clock domain')
                        if request['ordinal_start'] != len(record['tokens']):
                            raise ValueError('server token ordinals are incomplete for an episode request')
                        if not record['ready_ns']:
                            record['first_output_iteration'] = event['iteration']
                            record['first_output_batch_cuda_s'] = sum(event[phase + '_stream_ms'] for phase in ('forward', 'logits', 'sample')) / 1000
                        record['tokens'].extend(request['token_ids'])
                        record['ready_ns'].extend([event['output_ready_ns']] * len(request['token_ids']))
    for request_id, record in server.items():
        tokens = record.pop('tokens')
        expected = expected_tokens[request_id]
        record['local_tokens_complete'] = tokens == expected
        record['client_prefix_joined'] = tokens[:len(expected)] == expected
        record['worker_only_tokens'] = len(tokens) - len(expected)
        if record['ready_ns'] and (record['registration_ns'] > record['scheduled_ns']
                or record['scheduled_ns'] > record['ready_ns'][0] or record['ready_ns'] != sorted(record['ready_ns'])):
            raise ValueError('invalid same-clock episode server timing')
    return rows, engine, server, {'input_sha256': hashes, 'request_audit': audit_result,
        'global_telemetry_accepted': report['global_telemetry_accepted'],
        'telemetry_error_counts': report['telemetry_error_counts'],
        'endpoint_training_diagnostic_s': [{'cell': c['cell'], 'median': float(np.median([
            r['ttft_s'] - r['server_registration_to_first_ready_s'] for r in c['requests']]))}
            for c in certificate['warm_cells'] if c['cell'].endswith('-r1') and c['request_evidence_complete']]}


def server_conditioned(rows, server, coefficients, include_worker_extras=True):
    """Condition GPU queue execution on observed server registrations, without RTT fitting."""
    requests, selected = inputs(rows, 'miss')
    missing = [selected[r['request_id']] for r in requests if selected[r['request_id']]['request_id'] not in server]
    if any(r['done'] for r in missing):
        raise ValueError('completed destination demand lacks server registration')
    if missing and max(r['end_s'] for r in rows if r['cohort'] == 'resident') >= min(r['start_s'] for r in missing):
        raise ValueError('unregistered demand overlaps resident evidence')
    requests = [r for r in requests if selected[r['request_id']]['request_id'] in server]
    domains = {tuple(server[selected[r['request_id']]['request_id']]['domain'] or ()) for r in requests}
    if len(domains) != 1 or len(next(iter(domains))) != 4:
        raise ValueError('all destination registrations require the same verified clock domain')
    epoch = min(server[selected[r['request_id']]['request_id']]['registration_ns'] for r in requests)
    observations = {}
    for request in requests:
        raw = selected[request['request_id']]
        event = server[raw['request_id']]
        if not raw['done'] or raw['effective_cached_tokens'] is None or not event['client_prefix_joined']:
            raise ValueError('registered episode request lacks known completed demand and joined token prefix')
        if not event['local_tokens_complete'] and raw['cohort'] != 'migration':
            raise ValueError('service request lacks complete local server token evidence')
        if include_worker_extras:
            request['output_tokens'] = len(event['ready_ns'])
        request['arrival_s'] = (event['registration_ns'] - epoch) / 1e9
        observations[request['request_id']] = event
    result = simulate(requests, {**coefficients, 'endpoint_s': 0.}, rows[0]['duration_s'] + 10.)
    comparisons = []
    for prediction in result['requests']:
        raw, event = selected[prediction['request_id']], observations[prediction['request_id']]
        if not raw['done'] or not event['ready_ns']:
            continue
        count = prediction['output_tokens']
        first, last = event['ready_ns'][0], event['ready_ns'][count - 1]
        wait = (event['scheduled_ns'] - event['registration_ns']) / 1e9
        latency = (first - event['registration_ns']) / 1e9
        cadence = (last - first) / 1e9 / (count - 1) if count > 1 else None
        predicted_wait = prediction['admitted_s'] - prediction['arrival_s'] if prediction['admitted_s'] is not None else None
        comparisons.append({'row_id': raw['row_id'], 'cohort': raw['cohort'], 'scheduled_s': raw['scheduled_s'],
            'client_output_tokens': raw['output_tokens'], 'modeled_output_tokens': count,
            'worker_only_tokens': event['worker_only_tokens'], 'local_tokens_complete': event['local_tokens_complete'],
            'observed_generation_span_s': (last - first) / 1e9,
            'predicted_generation_span_s': prediction['last_token_s'] - prediction['first_s'] if prediction['done'] else None,
            'first_schedule_wait_s': comparison(wait, predicted_wait, .2),
            'registration_to_ready_s': comparison(latency, prediction['ttft_s'], .2),
            'server_ready_tpot_s': comparison(cadence, prediction['mean_tpot_s'], .005)})
    return {'episode': rows[0]['episode'], 'split': 'heldout', 'requests': comparisons,
            'excluded_unregistered_tail': [r['row_id'] for r in missing], 'include_worker_extras': include_worker_extras,
            'scope': 'Conditional GPU scheduling with observed destination registration and native cache inputs; endpoint zero because transport is outside this clock domain. Resident releases still serialize modeled predecessor completion. Worker-only migration tokens remain unexplained downstream evidence; they are included as observed GPU demand or explicitly bracketed. This does not predict offered-arrival or source queues.'}


def comparison(observed, predicted, absolute, relative=.25):
    tolerance = max(absolute, relative * observed) if observed is not None else None
    return {"observed": observed, "predicted": predicted, "tolerance": tolerance,
            "pass": None if observed is None else predicted is not None and abs(predicted - observed) <= tolerance}


def window(rows, start, end):
    eligible = [r for r in rows if start <= r["scheduled_s"] < end]
    first = [r for r in eligible if r["first_s"] is not None and r["first_s"] <= end]
    done = [r for r in eligible if r["done"] and r["end_s"] <= end]
    tpot = [r["mean_tpot_s"] for r in done if r["mean_tpot_s"] is not None]
    quantile = lambda values: float(np.quantile(values, .9)) if values else None
    missing = [r for r in eligible if r["first_s"] is None or r["first_s"] > end]
    return {"arrivals": len(eligible), "first_tokens_observed": len(first), "completed": len(done),
            "unfinished": len(eligible) - len(done), "tpot_samples": len(tpot),
            "ttft_p90_s": quantile([r["first_s"] - r["scheduled_s"] for r in first]),
            "tpot_p90_s": quantile(tpot),
            "known_ttft_violations": sum(r["first_s"] - r["scheduled_s"] > 1 for r in first)
                + sum(end - r["scheduled_s"] > 1 for r in missing),
            "unobserved_first_tokens": len(missing),
            "all_prior_outstanding": sum(r["scheduled_s"] < end and (not r["done"] or r["end_s"] > end) for r in rows)}


def inputs(rows, unknown_cache):
    if unknown_cache not in ("miss", "hit"):
        raise ValueError("unknown cache sensitivity must be miss or hit")
    selected, requests = {}, []
    for row in rows:
        resident = row["cohort"] == "resident"
        if not resident and (row["serving_role"] != "destination" or row["client_dispatch_s"] is None):
            continue
        arrival = row["scheduled_s"] if resident else row["client_dispatch_s"]
        if arrival < 0:
            continue  # Prewarming finished before epoch; its observed cache state is an input.
        prompt = row["submitted_prompt_tokens"] or row["planned_prompt_tokens"]
        output = row["output_tokens"] if row["done"] else row["planned_output_tokens"]
        if prompt is None or output is None:
            raise ValueError(f"unknown request shape: {row['row_id']}")
        cached = row["effective_cached_tokens"]
        if cached is None:
            cached = 0 if unknown_cache == "miss" else prompt - 1
        history = (f"resident:{row['session']}" if resident else f"incoming:{row['session']}"
                   if row["cohort"] == "incoming" or row["phase"] in ("initial", "catch_up") else row["row_id"])
        selected[row["row_id"]] = row
        requests.append(dict(request_id=row["row_id"], gpu="destination", history=history,
                             arrival_s=arrival, prompt_tokens=prompt, cached_tokens=cached, output_tokens=output))
    requests.sort(key=lambda r: (r["arrival_s"], selected[r["request_id"]]["turn"] or 0))
    return requests, selected


def evaluate(rows, engine, coefficients, endpoint_before_fraction=0., unknown_cache="miss"):
    requests, selected = inputs(rows, unknown_cache)
    duration = rows[0]["duration_s"]
    # ponytail: no KV eviction model; measured preemptions require another execution model.
    preemptions = [r["num_preemptions_total"] for r in engine]
    if not preemptions or None in preemptions or max(preemptions) != min(preemptions):
        raise ValueError("queue pilot requires complete, zero-preemption episode evidence")
    result = simulate(requests, coefficients, duration, endpoint_before_fraction=endpoint_before_fraction)
    predicted = {r["request_id"]: r for r in result["requests"]}
    residents = [r for r in rows if r["cohort"] == "resident"]
    simulated = [{**predicted[r["row_id"]], "scheduled_s": r["scheduled_s"]} for r in residents]
    intervals = [(30, 90)] if rows[0]["arm"] == "resident" else [(0, 60), (60, 90), (90, 120), (120, 180), (180, 240), (240, 300), (60, 300)]
    windows = []
    for start, end in intervals:
        if end > duration:
            continue
        observed, forecast = window(residents, start, end), window(simulated, start, end)
        samples = [r for r in engine if start <= r["time_s"] < end]
        if not samples or any(r["num_requests_waiting"] is None for r in samples):
            raise ValueError("missing engine queue samples")
        waiting = [sum(r["eligible_s"] is not None and r["eligible_s"] <= s["time_s"]
                       and (r["admitted_s"] is None or r["admitted_s"] > s["time_s"]) for r in predicted.values()) for s in samples]
        metrics = {"ttft_p90_s": comparison(observed["ttft_p90_s"], forecast["ttft_p90_s"], .2),
                   "tpot_p90_s": comparison(observed["tpot_p90_s"], forecast["tpot_p90_s"], .005),
                   "engine_waiting_peak": comparison(max(r["num_requests_waiting"] for r in samples), max(waiting), 2, 0),
                   "resident_outstanding": comparison(observed["all_prior_outstanding"], forecast["all_prior_outstanding"], 1, 0),
                   "known_ttft_violations": comparison(observed["known_ttft_violations"], forecast["known_ttft_violations"], 1, 0),
                   "first_token_coverage": comparison(observed["first_tokens_observed"], forecast["first_tokens_observed"], 1, 0)}
        windows.append(dict(start_s=start, end_s=end, observed=observed, predicted=forecast, metrics=metrics))
    migration = [r for r in rows if r["phase"] == "initial" and r["serving_role"] == "destination"]
    initial = {}
    if migration:
        start = min(r["client_dispatch_s"] for r in migration)
        for field in ("first_s", "end_s"):
            observed = max(r[field] for r in migration) - start if all(r[field] is not None for r in migration) else None
            forecast = max(predicted[r["row_id"]][field] for r in migration) - start if all(predicted[r["row_id"]][field] is not None for r in migration) else None
            initial[field] = comparison(observed, forecast, 2.)
    burst = [r for r in residents if 60 <= r["scheduled_s"] < 90]
    observed = max(r["end_s"] for r in burst) - 60 if burst and all(r["completed_within_observation"] for r in burst) else None
    forecast = max(predicted[r["row_id"]]["end_s"] for r in burst) - 60 if burst and all(predicted[r["row_id"]]["done"] for r in burst) else None
    return {"episode": rows[0]["episode"], "workload": rows[0]["workload"], "arm": rows[0]["arm"], "seed": rows[0]["seed"],
            "split": "heldout" if rows[0]["seed"] == 7102 else "training_control" if rows[0]["arm"] == "control" else "diagnostic",
            "endpoint_before_fraction": endpoint_before_fraction, "unknown_cache": unknown_cache,
            "unknown_cache_inputs": sum(r["effective_cached_tokens"] is None for r in selected.values()),
            "observed_failed_requests": sum(r["status"] in ("failed", "dependency_failed") for r in selected.values()),
            "windows": windows, "initial_migration": initial, "burst_resident_drain_s": comparison(observed, forecast, 2.),
            "gpu": result["gpus"]["destination"], "predictions": list(predicted.values())}


def checks(episodes):
    rows = [{"episode": r["episode"], "start_s": w["start_s"], "end_s": w["end_s"], "metric": k, **v}
            for r in episodes for w in r["windows"] for k, v in w["metrics"].items() if v["pass"] is not None]
    rows += [{"episode": r["episode"], "metric": "initial_migration_" + k, **v} for r in episodes for k, v in r["initial_migration"].items()]
    rows += [{"episode": r["episode"], "metric": "burst_resident_drain_s", **r["burst_resident_drain_s"]}
             for r in episodes if r["arm"] != "resident"]
    return {"checks": len(rows), "failures": [r for r in rows if r["pass"] is not True],
            "gate_pass": bool(rows) and all(r["pass"] is True for r in rows)}


def server_summary(result):
    metrics = ('first_schedule_wait_s', 'registration_to_ready_s', 'server_ready_tpot_s')
    def summarize(rows):
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric]['observed'] is not None]
            summary[metric] = {'requests': len(values), 'within_band': sum(v['pass'] is True for v in values),
                'p90': comparison(float(np.quantile([v['observed'] for v in values], .9)),
                    float(np.quantile([v['predicted'] for v in values], .9)) if all(v['predicted'] is not None for v in values) else None,
                    .005 if metric == 'server_ready_tpot_s' else .2) if values else None}
        return summary
    residents = [r for r in result['requests'] if r['cohort'] == 'resident']
    return {'all_requests': summarize(result['requests']), 'residents': summarize(residents),
        'resident_windows': [{'start_s': start, 'end_s': end,
            'metrics': summarize([r for r in residents if start <= r['scheduled_s'] < end])}
            for start, end in ((0, 60), (60, 90), (90, 120), (120, 180), (180, 240), (240, 300), (60, 300))]}


def validate_server(source, frozen_report, out):
    rows, engine, server, provenance = server_bundle(source)
    frozen = json.loads(frozen_report.read_text())
    fit = calibrate_server_decode(source / 'report.json')
    coefficients = {**frozen['fit']['coefficients'], **fit['coefficients']}
    endpoint = provenance['endpoint_training_diagnostic_s']
    n1 = [r['median'] for r in endpoint if '-n1-' in r['cell']]
    variants = [('frozen_germany', coefficients['endpoint_s']), ('zero', 0.),
                ('east_warm_n1_lower', min(n1)), ('east_warm_n1_upper', max(n1))]
    groups = [[r for r in rows if r['episode'] == name] for name in sorted({r['episode'] for r in rows})]
    episodes, sensitivities = [], []
    for label, delay in variants:
        for before, unknown in ((0., 'miss'), (1., 'miss'), (0., 'hit'), (1., 'hit')):
            cases = []
            for group in groups:
                samples = [r for r in engine if r['episode'] == group[0]['episode'] and 0 <= r['time_s'] < group[0]['duration_s']]
                case = evaluate(group, samples, {**coefficients, 'endpoint_s': delay}, before, unknown)
                case.pop('predictions')
                case['split'] = 'heldout'
                cases.append(case)
            gate = checks(cases)
            sensitivities.append({'endpoint': label, 'endpoint_s': delay, 'endpoint_before_fraction': before,
                'unknown_cache': unknown, 'checks': gate['checks'], 'failed_checks': len(gate['failures']), 'gate_pass': gate['gate_pass']})
            if label == 'frozen_germany' and before == 0 and unknown == 'miss':
                episodes = cases
    conditioned = [server_conditioned(group, server, coefficients) for group in groups]
    for case in conditioned:
        case['summary'] = server_summary(case)
    early = next(r for r in rows if r['row_id'] == 'episodes-coding-0-control-7102/resident/6/0')
    timing = server[early['request_id']]
    if timing['first_schedule_iteration'] != timing['first_output_iteration']:
        raise ValueError('early host-interval diagnostic requires a single prefill iteration')
    report = {'schema': 'a100-east-resident-queue-heldout-v1', 'campaign_ready': False,
        'resident_latency_validated': False, 'global_telemetry_accepted': provenance['global_telemetry_accepted'],
        'coefficients': coefficients, 'server_decode_fit': fit, 'provenance': provenance,
        'frozen_prefill_endpoint_report': {'path': str(frozen_report), 'sha256': digest(frozen_report)},
        'client_primary_checks': checks(episodes), 'client_primary_episodes': episodes,
        'client_endpoint_sensitivity': sensitivities, 'server_conditioned_episodes': conditioned,
        'early_host_interval_diagnostic': {'row_id': early['row_id'], 'request_id': early['request_id'],
            'iteration': timing['first_output_iteration'], 'scheduled_tokens': timing['first_scheduled_tokens'],
            'native_cached_tokens': early['effective_cached_tokens'],
            'first_schedule_to_ready_s': (timing['ready_ns'][0] - timing['scheduled_ns']) / 1e9,
            'forward_logits_sample_stream_sum_s': timing['first_output_batch_cuda_s'],
            'scope': 'Selected after evaluation to explain the early latency miss; no fit. The large host interval is outside recorded CUDA phase elapsed time; its cause remains unassigned. CUDA phases are stream elapsed intervals, not exclusive kernel busy time.'},
        'worker_token_count_sensitivity': [{'episode': group[0]['episode'],
            **server_summary(server_conditioned(group, server, coefficients, False))} for group in groups],
        'code_sha256': {name: digest(Path(name)) for name in ('pool_shed_resident_queue.py', 'pool_shed_resident_fit.py',
            'pool_shed_resident_validation.py', 'pool_shed_resident_data.py')},
        'scope': [
            'All four East episodes are held out from the warm-repeat-1 decode fit. Prefill and the primary endpoint constant are frozen from the previous report; no episode or policy-ranking fit is performed.',
            'Primary client windows retain the original TTFT/TPOT, queue, coverage and burst-drain gates. Warm N1 endpoint bounds are training-only sensitivity inputs, not a fitted transport model or a selected passing variant.',
            'Server registration, first schedule and output ready share a verified destination host clock domain. Client and server absolute clocks are never subtracted. Warm endpoint diagnostics subtract elapsed intervals only.',
            'Server-conditioned per-request and P90 summaries reuse numeric bands for diagnostics; they do not replace the original window gates or certify fleet SLOs. The 60–300 window overlaps five smaller windows; checks are not independent samples or confidence intervals.',
            'Server windows group requests by original offered time and use their complete server outcomes, including completions after the window ends. Primary client windows retain their original within-window observation rule.',
            'Server demand uses measured registrations and native cache counts, then serializes modeled history completion. This isolates destination GPU scheduling conditionally; it does not predict dispatch, source progress, WAN, placement or endpoint buffering.',
            'Worker-only migration tokens remain unresolved generated-without-delivery evidence. Primary GPU demand includes those recorded tokens; the explicit client-count sensitivity excludes them. Neither interpretation repairs failed global telemetry.',
            'Failed or censored requests without registrations are excluded only from the conditional server check, after verifying they start after every resident completion. The primary client evaluation retains their planned demand and cache bounds.',
            'Server-ready cadence includes host/scheduling gaps and differs from CUDA stream elapsed time. First-schedule timing can lead actual execution in the asynchronous pipeline; the simulator does not introduce a fitted pipeline delay.',
            'East A100 timing is a qualified GPU-local check. Cross-region endpoint timing and fleet-wide resident latency remain unvalidated; no additional GPU campaign is required by this report.']}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'server-heldout.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--server-data", type=Path)
    args = parser.parse_args()
    if args.server_data:
        report = validate_server(args.server_data, args.data / 'report.json', SERVER_OUTPUT if args.out == OUTPUT else args.out)
        print(json.dumps({'client_checks': report['client_primary_checks']['checks'],
            'client_failed_checks': len(report['client_primary_checks']['failures']), 'campaign_ready': False}))
        return
    manifest = json.loads((args.data / "data-manifest.json").read_text())
    if (set(manifest["output_sha256"]) != {"requests.csv", "engine.csv", "migration-events.csv"}
            or any(digest(args.data / name) != sha for name, sha in manifest["output_sha256"].items())
            or digest(Path("pool_shed_resident_data.py")) != manifest["reducer_sha256"]):
        raise ValueError("compact trace artifacts or extractor changed; regenerate before validation")
    rows = load_requests(args.data / "requests.csv")
    expected = {r["episode"] for r in manifest["episodes"]}
    if (len(expected) != 20 or {r["episode"] for r in rows} != expected
            or len({r["episode"] for r in rows if r["seed"] == 7102}) != 6):
        raise ValueError("validation requires all20 episodes including six held-out episodes")
    with (args.data / "engine.csv").open() as stream:
        engine = [{k: v if k in ("episode", "serving_role") else float(v) if v else None for k, v in r.items()} for r in csv.DictReader(stream)]
    fit = calibrate(rows, request_sha256=digest(args.data / "requests.csv"))
    episodes, predictions = [], []
    for episode in sorted({r["episode"] for r in rows}):
        requests = [r for r in rows if r["episode"] == episode]
        samples = [r for r in engine if r["episode"] == episode and r["serving_role"] == "destination" and 0 <= r["time_s"] < requests[0]["duration_s"]]
        for before, unknown in ((0., "miss"), (1., "miss"), (0., "hit"), (1., "hit")):
            result = evaluate(requests, samples, fit["coefficients"], before, unknown)
            prediction = result.pop("predictions")
            if before == 0 and unknown == "miss":
                predictions.extend({"episode": episode, **r} for r in prediction)
            episodes.append(result)
    gates = [{"endpoint_before_fraction": before, "unknown_cache": unknown,
              **{split: checks([r for r in episodes if r["endpoint_before_fraction"] == before and r["unknown_cache"] == unknown
                               and (r["split"] == split if split == "heldout" else r["split"] != "training_control")])
                 for split in ("heldout", "stress_and_heldout")}}
             for before, unknown in ((0., "miss"), (1., "miss"), (0., "hit"), (1., "hit"))]
    lookup = {r["request_id"]: r for r in predictions}
    long_outputs = [{k: r[k] for k in ("episode", "row_id", "cohort", "prompt_tokens", "effective_cached_tokens", "output_tokens",
                                     "scheduled_s", "first_s", "end_s", "mean_tpot_s", "submillisecond_token_gaps", "token_gap_count")}
                    | {"predicted_end_s": lookup[r["row_id"]]["end_s"], "predicted_mean_tpot_s": lookup[r["row_id"]]["mean_tpot_s"]}
                    for r in rows if r["cohort"] == "resident" and r["output_tokens"] == 1233 and r["completed_within_observation"]]
    bursts = [{k: r[k] for k in ("episode", "row_id", "output_tokens", "mean_tpot_s", "submillisecond_token_gaps",
                                "token_gap_count", "median_token_gap_s", "max_token_gap_s")}
              for r in rows if r["episode"] == "episodes-coding_long-0-replay-7101" and r["cohort"] == "resident"
              and 60 <= r["scheduled_s"] < 90 and r["done"] and r["submillisecond_token_gaps"]]
    cold_errors = [prefill_seconds(json.loads(r["prompt_counts"])[0], r["request_prefill_kv_computed_tokens_sum"], fit["coefficients"])
                   - r["request_prefill_time_seconds_sum"] for r in json.loads(UNLOADED.read_text())["rows"]
                   if r["seed"] == 7102 and r["width"] == 1 and r["phase_valid"]]
    report = {"campaign_ready": False, "resident_latency_validated": False,
              "conditional_heldout_gate_pass": all(g["heldout"]["gate_pass"] for g in gates),
              "conditional_stress_gate_pass": all(g["stress_and_heldout"]["gate_pass"] for g in gates),
              "gates": gates, "fit": fit, "episodes": episodes,
              "singleton_prefill_heldout": {"requests": len(cold_errors), "rmse_s": float(np.sqrt(np.mean(np.square(cold_errors)))),
                                             "max_absolute_error_s": max(abs(e) for e in cold_errors)},
              "long_generation_evidence": long_outputs, "client_token_bursts": bursts,
              "remaining_work": ["Measure per-request server arrival, first scheduled, first generated and last generated timestamps alongside current client SSE events, plus iteration start/end, scheduled prefill tokens and active decode request IDs.",
                  "A bounded follow-up can test warm independent prefixes at8K/30K, decode concurrency1/8/16 and1536 fixed output tokens, two repeats; then retain one coding1233-token-tail episode and one width-eight long replay/resident burst as workload checks. Freeze runtime and separate fitting from checks. A broad policy campaign is unnecessary.",
                  "Validate loaded generation and variable token-delivery delay before integrating persistent weighted GPU groups, per-GPU KV capacity and causal source queues into the fleet executor and planner forecasts."],
              "input_sha256": {str(args.data / name): digest(args.data / name) for name in ("requests.csv", "engine.csv", "data-manifest.json")},
              "code_sha256": {name: digest(Path(name)) for name in ("pool_shed_resident_queue.py", "pool_shed_resident_fit.py", "pool_shed_resident_validation.py", "pool_shed_resident_data.py")},
              "scope": ["Resident releases use original offered arrivals and modeled prior completion, with stationary GPU history. Incoming work and probes use measured dispatch eligibility, native cache observations and actual generation; censored service uses planned generation.",
                  "This is a conditional destination scheduler check, not a prediction of source quiescence, KV transfer, handoff, prefix eviction or fleet placement. Full-context submitted prompts remain intact; cached tokens only decode measured native usage.",
                  "Unfinished requests remain in arrival counts; first-token waiting over one second counts as a known violation. TPOT excludes unfinished completions. Missing cache inputs are bracketed by full miss and maximal hit, not assigned a fitted reuse fraction.",
                  "Observed transport and dependency failures remain explicit. The pilot executes their planned demand without a transport-failure model, so those scout discrepancies do not identify GPU capacity or a replay slowdown.",
                  "TTFT and TPOT bands are max(0.2s,25%) and max(0.005s,25%); sampled queue-peak error at most2; resident outstanding, observed-first and known-violation count errors at most1; migration and burst-drain error max(2s,25%). These are pilot acceptance bands, not probabilistic confidence or production tail guarantees.",
                  "Window cohorts overlap; metric-check counts are not independent experimental samples. The adverse seed7101 replay episodes and loading scouts remain stress checks even though only seed7102 is the formal held-out seed.",
                  "Burst drain means completion of resident arrivals in60–90s, while later arrivals continue. It does not assert all subsequent queues vanish or repair an earlier SLO miss.",
                  "Client token intervals include buffering and stalls. The decode coefficient is an effective client-derived approximation, not a measured GPU iteration duration; isolated requests also contain submillisecond delivery bursts.",
                  "Endpoint before/after placement is structurally unidentifiable here: it shifts GPU events but gives the same client delivery times. The variants test absolute engine sample alignment, not whether actual dynamic endpoint delay is harmless.",
                  "Engine waiting includes remote-KV readiness as well as compute waiting. Observed dispatch eligibility and cache hits condition KV timing; elapsed initial KV spans do not independently validate transfer or queue prediction.",
                  "Unknown-cache limits are reported without refitting. Whole-site spare capacity never repays another GPU's work; this pilot is not installed in the fleet executor."]}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "predictions.csv", predictions)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("campaign_ready", "conditional_heldout_gate_pass", "conditional_stress_gate_pass")}))


if __name__ == "__main__":
    main()
