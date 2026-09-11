"""Reconstruct observed initial states and offered load from the frozen 20 episodes."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pool_shed_calibration import calibration, service_work
from pool_shed_resident_data import checked_lines, digest, load_requests

SOURCE = ROOT / 'outputs/a100-replay-final-20260910T0352'
COMPACT = ROOT / 'outputs/a100-resident-queues-20260910'
SERVICE = ('resident', 'incoming')


def stats(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)} if values else {'n': 0}


def successful(row, now):
    return row['done'] and row['status'] == 200 and row['end_s'] is not None and row['end_s'] <= now


def state(rows, now):
    by_id = {r['row_id']: r for r in rows}
    arrived = [r for r in rows if r['scheduled_s'] <= now]
    completed = [r for r in arrived if successful(r, now)]
    ended = [r for r in arrived if not successful(r, now) and r['end_s'] is not None and r['end_s'] <= now]
    censored = [r for r in ended if r['status'] == 'censored']
    failed = [r for r in ended if r['status'] != 'censored']
    pending = [r for r in arrived if r not in ended and (r['client_dispatch_s'] is None or r['client_dispatch_s'] > now)]
    active = [r for r in arrived if r not in ended and r['client_dispatch_s'] is not None
              and r['client_dispatch_s'] <= now and (r['end_s'] is None or r['end_s'] > now)]
    blocked = [r for r in pending if r.get('predecessor_id') and not successful(by_id[r['predecessor_id']], now)]
    assert len(arrived) == len(completed) + len(failed) + len(censored) + len(pending) + len(active)
    return {'offered': len(arrived), 'successfully_completed': len(completed),
        'terminated_unsuccessfully': len(failed), 'right_censored': len(censored), 'client_requests_in_flight': len(active),
        'client_in_flight_before_first_output': sum(r['first_s'] is None or r['first_s'] > now for r in active),
        'client_in_flight_after_first_output': sum(r['first_s'] is not None and r['first_s'] <= now for r in active),
        'offered_awaiting_dispatch': len(pending), 'awaiting_predecessor_completion': len(blocked),
        'awaiting_dispatch_with_completed_or_no_predecessor': len(pending)-len(blocked),
        'pending_row_ids': [r['row_id'] for r in pending], 'failed_row_ids': [r['row_id'] for r in failed],
        'censored_row_ids': [r['row_id'] for r in censored],
        'in_flight': [{k: r.get(k) for k in ('row_id', 'request_id', 'episode', 'cohort', 'session', 'turn',
            'serving_role', 'scheduled_s', 'client_dispatch_s', 'start_s', 'first_s', 'last_token_s', 'end_s',
            'planned_output_tokens', 'observed_output_tokens', 'done', 'status')} for r in active]}


def window(rows, start, end, measured):
    offered = [r for r in rows if start <= r['scheduled_s'] < end]
    sent = [r for r in rows if r['client_dispatch_s'] is not None and start <= r['client_dispatch_s'] < end]
    completed = [r for r in rows if successful(r, end) and start <= r['end_s'] < end]
    cached = [r for r in completed if r['effective_cached_tokens'] is not None]
    return {'start_s': start, 'end_s': end, 'offered_requests': len(offered), 'dispatched_requests': len(sent),
        'completed_requests': len(completed), 'offered_rps': len(offered)/(end-start),
        'dispatched_rps': len(sent)/(end-start), 'completed_rps': len(completed)/(end-start),
        'dispatch_roles': dict(Counter(r['serving_role'] for r in sent)),
        'offered_shape': {k: stats(r[k] for r in offered) for k in
            ('planned_context_tokens', 'planned_append_tokens', 'planned_prompt_tokens', 'planned_output_tokens')},
        'offered_new_prompt_tokens_per_s': sum(r['planned_append_tokens'] for r in offered)/(end-start),
        'offered_output_tokens_per_s': sum(r['planned_output_tokens'] for r in offered)/(end-start),
        'model_normalized_offered_work_per_s': sum(float(service_work(r['planned_prompt_tokens'],
            r['planned_append_tokens'], r['planned_output_tokens'], measured)) for r in offered)/(end-start),
        'completed_prompt_tokens': sum(r['prompt_tokens'] for r in completed),
        'completed_output_tokens': sum(r['output_tokens'] for r in completed),
        'completed_output_tokens_per_s': sum(r['output_tokens'] for r in completed)/(end-start),
        'completed_known_cache_requests': len(cached),
        'completed_reported_cached_tokens': sum(r['effective_cached_tokens'] for r in cached),
        'completed_prompt_minus_cache_tokens': sum(r['prompt_tokens']-r['effective_cached_tokens'] for r in cached),
        'dispatch_lateness_s': stats(r['client_dispatch_s']-r['scheduled_s'] for r in sent),
        'client_wakeup_lateness_s': stats(r['client_wakeup_s']-r['scheduled_s'] for r in sent if r['client_wakeup_s'] is not None),
        'client_wakeup_to_dispatch_s': stats(r['client_dispatch_s']-r['client_wakeup_s'] for r in sent if r['client_wakeup_s'] is not None)}


def engine_state(rows, now):
    selected = sorted(rows, key=lambda r: r['time_s'])
    before = [r for r in selected if r['time_s'] <= now]
    after = [r for r in selected if r['time_s'] > now]
    baseline = [r for r in selected if 0 <= r['time_s'] < now]
    return {'last_poll_at_or_before_anchor': before[-1] if before else None,
        'first_poll_after_anchor': after[0] if after else None, 'baseline_polls': len(baseline),
        'baseline_sampled_running': stats(r['num_requests_running'] for r in baseline),
        'baseline_sampled_waiting': stats(r['num_requests_waiting'] for r in baseline),
        'baseline_idle_sample_fraction': sum(r['num_requests_running'] == r['num_requests_waiting'] == 0
            for r in baseline)/len(baseline) if baseline else None}


def delivered_at_anchors(targets, archive_manifest):
    archive_path = SOURCE/archive_manifest['archive']
    assert digest(archive_path) == archive_manifest['sha256']
    expected = next(r for r in archive_manifest['members'] if r['path'] == 'request-events.jsonl')
    total, before, seen = Counter(), Counter(), False
    process = subprocess.Popen(['zstd', '--long=27', '-dc', str(archive_path)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
        for member in archive:
            if member.name != expected['path']:
                continue
            assert not seen
            seen = True
            for line in checked_lines(archive.extractfile(member), expected):
                event = json.loads(line)
                key = tuple(event.get(k) for k in ('episode', 'cohort', 'session', 'turn'))
                if key not in targets or not event.get('data') or event['data'] == '[DONE]':
                    continue
                value = json.loads(event['data'])
                tokens = sum(len(choice.get('token_ids', [])) for choice in value.get('choices', []))
                if tokens:
                    assert value['id'] == targets[key][0]['request_id']
                    total[key] += tokens
                    before[key] += tokens * (event['monotonic_ns'] <= targets[key][1])
    process.stdout.read()
    process.stdout.close()
    assert process.wait() == 0 and seen
    for key, (row, _) in targets.items():
        assert total[key] == row['observed_output_tokens']
        assert 0 <= before[key] <= row['planned_output_tokens']
        row.update(client_delivered_output_tokens_at_anchor=before[key],
                   planned_output_tokens_not_client_delivered_at_anchor=row['planned_output_tokens']-before[key],
                   remaining_gpu_output_tokens=None)


def analyze():
    manifest = json.loads((COMPACT/'data-manifest.json').read_text())
    hashes = {str((COMPACT/'data-manifest.json').relative_to(ROOT)): digest(COMPACT/'data-manifest.json')}
    def read(path):
        checksum = digest(path)
        hashes[str(path.relative_to(ROOT))] = checksum
        if path.is_relative_to(SOURCE) and str(path.relative_to(SOURCE)) in manifest['input_sha256']:
            assert checksum == manifest['input_sha256'][str(path.relative_to(SOURCE))]
        return json.loads(path.read_text())
    for name in ('requests.csv', 'engine.csv', 'migration-events.csv'):
        hashes[str((COMPACT/name).relative_to(ROOT))] = digest(COMPACT/name)
        assert hashes[str((COMPACT/name).relative_to(ROOT))] == manifest['output_sha256'][name]
    rows = load_requests(COMPACT/'requests.csv')
    with (COMPACT/'engine.csv').open() as stream:
        engine = [{k: v if k in ('episode', 'serving_role') else float(v) for k, v in r.items()}
                  for r in csv.DictReader(stream)]
    assert len(rows) == manifest['request_rows'] and len(engine) == manifest['engine_rows']
    measured = calibration(0)
    hashes.update(measured['sources'])
    plan, selection = read(SOURCE/'plan.json'), read(SOURCE/'resident-rate-selection.json')
    episodes, targets = [], {}
    for item in manifest['episodes']:
        episode = item['episode']
        result = read(SOURCE/episode/'result.json')
        trace, physical = read(SOURCE/episode/'offered-trace.json'), read(SOURCE/episode/'physical-workload.json')
        spec = result['spec']
        selected = [r for r in rows if r['episode'] == episode]
        service = [r for r in selected if r['cohort'] in SERVICE]
        assert {(r['cohort'], r['session'], r['turn']) for r in service} == {(r['cohort'], r['session'], r['turn']) for r in trace}
        assert len(service) == len(trace)
        snapshots = [r['monotonic_ns'] for r in result['migration_events'] if r['kind'] == 'snapshot']
        anchor_ns = min(snapshots) if snapshots else result['epoch_ns']+30_000_000_000
        anchor = (anchor_ns-result['epoch_ns'])/1e9
        duration = (result['boundary_ns']-result['epoch_ns'])/1e9
        nominal = plan['workloads'][spec['workload']]['initial_scout_rps_per_gpu']
        mean_work = statistics.mean(statistics.mean(float(service_work(r['context']+r['prompt'], r['prompt'],
            r['output'], measured)) for r in sequence) for sequence in physical['turn_sequences'])
        report = {'episode': episode, 'spec': spec, 'duration_s': duration, 'anchor_s': anchor,
            'anchor_basis': 'earliest recorded migration snapshot' if snapshots else 'resident-only scout measurement begins; no migration',
            'nominal_migration_s': 60. if snapshots else None, 'placement': result['placement'],
            'all_request_status_counts': dict(Counter(str(r['status']) for r in selected)),
            'unfinished_service_requests': sum(not r['done'] for r in service),
            'nominal_fleet_50_percent_resident_rps': nominal,
            'configured_resident_rps_over_nominal_fleet_50_percent': spec['rate']/nominal,
            'physical_subset_model_normalized_cycle_offered_work_per_s': spec['rate']*mean_work,
            'selected_rate_screen': selection[spec['workload']], 'service': {}, 'engine': {}}
        for cohort in SERVICE:
            group = [r for r in service if r['cohort'] == cohort]
            if not group:
                continue
            initial = state(group, anchor)
            for active in initial['in_flight']:
                key = tuple(active[k] for k in ('episode', 'cohort', 'session', 'turn'))
                assert key not in targets
                targets[key] = active, anchor_ns
            future = [r['scheduled_s'] for r in group if r['scheduled_s'] > anchor]
            spans = [(0., anchor), (anchor, duration)] if not snapshots else [(0., anchor), (30., anchor), (0., duration)]
            report['service'][cohort] = {'initial_state': initial,
                'next_offered_arrival_after_anchor_s': min(future) if future else None,
                'next_offered_arrival_gap_s': min(future)-anchor if future else None,
                'offered_arrivals_in_first_20s_after_anchor': sum(anchor < r['scheduled_s'] <= anchor+20 for r in group),
                'windows': [window(group, a, b, measured) for a, b in spans]}
        for role in ('source', 'destination'):
            report['engine'][role] = engine_state([r for r in engine if r['episode'] == episode and r['serving_role'] == role], anchor)
        extra = [r for r in selected if r['cohort'] not in SERVICE]
        report['nonservice_requests'] = {'initial_state': state(extra, anchor),
            'baseline_dispatch_counts_by_phase': dict(Counter(r['phase'] for r in extra
                if r['client_dispatch_s'] is not None and 0 <= r['client_dispatch_s'] < anchor)),
            'baseline_completed_output_tokens': sum(r['output_tokens'] for r in extra if successful(r, anchor) and 0 <= r['end_s'] < anchor)}
        episodes.append(report)
    assert len(episodes) == 20 and len({r['episode'] for r in episodes}) == 20
    archive = read(SOURCE/'raw-telemetry-archive.json')
    delivered_at_anchors(targets, archive)
    hashes[str((SOURCE/archive['archive']).relative_to(ROOT))] = archive['sha256']
    for name in ('pool_shed_resident_data.py', 'pool_shed_calibration.py'):
        hashes[name] = digest(ROOT/name)
    hashes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    report = {'schema': 'queue-haul-initial-state-audit-v1', 'new_gpu_measurements': 0,
        'new_policy_evaluations': 0, 'episodes': episodes, 'episode_count': len(episodes),
        'active_service_requests_with_exact_client_output_counts': len(targets),
        'historical_loaded_reference_rps': measured['reference_rps'],
        'historical_loaded_50_percent_rps': measured['reference_rps']/2,
        'historical_loaded_reference_request_tokens': measured['reference_request_tokens'],
        'scope': ['All 20 compact episodes retained, including unsuccessful and unfinished requests.',
            'Main anchors are actual first snapshots near t60; scout anchors are their t30 measurement starts, with no migration.',
            'Client in-flight states do not identify server waiting, prefill or decode. Engine running/waiting are separate sampled aggregate counters.',
            'Engine poll times are controller-relative observations; they are not exact GPU state at the anchor or GPU utilization.',
            'Output counts are exact client-observed token IDs at the anchor, verified against the archived event member and compact totals. Buffering prevents inference of remaining GPU work.',
            'Rates count offers, dispatches and successful completion events separately; completed output counts are completion-accounted, not token emission rates.',
            'Model-normalized work uses the existing service curve and actual sampled trajectory subset. It is not measured GPU utilization or a certified capacity fraction.',
            'Quiet observed starts and seeded arrival gaps do not identify the distribution of initial fleet queues under real workload timestamps.',
            'The existing episodes already continue resident arrivals after migration. Cleanup drain does not prove recovery.'],
        'input_sha256': dict(sorted(hashes.items()))}
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'episodes': len(episodes), 'active_service_requests': len(targets),
                      'output': str(output.relative_to(ROOT)), 'sha256': digest(output)}))


def test_client_state_keeps_dependency_queue_and_failures_separate():
    def row(key, scheduled, dispatch, end, done=False, status=200, predecessor=None):
        return dict(row_id=key, scheduled_s=scheduled, client_dispatch_s=dispatch, end_s=end, done=done,
                    status=status, first_s=dispatch, predecessor_id=predecessor)
    result = state([row('done', 0, 0, 1, True), row('active', 1, 1, 5, True),
        row('blocked', 2, 5, 6, True, predecessor='active'), row('ready', 2, 4, 5, True),
        row('failed', 1, None, 2, status='dependency_failed'), row('future', 4, 4, 5, True),
        row('censored', 1, 1, 2, status='censored')], 3)
    assert (result['offered'], result['successfully_completed'], result['terminated_unsuccessfully'],
            result['client_requests_in_flight'], result['offered_awaiting_dispatch'],
            result['awaiting_predecessor_completion'], result['right_censored']) == (6, 1, 1, 1, 2, 1, 1)


if __name__ == '__main__':
    analyze()
