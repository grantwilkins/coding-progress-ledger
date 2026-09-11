"""Read frozen logs to distinguish handoff, resident delay, and continued service."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pool_shed_resident_data import load_requests

DATA = ROOT / 'outputs/a100-resident-queues-20260910'
SOURCE = ROOT / 'outputs/a100-replay-final-20260910T0352'
SERVER = ROOT / 'outputs/a100-replay-queue-resolution-20260910/server-heldout.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {'n': len(values), **{name: float(np.quantile(values, q)) if values else None
                               for name, q in (('p50', .5), ('p90', .9), ('max', 1.))}}


def finished(row, time):
    return row['done'] and row['status'] == 200 and row['end_s'] <= time


def identity(row):
    return row['cohort'], row['session'], row['turn']


def state(rows, time):
    lookup = {r['row_id']: r for r in rows}
    due = [r for r in rows if r['scheduled_s'] <= time]
    outstanding = [r for r in due if not finished(r, time)]
    terminal = [r for r in outstanding if r['status'] in ('failed', 'dependency_failed')
                and r['end_s'] is not None and r['end_s'] <= time]
    censored = [r for r in outstanding if r['status'] == 'censored'
                and time >= min(r['duration_s'], r['end_s'] if r['end_s'] is not None else r['duration_s'])]
    active = [r for r in outstanding if r not in terminal and r not in censored]
    pending = [r for r in active if r['client_dispatch_s'] is None or r['client_dispatch_s'] > time]
    blocked = [r for r in pending if r['predecessor_id'] in lookup and not finished(lookup[r['predecessor_id']], time)]
    waiting_first = [r for r in active if r not in pending and (r['first_s'] is None or r['first_s'] > time)]
    after_first = [r for r in active if r not in pending and r not in waiting_first]
    assert len(outstanding) == len(terminal) + len(censored) + len(pending) + len(waiting_first) + len(after_first)
    return {'time_s': time, 'offered': len(due), 'completed': len(due) - len(outstanding),
            'outstanding': len(outstanding), 'failed_client_terminal_obligations': len(terminal),
            'censored_observation_obligations': len(censored), 'pending_dispatch': len(pending),
            'predecessor_blocked': len(blocked), 'dispatched_without_first': len(waiting_first),
            'dispatched_after_first': len(after_first)}


def engine_window(rows, start, end):
    selected = [r for r in rows if start <= r['time_s'] < end]
    if not selected or any(r[k] is None for r in selected for k in ('num_requests_running', 'num_requests_waiting')):
        raise ValueError('missing destination engine queue samples')
    return {'samples': len(selected), 'running_samples': stats([r['num_requests_running'] for r in selected]),
            'waiting_samples': stats([r['num_requests_waiting'] for r in selected]),
            'sampled_idle_fraction': sum(r['num_requests_running'] + r['num_requests_waiting'] == 0 for r in selected) / len(selected),
            'preemptions_counter_change': max(r['num_preemptions_total'] for r in selected) - min(r['num_preemptions_total'] for r in selected)}


def window(rows, start, end, boundary, targets):
    arrivals = [r for r in rows if start <= r['scheduled_s'] < end]
    first = [r for r in arrivals if r['first_s'] is not None and r['first_s'] <= boundary]
    missing_first = [r for r in arrivals if r not in first]
    completed = [r for r in arrivals if finished(r, boundary)]
    interval_completions = [r for r in rows if finished(r, end) and r['end_s'] > start]
    tpot = [r['mean_tpot_s'] for r in completed if r['mean_tpot_s'] is not None]
    return {'start_s': start, 'end_s': end, 'arrivals': len(arrivals),
            'completed_arrival_cohort_by_boundary': len(completed),
            'arrival_cohort_unfinished_by_boundary': len(arrivals) - len(completed),
            'arrival_cohort_failed': sum(r['status'] in ('failed', 'dependency_failed') for r in arrivals),
            'arrival_cohort_censored': sum(r['status'] == 'censored' for r in arrivals),
            'first_token_coverage_by_boundary': len(first), 'missing_first_by_boundary': len(missing_first),
            'arrival_ttft_s': stats([r['first_s'] - r['scheduled_s'] for r in first]),
            'request_start_ttft_s': stats([r['first_s'] - r['start_s'] for r in first if r['start_s'] is not None]),
            'dispatch_lateness_s': stats([r['client_dispatch_s'] - r['scheduled_s'] for r in arrivals
                                          if r['client_dispatch_s'] is not None and r['client_dispatch_s'] <= boundary]),
            'known_ttft_over_target': sum(r['first_s'] - r['scheduled_s'] > targets['p90_ttft_s'] for r in first)
                + sum(r['status'] not in ('failed', 'dependency_failed') and boundary - r['scheduled_s'] > targets['p90_ttft_s'] for r in missing_first),
            'client_mean_tpot_s': stats(tpot),
            'known_mean_tpot_over_target': sum(v > targets['p90_mean_tpot_s'] for v in tpot),
            'completed_requests_in_interval': len(interval_completions),
            'completed_requests_per_s': len(interval_completions) / (end - start),
            'completed_response_output_token_credit': sum(r['output_tokens'] for r in interval_completions),
            'arrival_cohort_first_seen_by_window_end': sum(r['first_s'] <= end for r in first),
            'arrival_cohort_completed_by_window_end': sum(finished(r, end) for r in arrivals),
            'state_at_start': state(rows, start), 'state_at_end': state(rows, end)}


def clearance(rows, time, boundary):
    due = [r for r in rows if r['scheduled_s'] <= time]
    pending = [r for r in due if not finished(r, time)]
    cleared = all(finished(r, boundary) for r in due)
    clear_time = max([time] + [r['end_s'] for r in pending]) if cleared else None
    candidates = sorted({time, boundary} | {r['end_s'] for r in rows if finished(r, boundary) and time <= r['end_s'] <= boundary})
    first_empty = next((t for t in candidates if state(rows, t)['outstanding'] == 0), None)
    return {'anchor_s': time, 'already_offered_requests': len(due), 'already_offered_outstanding': len(pending),
            'finite_cohort_cleared_s': clear_time, 'finite_cohort_clear_delay_s': clear_time - time if cleared else None,
            'finite_cohort_uncleared_by_boundary': sum(not finished(r, boundary) for r in pending),
            'first_no_unresolved_offered_requests_s': first_empty,
            'new_arrivals_before_finite_cohort_clear': sum(time < r['scheduled_s'] <= clear_time for r in rows) if cleared else None}


def versus_control(rows, control, anchor, boundary, handoff):
    left, right = {identity(r): r for r in rows}, {identity(r): r for r in control}
    if left.keys() != right.keys():
        raise ValueError('control and action offered identities differ')
    shape = ('scheduled_s', 'planned_prompt_tokens', 'planned_append_tokens', 'planned_output_tokens', 'planned_reset')
    if any(any(left[k][f] != right[k][f] for f in shape) for k in left):
        raise ValueError('control and action offered schedule or planned shape differ')
    times = sorted({anchor, boundary, handoff} | {r['end_s'] for r in rows + control if finished(r, boundary) and anchor <= r['end_s'] <= boundary})
    series = []
    for t in times:
        action_done = {k for k, r in left.items() if finished(r, t)}
        control_done = {k for k, r in right.items() if finished(r, t)}
        behind = control_done - action_done
        series.append({'time_s': t, 'control_completed_minus_action': len(control_done) - len(action_done),
                       'matched_requests_completed_in_control_only': len(behind),
                       'matched_completed_response_output_token_credit_deficit': sum(right[k]['planned_output_tokens'] for k in behind)})
    delayed = [left[k]['first_s'] - right[k]['first_s'] for k in left
               if anchor <= left[k]['scheduled_s'] < boundary and left[k]['first_s'] is not None and right[k]['first_s'] is not None
               and max(left[k]['first_s'], right[k]['first_s']) <= boundary]
    area = sum((b['time_s'] - a['time_s']) * a['matched_requests_completed_in_control_only'] for a, b in zip(series, series[1:]))
    last_deficit = [b['time_s'] for a, b in zip(series, series[1:]) if a['matched_requests_completed_in_control_only']]
    return {'control_episode': control[0]['episode'], 'matched_offered_requests': len(left),
            'offered_schedule_and_shapes_match': True,
            'post_anchor_first_token_delay_relative_control_s': stats(delayed),
            'peak_matched_completion_deficit_requests': max(s['matched_requests_completed_in_control_only'] for s in series),
            'matched_completion_deficit_area_request_s': area,
            'last_observed_matched_completion_deficit_interval_end_s': max(last_deficit) if last_deficit else None,
            'unresolved_matched_completion_deficit_at_boundary': series[-1]['matched_requests_completed_in_control_only'],
            'completion_deficit_events': series}


def build():
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    manifest, plan = read(DATA / 'data-manifest.json'), read(SOURCE / 'plan.json')
    for name, checksum in manifest['output_sha256'].items():
        path = DATA / name
        if sha(path) != checksum:
            raise ValueError('compact table checksum mismatch: ' + name)
        sources[str(path.relative_to(ROOT))] = checksum
    rows = load_requests(DATA / 'requests.csv')
    if any(r['status'] not in (200, 'failed', 'dependency_failed', 'censored') for r in rows if r['cohort'] in ('resident', 'incoming')):
        raise ValueError('unrecognized offered service status')
    for name in ('pool_shed_resident_data.py', 'pool_shed_execution.py'):
        sources[name] = sha(ROOT / name)
    groups, engines, events = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row['episode']].append(row)
    with (DATA / 'engine.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['serving_role'] == 'destination':
                engines[row['episode']].append({k: (float(v) if v else None) for k, v in row.items() if k not in ('episode', 'serving_role')})
    with (DATA / 'migration-events.csv').open() as stream:
        for row in csv.DictReader(stream):
            events[row['episode']].append(row)
    if len(groups) != 20 or sum(r['cohort'] in ('resident', 'incoming') for r in rows) != manifest['service_requests']:
        raise ValueError('expected all twenty episodes and every offered service request')
    targets, episodes, primary = plan['analysis']['targets'], [], []
    for spec in manifest['episodes']:
        name, boundary = spec['episode'], spec['duration_s']
        resident = [r for r in groups[name] if r['cohort'] == 'resident']
        incoming = [r for r in groups[name] if r['cohort'] == 'incoming']
        snapshots = [float(e['time_s']) for e in events[name] if e['kind'] == 'snapshot']
        handoffs = [float(e['time_s']) for e in events[name] if e['kind'] == 'route_switch']
        anchor = min(snapshots) if snapshots else 30.
        handoff = max(handoffs) if handoffs else None
        spans = [(0., 30.), (30., 60.), (60., 90.), (90., 120.), (120., 180.), (180., 240.), (240., 300.)]
        spans = [(a, b) for a, b in spans if b <= boundary]
        windows = [{**window(resident, a, b, boundary, targets), 'engine': engine_window(engines[name], a, b)} for a, b in spans]
        report = {'episode': name, 'workload': spec['workload'], 'seed': spec['seed'], 'arm': spec['arm'],
                  'resident_offered_rps': spec['rate'], 'boundary_s': boundary, 'anchor_s': anchor,
                  'last_handoff_s': handoff, 'resident_at_anchor': state(resident, anchor),
                  'resident_windows': windows,
                  'all_service_at_boundary': {'resident': state(resident, boundary), 'incoming': state(incoming, boundary)},
                  'incoming_request_roles': {role: sum(r['serving_role'] == role for r in incoming) for role in ('source', 'destination')}}
        if handoff is not None:
            control_name = f"episodes-{spec['workload']}-0-control-{spec['seed']}"
            control = [r for r in groups[control_name] if r['cohort'] == 'resident']
            report['handoff_resident_finite_cohort'] = clearance(resident, handoff, boundary)
            report['handoff_incoming_finite_cohort'] = clearance(incoming, handoff, boundary)
            report['burst_60_90_finite_cohort'] = clearance([r for r in resident if 60 <= r['scheduled_s'] < 90], 90., boundary)
            report['relative_control'] = versus_control(resident, control, anchor, boundary, handoff)
            report['after_handoff'] = {cohort: [window(selected, a, b, boundary, targets) for a, b in
                ((handoff, min(handoff + 30., boundary)), (handoff, boundary)) if a < b]
                for cohort, selected in (('resident', resident), ('incoming_all_offered', incoming))}
            w = next(w for w in windows if w['start_s'] == 60.)
            primary.append({'episode': name, 'workload': spec['workload'], 'seed': spec['seed'], 'arm': spec['arm'],
                            'resident_arrivals_60_90': w['arrivals'], 'resident_first_coverage_60_90': w['first_token_coverage_by_boundary'],
                            'resident_unfinished_60_90_by_boundary': w['arrival_cohort_unfinished_by_boundary'],
                            'resident_ttft_p90_60_90_s': w['arrival_ttft_s']['p90'],
                            'known_ttft_over_target_60_90': w['known_ttft_over_target'],
                            'resident_tpot_p90_60_90_s': w['client_mean_tpot_s']['p90'], 'resident_tpot_samples_60_90': w['client_mean_tpot_s']['n'],
                            'known_mean_tpot_over_target_60_90': w['known_mean_tpot_over_target'],
                            'last_handoff_after_nominal_trigger_s': handoff - 60.,
                            **report['handoff_resident_finite_cohort']})
        episodes.append(report)
    server = read(SERVER)
    server_windows = []
    for episode in server['server_conditioned_episodes']:
        resident = [r for r in episode['requests'] if r['cohort'] == 'resident']
        server_windows.append({'episode': episode['episode'], 'resident_rows': len(resident), 'windows': [
            {'start_s': a, 'end_s': b, **{field: stats([r[field]['observed'] for r in resident if a <= r['scheduled_s'] < b and r[field]['observed'] is not None])
                 for field in ('first_schedule_wait_s', 'registration_to_ready_s', 'server_ready_tpot_s')}}
                 for a, b in ((0., 60.), (60., 90.), (90., 120.), (120., 180.), (180., 240.), (240., 300.))]})
    return {'schema': 'a100-observed-resident-recovery-v1', 'new_gpu_jobs': 0, 'new_policy_evaluations': 0,
            'sources': sources, 'reducer_sha256': sha(Path(__file__)), 'targets': targets,
            'source_episode_contract': plan['episode_contract'], 'episodes': episodes, 'primary_60_90_and_handoff': primary,
            'east_server_observations': {'source_global_telemetry_accepted': server['global_telemetry_accepted'],
                'resident_latency_validated': server['resident_latency_validated'], 'windows': server_windows,
                'scope': 'Existing complete local server joins, grouped by offered-arrival window; full observed generation may finish after the window. Server intervals share verified local clocks. No Germany/East clock subtraction or client-derived GPU iteration time.'},
            'scope': ['One measured source GPU and one destination GPU per episode; all twenty Germany episodes, including eight resident-only scouts, remain present.',
                'Arrival-window latency uses observations through the fixed episode boundary; window-end coverage is separately retained. Unfinished and undispatched arrivals remain in denominators. Known violation counts include right-censored TTFT lower bounds; failed/dependency-failed requests with no first token remain separate failures, not latency observations through the boundary.',
                'Completed-request throughput counts successful response completion events in each interval. Output-token credit is assigned at response completion and is not a token-delivery rate or exclusive GPU work.',
                'Outstanding counts unresolved offered service obligations. Explicit client failures and requests censored at the observation cutoff remain separate counts; neither is classified as a known live server request, and censoring does not establish terminal server execution. Client pending dispatch and predecessor blocking are distinct from sampled all-population engine running/waiting; client-after-first does not identify exclusive GPU decode activity.',
                'Finite handoff-cohort clearance excludes later arrivals. First empty resident state and the end of a matched completion deficit do not prove stationary recovery or a window SLO. Cleanup after the boundary is not recovery.',
                'Controls share offered resident shapes and arrivals but perform recorded destination mirroring; actual incoming source evolution can differ. Comparisons describe measured outcomes, not isolated pure migration causality.',
                'Client mean TPOT can contain buffered token delivery; only the separate East server-ready intervals diagnose server generation cadence.',
                'The current fleet starts resident fluid debt at zero, creates debt only during migration compute, and drains it using average headroom. That state cannot encode already-active requests, serialized pending turns, individual first-token deadlines or delayed long output completions. Clearing this scalar by D does not certify the posted P90 TTFT or TPOT targets.']}


def test_recovery_accounting():
    common = {'episode': 'test', 'cohort': 'resident', 'session': 0, 'status': 200, 'done': True,
              'planned_output_tokens': 2, 'output_tokens': 2, 'mean_tpot_s': .1, 'start_s': 0., 'duration_s': 7.,
              'planned_prompt_tokens': 4, 'planned_append_tokens': 1, 'planned_reset': False}
    a = {**common, 'row_id': 'a', 'turn': 0, 'predecessor_id': None, 'scheduled_s': 0., 'client_dispatch_s': 0., 'first_s': 1., 'end_s': 4.}
    b = {**common, 'row_id': 'b', 'turn': 1, 'predecessor_id': 'a', 'scheduled_s': 2., 'client_dispatch_s': 4., 'first_s': 5., 'end_s': 8.}
    assert state([a, b], 3.)['predecessor_blocked'] == 1
    assert state([a, b], 3.)['outstanding'] == 2
    assert clearance([a, b], 3., 10.)['finite_cohort_clear_delay_s'] == 5.
    assert clearance([a, b], 3., 7.)['finite_cohort_cleared_s'] is None
    c = {**b, 'done': False, 'status': 'timeout', 'end_s': 7.}
    assert state([a, c], 9.)['outstanding'] == 1
    w = window([a, {**c, 'status': 'censored'}], 0., 7., 7., {'p90_ttft_s': 1., 'p90_mean_tpot_s': .1})
    assert (w['arrival_cohort_failed'], w['arrival_cohort_censored'], w['arrival_cohort_unfinished_by_boundary']) == (0, 1, 1)
    s = state([{**c, 'status': 'censored', 'end_s': 7.05}], 7.)
    assert (s['outstanding'], s['failed_client_terminal_obligations'], s['censored_observation_obligations'], s['dispatched_after_first']) == (1, 0, 1, 0)
    failed = {**a, 'status': 'failed', 'done': False, 'first_s': None, 'end_s': .05}
    w = window([failed], 0., 7., 7., {'p90_ttft_s': 1., 'p90_mean_tpot_s': .1})
    assert (w['arrival_cohort_failed'], w['known_ttft_over_target']) == (1, 0)
    assert (state([failed], 7.)['failed_client_terminal_obligations'], state([failed], 7.)['dispatched_without_first']) == (1, 0)
    assert versus_control([a, b], [a, b], 0., 10., 3.)['matched_completion_deficit_area_request_s'] == 0


if __name__ == '__main__':
    result = build()
    path = Path(__file__).with_name('recovery.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'episodes': len(result['episodes']), 'primary_rows': len(result['primary_60_90_and_handoff']), 'path': str(path.relative_to(ROOT))}))
