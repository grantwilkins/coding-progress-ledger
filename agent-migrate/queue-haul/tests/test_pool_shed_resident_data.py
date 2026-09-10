import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pool_shed_resident_data import audit, checked_lines, compact_request, digest, load_requests, token_observation, write_csv


def test_archive_member_corruption_hard_fails():
    expected = {'path': 'requests.jsonl', 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
    assert b''.join(checked_lines(io.BytesIO(b'good'), expected)) == b'good'
    with pytest.raises(ValueError, match='hash mismatch'):
        list(checked_lines(io.BytesIO(b'evil'), expected))


def test_censored_tokens_keep_arrival_clock_and_unknown_usage():
    result = {'spec': {'episode': 'episode'}, 'epoch_ns': 10_000_000_000, 'boundary_ns': 20_000_000_000}
    observed = {}
    for stamp in (19_000_000_000, 21_000_000_000):
        token_observation(observed, {'episode': 'episode', 'cohort': 'resident', 'session': 0, 'turn': 0,
            'monotonic_ns': stamp, 'data': json.dumps({'id': 'partial', 'choices': [{'token_ids': [5]}]})}, result['boundary_ns'])
    raw = {'episode': 'episode', 'cohort': 'resident', 'session': 0, 'turn': 0, 'phase': 'service',
           'scheduled_ns': 11_000_000_000, 'client_dispatch_ns': 12_000_000_000, 'start_ns': 12_000_000_000,
           'end_ns': 20_000_000_000, 'done': False, 'status': 'censored', 'full_prompt_token_ids': [1, 2]}
    workload = {'turn_sequences': [[{'context': 1, 'prompt': 1, 'output': 4, 'reset': False}]], 'turn_offset': [0]}
    row = compact_request(raw, result, workload, observed)
    assert (row['scheduled_s'], row['first_s'], row['last_token_s'], row['arrival_ttft_s']) == (1, 9, 11, 8)
    assert (row['observed_output_tokens'], row['observed_tokens_before_boundary'], row['planned_output_tokens']) == (2, 1, 4)
    assert row['cached_tokens'] is None and row['output_tokens'] is None
    assert row['partial_exact_token_timestamps'] and not row['completed_within_observation']
    assert (row['token_gap_count'], row['median_token_gap_s'], row['max_token_gap_s']) == (1, 2, 2)


def test_audit_rejects_missing_arrivals_and_cross_gpu_residents():
    trace = {'episode': [{'cohort': 'resident', 'session': 0, 'turn': 0, 'offset_s': 1}]}
    with pytest.raises(ValueError, match='every unique offered arrival'):
        audit([], {}, trace)
    row = {'episode': 'episode', 'cohort': 'resident', 'session': 0, 'turn': 0, 'scheduled_s': 1,
           'dispatched': True, 'client_dispatch_s': 1, 'serving_role': 'source'}
    with pytest.raises(ValueError, match='changed GPU'):
        audit([row], {'episode': {'migration_events': []}}, trace)


def test_compact_csv_preserves_types(tmp_path):
    rows = [{'episode': 'episode', 'status': 200, 'done': True, 'cached_tokens': None, 'start_s': 1.2, 'output_tokens': 4},
            {'episode': 'episode', 'status': 'censored', 'done': False, 'cached_tokens': None, 'start_s': None, 'output_tokens': None}]
    path = tmp_path/'requests.csv'
    write_csv(path, rows)
    assert load_requests(path) == rows


def test_omitted_cache_zero_requires_completed_final_usage():
    result = {'spec': {'episode': 'episode'}, 'epoch_ns': 10_000_000_000, 'boundary_ns': 20_000_000_000}
    raw = {'cohort': 'migration', 'session': 0, 'phase': 'initial', 'request_id': 'request',
           'scheduled_ns': 11_000_000_000, 'start_ns': 11_000_000_000, 'end_ns': 12_000_000_000,
           'done': True, 'status': 200, 'prompt_tokens': 2, 'output_tokens': 1, 'cached_tokens': None}
    usage = {'request': {'prompt_tokens': 2, 'completion_tokens': 1}}
    row = compact_request(raw, result, {}, {}, usage)
    assert row['cached_tokens'] is None and row['effective_cached_tokens'] == 0
    assert row['cache_usage_basis'] == 'vllm_0_22_omitted_zero'
    assert compact_request(raw, result, {}, {})['effective_cached_tokens'] is None
    assert compact_request({**raw, 'done': False}, result, {}, {}, usage)['effective_cached_tokens'] is None


def test_compact_trace_reproduces_frozen_service_windows():
    rows = load_requests()
    reference = json.loads(Path('outputs/a100-replay-final-20260910T0352/service-analysis.json').read_text())
    for episode in reference['episodes']:
        for window in episode['windows']:
            cohort = [r for r in rows if r['episode'] == episode['spec']['episode'] and r['cohort'] == window['cohort']
                      and window['start_s'] <= r['scheduled_s'] < window['end_s']]
            done = [r for r in cohort if r['done'] and r['status'] == 200 and r['end_s'] <= window['end_s']]
            exact = [r for r in done if r['exact_token_timestamps']]
            first = [r['arrival_ttft_s'] for r in cohort if r['first_s'] is not None and r['first_s'] <= window['end_s']
                     and (r['exact_token_timestamps'] or r['partial_exact_token_timestamps'])]
            assert len(cohort) == window['offered_requests']
            assert len(done) == window['completed_arrival_cohort_requests']
            assert len(exact) == window['exact_requests']
            assert (float(np.quantile(first, .9)) if first else None) == pytest.approx(window['p90_original_arrival_ttft_s'])


def test_compact_artifact_digests_match():
    root = Path('outputs/a100-resident-queues-20260910')
    manifest = json.loads((root/'data-manifest.json').read_text())
    assert all(digest(root/name) == expected for name, expected in manifest['output_sha256'].items())
    assert digest(Path('pool_shed_resident_data.py')) == manifest['reducer_sha256']


def test_client_token_bursts_are_not_hidden_by_single_token_framing():
    result = {'spec': {'episode': 'episode'}, 'epoch_ns': 0, 'boundary_ns': 1_050_000_000}
    raw = {'cohort': 'resident', 'session': 0, 'turn': 0, 'phase': 'service', 'scheduled_ns': 0,
           'start_ns': 0, 'end_ns': result['boundary_ns'], 'done': False, 'status': 'censored'}
    workload = {'turn_sequences': [[{'context': 1, 'prompt': 1, 'output': 5, 'reset': False}]], 'turn_offset': [0]}
    observed = {}
    def event(stamp, tokens):
        return {'episode': 'episode', 'cohort': 'resident', 'session': 0, 'turn': 0, 'monotonic_ns': stamp,
                'data': json.dumps({'id': 'request', 'choices': [{'token_ids': tokens}]})}
    for stamp in (1_000_000_000, 1_000_200_000, 1_030_000_000, 1_060_000_000):
        token_observation(observed, event(stamp, [5]), result['boundary_ns'])
    row = compact_request(raw, result, workload, observed)
    assert row['token_gaps_exact_single_token_frames']
    assert (row['token_gap_count'], row['submillisecond_token_gaps'], row['observed_tokens_before_boundary']) == (3, 1, 3)
    assert row['median_token_gap_s'] == pytest.approx(.0298) and row['max_token_gap_s'] == pytest.approx(.03)
    token_observation(observed, event(1_090_000_000, [5, 6]), result['boundary_ns'])
    row = compact_request(raw, result, workload, observed)
    assert row['token_gaps_exact_single_token_frames'] is False and row['token_gap_count'] is None
    with pytest.raises(ValueError, match='reversed client time'):
        token_observation(observed, event(1_000_000_000, [5]), result['boundary_ns'])
