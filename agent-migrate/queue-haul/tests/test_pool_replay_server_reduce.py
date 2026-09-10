import json

from pool_replay_server_reduce import client_rows, reduce


def evidence(tmp_path, host='node', token=7, final=True):
    identity = dict(host=host, boot_id='boot', time_namespace='time:[1]', time_namespace_offsets='monotonic 0 0')
    scheduler = [dict(kind='schedule', iteration='iteration-1', start_ns=6, end_ns=7,
                      requests=[dict(request_id='internal', scheduled_tokens=32, computed_before=8160)])]
    worker = [dict(kind='worker_output', iteration='iteration-1', output_ready_ns=10,
                   forward_stream_ms=1.0, logits_stream_ms=0.5, sample_stream_ms=0.2,
                   requests=[dict(request_id='internal', ordinal_start=0, token_ids=[7])])]
    frontend = [dict(kind='http_ingress', ingress_id='ingress'),
                dict(kind='http_request_mapping', ingress_id='ingress', request_id='cmpl-a'),
                dict(kind='frontend_registration', request_id='internal', external_request_id='cmpl-a-0')]
    frontend += [dict(kind=kind, request_id='internal', ordinal_start=0, token_ids=[token], mono_ns=20 + i)
                 for i, kind in enumerate(('frontend_receipt', 'collector_put', 'collector_pop'))]
    files = []
    for name, rows in (('scheduler', scheduler), ('worker', worker), ('frontend', frontend)):
        rows = [dict(kind='process_start', **identity), dict(kind='module', path=name+'.py', sha256='abc')] + rows
        if final:
            rows.append(dict(kind='process_final', dropped_records=0, records_before_final=len(rows)))
        for sequence, row in enumerate(rows, 1):
            row['sequence'] = sequence
            row.setdefault('mono_ns', sequence)
        path = tmp_path / (name + '.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        files.append(path)
    client = dict(request_id='cmpl-a', start_ns=0, end_ns=35, status=200, done=True, output_tokens=1,
                  token_ids=[7], token_events=[dict(token_ids=[7], monotonic_ns=30)], scheduled_ns=-5)
    return files, client, identity


def test_complete_iteration_request_token_usage_and_clock_joins(tmp_path):
    files, client, identity = evidence(tmp_path)
    result = reduce(files, [client], identity)
    assert result['accepted'], result['errors']
    assert result['counts']['generated_tokens'] == 1
    assert result['gpu_stream_elapsed_ms']['forward']['median'] == 1.0
    assert result['same_clock_host_intervals_ms']['client_receive_from_collector_pop_ms']['median'] == 8 / 1e6
    assert result['causal_client_queue_s']['median'] == 5 / 1e9
    assert result['same_clock_host_intervals_ms']['registration_to_first_schedule_ms']['median'] == 1 / 1e6


def test_cross_host_monotonic_clocks_are_never_subtracted(tmp_path):
    files, client, identity = evidence(tmp_path)
    result = reduce(files, [client], dict(identity, host='different-client'))
    assert result['accepted']
    assert 'client_receive_from_collector_pop_ms' not in result['same_clock_host_intervals_ms']


def test_truncated_process_shutdown_fails_completeness(tmp_path):
    files, client, _ = evidence(tmp_path, final=False)
    result = reduce(files, [client])
    assert not result['accepted']
    assert sum(error['kind'] == 'missing_or_invalid_process_final' for error in result['errors']) == 3


def test_token_corruption_and_final_usage_mismatch_are_visible(tmp_path):
    files, client, _ = evidence(tmp_path, token=9)
    result = reduce(files, [dict(client, output_tokens=2)])
    assert not result['accepted']
    kinds = {error['kind'] for error in result['errors']}
    assert {'token_mismatch', 'client_token_not_joined', 'client_usage_mismatch'} <= kinds


def test_sequence_gap_and_scheduled_budget_are_rejected(tmp_path):
    files, client, _ = evidence(tmp_path)
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    rows[2]['sequence'] = 8
    rows[2]['requests'][0]['scheduled_tokens'] = 8193
    files[0].write_text(''.join(json.dumps(row)+'\n' for row in rows))
    kinds = {error['kind'] for error in reduce(files, [client])['errors']}
    assert {'record_sequence_gap', 'scheduled_token_budget'} <= kinds


def test_failed_and_uninstrumented_clients_remain_counted(tmp_path):
    files, client, _ = evidence(tmp_path)
    failed = dict(start_ns=40, end_ns=50, status='censored', done=False)
    off = dict(client, request_id='off-only', timing_mode='off')
    result = reduce(files, [client, failed, off])
    assert not result['accepted']
    assert result['counts']['failed_or_censored'] == 1
    assert result['counts']['uninstrumented_client_requests'] == 1
    assert any(error['kind'] == 'client_request_mapping_missing' for error in result['errors'])
    assert list(client_rows(dict(nested=[client, failed]))) == [client, failed]


def test_missing_collector_boundary_and_ordinal_gap_cannot_certify(tmp_path):
    files, client, _ = evidence(tmp_path)
    rows = [json.loads(line) for line in files[2].read_text().splitlines()]
    rows = [row for row in rows if row['kind'] != 'collector_put']
    next(row for row in rows if row['kind'] == 'collector_pop')['ordinal_start'] = 2
    files[2].write_text(''.join(json.dumps(row)+'\n' for row in rows))
    kinds = {error['kind'] for error in reduce(files, [client])['errors']}
    assert {'upstream_tokens_without_downstream', 'noncontiguous_token_ordinals'} <= kinds


def test_undispatched_censoring_is_retained_without_demanding_a_server_id(tmp_path):
    files, client, _ = evidence(tmp_path)
    undispatched = dict(status='censored', done=False, scheduled_ns=10, end_ns=30)
    result = reduce(files, [client, undispatched])
    assert result['accepted'], result['errors']
    assert result['counts']['undispatched_offered_requests'] == 1
    assert list(client_rows(dict(nested=undispatched))) == [undispatched]


def test_http_ingress_without_registration_is_visible(tmp_path):
    files, client, _ = evidence(tmp_path)
    rows = [json.loads(line) for line in files[2].read_text().splitlines()]
    rows.insert(-1, dict(kind='http_ingress', ingress_id='failed-before-registration', sequence=99))
    files[2].write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert any(error['kind'] == 'http_ingress_without_request_mapping' for error in reduce(files, [client])['errors'])


def test_impossible_same_clock_client_chronology_is_rejected(tmp_path):
    files, client, identity = evidence(tmp_path)
    client['token_events'][0]['monotonic_ns'] = 1
    result = reduce(files, [client], identity)
    assert not result['accepted']
    assert any(error['kind'] == 'negative_host_interval' and error['boundary'] == 'client_receive' for error in result['errors'])
