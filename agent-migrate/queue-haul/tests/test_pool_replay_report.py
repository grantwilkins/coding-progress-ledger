import json
import csv
from types import SimpleNamespace

import pool_replay_report as report


def test_missing_operating_point_never_starts_policy_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(report.q,'sample_fleet',lambda *a,**k:SimpleNamespace(metadata={'reference_rps':.4}))
    monkeypatch.setattr(report.q,'provenance',lambda c:{})
    def forbidden(*args,**kwargs):
        raise AssertionError('unqualified hardware evidence must not start simulation')
    monkeypatch.setattr(report.q,'network_samples',forbidden)
    monkeypatch.setattr(report.q,'execute_feedback',forbidden)
    report.policies(tmp_path,[],{})
    result=json.loads((tmp_path/'policy-verification.json').read_text())
    assert result['status']=='unmeasured_qualified_operating_point_missing' and result['cells']==[]
    assert not result['campaign_ready'] and not result['fleet_latency_validated']


def test_cleanup_completion_remains_censored_at_episode_boundary(tmp_path):
    report.write(tmp_path/'node-recovery.json', {'raw_pre_reboot': {}})
    for name in ('request-events.jsonl', 'request-events-recovered.jsonl'):
        (tmp_path/name).write_text('')
    root=tmp_path/'episodes-test';root.mkdir()
    report.write(root/'offered-trace.json', [])
    report.write(root/'result.json', {'spec':{'episode':root.name,'arm':'replay','trace_id':'test'},
        'epoch_ns':0,'boundary_ns':180_000_000_000,'migration_events':[]})
    report.csv_rows(root/'engine.csv', [{'monotonic_ns':t,'vllm:num_requests_running':0,
        'vllm:num_requests_waiting':0,'vllm:num_preemptions_total':0,
        'vllm:request_queue_time_seconds_sum':0} for t in (0,180_000_000_000)])
    (root/'power.csv').write_text('valid\n0\n')
    report.service(tmp_path,[{'episode':root.name,'cohort':'migration','phase':'catch_up','status':'complete',
        'start_ns':60_000_000_000,'end_ns':t} for t in (175_000_000_000,185_000_000_000)])
    rows=list(csv.DictReader((tmp_path/'migration-observations.csv').open()))
    assert [r['completed_within_observation'] for r in rows]==['True','False']
    assert [r['observation_status'] for r in rows]==['complete','censored_at_boundary']
    assert all(r['status']=='complete' for r in rows)
