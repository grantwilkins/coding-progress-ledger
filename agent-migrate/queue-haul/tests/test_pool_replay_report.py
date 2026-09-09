import json
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
