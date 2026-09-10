from types import SimpleNamespace
import subprocess
import sys

import pytest

import pool_replay_paired as paired


def test_bounded_counterbalanced_plan_never_resets_attached_stack():
    rows = paired.scenarios()
    assert len(rows) == 8
    assert [row['method'] for row in rows] == ['replay', 'kv_transfer']*2 + ['kv_transfer', 'replay']*2
    assert {(row['context_size'], row['activity_tokens']) for row in rows} == {(8192, 32), (30000, 2048)}
    assert all(row['concurrency'] == 8 and row['deadline_s'] == paired.SCENARIO_LIMIT_S for row in rows)
    assert all(not row['reset_caches'] and not row['wait_cache_idle'] and row['final_state'] == 'awake' for row in rows)
    assert len({session['session_id'] for row in rows for session in row['sessions']}) == 64


def session():
    result = paired.CheckedSession.__new__(paired.CheckedSession)
    result.cfg = SimpleNamespace(max_model_len=32768)
    result.session_id, result.state_code = 'test', '012345ABCDEF'
    result.event_log = SimpleNamespace(write=lambda *args, **kwargs: None)
    return result


def test_context_failure_preserves_probe_and_never_sends_cheaper_request(monkeypatch):
    seen = []
    monkeypatch.setattr(paired.b, 'mp_chat_tokens', lambda cfg, messages, max_tokens:
                        seen.append((messages, max_tokens)) or [1]*32257)
    monkeypatch.setattr(paired.p.LiveSession, 'request', lambda *args: pytest.fail('must not send oversized request'))
    with pytest.raises(ValueError, match='512-token'):
        session().request(8100, [{'role': 'user', 'content': 'retained'}], 'initial')
    assert seen[0][1] == 512
    assert seen[0][0][-1]['content'] == 'Reply with session state code 012345ABCDEF.'


def test_http_success_without_state_is_failure(monkeypatch):
    monkeypatch.setattr(paired.b, 'mp_chat_tokens', lambda *args: [1, 2])
    monkeypatch.setattr(paired.p.LiveSession, 'request', lambda *args:
                        (SimpleNamespace(request_id='r', prompt_tokens=2), ''))
    with pytest.raises(RuntimeError, match='state code'):
        session().request(8100, [], 'initial')


def test_actual_rendered_prompt_is_verified(monkeypatch):
    monkeypatch.setattr(paired.b, 'mp_chat_tokens', lambda *args: [1, 2])
    monkeypatch.setattr(paired.p.LiveSession, 'request', lambda *args:
                        (SimpleNamespace(request_id='r', prompt_tokens=3), '012345ABCDEF'))
    with pytest.raises(RuntimeError, match='rendered token count'):
        session().request(8100, [], 'initial')


def test_timeout_is_failure_and_does_not_retry(tmp_path):
    result = paired.run_worker([sys.executable, '-c', 'import time; time.sleep(10)'], .05, tmp_path/'worker.log')
    assert result['status'] == 'timeout' and result['returncode'] < 0


def test_same_gpu_rejected_before_runtime_access():
    with pytest.raises(ValueError, match='separate physical GPUs'):
        paired.validate_inventory({'source': {'gpu_uuid': 'same'}, 'destination': {'gpu_uuid': 'same'}})


def test_worker_config_is_copied_without_mutating_frozen_reference(monkeypatch, tmp_path):
    cfg = paired.b.Config()
    calls = []
    monkeypatch.setattr(paired, 'validate_inventory', lambda inventory: cfg)
    monkeypatch.setattr(paired, 'validate_completion', lambda *args: None)
    monkeypatch.setattr(paired.p, 'run_scenario', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(paired.p, 'LiveSession', paired.p.LiveSession)
    paired.worker({'stack_root': str(tmp_path), 'bandwidth_mbps': 1000}, paired.scenarios()[0], tmp_path)
    assert not cfg.architecture_campaign
    assert calls[0][0][1].architecture_campaign
    assert calls[0][1] == {'configure_proxy': False}


def test_method_filter_keeps_four_original_conditions_in_order():
    original = paired.scenarios()
    for method in ('replay', 'kv_transfer'):
        rows = paired.scenarios(method)
        assert rows == [r for r in original if r['method'] == method]
        assert len(rows) == 4
        assert [(r['seed'],r['context_size']) for r in rows] == [(7101,8192),(7101,30000),(7102,8192),(7102,30000)]


def test_method_filter_is_frozen_with_runtime_dependency_hashes(monkeypatch,tmp_path):
    import json
    from pathlib import Path
    evidence=tmp_path/'identity.json'; evidence.write_text('{}')
    inventory=tmp_path/'inventory.json'
    inventory.write_text(json.dumps({role:{'identity_evidence':str(evidence),'runtime_evidence':str(evidence)} for role in ('source','destination')}))
    monkeypatch.setattr(paired,'validate_inventory',lambda raw:paired.b.Config())
    args=['paired','freeze','--inventory',str(inventory),'--out',str(tmp_path),'--method','kv_transfer']
    monkeypatch.setattr(sys,'argv',args)
    paired.main()
    plan=json.loads((tmp_path/'paired-plan.json').read_text())
    assert plan['method']=='kv_transfer' and len(plan['scenarios'])==4
    for path in (paired.p.__file__,paired.b.__file__):
        assert plan['input_sha256'][path]==paired.p.file_hash(Path(path))
    monkeypatch.setattr(sys,'argv',['paired','worker','--inventory',str(inventory),'--out',str(tmp_path),'--method','replay','--index','0'])
    with pytest.raises(ValueError,match='frozen bounded plan'):
        paired.main()


def test_context_filter_keeps_both_seeds_without_extra_conditions():
    rows=paired.scenarios('kv_transfer',8192)
    assert [(r['seed'],r['context_size'],r['method']) for r in rows]==[(7101,8192,'kv_transfer'),(7102,8192,'kv_transfer')]


def test_original_wall_deadline_never_extends_local_budget(monkeypatch):
    monkeypatch.setattr(paired.time,'monotonic',lambda:200.)
    monkeypatch.setattr(paired.time,'time_ns',lambda:1_000_000_000_000)
    assert paired.remaining_seconds(100.)==1400
    assert paired.remaining_seconds(100.,1_010_000_000_000)==10
    assert paired.remaining_seconds(100.,900_000_000_000)==0
    assert paired.remaining_seconds(100.,9_000_000_000_000)==1400


def test_distinct_run_directories_have_disjoint_source_cache_histories(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(paired,'validate_inventory',lambda inventory:paired.b.Config())
    monkeypatch.setattr(paired,'validate_completion',lambda *args:None)
    monkeypatch.setattr(paired.p,'run_scenario',lambda *args,**kwargs:calls.append(args))
    monkeypatch.setattr(paired.p,'LiveSession',paired.p.LiveSession)
    original=paired.scenarios('kv_transfer',8192)[0]
    for label in ('first','corrected'):
        paired.worker({'stack_root':str(tmp_path),'bandwidth_mbps':1000},original,tmp_path/label/'scenario')
    ids=[{r['id'] for r in call[2]['sessions']} for call in calls]
    assert len(ids[0])==len(ids[1])==8 and ids[0].isdisjoint(ids[1])
    for call,keys in zip(calls,ids):
        assert keys=={r['session_id'] for r in call[3]['sessions']}=={r['session_id'] for r in call[3]['moves']}
    assert original['sessions'][0]['session_id'].startswith('paired-')


@pytest.mark.parametrize("remaining", [14.6, 329.9])
def test_insufficient_full_allowance_never_launches_worker(monkeypatch, tmp_path, remaining):
    import json
    evidence = tmp_path/"identity.json"; evidence.write_text("{}")
    inventory = tmp_path/"inventory.json"
    inventory.write_text(json.dumps({role: {"identity_evidence": str(evidence), "runtime_evidence": str(evidence)} for role in ("source", "destination")}))
    monkeypatch.setattr(paired, "validate_inventory", lambda raw: paired.b.Config())
    args = ["paired", "freeze", "--inventory", str(inventory), "--out", str(tmp_path), "--method", "kv_transfer"]
    monkeypatch.setattr(sys, "argv", args)
    paired.main()
    monkeypatch.setattr(paired, "remaining_seconds", lambda *args: remaining)
    monkeypatch.setattr(paired, "run_worker", lambda *args: pytest.fail("must reserve complete scenario"))
    monkeypatch.setattr(sys, "argv", [args[0], "run", *args[2:]])
    paired.main()
    attempts = json.loads((tmp_path/"paired-attempts.json").read_text())
    assert len(attempts) == 4 and all(row["status"] == "unmeasured_budget_limit" for row in attempts)


@pytest.mark.parametrize("missing", [None, "catch_up", "committed_state", "continuations"])
def test_complete_requires_catchup_state_and_destination_continuation(missing):
    move = {"move": {"session_id": "s"}, "initial": {"context_hash": "before"},
            "catch_up": {"context_hash": "after"}, "committed_state": {"context_hash": "after", "generation": 1},
            **dict(zip(("initial_end_ns", "pause_start_ns", "idle_ns", "catch_up_start_ns", "catch_up_end_ns", "switch_start_ns", "switch_end_ns"), range(1, 8)))}
    result = {"status": "complete", "migrations": [move], "continuations": [{"session_id": "s", "committed_context_hash": "after", "status_code": 200, "route_port": 8400, "start_ns": 8}]}
    if missing == "continuations": result[missing] = []
    elif missing: move[missing] = None
    if missing:
        with pytest.raises(RuntimeError): paired.validate_completion(result, {"sessions": [{"session_id": "s"}]}, 8400)
    else:
        paired.validate_completion(result, {"sessions": [{"session_id": "s"}]}, 8400)
