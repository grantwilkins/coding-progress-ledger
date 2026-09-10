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
    assert all(row['concurrency'] == 8 and row['deadline_s'] == 180 for row in rows)
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
