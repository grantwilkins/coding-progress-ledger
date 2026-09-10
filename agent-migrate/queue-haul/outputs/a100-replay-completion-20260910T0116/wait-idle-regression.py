"""Reproduce an invalid source handoff; QH_TEST_WAIT_IDLE_FIX=1 tests the proposed guard."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import migration_profiler as c

if os.environ.get('QH_TEST_WAIT_IDLE_FIX') == '1':
    source = Path(c.__file__).read_text().replace('            state = SessionState(\n                session.session_id, session.generation,', '            if session.activity_error:\n                raise session.activity_error\n            state = SessionState(\n                session.session_id, session.generation,')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LiveRuntime')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'wait_idle')
    namespace = dict(c.__dict__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<proposed-wait-idle>', 'exec'), namespace)
    c.LiveRuntime.wait_idle = namespace['wait_idle']


def test_invalid_source_activity_cannot_commit_destination(monkeypatch):
    events = []
    session = SimpleNamespace(session_id='s', generation=0, messages=[], timeout_s=1,
        activity_condition=threading.Condition(), activity_active=False,
        activity_error=RuntimeError('invalid source state response'), paused=False,
        route=1, cfg=SimpleNamespace(src_port=1))
    runtime = c.LiveRuntime.__new__(c.LiveRuntime)
    runtime.sessions, runtime.cfg = {'s':session}, SimpleNamespace(api_proxy_port=2,src_port=1)
    runtime.event_log = SimpleNamespace(write=lambda event, **fields: events.append(event))
    state = c.SessionState('s',0,(),c.messages_hash([]))
    monkeypatch.setattr(runtime,'snapshot',lambda move:state)
    monkeypatch.setattr(runtime,'prepare',lambda move,snapshot,phase:c.RequestResult('r',200,snapshot.context_hash,1,2))
    monkeypatch.setattr(runtime,'background',lambda move,snapshot:())
    result = c.MigrationController(runtime,1).run([c.Move('s','replay',0)])[0]
    assert result.error and 'invalid source state response' in result.error
    assert result.committed_state is None
    assert session.route == 1
    assert 'route_switch' not in events and 'idle' not in events
