import json
from pathlib import Path

import pytest

from pool_replay_server_acquire import can_start, checked_json, overhead_check, summarize_warm
from pool_replay_resident import arrival_trace, physical_workload
from pool_replay_measure import write


def test_smoke_interpolation_and_inconclusive_baselines():
    rows = [{'warm_valid': True, 'midpoint_ns': t, 'median_duration_s': d,
             'output_tokens_s': r} for t, d, r in [(10, 10, 100), (20, 10.2, 98), (30, 10.4, 96)]]
    assert overhead_check(rows)['status'] == 'passed'
    rows[1]['median_duration_s'] = 11
    assert overhead_check(rows)['status'] == 'failed'
    rows[2]['median_duration_s'] = 11
    assert overhead_check(rows)['status'] == 'inconclusive'
    rows[1]['warm_valid'] = False
    assert overhead_check(rows)['status'] == 'failed'


def test_reserves_include_cleanup():
    assert can_start(540, 420, now=0)
    assert not can_start(539.999, 420, now=0)
    assert not can_start(389, 270, now=0)


def test_warm_cache_misses_and_preemptions_are_invalid():
    cell = {'concurrency': 1, 'context_tokens': 8192}
    row = {'done': True, 'status': 200, 'error': '', 'prompt_tokens': 8192,
           'output_tokens': 1536, 'recorded_output_tokens': 1536, 'cached_tokens': 8160,
           'start_ns': 1_000_000_000, 'end_ns': 11_000_000_000, 'client_done_ns': 11_000_000_000}
    metrics = {'vllm:external_prefix_cache_hits_total': 0, 'vllm:num_preemptions_total': 0}
    assert summarize_warm([row], cell, 1536, metrics, metrics)['warm_valid']
    for key, value in [('cached_tokens', 8144), ('recorded_output_tokens', 1535), ('done', False)]:
        assert not summarize_warm([{**row, key: value}], cell, 1536, metrics, metrics)['warm_valid']
    for key in metrics:
        assert not summarize_warm([row], cell, 1536, metrics, {**metrics, key: 1})['warm_valid']


def test_frozen_episode_inputs_reproduce_hashes(tmp_path):
    plan = json.loads(Path('outputs/a100-resident-server-timing-plan/plan.json').read_text())
    reference = checked_json(plan['reference_plan'], plan['reference_plan_sha256'])
    for entry in plan['agentic_contract']['episodes']:
        spec = entry['spec']
        sampled = physical_workload(reference['workloads'][spec['workload']], spec['seed'])
        write(tmp_path/'physical.json', sampled)
        checked_json(tmp_path/'physical.json', entry['physical_workload_sha256'])
        trace = [{**r, 'cohort': 'resident'} for r in arrival_trace(spec['rate'], [1]*8, spec['seed'], 300)]
        trace += [{**r, 'cohort': 'incoming'} for r in arrival_trace(spec['incoming_session_rps']*8, [1]*8, spec['seed']+1, 300)]
        trace.sort(key=lambda r: (r['offset_s'], r['cohort'], r['session'], r['turn']))
        write(tmp_path/'trace.json', trace)
        checked_json(tmp_path/'trace.json', entry['offered_trace_sha256'])
    (tmp_path/'trace.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        checked_json(tmp_path/'trace.json', entry['offered_trace_sha256'])


def test_warm_cancellation_retains_partial_request(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    import pool_replay_server_acquire as module

    class Response:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def text(self):
            return 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'

    class Client(Response):
        def get(self, *args, **kwargs):
            return Response()
        post = get

    monkeypatch.setattr(module.aiohttp, 'ClientSession', lambda **kwargs: Client())
    monkeypatch.setattr(module.serving, 'MetricsSampler', lambda *args: SimpleNamespace(start=lambda: None, close=lambda: None))
    records = []
    acquisition = SimpleNamespace(cfg=SimpleNamespace(host='unused', sink_port=1, model='test'),
        deadline=float('inf'), requests='requests', events='events', record=lambda handle, row: records.append((handle, row)))

    async def completion(client, host, port, item, dispatch, timeout, event_sink):
        if item['session'].output_tokens == 1:
            return {'done': True, 'status': 200, 'error': '', 'finish_reason': 'length',
                    'planned_output_tokens': 1, 'output_tokens': 1, 'recorded_output_tokens': 1,
                    'prompt_tokens': len(item['prompt'])}
        event_sink({'monotonic_ns': dispatch+1, 'data': json.dumps({'id': 'partial', 'choices': [{'token_ids': [42]}]})})
        raise asyncio.CancelledError

    monkeypatch.setattr(module.headroom, 'async_completion', completion)
    cell = {'id': 'cancelled', 'context_tokens': 8192, 'concurrency': 1, 'seed': 8101}
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.warm(acquisition, cell, 1536, tmp_path/'cell', 'cancelled'))
    row = [r for handle, r in records if handle == 'requests' and r['phase'] == 'measurement'][0]
    assert row['status'] == 'censored'
    assert row['recorded_output_tokens'] == 1 and row['token_ids'] == [42]
    assert row['request_id'] == 'partial' and row['planned_output_tokens'] == 1536
    assert row['planned_prompt_tokens'] == 8192


def test_bounded_worker_sigterm_runs_cancellation_cleanup(tmp_path):
    import inspect
    import subprocess
    import sys
    import time
    from pool_replay_server_acquire import bounded_command, run_cancellable

    marker = tmp_path/'cancelled'
    code = ('import asyncio, signal\nfrom pathlib import Path\n'+inspect.getsource(run_cancellable)+'\n'
            'async def work():\n try: await asyncio.sleep(60)\n'
            f' finally: Path({str(marker)!r}).write_text("cancelled")\nrun_cancellable(work())\n')
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_command([sys.executable, '-c', code], 3, tmp_path/'worker.log')
    assert marker.read_text() == 'cancelled'
    assert time.monotonic()-started < 3


def test_startup_failure_cleans_partial_owned_stack(monkeypatch, tmp_path):
    import runpy
    import shutil
    import signal
    import subprocess
    import sys
    from types import SimpleNamespace

    scripts = Path('outputs/a100-resident-server-timing-20260910T1723')
    shutil.copy(scripts/'launch-stack.py', tmp_path/'launch-stack.py')
    shutil.copy('outputs/a100-resident-server-timing-plan/plan.json', tmp_path/'plan.json')
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=4242))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'scp':
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        runpy.run_path(str(tmp_path/'launch-stack.py'), run_name='__main__')
    assert json.loads((tmp_path/'owned-processes.json').read_text()) == {'source': 4242, 'tunnels_and_mirrors': []}
    report = json.loads((tmp_path/'acquisition-report.json').read_text())
    assert report['status'] == 'failed' and len(report['cells']) == 16
    assert all(cell['status'] == 'unstarted' for cell in report['cells'])
    assert commands[-1][-1] == 'cleanup'


def test_cleanup_without_launched_roles_does_not_contact_nodes(monkeypatch, tmp_path):
    import runpy
    import shutil
    import subprocess
    import sys

    shutil.copy('outputs/a100-resident-server-timing-20260910T1723/control-stack.py', tmp_path/'control-stack.py')
    (tmp_path/'owned-processes.json').write_text('{"tunnels_and_mirrors": []}')
    monkeypatch.setattr(sys, 'argv', ['control-stack.py', 'cleanup'])
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('no launched nodes to contact'))
    runpy.run_path(str(tmp_path/'control-stack.py'), run_name='__main__')
    assert json.loads((tmp_path/'cleanup-result.json').read_text())['status'] == 'complete'


def test_supervisor_sigterm_interrupts_startup_and_records_shutdown(monkeypatch, tmp_path):
    import os
    import runpy
    import shutil
    import signal
    import sys
    from types import SimpleNamespace

    shutil.copy('outputs/a100-resident-server-timing-20260910T1723/node-supervisor.py', tmp_path/'node-supervisor.py')
    (tmp_path/'runtime-launch.json').write_text('{"deadline_wall_ns": 9999999999999999999}')
    monkeypatch.setattr(sys, 'argv', ['node-supervisor.py', str(tmp_path), str(tmp_path), 'source'])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    handlers, children, events = {}, {}, []
    monkeypatch.setattr(signal, 'signal', lambda number, handler: handlers.update({number: handler}))
    class Child:
        def __init__(self, pid):
            self.pid, self.returncode = pid, None
        def poll(self):
            return self.returncode
        def wait(self, **kwargs):
            return self.returncode
    def launch(*args):
        child = Child(len(children)+1)
        children[child.pid] = child
        return child
    def kill(pid, number):
        events.append('child_stop')
        children[pid].returncode = 0
    monkeypatch.setattr(os, 'killpg', kill)
    b = SimpleNamespace(Config=lambda **kwargs: SimpleNamespace(host='localhost', **kwargs),
        start_logged=launch, redis_cmd=lambda cfg: ['redis'],
        wait_tcp_process=lambda *args: handlers[signal.SIGTERM](signal.SIGTERM, None))
    p = SimpleNamespace(PowerSampler=lambda *args: SimpleNamespace(start=lambda: None, close=lambda: events.append('power_stop')))
    for name, module in [('migration_testbed', b), ('migration_profiler', p), ('destination_runner', SimpleNamespace())]:
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(SystemExit, match='supervisor terminated'):
        runpy.run_path(str(tmp_path/'node-supervisor.py'), run_name='__main__')
    stop = json.loads((tmp_path/'stop.json').read_text())
    assert stop['forced_kills'] == stop['telemetry_errors'] == []
    assert set(stop['returncodes']) == {'gpu-clocks', 'redis'}
    assert events == ['power_stop', 'child_stop', 'child_stop']
