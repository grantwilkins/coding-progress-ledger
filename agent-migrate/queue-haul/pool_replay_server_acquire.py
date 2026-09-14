"""Bounded attach-only acquisition for the frozen resident server timing plan."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import aiohttp

import destination_runner as serving
import service_headroom_campaign as headroom
from pool_replay_measure import write
from pool_replay_paired import validate_inventory
from pool_replay_resident import ResidentAcquisition

PLAN = Path('outputs/a100-resident-server-timing-plan/plan.json')


def checked_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'frozen reference hash mismatch: {path}')
    return json.loads(raw)


def overhead_check(bursts):
    if len(bursts) != 3 or any(not b['warm_valid'] for b in bursts):
        return {'status': 'failed', 'reason': 'incomplete or nonwarm smoke requests'}
    left, middle, right = bursts
    fraction = (middle['midpoint_ns']-left['midpoint_ns'])/(right['midpoint_ns']-left['midpoint_ns'])
    comparisons = {}
    for metric in ('median_duration_s', 'output_tokens_s'):
        a, b, c = (row[metric] for row in bursts)
        baseline_delta = abs(c-a)/a
        interpolated = a+(c-a)*fraction
        comparisons[metric] = {'off_relative_difference': baseline_delta,
                               'on_relative_difference': abs(b-interpolated)/interpolated,
                               'interpolated_off': interpolated}
    status = ('inconclusive' if any(r['off_relative_difference'] > .05 for r in comparisons.values()) else
              'failed' if any(r['on_relative_difference'] > .05 for r in comparisons.values()) else 'passed')
    return {'status': status, 'comparisons': comparisons}


def can_start(deadline, cap, now=None):
    return deadline-(time.monotonic() if now is None else now) >= cap+120


def bounded_command(command, timeout, log):
    if timeout <= 0:
        raise TimeoutError('acquisition deadline exhausted')
    with Path(log).open('ab') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=max(.001, timeout-2))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=min(2, timeout))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def summarize_warm(rows, cell, output, before, after):
    external = after['vllm:external_prefix_cache_hits_total']-before['vllm:external_prefix_cache_hits_total']
    preemptions = after['vllm:num_preemptions_total']-before['vllm:num_preemptions_total']
    valid = len(rows) == cell['concurrency'] and external == preemptions == 0 and all(
        row['done'] and row['status'] == 200 and not row['error']
        and row['prompt_tokens'] == cell['context_tokens']
        and row['output_tokens'] == row['recorded_output_tokens'] == output
        and row.get('cached_tokens') is not None and row['cached_tokens'] >= cell['context_tokens']-32
        for row in rows)
    complete = [r for r in rows if r['done'] and r['status'] == 200]
    start = min((r['start_ns'] for r in rows), default=None)
    end = max((r['client_done_ns'] for r in complete), default=None)
    return {'cell': cell, 'warm_valid': bool(valid), 'external_cache_hits': external,
            'preemptions': preemptions, 'engine_before': before, 'engine_after': after,
            'midpoint_ns': (start+end)//2 if start is not None and end is not None else None,
            'median_duration_s': statistics.median((r['client_done_ns']-r['start_ns'])/1e9 for r in complete) if complete else None,
            'output_tokens_s': sum(r['output_tokens'] for r in complete)/((end-start)/1e9) if complete else None,
            'scheduled_prefill_validation': 'requires scheduler event join; no Q=32 fit is certified here',
            'requests': rows}


async def warm(acquisition, cell, output, root, prompt_label):
    root.mkdir(exist_ok=False)
    cfg = acquisition.cfg
    rows = []
    sessions = [serving.Session(f'{prompt_label}-lane{i}', cell['context_tokens']-32, 32,
                               output, 200000, cell['seed'], force_output=False)
                for i in range(cell['concurrency'])]
    prepared = [serving.prepare_issue(session, 0, cfg.model, True) for session in sessions]
    write(root/'prompts.json', [{'session_id': p['session'].session_id, 'prompt': p['prompt'],
                                'body': json.loads(p['body'])} for p in prepared])
    metrics = serving.MetricsSampler(cfg.host, cfg.sink_port, root/'engine.csv', .5)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True)) as client:
        async def metric():
            async with client.get(f'http://{cfg.host}:{cfg.sink_port}/metrics', timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                return serving.parse_metrics(await response.text())
        before = await metric()
        if before['vllm:num_requests_running'] or before['vllm:num_requests_waiting']:
            raise RuntimeError('warm cache reset requires an idle owned engine')
        async with client.post(f'http://{cfg.host}:{cfg.sink_port}/reset_prefix_cache', timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            write(root/'cache-reset.json', {'before': before, 'status': response.status, 'body': await response.text()})
        metrics.start()
        async def request(item, phase, deadline):
            dispatch = time.monotonic_ns()
            tags = {'cell': cell['id'], 'phase': phase, 'session_id': item['session'].session_id,
                    'timing_mode': cell.get('timing_mode', 'on'), 'telemetry_expected': cell.get('timing_mode', 'on') == 'on'}
            acquisition.record(acquisition.events, {**tags, 'kind': 'dispatch', 'monotonic_ns': dispatch})
            row = {**tags, 'start_ns': dispatch, 'status': 'censored', 'done': False,
                   'planned_prompt_tokens': len(item['prompt']), 'planned_output_tokens': item['session'].output_tokens,
                   'request_index': item['index'], 'prompt_sha256': item['prompt_sha256']}
            received = []
            def event_sink(event):
                received.append(event)
                acquisition.record(acquisition.events, {**tags, **event})
            try:
                result = await headroom.async_completion(client, cfg.host, cfg.sink_port, item, dispatch,
                    max(.001, deadline-time.monotonic()),
                    event_sink=event_sink)
                row.update(result)
                row['client_done_ns'] = next((e['monotonic_ns'] for e in received if e['data'] == '[DONE]'), None)
                return row
            finally:
                if row['status'] == 'censored':
                    usage, tokens, request_id = {}, [], ''
                    for event in received:
                        if event['data'] == '[DONE]':
                            continue
                        data = json.loads(event['data'])
                        usage = data.get('usage') or usage
                        request_id = data.get('id') or request_id
                        tokens += [{'monotonic_ns': event['monotonic_ns'], 'token_ids': c['token_ids']}
                                   for c in data.get('choices', []) if c.get('token_ids')]
                    row.update(serving.completion_row(0, dispatch, time.monotonic_ns(), usage, tokens, False, request_id))
                    row.update(status='censored', cancellation='whole_cell_deadline')
                row.setdefault('end_ns', time.monotonic_ns())
                acquisition.record(acquisition.requests, row)
                if phase == 'measurement':
                    rows.append(row)
                with (root/f'{phase}.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
        try:
            deadline = min(time.monotonic()+90, acquisition.deadline)
            # Same one-output prefix prewarm contract, using the async collector to retain raw SSE and failures.
            for item in prepared:
                prefix = item['prompt'][:-32]
                prefix_item = {**item, 'prompt': prefix,
                    'session': SimpleNamespace(session_id=item['session'].session_id, prefix_tokens=len(prefix), append_tokens=0, output_tokens=1),
                    'body': json.dumps(serving.completion_payload(cfg.model, prefix, 1, None, True)),
                    'prompt_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
                if time.monotonic() >= deadline:
                    raise TimeoutError('all-prewarming 90-second deadline')
                row = await request(prefix_item, 'prewarm', deadline)
                if not serving.service_completion(row) or row['prompt_tokens'] != len(prefix) or row['output_tokens'] != 1:
                    raise RuntimeError('prewarm failed; raw request retained')
            before = await metric()
            deadline = min(time.monotonic()+180, acquisition.deadline)
            await asyncio.gather(*(request(item, 'measurement', deadline) for item in prepared))
            result = summarize_warm(rows, cell, output, before, await metric())
            write(root/'result.json', result)
            return result
        finally:
            metrics.close()


def run_cancellable(coroutine):
    async def run():
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await coroutine
    return asyncio.run(run())


def worker(args, plan, inventory):
    acquisition = ResidentAcquisition(args.out, inventory)
    try:
        if args.episode is not None:
            entry = plan['agentic_contract']['episodes'][args.episode]
            reference = checked_json(plan['reference_plan'], plan['reference_plan_sha256'])
            spec = entry['spec']
            run_cancellable(acquisition.episode(spec, reference['workloads'][spec['workload']], 300, True))
            for name, key in (('physical-workload.json', 'physical_workload_sha256'), ('offered-trace.json', 'offered_trace_sha256')):
                checked_json(args.out/spec['episode']/name, entry[key])
        else:
            cell = json.loads(args.cell)
            run_cancellable(warm(acquisition, cell, args.output_tokens, args.out/cell['id'], args.prompt_label or cell['id']))
    finally:
        acquisition.requests.close()
        acquisition.events.close()


def acquire(args, plan, inventory):
    validate_inventory(inventory)
    launch = json.loads((args.out/'runtime-launch.json').read_text())
    start_wall = launch['start_wall_ns']
    expected_deadline = start_wall+5400*10**9
    if inventory['deadline_wall_ns'] != expected_deadline:
        raise ValueError('inventory deadline must equal fresh launch start plus 5400 seconds')
    deadline = time.monotonic()+(expected_deadline-time.time_ns())/1e9
    setup_deadline = deadline-5040
    commands = inventory['timing_mode_commands']
    cleanup = inventory['cleanup_command']
    if inventory['bandwidth_mbps'] != 1000:
        raise ValueError('frozen hardware proxy bandwidth is 1000 Mbit/s')
    checked_json(plan['reference_plan'], plan['reference_plan_sha256'])
    write(args.out/'client-clock.json', {'host': socket.gethostname(), 'pid': os.getpid(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'time_namespace': os.readlink('/proc/self/ns/time'),
        'time_namespace_offsets': Path('/proc/self/timens_offsets').read_text(),
        'clock': 'CLOCK_MONOTONIC', 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()})
    write(args.out/'plan.json', plan)
    write(args.out/'acquisition-inventory.json', inventory)
    base = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out), '--inventory', str(args.inventory), '--plan', str(args.plan), '--worker']
    states = [{'id': e['spec']['episode'], 'kind': 'episode', 'status': 'unstarted'} for e in plan['agentic_contract']['episodes']]
    states += [{'id': c['id'], 'kind': 'warm', 'status': 'unstarted'} for c in plan['warm_decode_contract']['cells']]
    report = {'data_acceptance': 'pending server event joins and telemetry completeness validation', 'status': 'running', 'cells': states, 'overhead': [], 'start_wall_ns': start_wall,
              'deadline_wall_ns': expected_deadline, 'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    def save():
        write(args.out/'acquisition-report.json', report)
    save()
    try:
        for attempt in range(2):
            bursts = []
            for index, mode in enumerate(('off', 'on', 'off')):
                remaining = setup_deadline-time.monotonic()
                if bounded_command(commands[mode], remaining, args.out/'mode-command.log'):
                    raise RuntimeError('timing mode command failed')
                cell = {'id': f'smoke-{attempt}-{index}-{mode}', 'context_tokens': 8192, 'concurrency': 8, 'seed': 8101, 'timing_mode': mode}
                code = bounded_command([*base, '--cell', json.dumps(cell), '--output-tokens', '256', '--prompt-label', 'matched-smoke'],
                                       setup_deadline-time.monotonic(), args.out/'launcher.log')
                if code:
                    raise RuntimeError(f'smoke worker failed with exit {code}')
                bursts.append(json.loads((args.out/cell['id']/'result.json').read_text()))
            check = overhead_check(bursts)
            report['overhead'].append(check)
            save()
            if check['status'] != 'inconclusive':
                break
        if check['status'] != 'passed':
            raise RuntimeError(f'instrumentation overhead gate {check["status"]}')
        if bounded_command(commands['on'], min(30, deadline-time.monotonic()-120), args.out/'mode-command.log'):
            raise RuntimeError('timing enable command failed')
        for index, state in enumerate(states):
            cap = 420 if state['kind'] == 'episode' else 270
            if not can_start(deadline, cap):
                state['reason'] = 'whole-cell plus cleanup reserve unavailable'
                save()
                continue
            state.update(status='running', start_wall_ns=time.time_ns())
            save()
            command = [*base, '--episode', str(index)] if state['kind'] == 'episode' else [*base, '--cell', json.dumps(plan['warm_decode_contract']['cells'][index-4])]
            state['command'] = command
            try:
                code = bounded_command(command, cap, args.out/'launcher.log')
                state.update(status='complete' if code == 0 else 'failed', exit_code=code)
                if state['kind'] == 'warm' and code == 0:
                    state['warm_valid'] = json.loads((args.out/state['id']/'result.json').read_text())['warm_valid']
            except subprocess.TimeoutExpired:
                state.update(status='censored', reason='whole-cell deadline; raw records retained',
                             telemetry_completeness='requires validation; forced termination can truncate sampler output')
            state['end_wall_ns'] = time.time_ns()
            save()
        report['status'] = 'complete' if all(s['status'] == 'complete' for s in states) else 'incomplete'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
        try:
            code = bounded_command(cleanup, min(120, deadline-time.monotonic()), args.out/'cleanup.log')
            report['cleanup_exit_code'] = code
            if code:
                raise RuntimeError('owned stack cleanup failed')
        finally:
            report['end_wall_ns'] = time.time_ns()
            save()
            write(args.out/'sha256-manifest.json', {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'sha256-manifest.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--episode', type=int)
    parser.add_argument('--cell')
    parser.add_argument('--output-tokens', type=int, default=1536)
    parser.add_argument('--prompt-label')
    args = parser.parse_args()
    plan, inventory = (json.loads(p.read_text()) for p in (args.plan, args.inventory))
    (worker if args.worker else acquire)(args, plan, inventory)


if __name__ == '__main__':
    main()
