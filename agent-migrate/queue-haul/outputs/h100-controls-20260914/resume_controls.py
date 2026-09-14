"""Resume unfinished controls after connectivity loss; preserve every attempt."""
import json
import os
import subprocess
import time
from pathlib import Path

STATE = Path('/datadrive/shared-resume-20260914.json')
SSH = ['ssh', '-i', '/home/azureuser/.ssh/azrs', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'azureuser@10.13.0.4', 'test -x /datadrive/qh0912/.venv/bin/python']
JOBS = [('gemma', 'google/gemma-4-26B-A4B-it', ['none', 'replay', 'kv', 'mixed'], .25, []),
        ('gpt', 'openai/gpt-oss-20b', ['mixed'], .25, []),
        ('qwen-baseline', 'Qwen/Qwen3.8-27B', ['none'], .05,
         ['--seconds', '240', '--warmup-s', '60', '--state-reference', '/datadrive/qf20/full_state_control.json'])]


def reachable():
    return subprocess.run(['timeout', '20', *SSH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def save(state):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {'attempts': [], 'status': 'running'}
    failed = False
    for slug, model, wanted, rate, extra in JOBS:
        while True:
            if any(a['job'] == slug and a.get('classification') == 'failure_requires_review' for a in state['attempts']):
                failed = True
                break
            completed = set()
            for attempt in state['attempts']:
                if attempt['job'] != slug: continue
                for arm in wanted:
                    path = Path(attempt['root']) / arm / 'complete.json'
                    if path.exists() and json.loads(path.read_text())['status'] == 'complete': completed.add(arm)
            pending = [arm for arm in wanted if arm not in completed]
            if not pending: break
            deadline = time.monotonic() + 21600
            while not reachable():
                state.update(status='waiting_for_southcentral', pending_job=slug)
                save(state)
                if time.monotonic() > deadline: raise TimeoutError('South Central unavailable for six hours')
                time.sleep(30)
            index = sum(a['job'] == slug for a in state['attempts'])
            root = Path(f'/datadrive/shared-resume-{slug}-20260914-a{index}')
            if root.exists(): raise FileExistsError(root)
            attempt = {'job': slug, 'root': str(root), 'arms': pending, 'started_wall_ns': time.time_ns()}
            state['attempts'].append(attempt); state.update(status='running', pending_job=slug); save(state)
            profile = 'qwen' if slug == 'qwen-baseline' else slug
            command = ['/datadrive/qh0912/.venv/bin/python', 'shared_load_controls.py', '--model', model,
                       '--cluster', '/datadrive/shared-controls-cluster.json', '--calibration', '/datadrive/c12.json',
                       '--resident-rps', str(rate), '--run-root', str(root), '--arms', *pending, *extra]
            print(json.dumps(attempt), flush=True)
            with root.with_suffix('.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'QH_MODEL_PROFILE': f'profiles/network-{profile}-h100.json'})
            attempt.update(returncode=result.returncode, ended_wall_ns=time.time_ns()); save(state)
            if result.returncode:
                gate_path = root / 'state_equivalence.json'
                invalid = gate_path.exists() and not json.loads(gate_path.read_text())['passed']
                invalid = invalid or any(json.loads(p.read_text())['status'] == 'invalid_measurement' for p in root.glob('*/complete.json'))
                if not invalid and not reachable():
                    attempt['classification'] = 'connectivity_interruption'; save(state)
                    continue
                attempt['classification'] = 'failure_requires_review'; save(state)
                failed = True
                break
            if any(not (root / arm / 'complete.json').exists() for arm in pending):
                raise RuntimeError('successful process omitted arm completion evidence')
    state['status'] = 'failed' if failed else 'request_arms_complete_pending_telemetry_reduction'; save(state)
    if failed: raise SystemExit(1)


if __name__ == '__main__':
    main()
