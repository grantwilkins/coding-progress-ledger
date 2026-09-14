"""Finish repeat zero, retaining failed episodes and requiring fresh GPT state checks."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/datadrive/qh0912/queue-haul')
RUN = Path('/datadrive/d19')
STATE = Path('/datadrive/d19-resume-20260914.json')
PYTHON = '/datadrive/qh0912/.venv/bin/python'
CLUSTER = ROOT / 'azure_network_cluster_southeastasia_southcentral.json'
ENV = {**os.environ, 'PYTHONPATH': str(ROOT), 'QH_RUNTIME': 'native',
       'QH_LMCACHE_MODE': 'mp', 'QH_NATIVE_RUNTIME_VERSIONS': '0.24.0,0.5.1',
       'HF_HOME': '/datadrive', 'QH_CACHE_ROOT': '/datadrive/queue-haul-cache',
       'PATH': '/home/azureuser/.local/bin:/datadrive/qh0912/.venv/bin:' + os.environ['PATH']}
ENV.pop('QH_FULL_KV_CONTROL', None)


def save(state):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def main():
    import model_hardware_drain_campaign as campaign
    import network_campaign as network
    original = json.loads(Path('/datadrive/c12.json').read_text())
    current = json.loads(Path('/datadrive/campaign-route-recheck-20260914.json').read_text())
    network.validate_resume(network.freeze_contract(original), network.freeze_contract(current))
    for index in range(3):
        profile = RUN / 'arms' / f'm{index}' / 'profile.json'
        campaign._gated_profile(profile, 'H100')
        gate = json.loads(profile.with_suffix('.gate.json').read_text())
        if gate['calibration_sha256'] != network.profiler.object_hash(original):
            raise ValueError('original profile calibration binding changed')
    state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}, 'repeats': 1, 'original_episodes_remaining': 80}
    if state.get('status') == 'failure_requires_review':
        raise RuntimeError('previous queue failure requires review; retained evidence will not be overwritten')
    jobs = []
    for slug, index in [('gemma', 1), ('gpt', 2)]:
        if slug == 'gpt':
            jobs.extend((name, slug, [PYTHON, f'/datadrive/{script}.py']) for name, script in (
                ('gpt_sea_full_reference', 'gpt_full_state_sea'),
                ('gpt_sea_compact_reference_check', 'gpt_compact_state_sea')))
        jobs.append((f'{slug}_repeat_zero', slug, [PYTHON, 'network_campaign.py', 'run',
            '--cluster', str(CLUSTER), '--current-calibration', '/datadrive/c12.json',
            '--plan', str(RUN / 'plans' / f'm{index}.json'),
            '--run-root', str(RUN / 'arms' / f'm{index}'), '--stack-block', '0']))
    jobs.extend((f'reduce_{slug}', slug, [PYTHON, 'network_campaign.py', 'reduce',
        '--plan', str(RUN / 'plans' / f'm{index}.json'),
        '--run-root', str(RUN / 'arms' / f'm{index}')])
        for index, slug in enumerate(('qwen', 'gemma', 'gpt')))
    jobs.append(('reduce_comparison', 'gpt', [PYTHON, 'model_hardware_drain_campaign.py',
        'reduce', '--run-root', str(RUN), '--out', str(RUN), '--repeats', '1']))
    for name, slug, command in jobs:
        if state['jobs'].get(name, {}).get('returncode') == 0:
            continue
        state.update(status='running', current_job=name)
        state['jobs'][name] = {'started_wall_ns': time.time_ns(), 'command': command}
        save(state)
        with (RUN / f'{name}-20260914.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**ENV, 'QH_MODEL_PROFILE': f'profiles/network-{slug}-h100.json'})
        state['jobs'][name].update(returncode=result.returncode, ended_wall_ns=time.time_ns())
        if result.returncode:
            state['status'] = 'failure_requires_review'; save(state)
            raise subprocess.CalledProcessError(result.returncode, command)
        save(state)
    state['status'] = 'selected_episodes_resolved_and_reduced'; save(state)


if __name__ == '__main__':
    main()
