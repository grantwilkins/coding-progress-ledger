"""Freeze bounded replay validation inputs and record acquisition prerequisites.

This preparation adapter does not launch serving engines or measurement sweeps.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import migration_profiler as profiler
import pool_shed_campaign as pool

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'outputs/service-admission-transition-a100-20260816/plan.json'
SEEDS = (7101, 7102)
WORKLOADS = ('coding', 'coding_long')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {'argv': argv, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def trajectories(fleet, raw, support):
    ids = {row['session_id'] for row in fleet.metadata['sampled_states']}
    coding_ids = set(sum(raw['manifest']['splits']['coding'].values(), []))
    excluded = []
    for session in sorted(coding_ids):
        rows = [r for r in raw['traces'] if r['session_id'] == session]
        over = [r['turn'] for r in rows if r['input_tokens_total'] + r['output_tokens'] > support]
        if over:
            excluded.append({'session_id': session, 'turns': over,
                             'reason': 'trajectory exceeds measured context support'})
    return {'sampled_states': fleet.metadata['sampled_states'],
            'cohort_counts': fleet.count.tolist(), 'turn_offset': fleet.metadata['turn_offset'],
            'turn_sequences': fleet.metadata['turn_sequences'],
            'recorded_rows': sorted([r for r in raw['traces'] if r['session_id'] in ids],
                                    key=lambda r: (r['session_id'], r['turn'])),
            'excluded_trajectories': excluded,
            'excluded_initial_states': fleet.metadata['excluded_states'],
            'initial_state_exclusion_rule': fleet.metadata['exclusion_reason'],
            'unselected_supported_trajectories': sorted(coding_ids - ids - {r['session_id'] for r in excluded}),
            'initial_scout_rps_per_gpu': .5 * fleet.metadata['reference_rps']}


def make_plan(calibration, fleets, raw):
    reference = json.loads(REFERENCE.read_text())
    trials, episodes = [], []
    for repeat, seed in enumerate(SEEDS):
        conditions = list(product((2048, 8192, 30000), (32, 2048), (1, 8)))
        random.Random(seed).shuffle(conditions)
        for context, append, width in conditions:
            trials.append({'seed': seed, 'retained_tokens': context, 'append_tokens': append,
                           'width': width, 'order': ['initial', 'catch_up', 'cold_updated']
                           if repeat == 0 else ['cold_updated', 'initial', 'catch_up']})
        for workload, rate_slot in product(WORKLOADS, (0, 1)):
            arms = ['control', 'replay', 'kv_transfer']
            if repeat:
                arms.reverse()
            for arm in arms:
                episodes.append({'seed': seed, 'workload': workload, 'rate_slot': rate_slot,
                                 'arm': arm, 'width': 8,
                                 'trace_id': f'{seed}-{workload}-{rate_slot}',
                                 'requires_separate_source': arm != 'control'})
    return {
        'schema': 'queue-haul-bounded-replay-validation-plan-v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'reference_plan': str(REFERENCE.relative_to(ROOT)),
        'reference_plan_sha256': profiler.file_hash(REFERENCE),
        'model': reference['model'], 'stack': reference['stack'],
        'hardware': 'NVIDIA A100 80GB', 'runtime_mode': 'native',
        'seeds': SEEDS, 'fit_seed': 7101, 'validation_seed': 7102,
        'budget': {'target_measurement_s': 7200, 'initial_acquisition_hard_cap_s': 9000,
                   'scope': 'wall time from first runtime startup, all attempts and cleanup count; GPU seconds reported separately',
                   'unloaded_cap_s': 2160, 'scout_cap_s': 720, 'main_cap_s': 4320,
                   'followup_cap_s': 720, 'startup_and_cleanup_reserve_s': 1080,
                   'unloaded_trial_cap_s': 90, 'performance_retries': 0,
                   'timeout_status': 'failed or censored, never passing'},
        'unloaded_trials': trials,
        'unloaded_contract': {'max_generation_tokens': profiler.PROBE_MAX_TOKENS,
            'generation': 'unchanged LiveSession.probe and chat_payload; temperature=0, reasoning_effort=low, existing EOS behavior',
            'contexts': 'render exact full chat plus probe; require prompt+512<=32768; no truncation',
            'cache': 'independent history per trial/lane; retain initial history for catch-up; separate cold-updated history; never flush between initial and catch-up',
            'verification': 'known cold then shared-prefix chat requests on actual migration path; absent telemetry unknown; native and external hits distinguished using engine evidence'},
        'scout': {'max_probes_per_workload': 4, 'warmup_s': 30, 'observation_s': 60,
            'rate_rule': 'start at exact sampled fleet 0.5*reference_rps; double after a pass, halve after a failure; at most four tested rates',
            'selection': 'two distinct explicitly tested rates per workload, prefer two stable rates including the heaviest stable tested rate; if unavailable report no stable pair and do not claim a boundary',
            'rates_selected': None, 'boundary_bracketed': None},
        'main_episodes': episodes,
        'episode_contract': {'baseline_s': 60, 'migration_start_s': 60, 'deadline_report_s': 90,
            'arrivals_end_s': 180, 'observation_end_s': 180,
            'control': 'resident and incoming state materialized before observation; incoming demand served at destination at the matched actual replay handoff time; record service starts and placements',
            'source': 'eight active sessions on a separate GPU continue through initial copy; pause at actual request boundary, await inflight completion, catch up, commit ownership, retain queued arrivals',
            'single_gpu': 'A/B and destination contention supported; source quiescence, paired KV, and equivalent source-active migration episodes remain unvalidated',
            'recovery': 'compare against matched control with arrivals continuing; retain unfinished work at t=180; cleanup drain is not recovery'},
        'followups': [{'kind': 'width16_replay', 'workload': 'coding_long', 'rate_slot': 1, 'seed': s} for s in SEEDS]
                     + [{'kind': 'bursty', 'workload': 'coding_long', 'rate_slot': 1, 'seed': 7102, 'arm': a}
                        for a in ('control', 'replay')],
        'arrival_contract': {'status': 'assumed; manifest has no timestamps',
            'baseline': 'equal cadence per session, seeded independent initial phases, weighted by frozen cohort counts; pair traces across arms',
            'bursty': 'release each 5-second block of the matched baseline schedule at its left boundary; same offered request count',
            'causality': 'enqueue all scheduled turns independently of server slowdown; dispatch only after previous turn completes; retain send lateness and dependency queue separately',
            'evolution': 'preserve ordered recorded turns, context growth, append/output lengths and resets; coding_long selects initial state only; reset on recorded cycle wrap',
            'content': 'manifest contains lengths and hashes, not messages; retained-history rendering needs verification against these shapes before acquisition'},
        'workloads': {w: trajectories(fleets[w], raw, max(calibration['replay_context_tokens'])) for w in WORKLOADS},
        'telemetry': {'sample_interval_s': 1, 'min_exact_token_timing_coverage': .99,
            'request': 'episode/arm/seed/session/turn/cohort/method/phase, scheduled/dispatch/first/last/completion times, full prompt and token IDs/events, context/append/output/cache counts, status/error/timeout/cancellation/unfinished',
            'migration': 'snapshot/copy/pause/idle/catch-up/commit/first-valid-response events, ownership/queues, context hashes/resets, actual payload/wire bytes and retrieval evidence',
            'engine': 'raw metrics for running/waiting, KV occupancy, preemption/recomputation and throughput; raw GPU power/utilization',
            'interpretation': 'client token events are not server execution; no queue-wait inference from TTFT; no cross-host monotonic subtraction; prompt-minus-cache is derived, not executed work'},
        'analysis': {'targets': reference['targets'],
            'tails': 'P90 across per-request mean TPOT, separate resident/migrated populations; include original-arrival migration waiting, coverage and sample counts; unfinished/short windows cannot certify SLOs',
            'corrections': 'fit only seed 7101, freeze before seed 7102 analysis; keep other coefficients fixed; report per-condition/optimistic/queue/recovery errors and false-feasible deadlines, retain counterexamples'},
        'simulator_verification': {'prerequisite': 'explicitly validated resident operating point; otherwise blocked',
            'source_gpus': 6666, 'gpus_per_destination': [6666, 6666], 'gpus_per_node': 8,
            'workloads': WORKLOADS, 'deadlines_s': [30, 120], 'policies': pool.POLICIES,
            'evaluations': 20, 'shared_wan_gbps': 1000, 'execution_draw': 0,
            'comparison': 'common candidates, traffic and planning clock per cell',
            'kv_wire': '0.80 decimal GB per 32768 tokens effective wire, no additional private discount; native serialized geometry separate, resident memory unchanged'},
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    calibration = pool.calibration(0)
    fleets = {w: pool.sample_fleet(w, snapshot=0, gpus=6666, gpus_per_node=8) for w in WORKLOADS}
    plan = make_plan(calibration, fleets, json.loads(pool.MANIFEST.read_text()))
    names = ('migration.py', 'migration_profiler.py', 'migration_testbed.py', 'destination_runner.py',
             'service_headroom_campaign.py', 'service_admission_transition_campaign.py',
             'pool_shed_replay_audit.py', Path(__file__).name)
    evidence = [REFERENCE, ROOT / 'outputs/a100-replay-realism/audit.json']
    for directory in ('policy-hardware-width8-packing-20260730', 'service-headroom-a100-20260815',
                      'service-admission-transition-a100-20260816', 'single-gpu-capacity-a100-20260815'):
        evidence += [p for name in ('plan.json', 'summary.json', 'run_metadata.json', 'confirmed.json', 'normalization.json')
                     if (p := ROOT / 'outputs' / directory / name).is_file()]
    plan['input_hashes'] = {**pool.provenance(calibration),
                          **{str(p.relative_to(ROOT)): profiler.file_hash(p) for p in [*(ROOT / n for n in names), *evidence]}}
    plan['checkout'] = {'commit': command(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
                        'dirty_state': command(['git', 'status', '--porcelain'])['stdout'],
                        'baseline_commit': 'e68f1d26'}
    write(out / 'plan.json', plan)
    (out / 'plan.sha256').write_text(profiler.file_hash(out / 'plan.json') + '\n')
    (out / 'source.diff').write_text(command(['git', 'diff', 'HEAD'])['stdout'])
    write(out / 'prepare-command.json', {'argv': [sys.executable, *sys.argv], 'cwd': str(Path.cwd()),
                                       'shell_command': shlex.join([sys.executable, *sys.argv])})
    print(out / 'plan.json')


def preflight(out):
    plan = json.loads((out / 'plan.json').read_text())
    if profiler.file_hash(out / 'plan.json') != (out / 'plan.sha256').read_text().strip():
        raise RuntimeError('frozen plan hash changed')
    changed = [name for name, sha in plan['input_hashes'].items() if profiler.file_hash(ROOT / name) != sha]
    versions = {name: importlib.metadata.version(name) for name in ('vllm', 'lmcache')}
    gpu = command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total', '--format=csv'])
    expected = plan['stack']['runtime_versions']['native']
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'executable': sys.executable, 'platform': platform.platform(), 'gpu_query': gpu,
              'versions': versions, 'expected_native_versions': expected, 'changed_inputs': changed,
              'runtime_versions_match': list(versions.values()) == expected,
              'hardware_measurements': 0, 'campaign_ready': False,
              'acquisition_status': 'not_started',
              'remaining_prerequisites': ['trusted accessible A100 endpoint', 'verified reference runtime and scheduler',
                  'retained-history rendering and causal arrival adapter', 'known cold/shared-prefix telemetry check',
                  'measurement execution adapter with global acquisition deadline'],
              'launch_command': [sys.executable, *sys.argv]}
    with (out / 'preflight.jsonl').open('a') as handle:
        handle.write(json.dumps(report) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1)  # Preparation alone never establishes acquisition readiness.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'preflight': preflight}[args.command](args.out)
