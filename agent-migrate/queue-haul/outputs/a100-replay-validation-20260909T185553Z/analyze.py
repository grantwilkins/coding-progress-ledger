"""Reproduce the blocked-acquisition report from frozen inputs; no policy runs."""
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import pool_shed_campaign as q
from migration_profiler import file_hash
from pool_shed_execution import catchup, initial_work

plan = json.loads((OUT / 'plan.json').read_text())
assert file_hash(OUT / 'plan.json') == (OUT / 'plan.sha256').read_text().strip()
assert all(file_hash(ROOT / name) == sha for name, sha in plan['input_hashes'].items())
audit = json.loads((ROOT / 'outputs/a100-replay-realism/audit.json').read_text())
c = q.calibration(0)
timing = c['timing'][0]
base = q.sample_fleet('coding', gpus=6666)


def write_csv(name, rows):
    with (OUT / name).open('w') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


predictions = []
for trial in plan['unloaded_trials']:
    context, append, width = [trial[k] for k in ('retained_tokens', 'append_tokens', 'width')]
    origin = np.full(len(base.context), float(context))
    fleet = replace(base, context=origin, t1=q.replay_seconds(origin, c), log=2*origin)
    counts = np.zeros(len(origin)); counts[0] = width
    initial = initial_work(fleet, counts, 0, 0, origin, timing, c)[1]
    warm = catchup(fleet, counts, 0, 0, origin + append, np.zeros(len(origin), bool),
                   timing, c, origin_context=origin)[1]
    predictions.append({k: v for k, v in trial.items() if k != 'order'} | {
        'initial_predicted_idle_work_s': initial, 'catchup_predicted_idle_work_s': warm,
        'cold_updated_predicted_idle_work_s': warm, 'hardware_requests': 0,
        'observed_s': None, 'rendered_prompt_tokens': None, 'cache_state': 'unmeasured',
        'scope': 'nominal-context East-route simulator primitive; not rendered chat elapsed time or a new fit'})
write_csv('nominal-replay-predictions.csv', predictions)

historical = []

def collect(value, path=''):
    if isinstance(value, dict):
        for row in value.get('predictions', []):
            if 'observed_s' in row:
                observed, predicted = row['observed_s'], row['predicted_s']
                historical.append({'condition': path, 'scenario_id': row['scenario_id'],
                    'policy': row['policy'] if 'policy' in row else row['method'], 'observed_s': observed, 'predicted_s': predicted,
                    'error_s': predicted-observed, 'optimistic_error_s': max(0, observed-predicted),
                    'relative_absolute_error': abs(predicted-observed)/observed,
                    'false_feasible_25s': predicted <= 25 < observed})
        for k, v in value.items():
            if isinstance(v, dict):
                collect(v, f'{path}/{k}')
collect(audit['timing_checks'])
write_csv('archived-per-condition-errors.csv', historical)
summary = []
for condition in sorted({r['condition'] for r in historical}):
    rows = [r for r in historical if r['condition'] == condition]
    summary.append({'condition': condition, 'episodes': len(rows),
        'observed_mean_s': np.mean([r['observed_s'] for r in rows]),
        'predicted_mean_s': np.mean([r['predicted_s'] for r in rows]),
        'mae_s': np.mean([abs(r['error_s']) for r in rows]),
        'p90_relative_error': np.quantile([r['relative_absolute_error'] for r in rows], .9),
        'max_optimistic_error_s': max(r['optimistic_error_s'] for r in rows),
        'false_feasible_25s': sum(r['false_feasible_25s'] for r in rows),
        'scope': 'previously inspected archived checks; overlapping conditions are not independent samples'})
write_csv('archived-observations-vs-predictions.csv', summary)
write_csv('archived-resident-deficits.csv', audit['resident_execution_check']['predictions'])

report = {
    'status': 'blocked_before_hardware_acquisition', 'campaign_ready': False,
    'plan_sha256': file_hash(OUT / 'plan.json'), 'checkout': plan['checkout'],
    'hardware_measurement_seconds': 0, 'hardware_requests': 0, 'new_exact_timing_coverage': None,
    'new_token_events': 0, 'new_migration_events': 0, 'new_engine_or_power_samples': 0,
    'runtime': json.loads((OUT / 'preflight.jsonl').read_text().splitlines()[-1]),
    'current_scout_start_rps_per_gpu': {w: v['initial_scout_rps_per_gpu'] for w, v in plan['workloads'].items()},
    'resident_rates_selected': None, 'boundary_bracketed': None,
    'simulator_changes': [], 'new_timing_fit': None, 'seed_7102_validation': 'not acquired',
    'bounded_policy_evaluations': {'completed': 0, 'planned': 20,
        'reason': 'no explicitly validated resident operating point; do not substitute historical 50% normalization'},
    'questions': {
        'resident_traffic': {'status': 'open', 'answer': 'Normalizer mismatch is verified, but whether coding traffic is too light or its service normalization is wrong requires the agentic scout. Labels alone do not establish GPU utilization or capacity.',
                             'archived_mismatch': audit['load_mismatch']},
        'initial_scaling': {'status': 'partly_supported_by_old_measurements', 'answer': 'Existing singleton/width-eight checks support their fixed workloads only; the new 2K/8K/30K, width1/8 matrix has no observations.',
                            'archived_width8': audit['measurement_review']['width8_replay'],
                            'long_cold_burst': audit['measurement_review']['cold_burst_32k']},
        'warm_catchup': {'status': 'open', 'answer': 'No current-stack same-history initial/catch-up/cold triplets acquired. Full-rebuild cost remains an unvalidated simulator assumption; no cache percentage or correction fitted.'},
        'delay_decomposition': {'status': 'open', 'answer': 'No new queue/compute/source-idle/transfer/decode/recovery decomposition. Client stream timestamps cannot identify GPU queue wait, and cross-host monotonic clocks are not aligned.'},
        'divisible_width8_work': {'status': 'open', 'answer': 'Reproducing fixed width-eight elapsed time does not establish divisibility under continuing resident service. Width16 long-context followups remain missing.'},
        'service_degradation_and_recovery': {'status': 'open', 'answer': 'The prior six-route/five-episode deficit check has MAE 0.84674 requests, but measures debt at handoff, not recovery under continuing arrivals or per-request latency. No new resident/migrated TTFT/TPOT screen is available.',
            'archived_resident_deficit_mae_requests': audit['resident_execution_check']['mae_requests']},
        'kv_wire_anchor': {'status': 'open', 'effective_wire_bytes_per_32768_tokens': 800000000,
            'native_serialized_bytes_per_32768_tokens': int(32768/c['kv_block_tokens']*c['kv_block_bytes']),
            'resident_kv_capacity_tokens_per_gpu': c['kv_capacity_tokens'],
            'answer': 'No paired KV path acquired. Effective decimal wire measurement, serialized geometry, and resident memory remain separate; no private discount was applied.'}},
    'original_gap_status': {
        'reasoning_first_token': 'software corrected and regression tested; hardware coverage open',
        'missing_cache_telemetry': 'software preserves unknown; cold/shared-prefix hardware verification open',
        'catchup_csv_work': 'software retains derived catch-up processed tokens; executed/recomputed work open',
        'evolving_agentic_workload': 'exact sample_fleet trajectories and exclusions frozen; rendering/causal execution adapter remains required',
        'source_quiescence_and_kv': 'open; separate trusted GPU required',
        'shared_service_and_recovery': 'open; matched episodes not acquired',
        'arrival_distribution': 'assumed; original timestamps unavailable',
        'service_normalization': 'open; four-probe scout per workload not acquired',
        'divisible_compute_and_packing': 'open; loaded width8/16 validation missing',
        'effective_wire': 'open; path wire bytes not acquired'},
    'known_software_failure': {'test': 'test_queue_drift_discards_an_incomplete_trailing_block',
        'reproduced_on_unmodified_commit': '35be3279', 'observed': .7549019607843137,
        'expected': 0, 'action': 'retained; no queue data suppressed to force a pass'},
    'missing_measurements': ['24 current-stack unloaded replay/warm-catchup/cold triplets',
        'up to eight evolving-agentic resident scout probes with explicit rates',
        'matched control/replay/KV continuing-service episodes with equivalent populations and placement logs',
        'width16 long-context and same-mean burst sensitivities',
        'actual source quiescence, paired KV wire bytes and external/native retrieval separation'],
    'recommendation': 'Do not launch full simulations. Restore a trusted A100 endpoint and the exact vLLM0.22.0/LMCache0.5.1 stack; finish and verify the retained-history/causal measurement adapter, run the frozen acquisition, then fit only supported corrections and validate seed7102 before the twenty policy evaluations.'}
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': report['status'], 'historical_prediction_rows': len(historical),
                  'nominal_prediction_rows': len(predictions), 'new_hardware_requests': 0}))
