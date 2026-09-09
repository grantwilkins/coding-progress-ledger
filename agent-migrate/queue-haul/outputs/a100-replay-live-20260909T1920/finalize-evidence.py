"""Assemble this run's report from retained reductions; no hardware requests."""
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from pool_replay_report import csv_rows, raw_bytes, requests

out = Path(__file__).resolve().parent
read = lambda name: json.loads((out/name).read_text())
write = lambda name, value: (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
raw = requests(out)+[json.loads(line) for line in raw_bytes(out, 'requests-recovered.jsonl').splitlines()]
unloaded = read('unloaded-analysis.json')
service = read('service-analysis.json')
policy = read('policy-verification.json')
stop = read('acquisition-stop.json')
assert stop['elapsed_from_original_start_s'] <= 9000
assert policy['status'] == 'complete' and policy['policy_evaluations'] == 20

main = [r for r in service['episodes'] if r['spec']['episode'].startswith('episodes-')]
followups = [r for r in service['episodes'] if r['spec']['episode'].startswith('followups-')]
scouts = [r for r in service['episodes'] if r['spec']['episode'].startswith('scout-')]
validation = [r for r in unloaded['rows'] if r['seed']==7102 and r['phase_valid']]
main_windows = [r for r in service['windows'] if r['episode'].startswith('episodes-')]
critical = [r for r in main_windows if r['arm']=='replay' and r['cohort']=='resident' and (r['window_start_s'],r['window_end_s'])==(60,90)]
known_critical = [r for r in critical if r['p90_arrival_ttft_s'] is not None]
counts = lambda rows: {'requests':len(rows), 'statuses':dict(Counter(str(r.get('status')) for r in rows)),
    'exact_token_timing_requests':sum(bool(r.get('exact_token_timestamps')) for r in rows)}

policy_rows = []
for cell in policy['cells']:
    for name, result in cell['results'].items():
        policy_rows.append({'workload':cell['workload'], 'deadline_s':cell['deadline_s'], 'policy':name,
            **{k:result[k] for k in ('shed_fraction','pending_buffered_requests','pending_buffered_work_s',
                'service_recovered_by_deadline','service_ready_s','resident_latency_validated')},
            'pending_resident_debt_work_s':sum(result['pending_resident_debt_work_s']),
            'handoff_is_not_latency_feasibility':True})
csv_rows(out/'policy-comparison.csv', policy_rows)
hardware_rows = [{k:r[k] for k in ('seed','context','append','width','phase','phase_valid','requests',
    'cache_state','elapsed_s','model_full_rebuild_local_s','model_minus_observed_s','optimistic_error_s')}
    for r in unloaded['rows']]
csv_rows(out/'hardware-summary.csv', hardware_rows)

scaling=[]
for context in (2048,8192,30000):
    rows=[next(r for r in unloaded['rows'] if r['seed']==7102 and r['context']==context
               and r['append']==32 and r['width']==width and r['phase']=='initial') for width in (1,8)]
    scaling.append({'context':context,'width1_s':rows[0]['elapsed_s'],'width8_s':rows[1]['elapsed_s'],
        'elapsed_ratio':rows[1]['elapsed_s']/rows[0]['elapsed_s'],'both_phases_valid':all(r['phase_valid'] for r in rows)})

report = {
    'campaign_ready':False,
    'recommendation':'Do not launch full simulations as validated SLO-feasible shedding results. The bounded diagnostic comparison is complete; remaining measurements are listed by modeling gap.',
    'acquisition':stop,
    'counts':{'unloaded_trials_attempted':len({r['episode'] for r in unloaded['rows']}),
        'unloaded_planned_request_slots':324, 'unloaded_valid_phases':sum(r['phase_valid'] for r in unloaded['rows']),
        'unloaded_state_valid_responses':sum(r['state_valid'] for r in unloaded['rows']),
        'unloaded_fully_valid_triplets':sum(all(r['phase_valid'] for r in unloaded['rows'] if r['episode']==episode) for episode in {r['episode'] for r in unloaded['rows']}),
        'unloaded_phases':len(unloaded['rows']),
        'unloaded_requests':counts([r for r in raw if 'retained_tokens' in r]),
        'scout_attempts':len(list(out.glob('scout-*/offered-trace.json'))),
        'scouts_with_complete_observation':len(scouts), 'main_destination_only_episodes':len(main),
        'targeted_followup_episodes':len(followups), 'paired_kv_episodes':0,
        'all_retained_request_records':counts(raw)},
    'budget_omissions':{p.name:json.loads(p.read_text()) for p in out.glob('*budget-omissions.json')},
    'sampling':{'engine_samples':sum(r['engine_samples'] for r in service['episodes']),
        'power_samples':sum(r['power_samples'] for r in service['episodes']),
        'max_engine_sample_gap_s':max(r['max_engine_sample_gap_s'] for r in service['episodes']),
        'max_power_sample_gap_s':max(r['max_power_sample_gap_s'] for r in service['episodes']),
        'scope':'Completed scout/main/followup observation windows; unloaded traces and interrupted attempts retained separately.'},
    'targeted_sensitivities':[{'spec':e['spec'],'admitted_by_90s':e['admitted_by_90s'],
        'admitted_by_boundary':e['admitted_by_boundary'],'admission_times_s':e['admission_times_s'],
        'windows':[r for r in service['windows'] if r['episode']==e['spec']['episode']
            and (r['window_start_s'],r['window_end_s']) in ((60,90),(150,180),(60,180))]}
        for e in followups],
    'migration_censoring':'Actual HTTP completions after t=180 remain in raw records and are marked censored_at_boundary in migration-observations.csv. They are cleanup, not admission or recovery evidence.',
    'runtime':{'model':'openai/gpt-oss-20b','vllm':'0.22.0','lmcache':'0.5.1',
        'gpu_identity_files':['gpu-identity.csv','recovered-gpu-identity.csv'],
        'reference_plan':'outputs/service-admission-transition-a100-20260816/plan.json',
        'settings':'TP1; len32768; seq256; batch8192; memory0.75; KVauto; block16; chunked prefill, native prefix caching and eager execution enabled; hybrid KV manager disabled',
        'differences':['Node reboot replaced the physical A100 UUID; repeats within A and within main episodes stay on their respective GPU.',
            'Isolated runtime FIFO patch preserves original engine token events; performance equivalence to unpatched vLLM is not assumed.',
            'Unloaded probes used 10 lowercase hex characters rather than the reference 12 uppercase; full messages and max512 generation retained. Main episodes use the reference format.'],
        'evidence':['runtime-launch.json','recovered-mp-launch.json','python-environment.json',
            'recovered-python-environment.json','model-input-hashes.json','stream-patch.json','probe-format-audit.json','node-recovery.json']},
    'answers':{
        'resident_traffic_and_normalization':'The tested finite traces are light enough to pass completed-request latency screens at declared nominal RPS outside migration. Neither scout brackets a capacity boundary. This does not establish a wrong normalization coefficient: 24 representative resident histories per GPU differ from modeled inventory and the short windows have small, varying output mixes. GPU utilization is not the request-rate normalization.',
        'initial_replay_scaling':{'width_scaling_second_repeat':scaling,
            'conclusion':'Elapsed width-eight replay is context-dependent and substantially sublinear versus eight singleton requests at short contexts. A scalar divisible-work interpretation does not establish queue delay or latency feasibility.'},
        'warm_full_message_catchup':'Naturally retained native prefixes sharply reduce recomputed prefill work. At C=30000,width8,seed7102: append32 takes about1.4s versus32.4s cold-updated; append2048 takes about5.0s versus35.9s. Engine counters independently show warm executed prefill work. Full rebuild is not a validated warm timing model; an assumed fleet cache-hit percentage would not repair it.',
        'delay_decomposition':'Client schedule lateness, dependency/admission queueing, dispatch, token events and completion are separate. Engine queue/prefill/decode histogram deltas are direct aggregate observations, not per-request attribution and not additive GPU work. Source quiescence, transfer and ownership switch are unmeasured. Incoming arrivals wait for destination admission from t=60; that waiting is not a measured outage of an active source.',
        'divisible_gpu_work':'The unloaded context/width surface and width16 sensitivities expose packing and queue effects. They do not validate fluid allocation across a 6666-GPU site, physical cache placement, or fractional sharing under concurrent migration.',
        'service_degradation':{'resident_replay_60_90_ttft_p90_range_s':[min(r['p90_arrival_ttft_s'] for r in known_critical),max(r['p90_arrival_ttft_s'] for r in known_critical)],
            'windows_without_resident_ttft_sample':len(critical)-len(known_critical),
            'evidence':'service-paired.csv reports every 30-second window through t=180, original-arrival latency, completion deficit, outstanding work, timing coverage and equivalent admitted population. An empty queue at t=90 can coexist with severe latency violations during t=60..90. Failed admission leaves the populations unequal and recovery unresolved.',
            'scope':'P90 is over request TTFT and per-request mean TPOT. Short windows, zero-arrival windows and censoring preclude tail guarantees; cleanup drain is excluded.'},
        'wire_anchor':'No actual paired KV payload or wire bytes were measured. The 800000000-byte effective-wire anchor remains an explicit decimal assumption, separate from 1610612736-byte native serialized geometry at32768 tokens and actual resident KV allocation. No private-KV discount is applied twice.'},
    'heldout':{'fit_seed':7101,'validation_seed':7102,'fitted_coefficients':[],
        'valid_second_repeat_phases':len(validation),
        'median_absolute_baseline_error_s':float(np.median([abs(r['model_minus_observed_s']) for r in validation])),
        'max_optimistic_baseline_error_s':max(r['optimistic_error_s'] for r in validation),
        'baseline_false_30s_completion_predictions':sum(r['model_predicts_30s_but_observed_exceeds'] for r in validation),
        'scope':'Per-condition signed errors and repeatability are in unloaded-heldout.csv. Invalid phases remain in the tables and are excluded from fitted-error conclusions. Existing holdouts use unchanged coefficients.',
        'queue_recovery_errors':'The fluid model has no validated request-level latency mapping or matched single-GPU episode predictor. Numerical GPU queue/recovery errors against this hardware cannot be identified; service-paired.csv retains measured control/replay counterexamples and policy-comparison.csv keeps fluid debt separate.',
        'false_feasible_deadlines':'No policy result is marked latency-feasible. Any interpretation of handoff alone as a service deadline is contradicted by the observed resident TTFT violations.'},
    'changes':[
        'Preserve reasoning token IDs/events on the actual migration request; leave absent cache counts unknown and retain derived catch-up work in CSV.',
        'Use retained, evolving recorded trajectory shapes, causal offered arrivals, cancellation records, matched incoming demand and independent session cache salts.',
        'Separate actual client wakeup lateness from dependency/admission queueing; keep request-level TPOT and observation-boundary outstanding work.',
        'Require a valid state-code response for destination admission; retain malformed/empty final responses as failures.',
        'Correct preflight to inspect actual static prerequisites instead of always failing. Correct the stale test that expected late queue growth to be discarded; queue implementation unchanged.',
        'No simulator timing, cache, fleet-size, normalization or policy resource coefficient changed.'],
    'gaps':[
        {'gap':'Full-context initial replay and packing','status':'verified within measured contexts,width1/8 and retained failed attempts','missing':'Reference-format repeat on the final FIFO runtime before fitting request/decode overhead.'},
        {'gap':'Naturally warm full-message catch-up','status':'verified locally; full-rebuild assumption contradicted','missing':'Measure reuse/eviction with physical per-GPU inventory and source-evolved context before changing fleet catch-up timing.'},
        {'gap':'Resident rate normalization','status':'open; explicit tested points only, no boundary bracket','missing':'Continue geometric scouts with exact timing and enough causal turns/output mix to bracket service degradation; match modeled resident inventory.'},
        {'gap':'Shared resident interference and continuing-load recovery','status':'measured destination-only; latency counterexamples retained','missing':'Longer paired windows for reliable tails and full admitted populations where state validation failed.'},
        {'gap':'Source quiescence and ownership buffering','status':'open; one GPU','missing':'A separate source GPU serving the same eight causal sessions through initial migration, boundary pause and real catch-up.'},
        {'gap':'KV transfer and decimal wire anchor','status':'open; zero paired KV episodes','missing':'Separate source endpoint plus actual payload/wire accounting and destination retrieval evidence on this path.'},
        {'gap':'Divisible fleet compute and cache placement','status':'open; local width sensitivity only','missing':'Matched physical inventory and bounded concurrent-batch placement measurements; no fleet-size penalty assumed.'},
        {'gap':'Token timing and cache telemetry','status':'verified for retained exact events on patched runtime; early gaps retained','missing':'Server per-request execution timestamps for detailed queue/compute attribution; performance equivalence check for FIFO event delivery.'},
        {'gap':'SLO-feasible five-policy shedding','status':'open; bounded twenty-evaluation diagnostic only','missing':'Resolve live-source/KV, physical placement and latency/recovery mappings before treating policy handoff as feasibility.'}],
    'policy_verification':{'evaluations':20,'source_gpus':6666,'destination_gpus_per_site':[6666,6666],
        'gpu_nameplate_mw_per_site':1.9998,'shared_wan_gbps':1000,'qualification':policy['qualification'],
        'qualification_scope':'Finite completed-request control screen at nominal tested RPS, with censoring disclosed; this is not a validated fleet SLO operating point.',
        'saved_audit_comparison':'The saved2-MW audit reuses the20-MW cross-variant candidate union. This verification constructs candidates for the current2-MW fleet and shares them across all five policies. This is not a controlled reproduction of the saved96.3-percent result; a unique cause for numerical differences is not inferred.',
        'table':'policy-comparison.csv','details':'policy-verification.json','all_latency_validated':False},
    'tests':read('test-results.json'),
    'reproduction':{'reduction_command':['/tmp/qh-replay-runtime/bin/python','pool_replay_report.py','--out',str(out.relative_to(Path.cwd())),'--policies'],
        'finalize_command':[sys.executable,*sys.argv], 'launch_evidence':'runtime-launch.json, recovered-mp-launch.json, unloaded-launch.json, stage launch JSON files and continue-acquisition.sh',
        'seeds':[7101,7102],'raw':'requests*.jsonl.gz and request-events*.jsonl.gz; lossless gzip retains the pre-reboot damaged tail, whose valid prefix/hash is recorded in node-recovery.json; per-episode metrics, power, traces and migration events remain separate.'}}
write('report.json', report)
write('provenance-final.json', {'commit_before_evidence_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'dirty_state_before_evidence_commit':subprocess.check_output(['git','status','--porcelain'],text=True),
    'baseline':'e68f1d26','autostash_fix':'35be3279',
    'commits_since_autostash':subprocess.check_output(['git','log','--format=%H %s','35be3279..HEAD'],text=True).splitlines()})
archives=[]
for name in ('requests.jsonl','request-events.jsonl','requests-recovered.jsonl','request-events-recovered.jsonl'):
    path=out/name
    data=raw_bytes(out,name)
    target=out/(name+'.gz')
    if path.exists():
        target.write_bytes(gzip.compress(data,mtime=0))
        assert gzip.decompress(target.read_bytes())==data
        path.unlink()
    archives.append({'file':target.name,'uncompressed_sha256':hashlib.sha256(data).hexdigest(),'uncompressed_bytes':len(data)})
write('raw-archives.json',archives)
write('artifact-sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json' and '__pycache__' not in p.parts})
print(json.dumps({'counts':report['counts'],'report':str(out/'report.json'),'campaign_ready':False},indent=2))
