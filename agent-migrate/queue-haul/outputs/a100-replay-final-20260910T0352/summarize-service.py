"""Produce compact service evidence tables; no fitting or policy simulation."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
paths=[root/'service-analysis.json',root/'kv-observations.json',root/'resident-rate-selection.json']
service,kv,rates=(json.loads(p.read_text()) for p in paths)
main=[e for e in service['episodes'] if e['spec']['arm']!='resident']
scouts=[e for e in service['episodes'] if e['spec']['arm']=='resident']
rows=[]
for episode in main:
    spec=episode['spec'];pause=episode['quiescence']
    for w in episode['windows']:
        if (w['start_s'],w['end_s'])!=(60,300):continue
        rows.append({**{k:spec[k] for k in ('episode','workload','arm','seed','rate')},'cohort':w['cohort'],
            **{k:w[k] for k in ('offered_rps','completed_rps','offered_requests','completed_arrival_cohort_requests','exact_requests','tpot_requests','exact_timing_coverage_of_completed','exact_completed_fraction_of_arrivals','p90_original_arrival_ttft_s','p90_request_mean_tpot_s','known_original_arrival_ttft_over_1s','outstanding_all_prior_arrivals')},
            'outstanding_at_migration_plus30':episode['outstanding_by_seconds_after_migration']['30'][w['cohort']],
            'outstanding_at_migration_plus120':episode['outstanding_by_seconds_after_migration']['120'][w['cohort']],
            'switches_at_migration_plus30':episode['switches_by_seconds_after_migration']['30'],
            'switches_at_migration_plus120':episode['switches_by_seconds_after_migration']['120'],
            'destination_dispatch_timing':episode['incoming_by_dispatch_placement']['destination'] if w['cohort']=='incoming' else None,
            'strict_source_token_stream_pause_overlaps':sum(r['client_token_stream_overlap_verified'] for r in pause),
            'tail_guarantee':False})
with (root/'service-observations.csv').open('w') as handle:
    writer=csv.DictWriter(handle,list(rows[0]) if rows else ['episode']);writer.writeheader();writer.writerows(rows)
verified_methods=sorted({e['spec']['arm'] for e in main if any(q['client_token_stream_overlap_verified'] for q in e['quiescence'])})
report={'acquisition_status':'in_progress' if len(main)<12 else 'all_twelve_main_observations_complete',
    'completed_main_episodes':len(main),'requested_main_episodes':12,'completed_scouts':len(scouts),
    'clean_controlled_KV_conditions':kv['clean_conditions'],'clean_KV_destination_continuations':kv['clean_destination_continuations'],
    'selected_resident_rates':rates,'service_observations':rows,
    'strict_client_token_stream_pause_overlap_methods':verified_methods,
    'previous_nine_scenario_events_archive':'../a100-replay-completion-20260910T0116/scenario-events-archive.json',
    'evidence_tables':{'controlled_KV':'kv-observations.csv','service':'service-observations.csv','complete_windows_phases_metrics':'service-analysis.json','recorded_history_validation':'resident-history-audit.json','raw_archive':'raw-telemetry-archive.json','GET_payload_origin_proof':'attribution-proof.json','loaded_KV_payload':'service-wire-analysis.json'},
    'modeling_gap_status':{
        'eight_physical_resident_sessions':'implemented; final recorded-history audit must verify all completed episodes',
        'evolving_history_and_queued_arrivals':'actual generated token IDs retained with recorded resets; per-session causality and queued arrivals preserved; raw history audit provides independent verification',
        'separate_source_destination':'verified distinct Sweden/Germany A10080GB GPUs; runtime-build differences remain explicit',
        'four_clean_KV_measurements':'verified four requested width-eight conditions; two original L1-assumption failures also retained',
        'real_source_quiescence':{'verified_client_token_stream_overlap_methods':verified_methods,'server_execution_timestamps':'unavailable; client overlap is not an aligned server execution trace'},
        'loaded_full_context_catchup':'measured full actual retained input with natural cache reuse and original512 maximum migration output; TTFT and full generation/completion time reported separately',
        'service_latency':'per-arrival TTFT and per-request mean TPOT with exact counts and censoring; no reliable tail guarantee from these short windows',
        'recovery_under_continuing_load':'matched finite windows through300s, per-cohort completion deficits and latency differences; cleanup is excluded',
        'wire_anchor':'native serialized payload49152B/token implies1610612736B/32768tokens before retransfers/protocol; historical800000000B effective-wire anchor remains a separate path assumption',
        'divisible_width_eight_GPU_work':'not validated as a fleet scaling law; shared request packing/queueing is directly observed',
        'resident_service_normalization':'actual RPS and finite-window brackets measured; no universal capacity percentage or fleet normalization fit',
        'GPU_queue_compute_decomposition':'aggregate engine queue/prefill/decode counters retained; no causal per-request execution-time decomposition available',
        'semantic_state_validation':'controlledKV preserves original chat state-code probe; loaded service adapter validates exact retained input hash/generation continuity, not semantic state-code recall',
        'simulator_fitting_and_campaign':'stopped pending user review; no coefficients changed and no full campaign launched'},
    'interpretation_limits':service['limitations']+[
        'Sweden/Germany match vLLM0.22.0 and LMCache0.5.1 plus reference serving settings; Python, Torch CUDA build and Transformers versions differ (runtime-builds.json). Identical model-file hashes and rendered-token check do not establish exact runtime-build equivalence.',
        'Arrival timestamps are assumed because recorded trajectories contain none. Eight physical per-GPU resident histories are sampled from the frozen trajectory distribution; physical source ownership continues until measured handoff.',
        'Loaded service uses exact-token completions to preserve evolving output history; this is an explicit adapter deviation from original full-message chat probes.',
        'Protocol bytes are observed RESP application commands/keys/framing, excluding TCP/IP retransmitted packets, HTTP/SSE and SSH overhead. Unique GET payload and repeated GET payload are separate. Archived persistent-pool connection-origin proof identifies source-local and Germany destination GET subsets; original unfiltered proxy totals remain distinct. The hardware proxy1000Mbit/s aggregate cap is not the archived simulator1000Gbit/s scenario.',
        'Original failed KV attempts and contaminated scout rates remain evidence. Instrumentation/transport amendments do not convert failed attempts into passing measurements.'],
    'recommendation':'Keep full simulations and fitting stopped until review; these bounded measurements do not establish fleet SLO feasibility.',
    'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'reporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'main_episodes':len(main),'scouts':len(scouts),'strict_pause_methods':verified_methods}))
