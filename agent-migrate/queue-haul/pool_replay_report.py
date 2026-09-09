"""Reduce bounded replay evidence without fitting simulator coefficients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pool_shed_campaign as q
from pool_shed_calibration import replay_seconds
from pool_shed_execution import initial_work
from pool_replay_measure import write
from pool_replay_resident import summarize


def requests(out):
    recovery = json.loads((out/'node-recovery.json').read_text())
    path = out/'requests.jsonl'
    raw = path.read_bytes()
    damaged = recovery['raw_pre_reboot'][path.name]
    assert hashlib.sha256(raw).hexdigest() == damaged['sha256']
    return [json.loads(line) for line in raw[:damaged['valid_prefix_bytes']].splitlines()]


def csv_rows(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def unloaded(out, raw, calibration):
    phases = {(path.stem, phase['phase']): phase
              for path in out.glob('unloaded*/unloaded-*.json')
              for phase in json.loads(path.read_text())['phases']}
    rows = []
    for episode in sorted({r['episode'] for r in raw if 'retained_tokens' in r}):
        trial = next(r for r in raw if r.get('episode') == episode)
        context, append, width = (trial[k] for k in ('retained_tokens','append_tokens','width'))
        for phase in ('initial','catch_up','cold_updated'):
            selected = [r for r in raw if r.get('episode') == episode and r['phase'] == phase]
            done = [r for r in selected if r['status'] == 'complete']
            valid = [r for r in done if r['messages'][-1]['content'].split('code ')[-1].rstrip('.') in r['response_text']]
            target = context if phase == 'initial' else context+append
            fleet = SimpleNamespace(context=np.array([target]), t1=replay_seconds([target],calibration), log=np.array([2*target]),
                metadata={'packing_context_tokens':calibration['packing_context_tokens'],
                          'batch_context_limit':calibration['batch_context_limit']})
            _, predicted = initial_work(fleet,np.array([width]),0,0,None,
                {**calibration['timing'][0],'regional_replay_factor':[1.,1.]},calibration)
            elapsed = (max(r['end_ns'] for r in selected)-min(r.get('start_ns',r['dispatch_ns']) for r in selected))/1e9 if selected else None
            row = {'episode':episode,'seed':trial['seed'],'context':context,'append':append,'width':width,'phase':phase,
                'requests':len(selected),'http_complete':len(done),'state_valid':len(valid),
                'phase_valid':len(valid)==width,'missing_requests':width-len(selected),
                'exact_requests':sum(r['exact_token_timestamps'] for r in done),
                'elapsed_s':elapsed,'model_full_rebuild_local_s':predicted,
                'model_minus_observed_s':predicted-elapsed if elapsed is not None else None,
                'optimistic_error_s':max(0,elapsed-predicted) if elapsed is not None else None,
                'model_predicts_30s_but_observed_exceeds':elapsed is not None and predicted<=30<elapsed,
                'reported_cached_tokens':json.dumps([r.get('cached_tokens') for r in selected]),
                'prompt_counts':json.dumps([r['prompt_tokens'] for r in selected]),
                'output_counts':json.dumps([r.get('output_tokens') for r in selected])}
            evidence = phases.get((episode,phase))
            for metric in ('prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_hits_total',
                           'request_prefill_kv_computed_tokens_sum','request_queue_time_seconds_sum',
                           'request_inference_time_seconds_sum','request_prefill_time_seconds_sum','request_decode_time_seconds_sum'):
                key = 'vllm:'+metric
                row[metric] = (evidence['engine_after'][key]-evidence['engine_before'][key]
                    if evidence and key in evidence['engine_after'] and key in evidence['engine_before'] else None)
            queries, hits = row['prefix_cache_queries_total'], row['prefix_cache_hits_total']
            row['cache_state'] = ('cold_verified' if queries == sum(r['prompt_tokens'] for r in selected) and hits == 0
                else 'native_reuse_verified' if hits is not None and hits>0 else 'unknown')
            rows.append(row)
    csv_rows(out/'unloaded-observations.csv',rows)
    heldout = []
    for row in rows:
        if row['seed'] != 7102: continue
        prior = next(r for r in rows if r['seed']==7101 and all(r[k]==row[k] for k in ('context','append','width','phase')))
        usable = prior['phase_valid'] and row['phase_valid']
        heldout.append({k:row[k] for k in ('context','append','width','phase')} | {
            'training_phase_valid':prior['phase_valid'],'validation_phase_valid':row['phase_valid'],
            'training_observed_s':prior['elapsed_s'],'validation_observed_s':row['elapsed_s'],
            'repeat_error_s':prior['elapsed_s']-row['elapsed_s'] if usable else None,
            'repeat_relative_error':prior['elapsed_s']/row['elapsed_s']-1 if usable else None,
            'baseline_model_error_s':row['model_minus_observed_s'],
            'optimistic_error_s':row['optimistic_error_s']})
    csv_rows(out/'unloaded-heldout.csv',heldout)
    write(out/'unloaded-analysis.json',{'rows':rows,'heldout':heldout,
        'fitting':'No fitted correction. First-repeat lookup versus independent second repeat is a repeatability diagnostic only.',
        'prediction_scope':'Existing full-context work and packing, one destination GPU, regional factor explicitly 1; no resident load. Warm phases compare with the existing full-rebuild catch-up assumption.',
        'timing_scope':'Elapsed client request interval includes validation/decode; server queue/prefill/decode sums are engine histogram deltas, not additive GPU work or per-request attribution.',
        'missing_phase_evidence':'Trial 08 seed 7101 aborted before its cold-updated batch; raw initial/catch-up requests survive but phase counter snapshots do not.',
        'state_validation':'HTTP completion and exact token timing do not establish a valid response; expected state code is checked independently.',
        'probe_format_deviation':'Initial trials used 10 lowercase hex characters instead of the reference 12 uppercase state code; full-message probe construction and 512-token generation settings were retained. Actual output lengths are recorded; do not infer a reference completion-overhead correction.',
        'wire_scope':'No paired KV transfer measured. 800000000 decimal effective-wire bytes per32768 tokens remains an assumption, separate from 1610612736 native serialized bytes and resident KV footprint.'})
    return rows


def service(out, raw):
    records, windows, pairs = [], [], []
    for path in sorted(out.glob('*/result.json')):
        result=json.loads(path.read_text());spec=result['spec'];episode=spec['episode']
        trace=json.loads((path.parent/'offered-trace.json').read_text())
        rows=[r for r in raw if r.get('episode')==episode and r.get('cohort') in ('resident','incoming')]
        metrics=[{k:float(v) for k,v in r.items() if v} for r in csv.DictReader((path.parent/'engine.csv').open())]
        power=[r for r in csv.DictReader((path.parent/'power.csv').open()) if r['valid']=='1']
        epoch=result['epoch_ns'];duration=(result['boundary_ns']-epoch)/1e9
        intervals=[(30,90)] if spec['arm']=='resident' else [(0,60),(60,90),(90,120),(120,150),(150,180)]
        for cohort in ('resident','incoming'):
            cohort_rows=[r for r in rows if r['cohort']==cohort]
            offered=[r for r in trace if r['cohort']==cohort]
            if not offered:continue
            for a,z in intervals:
                summary=summarize(cohort_rows,offered,epoch,(a,z),metrics)
                window={'episode':episode,'workload':spec['workload'],'seed':spec['seed'],'rate':spec['rate'],
                        'arm':spec['arm'],'cohort':cohort,'window_start_s':a,'window_end_s':z,**summary}
                window['outstanding_all_prior_arrivals']=sum(item['offset_s']<z for item in offered)-sum(
                    r.get('done') and r.get('status')==200 and r['end_ns']<=epoch+z*1e9 for r in cohort_rows)
                windows.append(window)
        admission=[(event['monotonic_ns']-epoch)/1e9 for event in result['migration_events'] if event['kind']=='destination_admission']
        after60=[r for r in metrics if r['monotonic_ns']>=epoch+60e9]
        record={'spec':spec,'epoch_ns':epoch,'duration_s':duration,'trace_sha256':hashlib.sha256((path.parent/'offered-trace.json').read_bytes()).hexdigest(),
            'admitted_by_90s':sum(v<=90 for v in admission),'admitted_by_boundary':len(admission),
            'admission_times_s':admission,'migration_events':result['migration_events'],
            'engine_samples':len(metrics),'power_samples':len(power),
            'max_engine_sample_gap_s':max(np.diff([r['monotonic_ns'] for r in metrics]),default=0)/1e9,
            'max_power_sample_gap_s':max(np.diff([int(r['monotonic_ns']) for r in power]),default=0)/1e9,
            'max_running':max((r['vllm:num_requests_running'] for r in metrics),default=None),
            'max_waiting':max((r['vllm:num_requests_waiting'] for r in metrics),default=None),
            'mean_power_w':float(np.mean([float(r['power_w']) for r in power])) if power else None,
            'mean_gpu_utilization_pct':float(np.mean([float(r['utilization_pct']) for r in power])) if power else None,
            'post_60_engine_queue_time_sum_s':after60[-1]['vllm:request_queue_time_seconds_sum']-after60[0]['vllm:request_queue_time_seconds_sum'] if after60 else None,
            'preemptions':metrics[-1]['vllm:num_preemptions_total']-metrics[0]['vllm:num_preemptions_total'],
            'source_quiescence_validated':False,'paired_kv_validated':False,'latency_tail_guarantee':False}
        records.append(record)
    for replay in records:
        if replay['spec']['arm']!='replay':continue
        control=next((r for r in records if r['spec']['arm']=='control' and r['spec']['trace_id']==replay['spec']['trace_id']),None)
        if control is None:continue
        assert replay['trace_sha256']==control['trace_sha256'],'paired offered arrivals differ'
        for row in [r for r in windows if r['episode']==replay['spec']['episode']]:
            reference=next(r for r in windows if r['episode']==control['spec']['episode'] and r['cohort']==row['cohort'] and r['window_start_s']==row['window_start_s'])
            pairs.append({k:row[k] for k in ('episode','workload','seed','rate','cohort','window_start_s','window_end_s')} | {
                'replay_arrival_ttft_p90_s':row['p90_arrival_ttft_s'],'control_arrival_ttft_p90_s':reference['p90_arrival_ttft_s'],
                'replay_request_tpot_p90_s':row['p90_request_mean_tpot_s'],'control_request_tpot_p90_s':reference['p90_request_mean_tpot_s'],
                'replay_outstanding':row['outstanding_all_prior_arrivals'],'control_outstanding':reference['outstanding_all_prior_arrivals'],
                'completion_deficit_requests':row['outstanding_all_prior_arrivals']-reference['outstanding_all_prior_arrivals'],
                'replay_completed_rps':row['completed_rps'],'control_completed_rps':reference['completed_rps'],
                'replay_screen_pass':row['screen_pass'],'control_screen_pass':reference['screen_pass'],
                'replay_exact_coverage':row['exact_timing_coverage'],'control_exact_coverage':reference['exact_timing_coverage'],
                'replay_completed_n':row['completed_requests'],'control_completed_n':reference['completed_requests']})
    csv_rows(out/'service-windows.csv',windows)
    csv_rows(out/'service-paired.csv',pairs)
    write(out/'service-analysis.json',{'episodes':records,'windows':windows,'pairs':pairs,
        'scope':'Destination-only synthetic content with recorded evolving shapes; paired arrivals and demand. No active source, ownership transfer or source quiescence measurement.',
        'recovery_scope':'Outstanding arrivals and completion deficit relative to matched control while arrivals continue; no cleanup-drain recovery claim.',
        'latency_scope':'Original-arrival TTFT and P90 per-request mean TPOT; exact client token-event coverage, not server execution timestamps. Short windows do not validate tails.',
        'engine_scope':'Queue time is directly measured aggregate engine histogram delta across all populations; no per-request queue attribution or queue inferred from TTFT. KV usage gauge excludes free evictable cached blocks; it is not resident tensor allocation.'})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--service',action='store_true')
    args=parser.parse_args()
    raw=requests(args.out)
    rows=unloaded(args.out,raw,q.calibration(0))
    print('Unloaded phase observations:',len(rows),'valid:',sum(r['phase_valid'] for r in rows))
    if args.service:
        raw += [json.loads(line) for line in (args.out/'requests-recovered.jsonl').read_text().splitlines()]
        print('Service episodes:',len(service(args.out,raw)))


if __name__=='__main__':main()
