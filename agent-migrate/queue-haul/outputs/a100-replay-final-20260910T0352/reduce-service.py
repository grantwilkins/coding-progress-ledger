"""Reduce source-active service evidence without fitting or simulator execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def percentile(values):
    return float(np.quantile(values, .9)) if values else None


def window(rows, trace, epoch, start, end):
    eligible = [r for r in rows if start <= (r['scheduled_ns']-epoch)/1e9 < end]
    done = [r for r in eligible if r.get('done') and r.get('status') == 200 and r['end_ns'] <= epoch+end*1e9]
    exact = [r for r in done if r.get('exact_token_timestamps') and r.get('first_ns') is not None]
    offered = sum(start <= r['offset_s'] < end for r in trace)
    ttft = [(r['first_ns']-r['scheduled_ns'])/1e9 for r in exact]
    tpot = [r['mean_tpot_s'] for r in exact if r.get('mean_tpot_s') is not None]
    completed = [r for r in rows if r.get('done') and r.get('status') == 200 and r['end_ns'] <= epoch+end*1e9]
    cached = [r for r in done if r.get('cached_tokens') is not None]
    coverage = len(exact)/len(done) if done else None
    return {'start_s':start,'end_s':end,'offered_requests':offered,'completed_arrival_cohort_requests':len(done),
        'unfinished_or_failed_arrival_cohort':offered-len(done),'exact_requests':len(exact),'tpot_requests':len(tpot),
        'exact_timing_coverage_of_completed':coverage,'exact_completed_fraction_of_arrivals':len(exact)/offered if offered else None,
        'offered_rps':offered/(end-start),'completed_rps':sum(r['end_ns'] >= epoch+start*1e9 for r in completed)/(end-start),
        'p90_original_arrival_ttft_s':percentile(ttft),'p90_request_mean_tpot_s':percentile(tpot),
        'p90_client_send_lateness_s':percentile([(r['start_ns']-r['scheduled_ns'])/1e9 for r in done]),
        'p90_client_queue_s':percentile([(r['client_dispatch_ns']-r['client_wakeup_ns'])/1e9 for r in done if r.get('client_wakeup_ns') is not None]),
        'p90_client_schedule_lateness_s':percentile([(r['client_wakeup_ns']-r['scheduled_ns'])/1e9 for r in done if r.get('client_wakeup_ns') is not None]),
        'outstanding_all_prior_arrivals':sum(r['offset_s'] < end for r in trace)-len(completed),
        'known_cache_requests':len(cached),'cached_tokens':sum(r['cached_tokens'] for r in cached) if cached else None,
        'derived_prompt_minus_cache_tokens':sum(r['prompt_tokens']-r['cached_tokens'] for r in cached) if cached else None,
        'completed_request_latency_screen':bool(coverage is not None and coverage >= .99 and ttft and tpot and percentile(ttft) <= 1 and percentile(tpot) <= .1),
        'tail_guarantee':False}


def engine_window(rows, epoch, start, end):
    selected = [r for r in rows if start <= (float(r['monotonic_ns'])-epoch)/1e9 < end]
    gauges = ('num_requests_running','num_requests_waiting','kv_cache_usage_perc')
    counters = ('num_preemptions_total','prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_hits_total',
                'request_prefill_kv_computed_tokens_sum','request_queue_time_seconds_sum','request_prefill_time_seconds_sum','request_decode_time_seconds_sum')
    return {'samples':len(selected),'max_sample_gap_s':max(((float(z['monotonic_ns'])-float(a['monotonic_ns']))/1e9 for a,z in zip(selected,selected[1:])),default=None),
        'gauges':{key:{'min':min(values,default=None),'max':max(values,default=None),'mean':float(np.mean(values)) if values else None} for key in gauges for values in ([float(r['vllm:'+key]) for r in selected if r.get('vllm:'+key)],)},
        'counter_deltas':{key:float(selected[-1]['vllm:'+key])-float(selected[0]['vllm:'+key]) if len(selected)>1 and selected[0].get('vllm:'+key) and selected[-1].get('vllm:'+key) else None for key in counters}}


def reduce(root):
    hashes = {}
    def raw(path):
        value = path.read_bytes(); hashes[str(path.relative_to(root))] = hashlib.sha256(value).hexdigest(); return value
    def read(path):
        return json.loads(raw(path))
    rows = [json.loads(line) for line in raw(root/'requests.jsonl').splitlines()]
    episodes = []
    for path in sorted(root.glob('*/result.json')):
        result = read(path)
        if 'spec' not in result: continue
        spec, epoch = result['spec'], result['epoch_ns']
        duration = (result['boundary_ns']-epoch)/1e9
        trace = read(path.parent/'offered-trace.json')
        selected = [r for r in rows if r.get('episode') == spec['episode']]
        intervals = [(30,90)] if spec['arm']=='resident' else [(0,60),(60,90),(90,120),(120,180),(180,240),(240,300),(60,300)]
        intervals = [(a,z) for a,z in intervals if z <= duration]
        tables = {p.stem:list(csv.DictReader(raw(p).decode().splitlines())) for p in path.parent.glob('engine*.csv')}
        windows = [{'cohort':cohort,**window([r for r in selected if r['cohort']==cohort],[r for r in trace if r['cohort']==cohort],epoch,a,z)}
                   for cohort in ('resident','incoming') if any(r['cohort']==cohort for r in trace) for a,z in intervals]
        checkpoints = {str(t-60):{cohort:window([r for r in selected if r['cohort']==cohort],[r for r in trace if r['cohort']==cohort],epoch,0,t)['outstanding_all_prior_arrivals'] for cohort in ('resident','incoming')} for t in (90,180) if t <= duration and spec['arm']!='resident'}
        events = result.get('migration_events',[])
        switches = [e for e in events if e['kind'] in ('route_switch','destination_admission')]
        episodes.append({'spec':spec,'duration_s':duration,'resident_population':result.get('resident_population'),'incoming_population':result.get('incoming_population'),
            'trace_sha256':hashes[str((path.parent/'offered-trace.json').relative_to(root))],
            'windows':windows,'outstanding_by_seconds_after_migration':checkpoints,
            'engine':{name:[{'start_s':a,'end_s':z,**engine_window(table,epoch,a,z)} for a,z in intervals] for name,table in tables.items()},
            'migration_events':events,'switches_by_seconds_after_migration':{str(t-60):sum(e['monotonic_ns'] <= epoch+t*1e9 for e in switches) for t in (90,180) if t <= duration},
            'control_materialization_requests':sum(r.get('cohort')=='control_materialization' for r in selected),
            'statuses':{str(status):sum(r.get('status')==status for r in selected) for status in {r.get('status') for r in selected}}})
    pairs = []
    for episode in episodes:
        if episode['spec']['arm'] not in ('replay','kv_transfer'): continue
        control = next((r for r in episodes if r['spec']['arm']=='control' and r['spec']['trace_id']==episode['spec']['trace_id']),None)
        if control is None: continue
        if control['trace_sha256'] != episode['trace_sha256']: raise ValueError('matched arms have different offered traffic')
        for w in episode['windows']:
            reference = next(r for r in control['windows'] if all(r[k]==w[k] for k in ('cohort','start_s','end_s')))
            pairs.append({'episode':episode['spec']['episode'],'control':control['spec']['episode'],**{k:w[k] for k in ('cohort','start_s','end_s')},
                'completion_deficit_requests':w['outstanding_all_prior_arrivals']-reference['outstanding_all_prior_arrivals'],
                'original_arrival_ttft_p90_difference_s':w['p90_original_arrival_ttft_s']-reference['p90_original_arrival_ttft_s'] if w['p90_original_arrival_ttft_s'] is not None and reference['p90_original_arrival_ttft_s'] is not None else None,
                'request_mean_tpot_p90_difference_s':w['p90_request_mean_tpot_s']-reference['p90_request_mean_tpot_s'] if w['p90_request_mean_tpot_s'] is not None and reference['p90_request_mean_tpot_s'] is not None else None})
    return {'episodes':episodes,'matched_comparisons':pairs,'input_sha256':hashes,'fitting':'stopped_pending_user_review','campaign_ready':False,
        'limitations':['Exact timing is client token arrival, not server execution. No GPU queue inferred from TTFT; aggregate engine histogram deltas cannot be added as elapsed time.',
            'Window percentiles cover arrival cohorts completed by that window end; unfinished arrivals remain outstanding. Completed-only timing coverage and offered-arrival coverage are separate; short windows establish no tail guarantee.',
            'Recovery comparisons use continuing arrivals through300; cleanup excluded. Control materialization adds destination work and must be read with baseline differences.',
            'Service migration adapter validates full retained token history/hash/generation, not semantic state-code recall; original full-message512 probe retained only in controlled KV diagnostics.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path); args = parser.parse_args()
    value = reduce(args.root); value['reducer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root/'service-analysis.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'episodes':len(value['episodes']),'matched_windows':len(value['matched_comparisons'])}))
