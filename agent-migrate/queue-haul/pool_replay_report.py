"""Reduce bounded replay evidence without fitting simulator coefficients."""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pool_shed_campaign as q
from pool_shed_calibration import replay_seconds
from pool_shed_execution import initial_work, kv_transfer_bytes, regional_execution_check
from pool_shed_calibration import loaded_execution_check, resident_execution_check
from loaded_service_model import historical_execution_check
from pool_replay_measure import write
from pool_replay_resident import summarize


def raw_bytes(out,name):
    path=out/name
    return path.read_bytes() if path.exists() else gzip.decompress((out/(name+'.gz')).read_bytes())


def requests(out):
    recovery = json.loads((out/'node-recovery.json').read_text())
    path = out/'requests.jsonl'
    raw = raw_bytes(out,path.name)
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
    wakeups={}
    recovery=json.loads((out/'node-recovery.json').read_text())
    for name in ('request-events.jsonl','request-events-recovered.jsonl'):
        data=raw_bytes(out,name)
        if name in recovery['raw_pre_reboot']:data=data[:recovery['raw_pre_reboot'][name]['valid_prefix_bytes']]
        for line in data.splitlines():
            if b'"kind":"scheduled_arrival"' not in line:continue
            r=json.loads(line)
            wakeups[(r['episode'],r['cohort'],r['session'],r['turn'])]=r['client_wakeup_ns']
    for r in raw:
        key=tuple(r.get(k) for k in ('episode','cohort','session','turn'))
        if key in wakeups:r['client_wakeup_ns']=wakeups[key]
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
                exact=[r for r in cohort_rows if r.get('done') and r.get('status')==200 and r.get('exact_token_timestamps')
                       and a<=(r['scheduled_ns']-epoch)/1e9<z and r['end_ns']<=epoch+z*1e9]
                window['ttft_over_1s_requests']=sum((r['first_ns']-r['scheduled_ns'])/1e9>1 for r in exact)
                window['request_tpot_over_100ms_requests']=sum(r['mean_tpot_s'] is not None and r['mean_tpot_s']>.1 for r in exact)
                window['exact_completed_fraction_of_arrivals']=len(exact)/summary['offered_requests'] if summary['offered_requests'] else None
                observed_power=[r for r in power if a<=(int(r['monotonic_ns'])-epoch)/1e9<z]
                window['mean_power_w']=float(np.mean([float(r['power_w']) for r in observed_power])) if observed_power else None
                window['mean_gpu_utilization_pct']=float(np.mean([float(r['utilization_pct']) for r in observed_power])) if observed_power else None
                windows.append(window)
        admission=[(event['monotonic_ns']-epoch)/1e9 for event in result['migration_events'] if event['kind']=='destination_admission']
        after60=[r for r in metrics if r['monotonic_ns']>=epoch+60e9]
        record={'spec':spec,'epoch_ns':epoch,'duration_s':duration,'trace_sha256':hashlib.sha256((path.parent/'offered-trace.json').read_bytes()).hexdigest(),
            'admitted_by_90s':sum(v<=90 for v in admission),'admitted_by_boundary':len(admission),
            'admission_times_s':admission,'migration_events':result['migration_events'],
            'first_incoming_service_token_s':{str(session):min((r['first_ns']-epoch)/1e9 for r in rows
                if r.get('cohort')=='incoming' and r['session']==session and r.get('first_ns') is not None)
                for session in {r['session'] for r in rows if r.get('cohort')=='incoming' and r.get('first_ns') is not None}},
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
            admitted=[sum(t<=row['window_start_s'] for t in r['admission_times_s']) for r in (control,replay)]
            pairs.append({k:row[k] for k in ('episode','workload','seed','rate','cohort','window_start_s','window_end_s')} | {
                'control_admitted_at_window_start':admitted[0],'replay_admitted_at_window_start':admitted[1],
                'equivalent_full_incoming_population':admitted[0]==admitted[1]==replay['spec'].get('width',8),
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
    migrations=[{k:r.get(k) for k in ('episode','workload','seed','rate','width','phase','session','status',
        'prompt_tokens','output_tokens','cached_tokens','derived_prompt_minus_cache_tokens',
        'start_ns','end_ns','ttft_s','mean_tpot_s','exact_token_timestamps','state_code_valid','context_hash')}
        for r in raw if r.get('cohort')=='migration' and r.get('episode','').startswith(('episodes-','followups-'))]
    csv_rows(out/'migration-observations.csv',migrations)
    write(out/'service-analysis.json',{'episodes':records,'windows':windows,'pairs':pairs,
        'scope':'Destination-only synthetic content with recorded evolving shapes; paired arrivals and demand. No active source, ownership transfer or source quiescence measurement.',
        'recovery_scope':'Outstanding arrivals and completion deficit relative to matched control while arrivals continue; no cleanup-drain recovery claim.',
        'latency_scope':'Original-arrival TTFT and P90 per-request mean TPOT; exact client token-event coverage, not server execution timestamps. Short windows do not validate tails.',
        'client_timing_scope':'Send lateness is scheduled arrival to send; scheduling lateness is scheduled arrival to client wakeup; client queue is wakeup to dispatch and includes prompt preparation. Earlier inline summaries labelled total send lateness as client queue; this reduction uses the retained wakeup events to separate them.',
        'engine_scope':'Queue time is directly measured aggregate engine histogram delta across all populations; no per-request queue attribution or queue inferred from TTFT. KV usage gauge excludes free evictable cached blocks; it is not resident tensor allocation.'})
    return records


def policies(out, raw, calibration):
    """Exactly four cells; qualification is a finite-window hardware screen."""
    qualification={}
    for workload in ('coding','coding_long'):
        nominal=.5*q.sample_fleet(workload,gpus=6666,gpus_per_node=8).metadata['reference_rps']
        qualification[workload]={'offered_rps_per_gpu':nominal,'repeats':{}}
        for seed in (7101,7102):
            rows=[r for r in raw if r.get('workload')==workload and r.get('arm')=='control'
                  and r.get('seed')==seed and r.get('cohort')=='resident' and np.isclose(r['rate'],nominal)
                  and r.get('episode','').startswith('episodes-')
                  and 60 <= (r['scheduled_ns']-r['episode_epoch_ns'])/1e9 < 180]
            done=[r for r in rows if r.get('done') and r.get('status')==200]
            exact=[r for r in done if r['exact_token_timestamps']]
            ttft=[(r['first_ns']-r['scheduled_ns'])/1e9 for r in exact]
            tpot=[r['mean_tpot_s'] for r in exact if r['mean_tpot_s'] is not None]
            coverage=len(exact)/len(done) if done else 0
            p90=lambda values:float(np.quantile(values,.9)) if values else None
            a,z=p90(ttft),p90(tpot)
            qualification[workload]['repeats'][str(seed)]={'completed_requests':len(done),'offered_requests':len(rows),
                'censored_or_failed':len(rows)-len(done),'exact_timing_coverage':coverage,'tpot_requests':len(tpot),
                'p90_arrival_ttft_s':a,'p90_request_mean_tpot_s':z,
                'completed_request_screen_pass':bool(coverage>=.99 and a is not None and a<=1 and z is not None and z<=.1)}
    report={'qualification':qualification,'scope':'Four central-parameter cells, common candidates and traffic within each cell; observed completed-request service screen at the declared RPS. Finite 24-session hardware mixture does not validate fleet placement, tails, source quiescence or KV transfer.',
        'campaign_ready':False,'fleet_latency_validated':False,'coefficient_changes':[],
        'wire_bytes_per_32768_tokens':800000000,'native_serialized_bytes_per_32768_tokens':1610612736,
        'wire_scope':'Decimal effective-wire assumption; no extra private-KV discount. Native resident memory unchanged.',
        'source_gpus':6666,'destination_gpus_per_site':[6666,6666],'installed_gpu_nameplate_mw_per_site':1.9998,
        'gpus_per_node':8,'shared_wan_gbps':1000,'sources':q.provenance(calibration),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cells':[]}
    if not all(r['completed_request_screen_pass'] for w in qualification.values() for r in w['repeats'].values()):
        report['status']='unmeasured_qualified_operating_point_missing'
        write(out/'policy-verification.json',report)
        return
    report['status']='running'
    endpoint=np.r_[np.median(q.network_samples()[:,:2],axis=0),0.];endpoint[2]=endpoint[:2].sum()
    timing=calibration['timing'][0]
    for workload in ('coding','coding_long'):
        fleet=q.sample_fleet(workload,gpus=6666,gpus_per_node=8)
        fleet=replace(fleet,metadata={**fleet.metadata,'planning_reference_s':4.,
            'kv_wire_scale':.8e9/(32768*49152),'replay_cached_tokens':np.zeros(len(fleet.count)).tolist(),
            'kv_shared_tokens':np.zeros(len(fleet.count)).tolist()})
        fleet=replace(fleet,kv=kv_transfer_bytes(fleet,fleet.context,calibration))
        budgets=q.bandwidth(endpoint,fleet.nodes,1000)
        replay,kv=q.library(fleet)
        replay,kv=q.include_isolated(replay,kv,q.isolated_methods(fleet,.5,endpoint,budgets,timing))
        candidate_hash=hashlib.sha256(replay.tobytes()+kv.tobytes()).hexdigest()
        for deadline in (30,120):
            table=q.schedule_table(fleet,replay,kv,.5,deadline,endpoint,budgets,timing)
            cell={'workload':workload,'deadline_s':deadline,'candidate_sha256':candidate_hash,
                  'resident_rps_per_gpu':qualification[workload]['offered_rps_per_gpu'],
                  'results':{},'budgets_bytes_per_s':budgets.tolist(),'fleet_metadata':fleet.metadata}
            report['cells'].append(cell)
            for policy in q.POLICIES:
                started=time.monotonic()
                result=q.execute_feedback(table,table,policy,timing,calibration)
                assert result['max_relative_residual']<=1e-8 and result['last_completion_s']<=deadline+1e-8
                assert not result['resident_latency_validated']
                cell['results'][policy]={**result,'verification_wall_s':time.monotonic()-started}
                write(out/'policy-verification.json',report)
                print(workload,deadline,policy,result['shed_fraction'],result['pending_buffered_requests'],flush=True)
    report['status']='complete';report['policy_evaluations']=sum(len(cell['results']) for cell in report['cells'])
    assert report['policy_evaluations']==20
    write(out/'policy-verification.json',report)
    regional=regional_execution_check(calibration)
    write(out/'existing-holdouts.json',{'loaded':loaded_execution_check(calibration),'regional':regional,
        'resident':resident_execution_check(calibration,regional),'historical':historical_execution_check(calibration),
        'scope':'Existing hardware holdouts, unchanged coefficients; these do not validate the new fleet latency or recovery assumptions.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--service',action='store_true')
    parser.add_argument('--policies',action='store_true')
    args=parser.parse_args()
    raw=requests(args.out)
    calibration=q.calibration(0)
    rows=unloaded(args.out,raw,calibration)
    print('Unloaded phase observations:',len(rows),'valid:',sum(r['phase_valid'] for r in rows))
    if args.service or args.policies:
        raw += [json.loads(line) for line in raw_bytes(args.out,'requests-recovered.jsonl').splitlines()]
        print('Service episodes:',len(service(args.out,raw)))
    if args.policies:policies(args.out,raw,calibration)


if __name__=='__main__':main()
