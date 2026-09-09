"""Causal evolving-trajectory traffic and destination-only shared-service episodes."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np

import destination_runner as serving
import migration_profiler as profiler
import service_headroom_campaign as headroom
from pool_replay_measure import Acquisition, write


class Trajectory:
    def __init__(self, label, sequence, offset, seed):
        self.label, self.sequence, self.offset, self.seed = label, sequence, offset, seed
        self.completed, self.history = 0, []
        self.initial_history = None
        self.lock = asyncio.Lock()
        self.failed = False

    def prompt(self):
        index = (self.offset+self.completed) % len(self.sequence)
        shape = self.sequence[index]
        reset = self.completed == 0 or index == 0 or shape['reset']
        if reset:
            self.history = (list(self.initial_history) if self.completed == 0 and self.initial_history is not None else
                            serving.deterministic_tokens(f'{self.label}-cycle{self.completed}', int(shape['context']), 200000, self.seed))
        target = int(shape['context']+shape['prompt'])
        if len(self.history) > target:
            raise ValueError('recorded prompt cannot retain the preceding generated history')
        added = serving.deterministic_tokens(f'{self.label}-turn{self.completed}', target-len(self.history), 200000, self.seed)
        prompt = self.history+added
        if len(prompt)+shape['output'] > 32768:
            raise ValueError('trajectory exceeds runtime support')
        return prompt, shape, index, reset, len(added)

    def accept(self, prompt, row):
        if not row['done'] or row['status'] != 200 or row['recorded_output_tokens'] != row['planned_output_tokens']:
            self.failed = True
            raise RuntimeError('incomplete trajectory dependency; later turns cannot execute')
        self.history = prompt+row['token_ids']
        self.completed += 1


def arrival_trace(rate, counts, seed, duration, start=0., burst=False):
    rng = random.Random(seed)
    rows = []
    total = sum(counts)
    for session, count in enumerate(counts):
        period = total/(rate*count)
        offset = start+rng.random()*period
        turn = 0
        while offset < duration:
            rows.append({'session':session,'turn':turn,'offset_s':start+math.floor((offset-start)/5)*5 if burst else offset})
            offset += period; turn += 1
    return sorted(rows,key=lambda r:(r['offset_s'],r['session'],r['turn']))


def summarize(rows, scheduled, epoch, window, metrics):
    eligible = [r for r in rows if window[0] <= (r['scheduled_ns']-epoch)/1e9 < window[1]]
    completed = [r for r in rows if r.get('done') and r.get('status') == 200]
    done = [r for r in eligible if r in completed and r['end_ns'] <= epoch+window[1]*1e9]
    completions = sum(window[0] <= (r['end_ns']-epoch)/1e9 < window[1] for r in completed)
    exact = [r for r in done if r.get('exact_token_timestamps')]
    ttft = [(r['first_ns']-r['scheduled_ns'])/1e9 for r in exact if r['first_ns'] is not None]
    tpot = [r['mean_tpot_s'] for r in exact if r['mean_tpot_s'] is not None]
    offered = sum(window[0] <= r['offset_s'] < window[1] for r in scheduled)
    queue = [r['vllm:num_requests_waiting'] for r in metrics if window[0] <= (r['monotonic_ns']-epoch)/1e9 < window[1]]
    percentile = lambda x: float(np.quantile(x,.9)) if x else None
    coverage = len(exact)/len(done) if done else 0.
    growth = float(np.mean(queue[-10:])-np.mean(queue[:10])) if len(queue) >= 20 else None
    ttft90, tpot90 = percentile(ttft), percentile(tpot)
    return {'offered_requests':offered,'completed_requests':len(done),'exact_requests':len(exact),
        'tpot_requests':len(tpot),'unfinished_or_failed_requests':offered-len(done),
        'offered_rps':offered/(window[1]-window[0]), 'completed_rps':completions/(window[1]-window[0]),
        'completions_in_window':completions, 'arrival_cohort_completion_cutoff_s':window[1],
        'exact_timing_coverage':coverage, 'p90_arrival_ttft_s':ttft90,'p90_request_mean_tpot_s':tpot90,
        'p90_client_queue_s':percentile([r['send_lateness_s'] for r in done]),
        'queue_growth_requests':growth,'max_engine_waiting':max(queue,default=None),
        'prompt_tokens':{'min':min((r['prompt_tokens'] for r in done),default=None),
                         'median':float(np.median([r['prompt_tokens'] for r in done])) if done else None,
                         'max':max((r['prompt_tokens'] for r in done),default=None)},
        'output_tokens':{'median':float(np.median([r['output_tokens'] for r in done])) if done else None,
                         'max':max((r['output_tokens'] for r in done),default=None)},
        'screen_pass': bool(coverage >= .99 and offered == len(done) and ttft90 is not None and ttft90 <= 1
                            and tpot90 is not None and tpot90 <= .1 and growth is not None and growth <= 1),
        'tail_guarantee':False,'scope':'finite-window request-level screen; sample counts and censoring retained'}


class ResidentAcquisition(Acquisition):
    async def episode(self, spec, workload, duration, migration=False):
        root = self.out/spec['episode'];root.mkdir(exist_ok=False)
        trajectories = [Trajectory(f"{spec['trace_id']}-resident-{i}", seq, offset, spec['seed'])
                        for i,(seq,offset) in enumerate(zip(workload['turn_sequences'],workload['turn_offset']))]
        resident = arrival_trace(spec['rate'],workload['cohort_counts'],spec['seed'],duration,burst=spec.get('burst',False))
        trace = [{**r,'cohort':'resident'} for r in resident]
        incoming = []
        if migration:
            for i in range(spec.get('width',8)):
                k=i%len(trajectories)
                incoming.append(Trajectory(f"{spec['trace_id']}-incoming-{i}",workload['turn_sequences'][k],workload['turn_offset'][k],spec['seed']))
            # Matched source-demand cadence uses the frozen simulator's source rate.
            incoming_rate = spec['incoming_session_rps']*len(incoming)
            trace += [{**r,'cohort':'incoming'} for r in arrival_trace(incoming_rate,[1]*len(incoming),spec['seed']+1,duration,60, spec.get('burst',False))]
        trace.sort(key=lambda r:(r['offset_s'],r['cohort'],r['session'],r['turn']))
        write(root/'offered-trace.json',trace)
        ready = [asyncio.Event() for _ in incoming]
        if spec['arm'] == 'control':
            for event in ready:event.set()
        histories = []
        if migration:
            for i,t in enumerate(incoming):
                context = int(t.sequence[t.offset]['context'])
                messages,code = self.history(f"{spec['trace_id']}-migration-{i}",max(2048,context))
                histories.append((messages,code))
                t.initial_history = self.render(messages,code)
            if spec['arm']=='control':
                # Materialize the actual incoming completion prefix before observation.
                with __import__('concurrent.futures',fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=len(incoming)) as executor:
                    futures=[]
                    for i,t in enumerate(incoming):
                        prompt=t.initial_history
                        body=serving.completion_payload(self.cfg.model,prompt,1,None,True)
                        body['cache_salt']=f"{spec['episode']}-incoming-{i}"
                        futures.append(executor.submit(serving._completion,self.cfg.host,self.cfg.sink_port,self.cfg.model,
                            prompt,1,None,min(120,self.remaining()),True,prepared_body=json.dumps(body)))
                    prewarm=[f.result() for f in futures]
                write(root/'control-prewarm.json',prewarm)
                if any(r['status']!=200 or not r['done'] for r in prewarm):raise RuntimeError('control materialization failed')
        # Establish continuing resident state before the prescribed traffic warmup/baseline.
        with __import__('concurrent.futures',fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=8) as executor:
            futures=[]
            for i,t in enumerate(trajectories):
                prompt,shape,_,_,_=t.prompt()
                prefix=prompt[:int(shape['context'])]
                body=serving.completion_payload(self.cfg.model,prefix,1,None,True)
                body['cache_salt']=f"{spec['episode']}-resident-{i}"
                futures.append(executor.submit(serving._completion,self.cfg.host,self.cfg.sink_port,self.cfg.model,
                    prefix,1,None,min(120,self.remaining()),True,prepared_body=json.dumps(body)))
            prewarm=[f.result() for f in futures]
        write(root/'resident-prewarm.json',prewarm)
        if any(r['status']!=200 or not r['done'] for r in prewarm):raise RuntimeError('resident materialization failed')
        metrics=serving.MetricsSampler(self.cfg.host,self.cfg.sink_port,root/'engine.csv',.5)
        power=profiler.PowerSampler(root/'power.csv',.5)
        metrics.start();power.start()
        epoch=time.monotonic_ns()+1_000_000_000
        boundary=epoch+int(duration*1e9)
        rows=[];tasks=[];migration_events=[]
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as client:
            async def one(item):
                scheduled_ns=epoch+int(item['offset_s']*1e9)
                tags={**spec,**item,'scheduled_ns':scheduled_ns,'episode_epoch_ns':epoch,
                      'server_queue_start_ns':None,'executed_tokens':None,'recomputed_tokens':None}
                t=(trajectories if item['cohort']=='resident' else incoming)[item['session']]
                row={**tags,'status':'unfinished','done':False,'start_ns':None}
                try:
                    await asyncio.sleep(max(0,(scheduled_ns-time.monotonic_ns())/1e9))
                    self.record(self.events,{**tags,'kind':'scheduled_arrival','client_wakeup_ns':time.monotonic_ns()})
                    if item['cohort']=='incoming':await ready[item['session']].wait()
                    async with t.lock:
                        if t.failed:
                            row.update(status='dependency_failed');return
                        prompt,shape,index,reset,added=t.prompt()
                        body=serving.completion_payload(self.cfg.model,prompt,int(shape['output']),None,True)
                        body['cache_salt']=f"{spec['episode']}-{item['cohort']}-{item['session']}"
                        prepared={'body':json.dumps(body),'prompt':prompt,'prompt_sha256':profiler.object_hash(prompt),
                            'index':item['turn'],'session':SimpleNamespace(session_id=t.label,append_tokens=shape['prompt'],
                                prefix_tokens=shape['context'],output_tokens=shape['output'])}
                        tags.update(recorded_turn=index,reset=reset,full_prompt_token_ids=prompt,
                            actual_new_tokens_excluding_retained_output=added,recorded_append_tokens=shape['prompt'],
                            client_dispatch_ns=time.monotonic_ns())
                        row.update(tags,start_ns=time.monotonic_ns())
                        self.record(self.events,{k:v for k,v in {**row,'kind':'request_dispatch'}.items() if k!='full_prompt_token_ids'})
                        result=await headroom.async_completion(client,self.cfg.host,self.cfg.sink_port,prepared,scheduled_ns,
                            max(.001,min(180,(boundary-time.monotonic_ns())/1e9,self.remaining())),
                            event_sink=lambda event:self.record(self.events,{**spec,'cohort':item['cohort'],
                                'session':item['session'],'turn':item['turn'],'scheduled_ns':scheduled_ns,**event}))
                        row={**tags,**result}
                        t.accept(prompt,row)
                except asyncio.CancelledError:
                    row.update(status='censored',end_ns=min(boundary,time.monotonic_ns()),cancellation='observation_boundary' if time.monotonic_ns()>=boundary else 'acquisition_interruption')
                    raise
                except Exception as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}',end_ns=time.monotonic_ns())
                    t.failed=True
                finally:
                    rows.append(row);self.record(self.requests,row)
            async def migrate():
                await asyncio.sleep(max(0,(epoch+60_000_000_000-time.monotonic_ns())/1e9))
                async def move_one(i,messages,code):
                    migration_events.append({'kind':'initial_start','session':i,'monotonic_ns':time.monotonic_ns(),'source_active':False})
                    if spec['arm']=='replay':
                        initial=await asyncio.to_thread(self.chat,messages,code,
                            {**spec,'phase':'initial','cohort':'migration','session':i},f"{spec['episode']}-incoming-{i}")
                        if initial['status']!='complete':raise RuntimeError('initial replay failed')
                        migration_events.append({'kind':'initial_end','session':i,'monotonic_ns':time.monotonic_ns(),'request_id':initial['request_id']})
                        migration_events.append({'kind':'catch_up_start','session':i,'monotonic_ns':time.monotonic_ns()})
                        result=await asyncio.to_thread(self.chat,self.append(messages,code,32),code,
                            {**spec,'phase':'catch_up','cohort':'migration','session':i},f"{spec['episode']}-incoming-{i}")
                        if result['status']!='complete':raise RuntimeError('destination catch-up failed')
                        migration_events.append({'kind':'catch_up_end','session':i,'monotonic_ns':time.monotonic_ns(),'request_id':result['request_id']})
                    migration_events.append({'kind':'destination_admission','session':i,'monotonic_ns':time.monotonic_ns(),
                        'source_quiescence_validated':False,'kv_transfer_validated':False})
                    ready[i].set()
                outcomes=await asyncio.gather(*(move_one(i,messages,code) for i,(messages,code) in enumerate(histories)),return_exceptions=True)
                for i,outcome in enumerate(outcomes):
                    if isinstance(outcome,BaseException):migration_events.append({'kind':'migration_failed','session':i,'error':str(outcome)})
            tasks=[asyncio.create_task(one(item)) for item in trace]
            mover=asyncio.create_task(migrate()) if migration else None
            try:
                await asyncio.sleep(max(0,(boundary-time.monotonic_ns())/1e9))
                for task in tasks:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                if mover:
                    if not mover.done():mover.cancel()
                    settled=await asyncio.gather(mover,return_exceptions=True)
                    if isinstance(settled[0],BaseException):migration_events.append({'kind':'migration_failed_or_censored','error':str(settled[0])})
            finally:
                metrics.close();power.close()
        windows=[(30,90)] if not migration else [(0,60),(60,90),(90,120),(120,150),(150,180)]
        summaries={cohort:{f'{a}-{z}':summarize([r for r in rows if r['cohort']==cohort],
            [r for r in trace if r['cohort']==cohort],epoch,(a,z),metrics.rows) for a,z in windows}
            for cohort in ('resident','incoming') if any(r['cohort']==cohort for r in trace)}
        result={'spec':spec,'epoch_ns':epoch,'boundary_ns':boundary,'summaries':summaries,
            'migration_events':migration_events,'scope':'destination-only replay contention; no active source GPU or paired KV; synthetic content, recorded lengths and causal retained history',
            'resident_population':len(trajectories),'incoming_population':len(incoming)}
        write(root/'result.json',result)
        # Cleanup is explicitly outside the recovery observation.
        stop=min(time.monotonic()+60,self.deadline)
        while time.monotonic()<stop:
            state=self.engine_metrics()
            if state['vllm:num_requests_running']==state['vllm:num_requests_waiting']==0:break
            await asyncio.sleep(.5)
        else:raise RuntimeError('engine did not clear cancelled requests during cleanup')
        return result

    def scout(self,plan):
        results=json.loads((self.out/'scout-results.json').read_text()) if (self.out/'scout-results.json').exists() else []
        for workload in ('coding','coding_long'):
            rate=plan['workloads'][workload]['initial_scout_rps_per_gpu']
            first = max((int(path.name.rsplit('-',1)[1]) for path in self.out.glob(f'scout-{workload}-*') if path.is_dir()),default=-1)+1
            for probe in range(first,4):
                self.remaining()
                spec={'episode':f'scout-{workload}-{probe}','trace_id':f'scout-{workload}-{probe}',
                      'seed':7101,'workload':workload,'rate':rate,'arm':'resident','probe':probe}
                result=asyncio.run(self.episode(spec,plan['workloads'][workload],90))
                results.append(result);write(self.out/'scout-results.json',results)
                screen=result['summaries']['resident']['30-90']
                print(spec['episode'],rate,json.dumps(screen),flush=True)
                if screen['exact_timing_coverage'] < .99:
                    raise RuntimeError('scout timing prerequisite failed; no service degradation inferred')
                rate*=2 if screen['screen_pass'] else .5
        selection={}
        for workload in ('coding','coding_long'):
            trials=[r for r in results if r['spec']['workload']==workload]
            stable=sorted({r['spec']['rate'] for r in trials if r['summaries']['resident']['30-90']['screen_pass']})
            tested=sorted({r['spec']['rate'] for r in trials})
            if len(tested)<2:raise RuntimeError('two distinct resident rates have not been tested')
            selection[workload]={'rates':[stable[0],stable[-1]] if len(stable)>=2 else tested[-2:],
                'stable_rates':stable,'boundary_bracketed':bool(stable) and any(
                    s['exact_timing_coverage']>=.99 and ((s['p90_arrival_ttft_s'] or 0)>1 or
                    (s['p90_request_mean_tpot_s'] or 0)>.1 or (s['queue_growth_requests'] or 0)>1)
                    for s in (r['summaries']['resident']['30-90'] for r in trials)),
                'validated_capacity_boundary':False}
        write(self.out/'resident-rate-selection.json',selection)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['scout','episodes','followups'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());a=ResidentAcquisition(args.out)
    write(args.out/f'{args.stage}-launch-{time.monotonic_ns()}.json',{'argv':__import__('sys').argv,'source_sha256':profiler.file_hash(Path(__file__)),
        'runtime_patch':json.loads((args.out/'stream-patch.json').read_text()) if (args.out/'stream-patch.json').exists() else None,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'plan_sha256':profiler.file_hash(args.plan),'start_ns':time.monotonic_ns()})
    if args.stage=='scout':a.scout(plan)
    else:
        import pool_shed_campaign as pool
        selection=json.loads((args.out/'resident-rate-selection.json').read_text())
        episodes = plan['main_episodes'] if args.stage=='episodes' else [
            {'workload':'coding_long','rate_slot':1,'seed':seed,'arm':'replay','width':16,
             'trace_id':f'width16-{seed}'} for seed in (7101,7102)] + [
            {'workload':'coding_long','rate_slot':1,'seed':7102,'arm':arm,'width':8,
             'trace_id':'burst-7102','burst':True} for arm in ('control','replay')]
        for index,row in enumerate(episodes):
            if row['arm']=='kv_transfer':continue
            reserve=630 if row.get('burst') and row['arm']=='control' else 300
            if a.remaining()<reserve:
                write(args.out/f'{args.stage}-budget-omissions.json',{'remaining_s':a.remaining(),
                    'required_reserve_s':reserve,'unstarted':episodes[index:],'status':'unmeasured_budget_limit'})
                break
            w=row['workload'];slot=row['rate_slot'];seed=row['seed']
            spec={**row,'episode':f"{args.stage}-{w}-{slot}-{row['arm']}-{seed}" + (f"-w{row.get('width',8)}" if args.stage=='followups' else '') + ('-burst' if row.get('burst') else ''),
                'rate':selection[w]['rates'][slot],'incoming_session_rps':pool.sample_fleet(w).metadata['source_session_rps']}
            result=asyncio.run(a.episode(spec,plan['workloads'][w],180,True))
            print(spec['episode'],'complete',flush=True)


if __name__=='__main__':main()
