"""Causal evolving-trajectory traffic and paired shared-service episodes."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import replace
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
import migration_testbed as testbed
import service_headroom_campaign as headroom
from pool_replay_measure import Acquisition, write


class Trajectory:
    def __init__(self, label, sequence, offset, seed):
        self.label, self.sequence, self.offset, self.seed = label, sequence, offset, seed
        self.completed, self.history = 0, []
        self.initial_history = None
        self.lock = asyncio.Lock()
        self.queue_lock = asyncio.Lock()
        self.failed = False
        self.admission = asyncio.Event()
        self.admission.set()
        self.route = None

    @asynccontextmanager
    async def owned(self):
        async with self.queue_lock:
            while True:
                await self.admission.wait()
                await self.lock.acquire()
                if self.admission.is_set():break
                self.lock.release()
            try:yield
            finally:self.lock.release()

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
        'p90_client_send_lateness_s':percentile([r['send_lateness_s'] for r in done]),
        'p90_client_queue_s':percentile([(r['client_dispatch_ns']-r['client_wakeup_ns'])/1e9 for r in done if 'client_wakeup_ns' in r]),
        'p90_client_schedule_lateness_s':percentile([(r['client_wakeup_ns']-r['scheduled_ns'])/1e9 for r in done if 'client_wakeup_ns' in r]),
        'queue_growth_requests':growth,'max_engine_waiting':max(queue,default=None),
        'prompt_tokens':{'min':min((r['prompt_tokens'] for r in done),default=None),
                         'median':float(np.median([r['prompt_tokens'] for r in done])) if done else None,
                         'max':max((r['prompt_tokens'] for r in done),default=None)},
        'output_tokens':{'median':float(np.median([r['output_tokens'] for r in done])) if done else None,
                         'max':max((r['output_tokens'] for r in done),default=None)},
        'screen_pass': bool(coverage >= .99 and offered == len(done) and ttft90 is not None and ttft90 <= 1
                            and tpot90 is not None and tpot90 <= .1 and growth is not None and growth <= 1),
        'tail_guarantee':False,'scope':'finite-window request-level screen; sample counts and censoring retained'}


def physical_workload(workload, seed, count=8):
    supported = [i for i,seq in enumerate(workload['turn_sequences']) if max(r['context']+r['prompt']+r['output'] for r in seq)+512 <= 32768]
    if not supported:raise ValueError('no frozen trajectories support full-history 512-token migration probes')
    indices = random.Random(seed).choices(supported, weights=[workload['cohort_counts'][i] for i in supported], k=count)
    return {**workload, 'turn_sequences':[workload['turn_sequences'][i] for i in indices],
            'turn_offset':[workload['turn_offset'][i] for i in indices], 'cohort_counts':[1]*count,
            'sampled_trajectory_indices':indices,'migration_context_excluded_indices':[i for i in range(len(workload['turn_sequences'])) if i not in supported],
            'migration_context_exclusion':'entire trajectory must leave 512 output tokens; no truncation'}


def service_degraded(screen):
    return screen['exact_timing_coverage']>=.99 and any((screen[key] or 0)>limit for key,limit in
        (('p90_arrival_ttft_s',1),('p90_request_mean_tpot_s',.1),('queue_growth_requests',1)))


def cache_idle(status):
    if not status['is_healthy']:raise RuntimeError('LMCache status is unhealthy')
    state=status['storage_manager']
    return all(state[group][key]==0 for group,keys in (
        ('l1_manager',('write_locked_count','read_locked_count','temporary_count')),
        ('store_controller',('pending_keys_count','in_flight_task_count')),
        ('prefetch_controller',('submission_queue_size','pending_queue_size','in_flight_request_count')))
        for key in keys)


class ResidentAcquisition(Acquisition):
    def __init__(self, out, inventory=None):
        super().__init__(out)
        self.inventory = inventory
        if inventory:
            from pool_replay_paired import validate_inventory
            self.cfg = validate_inventory(inventory)
            self.deadline = time.monotonic() + (inventory['deadline_wall_ns']-time.time_ns())/1e9

    async def episode(self, spec, workload, duration, migration=False):
        if migration and not self.inventory:raise ValueError('paired source/destination inventory required')
        if migration and self.cfg.src_port == self.cfg.sink_port:raise ValueError('distinct source and destination required')
        root = self.out/spec['episode'];root.mkdir(exist_ok=False)
        workload = physical_workload(workload,spec['seed'])
        write(root/'physical-workload.json',workload)
        trajectories = [Trajectory(f"{spec['trace_id']}-resident-{i}",seq,offset,spec['seed'])
                        for i,(seq,offset) in enumerate(zip(workload['turn_sequences'],workload['turn_offset']))]
        incoming = [Trajectory(f"{spec['trace_id']}-incoming-{i}",workload['turn_sequences'][i%8],
                              workload['turn_offset'][i%8],spec['seed']) for i in range(spec.get('width',8))] if migration else []
        for t in trajectories:t.route=self.cfg.sink_port
        for t in incoming:t.route=self.cfg.src_port
        trace = [{**r,'cohort':'resident'} for r in arrival_trace(spec['rate'],[1]*8,spec['seed'],duration,burst=spec.get('burst',False))]
        if migration:
            trace += [{**r,'cohort':'incoming'} for r in arrival_trace(spec['incoming_session_rps']*len(incoming),
                       [1]*len(incoming),spec['seed']+1,duration,burst=spec.get('burst',False))]
        trace.sort(key=lambda r:(r['offset_s'],r['cohort'],r['session'],r['turn']))
        write(root/'offered-trace.json',trace)
        rows=[];migration_events=[];epoch=0;boundary=0
        layout=testbed.mp_model_layout(Path(self.inventory['stack_root'])/'lmcache-sink.log') if migration else None
        def event(kind, **fields):
            row={'kind':kind,'monotonic_ns':time.monotonic_ns(),**fields}
            migration_events.append(row);self.record(self.events,{**spec,**row})
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0,force_close=True)) as client:
            ports=(self.cfg.src_port,self.cfg.sink_port) if self.inventory else (self.cfg.sink_port,)
            before={str(port):serving.parse_metrics(testbed.http_text(self.cfg.host,port,'GET','/metrics')) for port in ports}
            if any(r['vllm:num_requests_running'] or r['vllm:num_requests_waiting'] for r in before.values()):raise RuntimeError('cache isolation requires idle engines before episode')
            cache_samples=[]
            async def cache_states():
                values={}
                for port in (self.cfg.src_lmc_http_port,self.cfg.sink_lmc_http_port):
                    async with client.get(f'http://{self.cfg.host}:{port}/status',timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status!=200:raise RuntimeError('LMCache status unavailable before cache isolation')
                        values[str(port)]=await response.json()
                cache_samples.append({'monotonic_ns':time.monotonic_ns(),'states':values})
                write(root/'cache-isolation-status.json',cache_samples)
                return values
            if self.inventory:
                stop=min(time.monotonic()+30,self.deadline);quiet=0
                while time.monotonic()<stop:
                    quiet=quiet+1 if all(cache_idle(v) for v in (await cache_states()).values()) else 0
                    if quiet==3:break
                    await asyncio.sleep(.5)
                else:raise TimeoutError('LMCache work did not quiesce before force-clear')
            cleared=[]
            endpoints=[(port,'/reset_prefix_cache') for port in ports]
            if self.inventory:endpoints += [(port,'/cache/clear') for port in (self.cfg.src_lmc_http_port,self.cfg.sink_lmc_http_port)]
            for port,path in endpoints:
                async with client.post(f'http://{self.cfg.host}:{port}{path}',timeout=aiohttp.ClientTimeout(total=30)) as response:
                    body=await response.text();cleared.append({'port':port,'path':path,'status':response.status,'body':body})
                    if response.status!=200:raise RuntimeError(f'cache isolation failed: {cleared[-1]}')
            if self.inventory:
                reader,writer=await asyncio.wait_for(asyncio.open_connection(self.cfg.host,self.cfg.lmc_port),10)
                writer.write(b'*1\r\n$8\r\nFLUSHALL\r\n');await writer.drain()
                reply=await asyncio.wait_for(reader.readline(),10)
                writer.close();await writer.wait_closed()
                if reply!=b'+OK\r\n':raise RuntimeError('isolated campaign Redis FLUSHALL failed')
                cleared.append({'redis_flushall':reply.decode()})
                after_cache=await cache_states()
                if any(not cache_idle(v) or v['storage_manager']['l1_manager']['total_object_count'] or v['storage_manager']['l1_manager']['memory_used_bytes'] for v in after_cache.values()):
                    raise RuntimeError('LMCache retained objects or asynchronous work after clear')
            write(root/'pre-episode-cache-isolation.json',{'before':before,'responses':cleared,
                'after':{str(port):serving.parse_metrics(testbed.http_text(self.cfg.host,port,'GET','/metrics')) for port in ports},
                'scope':'outside observation, before any resident or incoming materialization'})
            async def request(t,prompt,output,port,tags,bypass):
                if len(prompt)+output>self.cfg.max_model_len:raise ValueError('full retained context plus generation exceeds runtime limit')
                body=serving.completion_payload(self.cfg.model,prompt,output,None,bypass)
                if tags['cohort']=='migration' and output==512:body['ignore_eos']=False
                prepared={'body':json.dumps(body),'prompt':prompt,'prompt_sha256':profiler.object_hash(prompt),
                    'index':tags.get('turn',-1),'session':SimpleNamespace(session_id=t.label,
                    append_tokens=tags.get('recorded_append_tokens',0),prefix_tokens=len(prompt),output_tokens=output)}
                now=time.monotonic_ns()
                tags={**spec,**tags,'route_port':port,'serving_role':'source' if port==self.cfg.src_port else 'destination',
                    'client_dispatch_ns':now,'scheduled_ns':tags.get('scheduled_ns',now),'episode_epoch_ns':epoch,
                    'full_prompt_token_ids':prompt,'context_hash':profiler.object_hash(prompt),'server_queue_start_ns':None,
                    'executed_tokens':None,'recomputed_tokens':None,'request_path':'full_retained_token_completion',
                    'probe_deviation':'service adapter preserves exact trajectory token history; not unchanged chat state-code probe'}
                self.record(self.events,{k:v for k,v in {**tags,'kind':'request_dispatch'}.items() if k!='full_prompt_token_ids'})
                row={**tags,'status':'unfinished','done':False,'start_ns':now}
                active={'dispatch_ns':now,'request_id':None,'first_token_ns':None}
                if tags['cohort']=='incoming':t.active=active
                def token_event(e):
                    data=e.get('data','')
                    if data and data!='[DONE]':
                        payload=json.loads(data)
                        active['request_id']=payload.get('id') or active['request_id']
                        if any(c.get('token_ids') for c in payload.get('choices',[])) and active['first_token_ns'] is None:active['first_token_ns']=e['monotonic_ns']
                    self.record(self.events,{**spec,'cohort':tags['cohort'],'session':tags['session'],
                        'turn':tags.get('turn',-1),'phase':tags['phase'],'route_port':port,**e})
                try:
                    result=await headroom.async_completion(client,self.cfg.host,port,prepared,tags['scheduled_ns'],
                        max(.001,min(180,self.remaining(),(boundary-now)/1e9 if boundary else 180)),
                        event_sink=token_event)
                    row.update(result,server_status=result['status'] or None,transport_error=result['error'] or None)
                    row.update(derived_prompt_minus_cache_tokens=None if row.get('cached_tokens') is None else row['prompt_tokens']-row['cached_tokens'],
                               processed_tokens_basis='derived_prompt_minus_cache',external_cache_bypassed=bypass)
                    if not row['done'] or row['status']!=200 or row['prompt_tokens']!=len(prompt) or row['recorded_output_tokens']!=row['output_tokens'] or not 0<row['output_tokens']<=output or (tags['phase']=='service' and row['output_tokens']!=output):
                        raise RuntimeError('request did not complete exact requested input/output state')
                    return row
                except asyncio.CancelledError:
                    row.update(status='censored',done=False,end_ns=time.monotonic_ns(),cancellation='observation_boundary');raise
                except Exception as exc:
                    row.update(status='failed',done=False,error=f'{type(exc).__name__}: {exc}',end_ns=time.monotonic_ns());raise
                finally:
                    if tags['cohort']=='incoming':t.active=None;t.last_request=row
                    rows.append(row);self.record(self.requests,row)

            async def materialize(t,i,prompt,port,phase,output=512,bypass=False):
                return await request(t,list(prompt),output,port,{'cohort':'migration','session':i,'phase':phase},bypass)

            async def prewarm(t,i,cohort):
                prompt,shape,_,_,_=t.prompt()
                t.history=list(prompt[:int(shape['context'])])
                t.initial_history=list(t.history)
                await request(t,t.history or prompt,1,t.route,{'cohort':'materialization','session':i,'phase':f'{cohort}_prewarm'},cohort=='resident')
                if cohort=='incoming' and spec['arm']=='control':
                    await materialize(t,i,t.history or prompt,self.cfg.sink_port,'control_preepoch_materialization',1,True)
            await asyncio.gather(*(prewarm(t,i,c) for c,ts in (('resident',trajectories),('incoming',incoming)) for i,t in enumerate(ts)))
            if self.remaining()<duration+60:raise TimeoutError('insufficient full observation and cleanup reserve after materialization')
            metrics=serving.MetricsSampler(self.cfg.host,self.cfg.sink_port,root/'engine.csv',.5)
            source_metrics=serving.MetricsSampler(self.cfg.host,self.cfg.src_port,root/'engine-source.csv',.5) if migration else None
            power=profiler.PowerSampler(root/'power.csv',.5)
            metrics.start();power.start()
            if source_metrics:source_metrics.start()
            epoch=time.monotonic_ns()+1_000_000_000;boundary=epoch+int(duration*1e9)
            async def one(item):
                scheduled_ns=epoch+int(item['offset_s']*1e9)
                t=(trajectories if item['cohort']=='resident' else incoming)[item['session']]
                tags={**item,'scheduled_ns':scheduled_ns,'phase':'service','client_wakeup_ns':None}
                dispatched=False
                try:
                    await asyncio.sleep(max(0,(scheduled_ns-time.monotonic_ns())/1e9))
                    tags['client_wakeup_ns']=time.monotonic_ns()
                    self.record(self.events,{**spec,**tags,'kind':'scheduled_arrival'})
                    async with t.owned():
                        if t.failed:raise RuntimeError('incomplete trajectory dependency')
                        prompt,shape,index,reset,added=t.prompt()
                        tags.update(recorded_turn=index,reset=reset,actual_new_tokens_excluding_retained_output=added,
                                    recorded_append_tokens=shape['prompt'],retained_context_hash=profiler.object_hash(t.history))
                        dispatched=True
                        result=await request(t,prompt,int(shape['output']),t.route,tags,item['cohort']=='resident' or (t.route==self.cfg.sink_port and spec['arm']!='kv_transfer'))
                        t.accept(prompt,result)
                        if item['cohort']=='incoming' and t.route==self.cfg.sink_port:
                            event('first_valid_destination_response' if not getattr(t,'continued',False) else 'destination_response',
                                  session=item['session'],request_id=result['request_id'],context_hash=profiler.object_hash(prompt))
                            t.continued=True
                        mirror=list(t.history) if item['cohort']=='incoming' and t.route==self.cfg.src_port and spec['arm']=='control' else None
                    if mirror is not None:await materialize(t,item['session'],mirror,self.cfg.sink_port,'control_baseline_materialization',1,True)
                except asyncio.CancelledError:
                    if not dispatched:
                        row={**spec,**tags,'status':'censored','done':False,'end_ns':boundary,'cancellation':'observation_boundary',
                             'route_port':t.route,'ownership':'source' if t.route==self.cfg.src_port else 'destination'}
                        rows.append(row);self.record(self.requests,row)
                    raise
                except Exception as exc:
                    t.failed=True
                    if not dispatched:
                        row={**spec,**tags,'status':'dependency_failed','done':False,'end_ns':time.monotonic_ns(),'error':str(exc)}
                        rows.append(row);self.record(self.requests,row)
                    event('service_failed',session=item['session'],cohort=item['cohort'],error=str(exc))

            async def copy(t,i,prompt,phase):
                event(f'{phase}_start',session=i,context_hash=profiler.object_hash(prompt),context_tokens=len(prompt),full_prompt_token_ids=prompt)
                if spec['arm']=='kv_transfer':
                    # Generated output is part of state: export its newly sealed blocks before retrieval.
                    await materialize(t,i,prompt,self.cfg.src_port,f'{phase}_source_export')
                    url=f'http://{self.cfg.host}:{self.cfg.sink_lmc_http_port}/cache/prefetches'
                    async with client.post(url,json={'model_name':layout[0],'world_size':layout[1],'token_ids':prompt},timeout=aiohttp.ClientTimeout(total=min(30,self.remaining()))) as response:
                        if response.status!=202:raise RuntimeError(f'prefetch HTTP {response.status}')
                        submitted=await response.json()
                    while True:
                        async with client.get(url+'/'+submitted['request_id'],timeout=aiohttp.ClientTimeout(total=min(30,self.remaining()))) as response:
                            if response.status!=200:raise RuntimeError(f'prefetch status HTTP {response.status}')
                            warm=await response.json()
                        if warm['status']=='completed':break
                        if warm['status'] in ('failed','error'):raise RuntimeError(f'prefetch failed: {warm}')
                        await asyncio.sleep(.05)
                    event('kv_prefetch',session=i,phase=phase,**warm)
                    if not 0 <= warm['found_keys'] <= warm['total_keys']:raise RuntimeError('inconsistent warm-prefetch cache telemetry')
                sink_log=Path(self.inventory['stack_root'])/'lmcache-sink.log'
                log_offset=sink_log.stat().st_size
                result=await materialize(t,i,prompt,self.cfg.sink_port,phase,512,spec['arm']!='kv_transfer')
                if spec['arm']=='kv_transfer' and (phase=='initial' or result['request_id'] in testbed.read_after(sink_log,log_offset)):
                    external=testbed.mp_request_hit(sink_log,log_offset,result['request_id'],False,testbed.model_chunk_tokens(self.cfg),require_l1=False,
                        event_sink=lambda tiers:event('cache_request_tiers',session=i,phase=phase,request_id=result['request_id'],**tiers))
                    if phase=='initial' and external<=0:raise RuntimeError('KV initial request lacks external retrieval evidence')
                if spec['arm']=='kv_transfer' and result.get('cached_tokens') is None:raise RuntimeError('KV destination omitted cache telemetry')
                event(f'{phase}_end',session=i,context_hash=profiler.object_hash(prompt),context_tokens=len(prompt),
                      request_id=result['request_id'],cached_tokens=result.get('cached_tokens'))

            async def move_one(i,t):
                initial=list(t.history)
                event('snapshot',session=i,context_hash=profiler.object_hash(initial),context_tokens=len(initial),generation=t.completed)
                try:
                    if spec['arm']!='control':await copy(t,i,initial,'initial')
                    t.admission.clear()
                    event('pause',session=i,source_request_executing=bool(getattr(t,'active',None)),
                          in_flight_request=dict(t.active) if getattr(t,'active',None) else None,
                          source_decode_observed=bool(getattr(t,'active',None) and t.active['first_token_ns']),generation=t.completed)
                    async with t.lock:
                        if t.failed:raise RuntimeError('source dependency failed before switch')
                        actual=list(t.history);generation=t.completed
                        event('source_idle',session=i,context_hash=profiler.object_hash(actual),context_tokens=len(actual),
                              generation=generation,last_source_request={k:getattr(t,'last_request',{}).get(k) for k in ('request_id','first_ns','last_token_ns','end_ns')},actual_context_growth_tokens=len(actual)-len(initial),reset_since_snapshot=actual[:len(initial)]!=initial)
                        if spec['arm']=='control':
                            await materialize(t,i,actual,self.cfg.sink_port,'control_final_materialization',1,True)
                        else:await copy(t,i,actual,'catch_up')
                        if actual!=t.history or generation!=t.completed:raise RuntimeError('source state changed before ownership switch')
                        t.route=self.cfg.sink_port
                        event('route_switch',session=i,context_hash=profiler.object_hash(actual),context_tokens=len(actual),
                              generation=generation,source_port=self.cfg.src_port,destination_port=self.cfg.sink_port,
                              validation='exact retained token history and generation; continuation reported separately')
                        t.admission.set()
                except Exception as exc:
                    event('migration_failed',session=i,error=f'{type(exc).__name__}: {exc}')
                    t.failed=True

            async def migrate():
                await asyncio.sleep(max(0,(epoch+60_000_000_000-time.monotonic_ns())/1e9))
                await asyncio.gather(*(move_one(i,t) for i,t in enumerate(incoming)))
            tasks=[asyncio.create_task(one(item)) for item in trace]
            mover=asyncio.create_task(migrate()) if migration else None
            try:
                await asyncio.sleep(max(0,(boundary-time.monotonic_ns())/1e9))
            finally:
                for task in tasks+([mover] if mover else []):
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,*([mover] if mover else []),return_exceptions=True)
                metrics.close();power.close()
                if source_metrics:source_metrics.close()
        windows=[(30,90)] if not migration else [(a,min(a+30,duration)) for a in range(0,int(duration),30)]
        summaries={cohort:{f'{a}-{z}':summarize([r for r in rows if r['cohort']==cohort],
            [r for r in trace if r['cohort']==cohort],epoch,(a,z),metrics.rows) for a,z in windows}
            for cohort in ('resident','incoming') if any(r['cohort']==cohort for r in trace)}
        checkpoints={str(offset):{cohort:sum(r['cohort']==cohort and r['offset_s']<=60+offset for r in trace)-
            sum(r['cohort']==cohort and r.get('done') and r.get('status')==200 and r['scheduled_ns']<=epoch+int((60+offset)*1e9)
                and r['end_ns']<=epoch+int((60+offset)*1e9) for r in rows) for cohort in ('resident','incoming')}
            for offset in (30,120)} if migration else {}
        result={'spec':spec,'epoch_ns':epoch,'boundary_ns':boundary,'summaries':summaries,
            'migration_events':migration_events,'outstanding_at_migration_offset_s':checkpoints,
            'scope':'full exact-token evolving trajectory service adapter; original chat probes retained in separate controlled KV measurements',
            'arrival_schedule':'assumed independent per-session periodic offsets; causality and queued arrivals preserved',
            'control_materialization':'initial preepoch and updated baseline/final context explicitly recorded; extra control baseline cost is not resident service',
            'resident_population':len(trajectories),'incoming_population':len(incoming),
            'placement':{'resident':self.cfg.sink_port,'incoming_before_switch':self.cfg.src_port,'incoming_after_switch':self.cfg.sink_port}}
        write(root/'result.json',result)
        stop=min(time.monotonic()+60,self.deadline)
        while time.monotonic()<stop:
            ports=(self.cfg.src_port,self.cfg.sink_port) if migration else (self.cfg.sink_port,)
            states=[serving.parse_metrics(testbed.http_text(self.cfg.host,port,'GET','/metrics')) for port in ports]
            if all(state['vllm:num_requests_running']==state['vllm:num_requests_waiting']==0 for state in states):break
            await asyncio.sleep(.5)
        else:raise RuntimeError('engine did not clear cancelled requests during cleanup')
        return result

    def scout(self,plan,workloads=('coding_long','coding')):
        results=json.loads((self.out/'scout-results.json').read_text()) if (self.out/'scout-results.json').exists() else []
        for workload in workloads:
            prior=[r for r in results if r['spec']['workload']==workload]
            if any(r['summaries']['resident']['30-90']['screen_pass'] for r in prior) and any(service_degraded(r['summaries']['resident']['30-90']) for r in prior):continue
            rate=(prior[-1]['spec']['rate']*(.5 if service_degraded(prior[-1]['summaries']['resident']['30-90']) else 2)) if prior else plan['workloads'][workload]['initial_scout_rps_per_gpu']
            first = max((int(path.name.rsplit('-',1)[1]) for path in self.out.glob(f'scout-{workload}-*') if path.is_dir()),default=-1)+1
            for probe in range(first,4):
                if self.remaining()<150:raise TimeoutError('insufficient reserve for scout and cleanup')
                spec={'episode':f'scout-{workload}-{probe}','trace_id':f'scout-{workload}',
                      'seed':7101,'workload':workload,'rate':rate,'arm':'resident','probe':probe}
                result=asyncio.run(self.episode(spec,plan['workloads'][workload],90))
                results.append(result);write(self.out/'scout-results.json',results)
                screen=result['summaries']['resident']['30-90']
                print(spec['episode'],rate,json.dumps(screen),flush=True)
                if screen['completed_requests'] and screen['exact_timing_coverage'] < .99:
                    raise RuntimeError('scout timing prerequisite failed; no service degradation inferred')
                seen=[r for r in results if r['spec']['workload']==workload]
                if any(r['summaries']['resident']['30-90']['screen_pass'] for r in seen) and any(service_degraded(r['summaries']['resident']['30-90']) for r in seen):break
                rate*=.5 if service_degraded(screen) else 2
        selection={}
        for workload in ('coding_long','coding'):
            trials=[r for r in results if r['spec']['workload']==workload]
            stable=sorted({r['spec']['rate'] for r in trials if r['summaries']['resident']['30-90']['screen_pass']})
            tested=sorted({r['spec']['rate'] for r in trials})
            if len(tested)<2:raise RuntimeError('two distinct resident rates have not been tested')
            selection[workload]={'rates':[stable[-1] if stable else tested[0]],'selected_rate_rps':stable[-1] if stable else tested[0],
                'selected_rate_screen_pass':bool(stable),'selection_basis':'heaviest tested passing screen; lowest measured rate if no passing point',
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
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--workload',choices=('coding','coding_long'))
    args=parser.parse_args();plan=json.loads(args.plan.read_text());a=ResidentAcquisition(args.out,json.loads(args.inventory.read_text()) if args.inventory else None)
    write(args.out/f'{args.stage}-launch-{time.monotonic_ns()}.json',{'argv':__import__('sys').argv,'source_sha256':profiler.file_hash(Path(__file__)),
        'runtime_patch':json.loads((args.out/'stream-patch.json').read_text()) if (args.out/'stream-patch.json').exists() else None,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'plan_sha256':profiler.file_hash(args.plan),'start_ns':time.monotonic_ns()})
    if args.stage=='scout':a.scout(plan,(args.workload,) if args.workload else ('coding_long','coding'))
    else:
        import pool_shed_campaign as pool
        selection=json.loads((args.out/'resident-rate-selection.json').read_text())
        episodes = plan['main_episodes'] if args.stage=='episodes' else [
            {'workload':'coding_long','rate_slot':1,'seed':seed,'arm':'replay','width':16,
             'trace_id':f'width16-{seed}'} for seed in (7101,7102)] + [
            {'workload':'coding_long','rate_slot':1,'seed':7102,'arm':arm,'width':8,
             'trace_id':'burst-7102','burst':True} for arm in ('control','replay')]
        for index,row in enumerate(episodes):
            reserve=420
            if a.remaining()<reserve:
                write(args.out/f'{args.stage}-budget-omissions.json',{'remaining_s':a.remaining(),
                    'required_reserve_s':reserve,'unstarted':episodes[index:],'status':'unmeasured_budget_limit'})
                break
            w=row['workload'];slot=row['rate_slot'];seed=row['seed']
            spec={**row,'episode':f"{args.stage}-{w}-{slot}-{row['arm']}-{seed}" + (f"-w{row.get('width',8)}" if args.stage=='followups' else '') + ('-burst' if row.get('burst') else ''),
                'rate':selection[w]['rates'][slot],'incoming_session_rps':pool.sample_fleet(w).metadata['source_session_rps']}
            result=asyncio.run(a.episode(spec,plan['workloads'][w],300,True))
            print(spec['episode'],'complete',flush=True)


if __name__=='__main__':main()
