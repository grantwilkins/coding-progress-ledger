import collections, gzip, hashlib, json, pathlib, statistics, sys
root=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2])
plan=json.loads((root/'plan.json').read_text())
def read(path):return gzip.open(path,'rt') if path.suffix=='.gz' else path.open()
client_path=root/'requests.jsonl'
if not client_path.exists():client_path=root/'requests.jsonl.gz'
clients=[json.loads(line) for line in read(client_path)]
warm={c['id']:c for c in plan['warm_decode_contract']['cells']}
selected=[r for r in clients if (r.get('cell') in warm and r.get('phase')=='measurement') or (r.get('episode') and r.get('phase')=='service')]
external={r['request_id'] for r in selected if r.get('request_id')}
mapping=collections.defaultdict(set);registrations={}; files=sorted(p for d in (root/'stack/timing',root/'destination/timing') for p in d.iterdir() if p.name.endswith(('.jsonl','.jsonl.gz')) and not (p.suffix=='.gz' and p.with_suffix('').exists()))
assert files,'no timing logs'
processes={}; integrity=[]
for path in files:
 rows=map(json.loads,read(path)); first=next(rows); seq=first['sequence']; last=first; modules=[]; processes[str(path)]=first
 assert first['kind']=='process_start'
 for r in rows:
  if r['sequence']!=seq+1:integrity.append({'file':str(path),'sequence_gap':[seq,r['sequence']]})
  seq=r['sequence'];last=r
  if r['kind']=='module':modules.append(r)
  if r['kind']=='frontend_registration' and (r['external_request_id'] in external or r['external_request_id'].removesuffix('-0') in external):mapping[r['external_request_id']].add(r['request_id']);registrations[r['request_id']]=(r,str(path))
 if last['kind']!='process_final' or last.get('records_before_final')!=seq-1 or last.get('dropped_records')!=0:integrity.append({'file':str(path),'missing_final':True})
 if not modules:integrity.append({'file':str(path),'missing_modules':True})
lookup={e: mapping.get(e+'-0',set()) or mapping.get(e,set()) for e in external}
wanted=set().union(*lookup.values()) if lookup else set()
scheduled=collections.defaultdict(list); generated=collections.defaultdict(list); frontend={k:collections.defaultdict(list) for k in ('frontend_receipt','collector_put','collector_pop')}; batches={}; qviolations=[]
for path in files:
 for line in read(path):
  r=json.loads(line);kind=r['kind']
  if kind=='schedule':
   if sum(x['scheduled_tokens'] for x in r['requests'])>8192:qviolations.append(r['iteration'])
   for x in r['requests']:
    if x['request_id'] in wanted and x['scheduled_tokens']:
     scheduled[x['request_id']].append(dict(x,iteration=r['iteration'],start_ns=r['start_ns'],process=str(path),prefill_q=min(x['scheduled_tokens'],max(x['prompt_tokens']-x['computed_before'],0))))
  elif kind=='worker_output':
   if any(x['request_id'] in wanted for x in r['requests']):batches[r['iteration']]=r
   for x in r['requests']:
    if x['request_id'] in wanted:
     for i,t in enumerate(x['token_ids']):generated[x['request_id']].append((x['ordinal_start']+i,t,r['output_ready_ns'],r['iteration'],str(path)))
  elif kind in frontend and r['request_id'] in wanted:
   frontend[kind][r['request_id']].extend((r['ordinal_start']+i,t,r['mono_ns'],str(path)) for i,t in enumerate(r['token_ids']))
def stats(values):
 if not values:return {'n':0}
 s=sorted(values);pos=.9*(len(s)-1);lo=int(pos)
 return {'n':len(s),'min':s[0],'median':statistics.median(s),'p90':s[lo]+(s[min(lo+1,len(s)-1)]-s[lo])*(pos-lo),'max':s[-1]}
def resolve(r):
 ids=lookup.get(r.get('request_id'),set())
 return next(iter(ids)) if len(ids)==1 else None
def domain(path):
 p=processes[path];d=tuple(p.get(k) for k in ('host','boot_id','time_namespace','time_namespace_offsets'));return d if all(x is not None for x in d) else None
def details(r):
 internal=resolve(r);ss=scheduled[internal];gg=sorted(generated[internal]);out={k:r.get(k) for k in ('request_id','session_id','session','turn','cohort','episode','status','done','output_tokens','recorded_output_tokens','planned_output_tokens','cached_tokens','mean_tpot_s','ttft_s')};out['internal_id']=internal
 out['prefill_q']=sum(x['prefill_q'] for x in ss);out['scheduled_q']=sum(x['scheduled_tokens'] for x in ss)
 out['native_cache_evidence']=[x['native_cached_tokens'] for x in ss if x.get('native_cached_tokens') is not None]
 out['external_cache_evidence']=sorted(set(x['external_cached_tokens'] for x in ss if x.get('external_cached_tokens') is not None));out['max_preemptions']=max((x['preemptions'] for x in ss),default=None)
 tokens=[x[1] for x in gg];ordinal=[x[0] for x in gg]
 out['generated_count']=len(tokens);out['worker_client_tokens_match']=tokens==r.get('token_ids',[]);out['ordinal_contiguous']=ordinal==list(range(len(gg)));out['ready_monotonic']=all(a[2]<=b[2] for a,b in zip(gg,gg[1:]))
 out['boundary_matches']={k:[(x[0],x[1]) for x in sorted(v[internal])]==[(x[0],x[1]) for x in gg] for k,v in frontend.items()}
 out['server_ready_mean_tpot_s']=(gg[-1][2]-gg[0][2])/1e9/(len(gg)-1) if len(gg)>1 else None
 out['all_iterations_joined']=all(x['iteration'] in batches for x in ss)
 out['server_worker_processes']=sorted(set(x[4] for x in gg))
 out['arrival_ttft_s']=(r['first_ns']-r['scheduled_ns'])/1e9 if r.get('first_ns') and 'scheduled_ns' in r else None
 out['client_causal_queue_s']=(r['start_ns']-r['scheduled_ns'])/1e9 if 'start_ns' in r and 'scheduled_ns' in r else None
 if internal in registrations and ss and gg:
  reg,rp=registrations[internal];first=min(ss,key=lambda x:x['start_ns'])
  if domain(rp) is not None and domain(rp)==domain(first['process'])==domain(gg[0][4]):
   out['server_registration_to_first_schedule_s']=(first['start_ns']-reg['mono_ns'])/1e9
   out['server_first_schedule_to_first_ready_s']=(gg[0][2]-first['start_ns'])/1e9
   out['server_registration_to_first_ready_s']=(gg[0][2]-reg['mono_ns'])/1e9
  for kind in ('frontend_receipt','collector_pop'):
   tokens=sorted(frontend[kind][internal])
   if tokens and domain(tokens[0][3]) is not None and domain(tokens[0][3])==domain(gg[0][4]):out['first_ready_to_'+kind+'_s']=(tokens[0][2]-gg[0][2])/1e9
 return out
prefill_by_iteration=collections.Counter()
for values in scheduled.values():
 for row in values:prefill_by_iteration[row['iteration']]+=row['prefill_q']
cells=[]
for key,c in warm.items():
 rows=[r for r in selected if r.get('cell')==key];rr=[details(r) for r in rows]; reasons=[]
 if len(rows)!=c['concurrency']:reasons.append('measurement_request_count')
 for r,d in zip(rows,rr):
  checks={'internal_id':bool(d['internal_id']),'request_complete':r.get('status')==200 and r.get('done') and not r.get('error'),'exact_output':r.get('output_tokens')==r.get('recorded_output_tokens')==len(r.get('token_ids',[]))==1536,'prompt':r.get('prompt_tokens')==c['context_tokens'],'usage_native':(r.get('cached_tokens') or -1)>=c['context_tokens']-32,'scheduler_native':bool(d['native_cache_evidence']) and min(d['native_cache_evidence'])>=c['context_tokens']-32,'external_zero':d['external_cache_evidence']==[0],'no_preemptions':d['max_preemptions']==0,'prefill32':d['prefill_q']==32,'scheduled1567':d['scheduled_q']==1567,'generated_exact':d['generated_count']==1536,'ready_monotonic':d['ready_monotonic'],'one_worker_process':len(d['server_worker_processes'])==1,'tokens_joined':d['worker_client_tokens_match'] and d['ordinal_contiguous'] and all(d['boundary_matches'].values()) and d['all_iterations_joined']}
  d['checks']=checks;d['certified']=all(checks.values());d['native_cache_evidence']=sorted(set(d['native_cache_evidence']))
  reasons.extend(f"{r['session_id']}:{k}" for k,v in checks.items() if not v)
 ids={resolve(r) for r in rows};iterations={x['iteration'] for i in ids for x in scheduled[i]};batch_rows=[batches[i] for i in iterations if i in batches]
 cells.append({'cell':key,'request_evidence_complete':not reasons,'certified':not reasons and not integrity and not qviolations,'reasons':reasons,'request_count':len(rows),'client_tpot_s':stats([r['mean_tpot_s'] for r in rows if r.get('mean_tpot_s') is not None]),'server_ready_tpot_s':stats([d['server_ready_mean_tpot_s'] for d in rr if d['server_ready_mean_tpot_s'] is not None]),'iteration_cuda_stream_ms':{p:stats([b[p+'_stream_ms'] for b in batch_rows]) for p in ('forward','logits','sample')},'decode_only_iteration_cuda_stream_ms':{p:stats([b[p+'_stream_ms'] for b in batch_rows if not prefill_by_iteration[b['iteration']]]) for p in ('forward','logits','sample')},'requests':rr})
special=[details(r) for r in selected if r.get('planned_output_tokens')==1233]
burst=[details(r) for r in sorted([r for r in selected if r.get('episode')=='episodes-coding_long-0-replay-7101' and r.get('cohort')=='resident' and r.get('first_ns')],key=lambda r:r['first_ns']-r['scheduled_ns'],reverse=True)[:10]]
services=[]
for e in plan['agentic_contract']['episodes']:
 name=e['spec']['episode'];rows=[r for r in selected if r.get('episode')==name];cohorts={}
 for cohort in ('resident','incoming'):
  rr=[r for r in rows if r['cohort']==cohort];done=[r for r in rr if r.get('status')==200 and r.get('done')];post=[r for r in done if r['scheduled_ns']>=r['episode_epoch_ns']+60_000_000_000]
  cohorts[cohort]={'terminal':len(rr),'statuses':dict(collections.Counter(str(r['status']) for r in rr)),'arrival_ttft_s':stats([(r['first_ns']-r['scheduled_ns'])/1e9 for r in done if r.get('first_ns')]),'post_migration_arrival_ttft_s':stats([(r['first_ns']-r['scheduled_ns'])/1e9 for r in post if r.get('first_ns')]),'client_tpot_s':stats([r['mean_tpot_s'] for r in done if r.get('mean_tpot_s') is not None]),'server_ready_tpot_s':stats([d['server_ready_mean_tpot_s'] for r in done for d in [details(r)] if d['server_ready_mean_tpot_s'] is not None])}
 services.append({'episode':name,'cohorts':cohorts,'frozen_hashes':{f:hashlib.sha256((root/name/f).read_bytes()).hexdigest()==e[k] for f,k in [('physical-workload.json','physical_workload_sha256'),('offered-trace.json','offered_trace_sha256')]}})
result={'scope':'Independent read-only warm Q32/token certificate; unqualified CUDA values are batch stream elapsed, not exclusive kernel busy or per-request attribution. No cross-host subtraction.','telemetry_files':[str(p) for p in files],'global_stream_integrity_passed':not integrity and not qviolations,'stream_integrity_errors':integrity,'iteration_budget_violations':qviolations,'warm_cells':cells,'long_coding_turns':special,'long_replay_top10_resident_ttft':burst,'episode_service':services}
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cells':[(c['cell'],c['certified'],c['reasons'][:2]) for c in cells],'stream_integrity_errors':len(integrity),'q_violations':len(qviolations),'output':str(target)}))
