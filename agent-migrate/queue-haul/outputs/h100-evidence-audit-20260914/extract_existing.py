import csv,json,statistics,hashlib
from pathlib import Path
out=Path('outputs/h100-evidence-audit-20260914');out.mkdir(exist_ok=True)
def write(name,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
def q(values,p):
 if not values:return None
 a=sorted(values);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
rps=[]
root=Path('/datadrive/agentic-rps-sweep-h100-quick-v4-gpu-0ec0a705')
for f in root.glob('cells/*/attempt-*/result.json'):
 d=json.loads(f.read_text());rq=json.loads((f.parent/'requests.json').read_text());span=(max(x['end_ns'] for x in rq)-min(x['start_ns'] for x in rq))/1e9
 rps.append(dict(offered_rps=d['offered_rps'],observed_arrival_rps=d['realized_rps'],arrival_span_s=d['actual_start_span_s'],episode_through_drain_s=span,completed=d['completed'],failed=d['failed'],finite_episode_completion_rps=d['completed']/span,p90_ttft_s=d['p90_ttft_s'],p90_request_mean_tpot_s=d['p90_tpot_s'],slo_violation=d['slo_violation'],evidence=str(f)))
write('optimized_gpt_rps_reaudit.csv',sorted(rps,key=lambda x:x['offered_rps']))
root=Path('/datadrive/queue-haul-network/hardware-gap-h100-002');stacks=[]
for p in root.glob('stacks/*/power.csv'):
 with p.open() as f:rows=list(csv.DictReader(f))
 rows=[{k:float(r[k]) for k in ['monotonic_ns','wall_ns','power_w']} for r in rows if r['valid']=='1']
 stacks.append((p.parent,rows))
latencies=[];powers=[]
for f in root.glob('scenarios/*/attempt-*/result.json'):
 d=json.loads(f.read_text());s=json.loads((f.parent/'scenario.json').read_text());a,b=d['started_ns'],d['ended_ns']
 assert d['status']=='complete'
 base=dict(scenario_id=s['scenario_id'],condition=s['condition_id'],policy=s['policy'],repeat=s['repeat'],migration_s=d['migration_s'],deadline_met=d['deadline_met'])
 for node in ['east','germany']:
  p=f.parent/f'sink_load_{node}.jsonl'
  if not p.exists():continue
  rr=[json.loads(x) for x in p.read_text().splitlines()]
  before_arrivals=[x for x in rr if a-30e9<=x['start_ns']<a]
  before=[x for x in before_arrivals if x['end_ns']<=a]
  during=[x for x in rr if a<=x['start_ns']<b]
  bv=[(x['end_ns']-x['start_ns'])/1e9 for x in before if x['status_code']==200];dv=[(x['end_ns']-x['start_ns'])/1e9 for x in during if x['status_code']==200]
  ba=[(x['end_ns']-x['start_ns'])/1e9 for x in before_arrivals if x['status_code']==200]
  latencies.append({**base,'route':node,'before_arrival_requests':len(before_arrivals),'before_crossing_migration_start':sum(x['end_ns']>a for x in before_arrivals),'before_complete_requests':len(bv),'during_arrival_requests':len(during),'during_successful_requests':len(dv),'before_arrival_success_p50_response_s':q(ba,.5),'before_arrival_success_p90_response_s':q(ba,.9),'before_failed_requests':sum(x['status_code']!=200 for x in before_arrivals),'before_completed_success_p50_response_s':q(bv,.5),'during_arrival_success_p50_response_s':q(dv,.5),'before_completed_success_p90_response_s':q(bv,.9),'during_arrival_success_p90_response_s':q(dv,.9),'during_failed_requests':sum(x['status_code']!=200 for x in during),'evidence':str(p)})
 matches=[(p,rr) for p,rr in stacks if rr[0]['monotonic_ns']<=a and rr[-1]['monotonic_ns']>=b and (not d.get('stack_id') or p.name==d['stack_id'])]
 if len(matches)!=1:continue
 p,source=matches[0];offset=statistics.median(x['wall_ns']-x['monotonic_ns'] for x in source)
 for node in ['source','east','germany']:
  path=p/'power.csv' if node=='source' else p/'nodes'/node/'power.csv'
  if not path.exists():continue
  with path.open() as h:rr=list(csv.DictReader(h))
  for phase,start,end in [('before',a-30e9,a),('migration',a,b)]:
   samples=[x for x in rr if x['valid']=='1' and start+offset<=int(x['wall_ns'])<=end+offset]
   if len(samples)<3:continue
   powers.append({**base,'node':node,'phase':phase,'samples':len(samples),'window_s':(end-start)/1e9,'sample_span_s':(int(samples[-1]['wall_ns'])-int(samples[0]['wall_ns']))/1e9,'source_clock_offset_spread_ms':(max(x['wall_ns']-x['monotonic_ns'] for x in source)-min(x['wall_ns']-x['monotonic_ns'] for x in source))/1e6,'cross_host_clock_error_verified':False,'mean_power_w':statistics.mean(float(x['power_w']) for x in samples),'mean_utilization_pct':statistics.mean(float(x['utilization_pct']) for x in samples),'sampled_peak_framebuffer_used_mib':max(float(x['memory_mib']) for x in samples),'evidence':str(path)})
write('legacy_background_latency.csv',latencies);write('legacy_background_power.csv',powers)
print('rows',len(rps),len(latencies),len(powers))

power_cells=[]
for model,path in [("GPT-OSS","/datadrive/queue-haul-power/h100-realized-20260814-005/cells.jsonl"),("Gemma","/datadrive/h100-serving-gemma-vllm024-20260819-r3/power/cells.jsonl")]:
 for line in Path(path).read_text().splitlines():
  d=json.loads(line)
  power_cells.append(dict(model=model,workload=d['family'],prompt_tokens=d['prompt_tokens'],output_tokens=d['output_tokens'],concurrency=d['concurrency'],repeat=d['replicate'],mean_power_w=d['power_mean_w'],prefill_tps=d['realized_prefill_tps'],decode_tps=d['realized_decode_tps'],window_s=d['window_s'],power_samples=d['power_samples'],evidence=path,power_trace=d['power_path']))
path='/datadrive/h100-phase-power-20260820-r5/qwen3_8_27b/measurements.csv'
for d in csv.DictReader(Path(path).open()):
 power_cells.append(dict(model='Qwen',workload=d['mixture'],prompt_tokens='',output_tokens='',concurrency='',repeat=d['repeat'],mean_power_w=float(d['power_mean_w']),prefill_tps=float(d['f_tps']),decode_tps=float(d['g_tps']),window_s=(int(d['end_ns'])-int(d['start_ns']))/1e9,power_samples=int(d['power_samples']),evidence=path,power_trace=d['power_path']))
write('historical_power_conditions.csv',power_cells)
