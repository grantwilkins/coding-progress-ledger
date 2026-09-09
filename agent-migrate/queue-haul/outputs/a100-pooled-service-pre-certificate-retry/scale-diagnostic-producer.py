import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import pool_shed_campaign as q

out=Path('outputs/a100-pooled-service')
base=q.load_plan(out);target=out/'scale-diagnostic.json';start=time.perf_counter();rows=[]
source_hashes=lambda:{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('pool_shed_execution.py','pool_shed_planner.py','pool_shed_campaign.py','pool_shed_calibration.py')}
before=source_hashes();producer=Path(__file__).read_bytes();producer_sha=hashlib.sha256(producer).hexdigest()
sizes=[(q.GPUS,1000.,'shared_baseline'),(6666,1000.,'fixed_wan'),(6666,1000.*6666/q.GPUS,'proportional_wan')]
cases=list(itertools.product(('coding','coding_long'),(.5,.95),(30,300),sizes))
for workload,load,deadline,(gpus,wan,mode) in cases:
 plan={**base,'config':{**base['config'],'gpus':gpus,'installed_gpu_w':gpus*300}}
 begin=time.perf_counter();result=q.run_cell(plan,((workload,0),load,0,wan,deadline));elapsed=time.perf_counter()-begin
 fleet=q.sample_fleet(workload,0,gpus,base['config']['gpus_per_node']);power=fleet.metadata['source_power']
 policies={};tol=max(1e-7,2*gpus*1e-9)
 for policy,v in result['results'].items():
  assert v['resident_debt_generated_work_s']==[0,0] and v['pending_resident_debt_work_s']==[0,0]
  assert v['pending_backlog_reference_work_s']<=tol,(workload,load,deadline,gpus,wan,policy,'imported buffer')
  assert np.all(np.array(v['protected_serving_load'])<=1+1e-8)
  assert np.all(np.array(v['batch_replica_seconds'])<=gpus*(1-load)*deadline+1e-5)
  assert np.all(np.array(v['transferred_bytes'])<=np.array(result['budgets_gbps'])*1e9/8*deadline*(1+1e-10)+1)
  assert np.isclose(v['pending_buffered_work_s'],v['pending_source_buffer_work_s']+v['pending_backlog_reference_work_s'])
  assert np.isclose(v['buffered_requests'],v['pending_buffered_requests']+v['completed_buffered_requests'])
  assert np.isclose(v['buffered_requests'],v['source_buffered_requests']+v['transferred_buffered_requests'])
  assert v['shed_fraction']<=v['admitted_shed_fraction']+1e-8 and v['max_relative_residual']<=1e-8
  assert 0<=v['last_completion_s']<=deadline+1e-8
  policies[policy]={k:v[k] for k in ('shed_fraction','admitted_shed_fraction','action_fractions','admitted_action_fractions','resident_debt_generated_work_s','pending_resident_debt_work_s','pending_backlog_reference_work_s','pending_source_buffer_work_s','protected_serving_load','planning_steps','service_ready_s','last_completion_s','transferred_bytes','batch_replica_seconds','max_relative_residual','completed_sessions')}
  policies[policy].update(kv_shed_fraction=sum(v['action_fractions'][1::2]),admitted_kv_shed_fraction=sum(v['admitted_action_fractions'][1::2]),measured_power_shed_mw=v['shed_fraction']*gpus*power['delta_w']/1e6)
 row={'workload':workload,'snapshot':0,'resident_load':load,'deadline_s':deadline,'gpus_per_site':gpus,'sites':3,'nodes_per_site':fleet.nodes,'gpus_per_node':fleet.gpus_per_node,'nameplate_mw_per_site':gpus*300/1e6,'wan_mode':mode,'wan_gbps':wan,'actual_budgets_gbps':result['budgets_gbps'],'elapsed_s':elapsed,'policies':policies,'power':{k:power[k] for k in ('active_w','idle_w','delta_w','prefill_tokens_per_s_per_gpu','decode_tokens_per_s_per_gpu','power_load','grouped_cv_rmse_w','within_5w_fraction')},'source_active_mw':power['active_w']*gpus/1e6,'source_idle_mw':power['idle_w']*gpus/1e6}
 rows.append(row)
 print(json.dumps({'completed':len(rows),'total':len(cases),'case':[workload,load,deadline,gpus,wan],'elapsed_s':elapsed,'qh_shed':policies['queue_haul']['shed_fraction'],'qh_kv':policies['queue_haul']['kv_shed_fraction']}),flush=True)
 Path('/tmp/qh-scale-diagnostic-progress.json').write_text(json.dumps({'identity':base['identity'],'rows':rows},allow_nan=False))
assert len(rows)==24 and q.load_plan(out)['identity']==base['identity']
report={'schema':'queue-haul-a100-supplemental-scale-v1','identity':base['identity'],'complete':True,'pass':True,'cells':len(rows),'policy_evaluations':len(rows)*5,'elapsed_s':time.perf_counter()-start,'source_plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),'rows':rows,'scope':'Central-calibration supplemental diagnostic; source and both destinations have equal GPU counts. Measured per-node endpoint caps stay fixed; only shared WAN allocation is fixed or proportional. 20MW baseline appears once for both comparisons. No measured region-pair WAN capacity is inferred.','power_scope':'Nameplate uses300W per A100. Power shed linearly allocates the existing measured source active-to-awake-idle curve difference by completed serving fraction; this is a transfer proxy, not shutdown or whole-facility power.','numeric_tolerances':{'shed_fraction':1e-8,'pending_import_work_s':'max(1e-7,2*gpus*1e-9)'}}
report.update(source_sha256_start=before,source_sha256_end=source_hashes(),producer_sha256=producer_sha,producer='scale-diagnostic-producer.py')
assert report['source_sha256_start']==report['source_sha256_end']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==producer_sha
(out/'scale-diagnostic-producer.py').write_bytes(producer)
target.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('complete','pass','cells','policy_evaluations','elapsed_s')}),flush=True)
