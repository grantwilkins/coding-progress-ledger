"""Run the remaining inherited campaign through nine existing native scenario shards."""
import gzip, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import pool_shed_campaign as c

out=c.OUT;started=time.perf_counter();plan=c.load_plan(out);ledger=json.loads((out/'resume-stages.json').read_text())
assert ledger['identity']==plan['identity'] and ledger['inherited_cells']==len(plan['inherited_checkpoints']['cells_sha256'])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
script=Path(__file__).read_bytes();script_sha=hashlib.sha256(script).hexdigest();ledger_sha=sha(out/'resume-stages.json')
(out/'native-nine-shard-driver.py').write_bytes(script)
env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},'MPLCONFIGDIR':'/tmp/qh-mpl'}
logs=[(out/f'resumed-run-{i}.log').open('w') for i in range(9)];processes=[]
try:
 for i in range(9):processes.append(subprocess.Popen([sys.executable,'pool_shed_campaign.py','run','--out',str(out),'--shard',str(i),'--shards','9'],stdout=logs[i],stderr=subprocess.STDOUT,env=env))
 reported=0.
 while any(p.poll() is None for p in processes):
  failed=[i for i,p in enumerate(processes) if p.poll() not in (None,0)]
  if failed:raise RuntimeError({i:(out/f'resumed-run-{i}.log').read_text()[-6000:] for i in failed})
  elapsed=time.perf_counter()-started
  if elapsed-reported>=30:
   print(json.dumps({'cells':len(list((out/'cells').glob('*.json.gz'))),'elapsed_s':elapsed}),flush=True);reported=elapsed
  time.sleep(1)
 assert all(p.returncode==0 for p in processes)
finally:
 for p in processes:
  if p.poll() is None:p.terminate()
 for p in processes:p.wait()
 for f in logs:f.close()
compute=time.perf_counter()-started
assert sha(Path(__file__))==script_sha and sha(out/'resume-stages.json')==ledger_sha and c.load_plan(out)['identity']==plan['identity']
assert all(sha(out/'cells'/n)==h for n,h in plan['inherited_checkpoints']['cells_sha256'].items())
c.reduce(out)
with (out/'scenarios.csv').open('rb') as source,(out/'scenarios.csv.gz').open('wb') as target:
 with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as compressed:shutil.copyfileobj(source,compressed)
current=time.perf_counter()-started;prior=ledger['original_stage']['wall_s_approx']+ledger['failed_recovery_stage']['wall_s_approx']
performance={'identity':plan['identity'],'workers':9,'local_workers':9,'remote_workers':0,'runtime_partition':'modulo_scenarios',
 'runtime_files':[f'runtime-{i}.json' for i in range(9)],'preexisting_completed_cells':ledger['inherited_cells'],
 'computed_cells':ledger['current_stage']['expected_cells'],'scenarios':ledger['total_cells'],'policy_evaluations':5*ledger['total_cells'],
 'stage_ledger_sha256':ledger_sha,'driver_sha256':script_sha,'prior_stage_elapsed_s':prior,'compute_wall_s':compute,
 'current_stage_wall_s':current,'total_wall_s':prior+current,'scope':'Current stage is nine local native shards. Prior accepted remote/local results retain their original identity and archived platform provenance; parallel worker times are not added to wall time.'}
c.write_json(out/'performance.json',performance);print(json.dumps(performance),flush=True)
