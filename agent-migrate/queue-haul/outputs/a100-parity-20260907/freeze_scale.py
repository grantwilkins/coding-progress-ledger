import json,time,hashlib
from pathlib import Path
root=Path('/datadrive/queue-haul-network/a100-timing-r3')
plan=json.loads((root/'plan.json').read_text());train=plan['scenarios'][:80];holdout=plan['scenarios'][96:]
protocol={'schema':'queue-haul-prospective-scale-v1','created_wall_ns':time.time_ns(),'plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),'method':'Per-action positive least-squares scale through origin; no intercept, selection or refitting','training_ids':[r['scenario_id'] for r in train],'holdout_ids':[r['scenario_id'] for r in holdout],'gates':{'mae_s':3,'r2':0.8}}
with (root/'scale-protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
while True:
 progress=json.loads((root/'progress.json').read_text())
 if progress['completed']>=80:break
 time.sleep(5)
assert not any((root/'scenarios'/r['scenario_id']).exists() for r in holdout),'Holdout started before scale freeze'
pairs={a:[] for a in ('replay','kv_transfer','mixed')}
for scenario in train:
 paths=sorted((root/'scenarios'/scenario['scenario_id']).glob('attempt-*/result.json'))
 result=json.loads(paths[-1].read_text());assert result['status']=='complete'
 measured=(max(r['request']['stream_chunks'][0]['monotonic_ns'] for r in result['requests'])-result['started_ns'])/1e9
 prediction=scenario['parity_prediction'];pairs[prediction['action']].append((prediction['predicted_s'],measured))
scales={a:sum(p*m for p,m in rows)/sum(p*p for p,m in rows) for a,rows in pairs.items()}
assert all(v>0 for v in scales.values())
output={'protocol_sha256':hashlib.sha256((root/'scale-protocol.json').read_bytes()).hexdigest(),'frozen_wall_ns':time.time_ns(),'scales':scales,'training_counts':{a:len(v) for a,v in pairs.items()},'predictions':[{'scenario_id':r['scenario_id'],'action':r['parity_prediction']['action'],'predicted_s':r['parity_prediction']['predicted_s']*scales[r['parity_prediction']['action']]} for r in holdout]}
with (root/'scale-fit.json').open('x') as f:json.dump(output,f,indent=2)
print(json.dumps({'frozen_wall_ns':output['frozen_wall_ns'],'scales':scales,'training_counts':output['training_counts']}),flush=True)
