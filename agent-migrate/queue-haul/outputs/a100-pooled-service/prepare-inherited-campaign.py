"""Prepare an inherited plan only after reviewed certificate-only recovery and writer shutdown."""
import argparse, hashlib, json, shutil
from pathlib import Path
import pool_shed_campaign as c

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--recovery-wall-s',type=float,required=True)
p.add_argument('--stopped-checkpoints',type=Path,default=Path('/tmp/qh-ab507-stopped-checkpoints.json'))
p.add_argument('--retry-proof',type=Path,required=True)
a=p.parse_args();out=c.OUT;archive=a.archive.resolve();root=c.ROOT.resolve()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
parent=json.loads((archive/'plan.json').read_text());recovery=json.loads((archive/'recovery-manifest.json').read_text())
assert parent['identity']=='ab50712399d61c333e34f3709cdad20c58b8dd62ae56c6051fff10af6a4de3b0'
assert parent['identity']==c.digest({'config':parent['config'],'sources':parent['sources']}) and 'inherited_checkpoints_sha256' not in parent
assert parent['config']==c.configuration() and a.recovery_wall_s>=0 and archive.is_dir() and not out.exists()
stopped=json.loads(a.stopped_checkpoints.read_text());proof=json.loads(a.retry_proof.read_text())
assert stopped['identity']==parent['identity'] and len(stopped['cells_sha256'])==9486
assert proof['status']=='pass' and proof['previous_successful_path_unchanged'] is True
sources=c.provenance(c.calibration(parent['config']['draws']))
assert set(sources)==set(parent['sources']) and {k for k in sources if sources[k]!=parent['sources'][k]}=={'pool_shed_campaign.py'}
assert proof['parent_campaign_sha256']==parent['sources']['pool_shed_campaign.py'] and proof['candidate_campaign_sha256']==sources['pool_shed_campaign.py']
grid=c.cells(parent['config']);variants=len(parent['config']['deadlines'])*len(parent['config']['wan_gbps'])
group=lambda i:i//((parent['config']['draws']+1)*variants)*variants+i%variants
paths=sorted((archive/'cells').glob('*.json.gz'));counts={};pins={}
for path in paths:
 i=int(path.name.split('.')[0]);assert path.name==f'{i:06d}.json.gz' and 0<=i<len(grid)
 c.read_checkpoint(path,parent,grid[i]);pins[path.name]=sha(path);counts[group(i)]=counts.get(group(i),0)+1
assert pins==stopped['cells_sha256'] and all(n==parent['config']['draws']+1 for n in counts.values())
assert all(str(g) in recovery['original_accepted_groups'] or str(g) in recovery['replacement_groups'] for g in counts)
metadata=('plan.json','recovery-manifest.json','cpu-assignments.json','cpu-initial-checkpoints.json')
manifest={'parent_identity':parent['identity'],'parent_config':parent['config'],'parent_sources':parent['sources'],'cells_sha256':pins,
 'parent_archive':str(archive.relative_to(root)),'parent_plan_sha256':sha(archive/'plan.json'),
 'parent_records_sha256':{n:sha(archive/n) for n in metadata},'paired_groups_all_complete':True}
original=recovery['original_computed_cells'];assert len(pins)>=original
assert all(not (archive/n).exists() for n in ('accepted-checkpoints.json','stopped-checkpoints.json','certificate-retry-proof.json'))
c.write_json(archive/'accepted-checkpoints.json',manifest)
shutil.copy2(a.stopped_checkpoints,archive/'stopped-checkpoints.json');shutil.copy2(a.retry_proof,archive/'certificate-retry-proof.json')
new=c.prepare(out);assert new['config']==parent['config'] and new['network_indices']==parent['network_indices'] and new['identity']!=parent['identity']
(out/'cells').mkdir()
for name in pins:shutil.copy2(archive/'cells'/name,out/'cells'/name)
c.write_json(out/'inherited-checkpoints.json',manifest)
new['inherited_checkpoints_sha256']=sha(out/'inherited-checkpoints.json');c.write_json(out/'plan.json',new)
ledger={'identity':new['identity'],'parent_identity':parent['identity'],'inherited_cells':len(pins),'total_cells':len(grid),
 'parent_archive':manifest['parent_archive'],'archive_manifest_sha256':sha(archive/'accepted-checkpoints.json'),
 'stopped_checkpoints_sha256':sha(archive/'stopped-checkpoints.json'),'retry_proof_sha256':sha(archive/'certificate-retry-proof.json'),
 'original_stage':{'status':'interrupted','accepted_cells':original,'wall_s_approx':recovery['stage1_wall_s_approx']},
 'failed_recovery_stage':{'status':'failed_certificate','accepted_cells':len(pins)-original,'wall_s_approx':a.recovery_wall_s,'old_manifest_planned_cells':recovery['recovery_computed_cells']},
 'current_stage':{'expected_cells':len(grid)-len(pins),'workers':9,'host':'local'},
 'scope':'Accepted checkpoint identities and bytes are unchanged. Prior interrupted wall intervals include outages and are separate from the new run; old planned recovery counts are not completed counts.'}
c.write_json(out/'resume-stages.json',ledger)
loaded=c.load_plan(out)
assert all(sha(out/'cells'/n)==h==sha(archive/'cells'/n) for n,h in pins.items())
assert loaded['identity']==new['identity']
print(json.dumps({'identity':new['identity'],'inherited_cells':len(pins),'remaining_cells':len(grid)-len(pins),'archive':str(archive)}))
