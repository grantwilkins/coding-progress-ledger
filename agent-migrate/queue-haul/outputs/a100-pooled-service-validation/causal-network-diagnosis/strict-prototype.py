import inspect,json,time,hashlib
from pathlib import Path
import pool_shed_campaign as c
import pool_shed_planner as p
source=inspect.getsource(p.phase_profile).replace('recovery_sharing=1., primitive_cache=None):', 'recovery_sharing=1., primitive_cache=None, causal_network=False):').replace('            now = end\n            return','            now = float(edges[min(np.searchsorted(edges, end, side="left"), bins)]) if causal_network and 0 < work and end <= edges[-1] else end\n            return')
source=source.replace('        if not compute:\n            end =', '        if not compute:\n            if causal_network and work > 0 and now < edges[-1]:\n                now = float(edges[np.searchsorted(edges, now, side="left")])\n            end =')
exec(compile(source,'<network-bin-prototype>','exec'),p.__dict__)
planner_source=inspect.getsource(p.plan_admission).replace('sharing=max(1e-30, fixed_sharing[table.route[j]]) if protected else 1.,','sharing=max(1e-30, fixed_sharing[table.route[j]]) if protected else 1., causal_network=protected,')
exec(compile(planner_source,'<network-bin-planner-prototype>','exec'),p.__dict__)
c.initial_admission.cache_clear()
plan=json.loads(Path('outputs/a100-pooled-service/plan.json').read_text());rows=[]
for load,draw in [(.5,0),(.5,3),(.25,5)]:
    cell=(('measured_pack',0),load,draw,40,10);start=time.perf_counter();r=c.run_cell(plan,cell);row={'cell':cell,'elapsed_s':time.perf_counter()-start,'results':{policy:{key:z[key] for key in ['shed_fraction','admitted_shed_fraction','max_relative_residual','planning_steps']} for policy,z in r['results'].items()}};rows.append(row);print(json.dumps(row),flush=True)
Path('/tmp/qh-network-bin-prototype-isolated.json').write_text(json.dumps({'scope':'Read-only candidate network dependency rounding prototype','prototype_sha256':hashlib.sha256(source.encode()).hexdigest(),'rows':rows},indent=2))
