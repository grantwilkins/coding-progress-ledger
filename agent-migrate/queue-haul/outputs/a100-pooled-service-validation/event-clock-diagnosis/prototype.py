import sys,inspect,json,time
sys.path.insert(0,'/Users/grantwilkins/houdini/agent-migrate/queue-haul')
import pool_shed_planner as p
from pool_shed_calibration import calibration
from pool_shed_campaign import forecast,execute_feedback
s=inspect.getsource(p.plan_admission);line='    next_decision, start_times = float(edges[1]), edges[:-1][:3].copy()\n';assert line in s;s=s.replace(line,'').replace('    bins, columns = len(edges) - 1, len(table.route)',line+'    bins, columns = len(edges) - 1, len(table.route)');exec(s,p.__dict__)
c=calibration(0);t=forecast('coding',0,66666,8,.5,1000,3600)[0];rows=[]
for chunks,iterations in [(64,3),(128,3),(64,6)]:
 start=time.perf_counter();r=execute_feedback(t,t,'kv_only',t.timing,c,chunks=chunks,iterations=iterations)
 row=dict(chunks=chunks,iterations=iterations,runtime=time.perf_counter()-start,result=r);rows.append(row);json.dump(rows,open('/tmp/qh-event-clock-probe.json','w'))
 print(chunks,iterations,row['runtime'],r['planning_steps'],r['shed_fraction'],r['admitted_shed_fraction'],flush=True)
