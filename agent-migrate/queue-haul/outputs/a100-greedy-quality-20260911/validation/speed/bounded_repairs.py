import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from pool_shed_planner import _choose

ROOT=Path('outputs/a100-greedy-quality-20260911/matrix')

def pair_exchange(A,b,g,debt,old,static_rows,source_rows):
    replicas=A[static_rows-2:static_rows].sum(0)
    allowed=(g>0)&~np.any((A>0)&(b[:,None]==0),axis=0)
    ids=np.flatnonzero(allowed);rows=b>0
    B=A[rows][:,ids]/b[rows,None]/replicas[ids]
    value=g[ids]/replicas[ids];y=old[ids]*replicas[ids]
    routes=A[static_rows-2:static_rows,ids]>0
    for step in range(2):
        left=np.maximum(1-B@y,0.);best=(0.,None)
        for route in (0,1):
            active=np.flatnonzero((y>1e-8)&routes[route]);active=active[np.argsort(value[active],kind='stable')[:4]]
            for k,l in itertools.combinations(active,2):
                for width in (1.,2.):
                    delta=width*B-B[:,k,None]-B[:,l,None]
                    take=np.minimum(min(y[k],y[l]),np.min(np.divide(left[:,None],delta,out=np.full_like(delta,np.inf),where=delta>1e-14),axis=0))
                    profit=take*(width*value-value[k]-value[l]);profit[[k,l]]=-np.inf
                    j=int(np.argmax(profit))
                    if profit[j]>best[0]:best=(float(profit[j]),(k,l,j,float(take[j]),width))
        if best[0]<=1e-12:break
        k,l,j,take,width=best[1];y[k]-=take;y[l]-=take;y[j]+=width*take
    result=np.zeros(len(g));result[ids]=y/replicas[ids]
    return result


def joint_refill(A,b,g,debt,old,static_rows,source_rows):
    replicas=A[static_rows-2:static_rows].sum(0);value=np.divide(g,replicas,out=np.zeros_like(g),where=replicas>0)
    groups=[]
    for count in (2,3):
        removed=[]
        for route in (0,1):
            active=np.flatnonzero((old>1e-8)&(A[static_rows-2+route]>0))
            removed.extend(active[np.argsort(value[active],kind='stable')[:count]])
        groups.append(np.asarray(removed,int))
    best=old.copy();objective=float(g@best)
    for removed in groups:
        candidate=old.copy();candidate[removed]=0
        banned=np.any(np.all(A[:source_rows,:,None]==A[:source_rows,removed][:,None,:],axis=0),axis=1)
        candidate+=_choose(A,np.maximum(b-A@candidate,0.),g*(~banned),debt,None,True)
        if g@candidate>objective:best=candidate;objective=float(g@candidate)
    return best


def main():
    import pool_shed_campaign as q
    def forbidden(*a,**k):raise AssertionError('generic LP called')
    q.solve_lp=forbidden
    report=[]
    for path in sorted(ROOT.glob('*.npz')):
        f=np.load(path);A,b,g,d=(f[k] for k in ('matrix','capacity','gains','debt'))
        optimum=float(g@f['lp']);static=int(f['static_rows']);source_rows=f['replay'].shape[1]
        row=dict(case=path.stem,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),lp_gain=optimum,lp_wall_ms=1000*float(f['lp_s']),old_gain=float(g@f['old']))
        for name,method in [('pair_exchange',pair_exchange),('joint_refill',joint_refill)]:
            times=[]
            for repeat in range(3):
                start=time.perf_counter();old=_choose(A,b,g,d,None,True);chosen=method(A,b,g,d,old,static,source_rows);times.append(1000*(time.perf_counter()-start))
                usage=A@chosen;assert np.isfinite(chosen).all() and chosen.min()>=-1e-10
                assert np.max(usage[b>0]/b[b>0],initial=0.)<=1+1e-8 and not np.any(usage[b==0]>0)
                assert g@chosen>=g@old-1e-10 and g@chosen<=optimum+1e-8
            row[name]=dict(gain=float(g@chosen),gap_pp=100*(optimum-float(g@chosen)),relative_gap_percent=100*(1-float(g@chosen)/optimum),wall_ms_median=float(np.median(times)),wall_ms_range=[min(times),max(times)])
        report.append(row);print(json.dumps(row),flush=True)
    assert len(report)==12
    output=Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(dict(scope='Twelve saved fixed matrices; two bounded heuristics; all constraints retained. Three repeats include incumbent construction; stored LP timings are single-run references. No LP/policy solves or model changes.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=report),indent=2)+'\n')

if __name__=='__main__':main()
