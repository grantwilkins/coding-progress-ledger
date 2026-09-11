import json,time
from pathlib import Path
import numpy as np

def scan(A,g,debt,price=None):
    b=np.ones(A.shape[0]);x=np.zeros(len(g));steps=[]
    for _ in range(len(b)+1):
        ok=~np.any((A>0)&(b[:,None]<=1e-10),axis=0)
        if not ok.any():break
        cost=np.max(A/np.maximum(b[:,None],1e-30),axis=0) if price is None else A.T@price
        score=np.where(ok,g/np.maximum(cost,1e-30),-np.inf)
        ids=np.flatnonzero(ok & np.isclose(score,score.max(),rtol=1e-12,atol=0))
        j=int(ids[np.argmin(debt[ids]/g[ids])])
        used=np.flatnonzero(A[:,j]>0);r=int(used[np.argmin(b[used]/A[used,j])]);take=b[r]/A[r,j]
        x[j]+=take;b=np.maximum(b-take*A[:,j],0);steps.append((r,j))
    y=np.zeros(len(b))
    for r,j in reversed(steps):y[r]=(g[j]-A[:,j]@y)/A[r,j]
    return x,np.maximum(y,0)

rows=[]
for path in sorted(Path('outputs/a100-greedy-quality-20260911/matrix').glob('*.npz')):
 d=np.load(path);A,b,g,debt=[d[k] for k in ('matrix','capacity','gains','debt')]
 ids=(g>0)&~np.any((A>0)&(b[:,None]==0),axis=0);B=A[b>0][:,ids]/b[b>0,None];u=1/B.max(0);B*=u;c=g[ids]*u;cost=debt[ids]*u
 for alpha in (1.,.5,.2):
  t=time.perf_counter();x,p=scan(B,c,cost);best=x.copy();value=float(c@best)
  for it in range(1,9):
   y,prices=scan(B,c,cost,p)
   if c@y>value:best=y.copy();value=float(c@best)
   p=(1-alpha)*p+alpha*prices
   if it in (2,4,8):
    rows.append(dict(case=path.stem,alpha=alpha,passes=it+1,objective=value,lp=float(g@d['lp']),old=float(g@d['old']),gap_pp=100*(g@d['lp']-value),seconds=time.perf_counter()-t))
   assert np.max(B@best)<=1+1e-8
   chosen=np.zeros(len(g));chosen[ids]=best*u
   assert np.isfinite(chosen).all() and chosen.min()>=0
   assert np.max((A@chosen)[b>0]/b[b>0])<=1+1e-8 and not np.any((A@chosen)[b==0]>0)
   assert value>=float(g@d['old'])-1e-10 and value<=float(g@d['lp'])+1e-8
assert len(rows)==108
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2))
for alpha in (1.,.5,.2):
 for passes in (3,5,9):
  s=[r for r in rows if r['alpha']==alpha and r['passes']==passes]
  print(alpha,passes,'maxgap',max(r['gap_pp'] for r in s),'median',np.median([r['gap_pp'] for r in s]),'msrange',min(r['seconds'] for r in s)*1000,max(r['seconds'] for r in s)*1000,flush=True)
