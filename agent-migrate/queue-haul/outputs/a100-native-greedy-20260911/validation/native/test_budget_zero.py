from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csc_matrix
from _queue_haul_native import packing_coordinate
from pool_shed_planner import _choose

def compare(A,g,secondary):
 S=csc_matrix(A);inc=np.zeros(len(g));args=[S.indptr,S.indices,S.data,g,inc,secondary]
 frozen=[v.copy() for v in args]
 expected=_choose(A,np.ones(len(A)),g,g*secondary,SimpleNamespace(),True)
 chosen,dual,iterations=packing_coordinate(*S.shape,*args,0,0.,0.)
 assert iterations==0
 np.testing.assert_allclose(chosen,expected,rtol=1e-11,atol=1e-11)
 assert np.max(A@chosen)<=1+1e-8
 assert np.all(A.T@dual>=g*(1-1e-10))
 for current,old in zip(args,frozen):np.testing.assert_array_equal(current,old)

@pytest.mark.parametrize('seed',range(36))
def test_budget_zero_matches_exhaustive_greedy(seed):
 rng=np.random.default_rng(seed);m,n=((3,12),(12,40),(30,100))[seed%3]
 A=rng.uniform(.01,1,(m,n));A*=rng.uniform(size=(m,n))<(.1,.5,1.)[seed%3]
 for j in range(n):A[rng.integers(m),j]=1
 A/=A.max(0)
 compare(A,rng.uniform(.001,.1,n),rng.uniform(0,2,n))

@pytest.mark.parametrize('g,secondary',[
 ([1.,1.,1.],[1.,1.,1.]),
 ([1.,1-8e-13,1-12e-13],[10.,0.,0.]),
 ([1.,1.,1.],[1+8e-13,1.,1+12e-13]),
 ([1.,1-8e-13,1-12e-13],[1+8e-13,1.,0.]),
])
def test_tolerance_ties(g,secondary):
 compare(np.ones((1,3)),np.array(g),np.array(secondary))

def test_later_scores_and_fractional_remaining_resources():
 A=np.array([[1.,.25,0.,.4],[0.,1.,1.,.4],[.1,0.,.25,1.]])
 compare(A,np.array([1.,1.1,1.,.9]),np.array([1.,.5,1.,0.]))
