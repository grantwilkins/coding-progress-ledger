"""
Claim:
  token_weighted_evacuation = sum_q T_q (n_q - z_q) / sum_q T_q n_q
  kv_weighted_evacuation    = sum_q eta_q T_q (n_q - z_q) / sum_q eta_q T_q n_q
  kv_stranded_bytes         = sum_q eta_q T_q z_q
  residency_bound_fraction  = min(1, C_res / sum_q eta_q T_q n_q)

Plausible wrong implementations:
- weight by n (job count) instead of T / eta*T  -> collapses to job fraction
- KV forgets eta (uses T only) or uses beta instead of eta
- uses z instead of (n - z), or normalizes by sum(n-z) not the weighted total
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance
from objective_metrics import evac_summary


def _inst(T, eta, n, C_res=1e12):
    Q = len(T)
    return ProblemInstance(
        model_idx=np.zeros(Q, int), T=np.array(T, float), beta=np.full(Q, 4.0),
        eta=np.array(eta, float), rho=np.ones(Q), n=np.array(n, float),
        lambda_bps=np.array([1.0]), W=np.ones((1, 1)), W_ing=np.ones((1, 1)),
        C_res=np.array([C_res]), mu_ing=1.0, D=1.0,
        M_names=("m",), L_names=("d",))


def test_hand_worked_separates_job_token_kv():
    # class0: big context, tiny KV/tok, left behind; class1: small context, big
    # KV/tok, evacuated. Job=0.5, token=100/1100, KV=1e4/2e4 are all distinct.
    inst = _inst(T=[1000.0, 100.0], eta=[10.0, 100.0], n=[1.0, 1.0], C_res=1e4)
    s = evac_summary(inst, np.array([1.0, 0.0]))
    assert np.isclose(s["evacuated_fraction_total"], 0.5)
    assert np.isclose(s["token_weighted_evacuation"], 100.0 / 1100.0)
    assert np.isclose(s["kv_weighted_evacuation"], 0.5)
    assert np.isclose(s["kv_stranded_bytes"], 1e4)         # eta*T of class 0
    assert np.isclose(s["residency_bound_fraction"], 0.5)  # C_res / (2e4 total KV)


def test_boundaries():
    inst = _inst(T=[1000.0, 100.0], eta=[10.0, 100.0], n=[3.0, 5.0])
    full = evac_summary(inst, np.zeros(2))      # all evacuated
    none = evac_summary(inst, inst.n.copy())    # none evacuated
    for k in ("token_weighted_evacuation", "kv_weighted_evacuation"):
        assert np.isclose(full[k], 1.0) and np.isclose(none[k], 0.0)
    assert full["residency_bound_fraction"] == 1.0  # slack C_res never caps


def test_uniform_weights_collapse_to_job_fraction():
    inst = _inst(T=[500.0, 500.0, 500.0], eta=[7.0, 7.0, 7.0], n=[2.0, 1.0, 4.0])
    s = evac_summary(inst, np.array([1.0, 0.0, 3.0]))
    f = s["evacuated_fraction_total"]
    assert np.isclose(s["token_weighted_evacuation"], f)
    assert np.isclose(s["kv_weighted_evacuation"], f)


def test_scale_invariance():
    # token metric is a ratio in T -> invariant to scaling T; KV ratio in eta*T.
    base = _inst(T=[1000.0, 100.0], eta=[10.0, 100.0], n=[2.0, 3.0])
    z = np.array([1.0, 2.0])
    s0 = evac_summary(base, z)
    s1 = evac_summary(_inst(T=[3000.0, 300.0], eta=[50.0, 500.0], n=[2.0, 3.0]), z)
    assert np.isclose(s0["token_weighted_evacuation"], s1["token_weighted_evacuation"])
    assert np.isclose(s0["kv_weighted_evacuation"], s1["kv_weighted_evacuation"])
