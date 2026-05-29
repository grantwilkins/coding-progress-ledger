"""
Claim:
  token_weighted_evacuation = sum_q T_q (n_q - z_q) / sum_q T_q n_q
  kv_weighted_evacuation    = sum_q eta_q T_q (n_q - z_q) / sum_q eta_q T_q n_q

Plausible wrong implementations:
- weight by n (job count) instead of T / eta*T  -> collapses to job fraction
- KV forgets eta (uses T only) or uses beta instead of eta
- uses z instead of (n - z), or normalizes by sum(n-z) not the weighted total
"""

from __future__ import annotations

import numpy as np

from instance import ProblemInstance, build_instance
from objective_metrics import evac_summary, model_token_grid
from stage1 import solve_stage1


def _inst(T, eta, n):
    Q = len(T)
    return ProblemInstance(
        model_idx=np.zeros(Q, int), T=np.array(T, float), beta=np.full(Q, 4.0),
        eta=np.array(eta, float), rho=np.ones(Q), n=np.array(n, float),
        lambda_bps=np.array([1.0]), W=np.ones((1, 1)), mu_ing=1.0, D=1.0,
        M_names=("m",), L_names=("d",))


def test_hand_worked_separates_job_token_kv():
    # class0: big context, tiny KV/tok, left behind; class1: small context, big
    # KV/tok, evacuated. Job=0.5, token=100/1100, KV=1e4/2e4 are all distinct.
    inst = _inst(T=[1000.0, 100.0], eta=[10.0, 100.0], n=[1.0, 1.0])
    s = evac_summary(inst, np.array([1.0, 0.0]))
    assert np.isclose(s["evacuated_fraction_total"], 0.5)
    assert np.isclose(s["token_weighted_evacuation"], 100.0 / 1100.0)
    assert np.isclose(s["kv_weighted_evacuation"], 0.5)


def test_boundaries():
    inst = _inst(T=[1000.0, 100.0], eta=[10.0, 100.0], n=[3.0, 5.0])
    full = evac_summary(inst, np.zeros(2))      # all evacuated
    none = evac_summary(inst, inst.n.copy())    # none evacuated
    for k in ("token_weighted_evacuation", "kv_weighted_evacuation"):
        assert np.isclose(full[k], 1.0) and np.isclose(none[k], 0.0)


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


def test_grid_places_cells_by_model_and_bucket():
    # model0 @ T=2000 -> bucket 0 (u=0.75); model1 @ T=20000 -> bucket 2 (u=0).
    inst = ProblemInstance(
        model_idx=np.array([0, 1]), T=np.array([2000.0, 20000.0]), beta=np.full(2, 4.0),
        eta=np.full(2, 10.0), rho=np.ones(2), n=np.array([4.0, 2.0]),
        lambda_bps=np.array([1.0]), W=np.ones((1, 2)), mu_ing=1.0, D=1.0,
        M_names=("a", "b"), L_names=("d",))
    g = model_token_grid(inst, np.array([1.0, 2.0]))
    assert g.shape == (2, 5)
    assert np.isclose(g[0, 0], 0.75) and np.isclose(g[1, 2], 0.0)
    assert np.isnan(g[0, 2]) and np.isnan(g[1, 0]) and np.isnan(g).sum() == 8


def test_grid_is_per_class_u_at_matching_bins():
    # With n_bins == #buckets, each class is one finite cell carrying its u_q.
    inst = build_instance(D=120.0, n_bins=5)
    s1 = solve_stage1(inst, "throughput")
    g = model_token_grid(inst, s1.z)
    finite = g[~np.isnan(g)]
    assert finite.size == inst.n.size
    np.testing.assert_allclose(np.sort(finite), np.sort(1.0 - s1.z / inst.n), atol=1e-9)
