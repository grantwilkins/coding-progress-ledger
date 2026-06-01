"""Semantic tests for the heuristic baseline allocators.

Targets believable errors in the rule/engine wiring:
  1. cap-hard conserves mass and never overloads a resource
  2. cap-soft moves everything (z == 0) and, at a tight deadline, busts phi > 1
  3. replay_only / state_only really pin one action to zero
  4. random is seed-reproducible and seed-sensitive; greedies are deterministic
  5. the throughput optimizer evacuates at least as much as any baseline
  6. greedy minimizes recon cost over BOTH actions (not collapsed to one action,
     not argmax) -- tested by the cost-domination invariant g <= replay, state
  7. round_robin spreads a model's classes across distinct destinations (it
     cycles with q) rather than concentrating them on one
"""

from __future__ import annotations

import numpy as np

from baselines import BASELINES, R, S, allocate, pressures
from instance import build_instance
from loads import loads
from objective_metrics import evac_summary
from stage1 import solve_stage1
from stage3 import recon_costs

INST = build_instance(D=80.0, n_bins=5)
TIGHT = build_instance(D=40.0, n_bins=5)  # tight enough that moving all overloads
ABUNDANT = build_instance(D=1e7, n_bins=5)  # caps astronomically slack: each class
#                                             lands fully on its first-choice (l, a)


def _recon_cost(inst, s):
    """Total unloaded reconstruction time of an allocation (z=0 under ABUNDANT)."""
    c_R, c_S = recon_costs(inst)
    return float((c_R * s.x_R).sum() + (c_S * s.x_S).sum())


def test_hard_conserves():
    for name in BASELINES:
        s = allocate(INST, name, hard=True)
        np.testing.assert_allclose(s.x_R.sum(1) + s.x_S.sum(1) + s.z, INST.n, atol=1e-6)


def test_soft_moves_all():
    for name in BASELINES:
        s = allocate(INST, name, hard=False)
        np.testing.assert_allclose(s.z, 0.0, atol=1e-9)
        np.testing.assert_allclose(s.x_R.sum(1) + s.x_S.sum(1), INST.n, atol=1e-6)


def test_hard_respects_caps():
    C_net, C_pfill, C_ing, S_pfill, S_ing, b_net_R, b_net_S = loads(INST)
    for name in BASELINES:
        s = allocate(INST, name, hard=True)
        assert np.all(b_net_R @ s.x_R + b_net_S @ s.x_S <= C_net * (1 + 1e-9) + 1e-3)
        assert np.all(S_pfill @ s.x_R <= C_pfill * (1 + 1e-9) + 1e-3)
        assert np.all(S_ing @ s.x_S <= C_ing * (1 + 1e-9) + 1e-3)


def test_action_isolation():
    for hard in (True, False):
        assert np.all(allocate(INST, "replay_only", hard=hard).x_S == 0)
        assert np.all(allocate(INST, "state_only", hard=hard).x_R == 0)


def test_random_seeding():
    a, b = allocate(INST, "random", seed=3), allocate(INST, "random", seed=3)
    np.testing.assert_allclose(a.x_R, b.x_R)
    np.testing.assert_allclose(a.z, b.z)
    assert not np.allclose(a.x_R, allocate(INST, "random", seed=7).x_R)


def test_greedies_deterministic():
    for name in ("greedy", "round_robin", "replay_only", "state_only", "least_loaded"):
        a, b = allocate(INST, name), allocate(INST, name)
        np.testing.assert_array_equal(a.x_R, b.x_R)
        np.testing.assert_array_equal(a.x_S, b.x_S)


def test_greedy_minimizes_recon_cost_over_both_actions():
    # greedy picks the per-class cheapest (dest, action) over the UNION of actions,
    # so its total cost must be <= either single-action restriction. A greedy that
    # sorts only one action equals that baseline's cost and so exceeds the cheaper
    # one; an argmax greedy blows past both. Both are caught here.
    g = _recon_cost(ABUNDANT, allocate(ABUNDANT, "greedy"))
    r = _recon_cost(ABUNDANT, allocate(ABUNDANT, "replay_only"))
    st = _recon_cost(ABUNDANT, allocate(ABUNDANT, "state_only"))
    assert g <= r + 1e-3 and g <= st + 1e-3
    assert min(r, st) > 0  # the comparison is non-vacuous


def test_round_robin_spreads_across_destinations():
    # Cycling the first choice with q must scatter a model's classes over several
    # destinations; a non-cycling rule (always dest 0 / cheapest) concentrates them.
    s = allocate(ABUNDANT, "round_robin")
    moved = s.x_R + s.x_S  # (Q, L); abundant => one nonzero dest per class
    spreads = {}
    for m in range(len(ABUNDANT.M_names)):
        q = ABUNDANT.model_idx == m
        if (ABUNDANT.W[:, m] > 0).sum() >= 2 and q.sum() >= 2:
            spreads[m] = int((moved[q].sum(0) > 0).sum())
    assert spreads, "no multi-destination model to exercise round-robin"
    assert max(spreads.values()) >= 2


def test_optimizer_evacuates_at_least_as_much():
    opt = evac_summary(INST, solve_stage1(INST, "throughput").z)["evacuated_fraction_total"]
    for name in BASELINES:
        for seed in (range(5) if name == "random" else (0,)):
            bl = evac_summary(INST, allocate(INST, name, seed=seed).z)["evacuated_fraction_total"]
            assert bl <= opt + 1e-6


def test_soft_mode_busts_deadline():
    phis = [pressures(TIGHT, s.x_R, s.x_S).phi_star
            for s in (allocate(TIGHT, n, hard=False) for n in BASELINES)]
    assert max(phis) > 1.0
