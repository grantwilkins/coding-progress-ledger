"""Synthetic-destination construction (build_instance n_dest).

  1. shapes/positivity  — L grows to n_dest, W is (n_dest, M), lambda > 0.
  2. real sites preserved — first 3 sites match the default instance exactly.
  3. workload untouched  — same seed gives identical jobs across n_dest.
  4. feasible            — added capacity keeps Z* = 0 (ADMM precondition).
  5. guards              — n_dest < 3 or explicit W with n_dest hard-fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from instance import DESTINATIONS, MODELS, build_instance
from stage1 import solve_stage1


def test_shapes_and_positivity():
    inst = build_instance(total_jobs=200, seed=0, n_dest=10)
    assert inst.lambda_bps.shape == (10,)
    assert inst.W.shape == (10, len(MODELS))
    assert (inst.lambda_bps > 0).all()
    assert (inst.W >= 0).all()
    assert len(inst.L_names) == 10


def test_real_sites_preserved():
    base = build_instance(total_jobs=200, seed=0)
    syn = build_instance(total_jobs=200, seed=0, n_dest=6)
    n_real = len(DESTINATIONS)
    np.testing.assert_array_equal(syn.lambda_bps[:n_real], base.lambda_bps)
    np.testing.assert_array_equal(syn.W[:n_real], base.W)


def test_workload_identical_across_n_dest():
    base = build_instance(total_jobs=200, seed=0)
    syn = build_instance(total_jobs=200, seed=0, n_dest=10)
    np.testing.assert_array_equal(syn.model_idx, base.model_idx)
    np.testing.assert_array_equal(syn.T, base.T)
    np.testing.assert_array_equal(syn.n, base.n)


def test_added_capacity_keeps_Z_star_zero():
    for n_dest in (3, 6, 10):
        inst = build_instance(total_jobs=200, seed=0, n_dest=n_dest)
        assert solve_stage1(inst).Z_star < 1e-6


def test_guards():
    with pytest.raises(AssertionError):
        build_instance(n_dest=2)
    with pytest.raises(AssertionError):
        build_instance(n_dest=6, W=np.ones((6, len(MODELS))))
