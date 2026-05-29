"""CVXPY pressure-ceiling duals (stage2_duals.ceiling_duals).

  1. simplex      — duals are nonnegative and sum to 1 (phi-stationarity).
  2. strong duality — the CVXPY duals, fed through the decomposition's per-class
                      inner min, recover phi* (D(pi*) = phi*). This ties the
                      shadow prices to the simplex prices the solvers converge to.
"""

from __future__ import annotations

import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2_dual import build_dual_structure, per_class_assign
from stage2_duals import ceiling_duals


def test_duals_are_on_the_simplex():
    inst = build_instance(total_jobs=200, seed=0)
    s1 = solve_stage1(inst)
    pi, _ = ceiling_duals(inst, s1)
    vals = np.array(list(pi.values()))
    assert (vals >= -1e-9).all()
    assert abs(vals.sum() - 1.0) < 1e-6


def test_duals_recover_phi_star_through_decomposition():
    inst = build_instance(total_jobs=200, seed=0)
    s1 = solve_stage1(inst)
    assert s1.Z_star < 1e-6                       # mu = None branch
    s2 = solve_stage2(inst, s1)
    pi_dict, phi = ceiling_duals(inst, s1)

    A, C, I_meta, feasible = build_dual_structure(inst)
    pi = np.array([pi_dict[(k, l) if k == "net" else (k, l, m)] for (k, l, m) in I_meta])
    _, _, _, D_pi = per_class_assign(A, inst.n, pi, None, feasible)
    assert abs(D_pi - s2.phi_star) < 1e-4
