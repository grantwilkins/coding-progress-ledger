"""Flagship-only rack/pod instance construction.

  1. rack arithmetic   — n_rack matches hand arithmetic from the HBM constants.
  2. occupancy scaling — N = round(o * SRC_RACKS * n_rack), linear in o.
  3. realized KV mass  — sum eta*T*n ~ o * C_res (the fitted-occupancy design).
  4. shapes/constants  — M = L = 1, C_res = 24 x 170 GB, W/W_ing = 8/24 nodes.
  5. binning           — n_bins aggregation preserves jobs and token mass.
"""

from __future__ import annotations

import numpy as np

from instance import (CONTEXT_DIST, DECODE_NODES_PER_RACK, ETA_BYTES_PER_TOK,
                      KV_FREE_PER_NODE_B, LAMBDA_BPS, SRC_RACKS, build_instance,
                      n_rack)


def test_n_rack_hand_arithmetic():
    expect = int(DECODE_NODES_PER_RACK * KV_FREE_PER_NODE_B
                 // (ETA_BYTES_PER_TOK * CONTEXT_DIST.mean()))
    assert n_rack() == expect
    assert KV_FREE_PER_NODE_B == 170e9  # 640 GB node - 470 GB weights


def test_occupancy_scales_jobs_linearly():
    for o in (0.5, 1.0, 1.5):
        inst = build_instance(occupancy=o, seed=0)
        assert inst.T.size == round(o * SRC_RACKS * n_rack())


def test_realized_kv_tracks_occupancy():
    ratios = [(lambda i: (i.eta * i.T * i.n).sum() / i.C_res[0])(
        build_instance(occupancy=1.0, seed=s)) for s in range(8)]
    assert abs(np.mean(ratios) - 1.0) < 0.1  # mean KV mass ~ C_res at o = 1


def test_shapes_and_constants():
    inst = build_instance(seed=0)
    assert len(inst.M_names) == 1 and len(inst.L_names) == 1
    assert inst.W.shape == (1, 1) and inst.W[0, 0] == 8           # prefill nodes
    assert inst.W_ing.shape == (1, 1) and inst.W_ing[0, 0] == 24  # decode nodes
    np.testing.assert_allclose(inst.C_res, [24 * 170e9])
    np.testing.assert_allclose(inst.lambda_bps, [LAMBDA_BPS])
    assert (inst.eta == ETA_BYTES_PER_TOK).all()
    assert inst.d_miss == 2.0 * inst.D


def test_binning_preserves_population_and_tokens():
    flat = build_instance(seed=0)
    binned = build_instance(seed=0, n_bins=5)
    assert binned.T.size <= 5
    assert binned.n.sum() == flat.n.sum()
    np.testing.assert_allclose((binned.T * binned.n).sum(), flat.T.sum(), rtol=1e-12)
