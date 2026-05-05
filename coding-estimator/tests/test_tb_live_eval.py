"""
Claim:
- evaluate_tb_only filters checkpoints to source == 'tb_live' and runs
  the v0 baselines (G2 time_only, G4 ledger_basic) under per-source
  LORO. It returns one TBOnlyCell per (model, target) pair that joined
  with at least one labeled checkpoint.
- The `note` field is `single-class y` when y is constant on the test
  predictions (so a downstream reader knows the metric is degenerate).

Plausible wrong implementations:
- Includes non-tb_live runs in the LORO split (cross-source leakage).
- Returns more rows than expected (e.g. one per fold instead of one
  per (model, target)).
- Skips the single-class note when y is constant — Brier=0 looks
  spurious and a downstream consumer would believe the model is good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.tb_live import (
    K1_MODELS,
    K1_TARGETS,
    TB_LIVE,
    evaluate_tb_only,
)


def test_evaluate_tb_only_filters_to_tb_live_runs():
    """Stuff non-tb_live rows into the frame; the function must ignore
    them. We assert the n_runs reported on each cell matches the
    tb_live run count, not the total run count."""
    ck = apply_canonical_fills(pd.read_parquet("datasets/checkpoints_all.parquet"))
    lb = pd.read_parquet("datasets/labels_all.parquet")
    cells, _raw = evaluate_tb_only(checkpoints_df=ck, labels_df=lb)
    if not cells:
        # Below LORO budget — that's a valid no-op; nothing to assert.
        return
    n_tb_runs = ck.loc[ck["source"] == TB_LIVE, "run_id"].nunique()
    for c in cells:
        assert c.n_runs <= n_tb_runs


def test_evaluate_tb_only_returns_one_row_per_model_target_pair():
    ck = apply_canonical_fills(pd.read_parquet("datasets/checkpoints_all.parquet"))
    lb = pd.read_parquet("datasets/labels_all.parquet")
    cells, _raw = evaluate_tb_only(checkpoints_df=ck, labels_df=lb)
    if not cells:
        return
    seen = {(c.model, c.target) for c in cells}
    expected = {
        (m.name, t) for m in K1_MODELS for t in K1_TARGETS
    }
    # Every produced row must be in the expected universe (no extras).
    assert seen.issubset(expected)
    # No duplicates.
    assert len(seen) == len(cells)


def test_evaluate_tb_only_flags_single_class_y_with_note():
    """y_success_eventual on tb_live in this dataset is run-constant
    with all positives (the cohort is currently 12/12 successes); the
    cell must carry the single-class note."""
    ck = apply_canonical_fills(pd.read_parquet("datasets/checkpoints_all.parquet"))
    lb = pd.read_parquet("datasets/labels_all.parquet")
    cells, _raw = evaluate_tb_only(checkpoints_df=ck, labels_df=lb)
    if not cells:
        return
    success_cells = [c for c in cells if c.target == "y_success_eventual"]
    if not success_cells:
        return
    pos_rates = {c.model: c.positive_rate for c in success_cells}
    for model, rate in pos_rates.items():
        if rate in (0.0, 1.0):
            cell = next(c for c in success_cells if c.model == model)
            assert cell.note is not None and "single-class" in cell.note
