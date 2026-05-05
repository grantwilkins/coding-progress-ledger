"""Workstream H — semantic-error tests that complement the existing H
test files (test_splits_builder, test_eval_slices, test_eval_report_render,
test_baseline_ladder).

Each section below targets a specific class of plausible wrong
implementation that the existing tests don't catch. The structure is:
top-of-section docstring states the claim and the impostors; tests
follow.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from coding_estimator.baselines.time_only import TIME_ONLY
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.slices import assign_phase
from coding_estimator.splits.builder import build_split
from coding_estimator.splits.protocol import Fold, Split
from scripts.run_baselines import _apply_canonical_fills


# =============================================================================
# 1) predict_cell — train-row leakage and run-disjoint guard
# =============================================================================
"""
Claim:
    `predict_cell` returns only test-fold rows (run_id ∈ test_run_ids
    of some fold), and raises ValueError if a run_id appears in two
    test folds.

Plausible wrong implementations:
    - Appends `train` rows alongside (or instead of) `test` rows.
    - Forgets the run-disjoint guard, silently double-counting a run
      whose id appears in two test folds.
    - Swaps train/test set membership in `isin(...)`.
"""


def _toy_frames(n_runs: int = 4, n_steps: int = 5):
    rows, labs = [], []
    for k in range(n_runs):
        rid = f"r{k}"
        for t in range(n_steps):
            cid = f"{rid}::{t}"
            rows.append({
                "run_id": rid, "source": "src",
                "checkpoint_id": cid, "checkpoint_step": t,
                "elapsed_steps": float(t),
            })
            labs.append({
                "run_id": rid, "source": "src",
                "checkpoint_id": cid, "target_name": "y",
                "label_value": float((k + t) % 2),
                "is_masked": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(labs)


def test_predict_cell_returns_only_test_runs():
    ck, lab = _toy_frames()
    fold = Fold("h", train_run_ids=("r0", "r1"), test_run_ids=("r2", "r3"))
    split = Split("holdout", 0, (fold,))
    preds = predict_cell(
        checkpoints_df=ck, labels_df=lab, target="y",
        spec=TIME_ONLY, split=split, sources_in_train=("src",),
    )
    assert set(preds["run_id"].unique()) == {"r2", "r3"}
    # Any train run id leaking through would make this set larger.
    assert "r0" not in set(preds["run_id"])
    assert "r1" not in set(preds["run_id"])


def test_predict_cell_raises_when_a_run_appears_in_two_test_folds():
    ck, lab = _toy_frames(n_runs=3)
    fold_a = Fold("a", train_run_ids=("r2",), test_run_ids=("r0", "r1"))
    # r0 reappears in fold_b's test partition — disjointness violation.
    fold_b = Fold("b", train_run_ids=("r2",), test_run_ids=("r0",))
    split = Split("loro", 0, (fold_a, fold_b))
    with pytest.raises(ValueError, match="r0"):
        predict_cell(
            checkpoints_df=ck, labels_df=lab, target="y",
            spec=TIME_ONLY, split=split, sources_in_train=("src",),
        )


# =============================================================================
# 2) assign_phase — exact boundary at frac = 1/3 and 2/3
# =============================================================================
"""
Claim:
    Phase = 'early'  iff frac ≤ 1/3
            'middle' iff 1/3 < frac ≤ 2/3
            'late'   iff frac > 2/3.
    A 4-step run (steps 0..3, span=3) puts step 1 at frac=1/3 (early)
    and step 2 at frac=2/3 (middle). These are the seams.

Plausible wrong implementations:
    - `<` everywhere → step 1 (frac=1/3) lands in 'middle'.
    - `>=` for late → step 2 (frac=2/3) lands in 'late'.
    - `<` for middle and `>=` for late → both boundaries shift.
"""


def _fourstep_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": "r", "checkpoint_step": s, "_y": 0, "_p": 0.5,
         "source": "s", "checkpoint_id": f"r::{s}"}
        for s in range(4)
    ])


def test_assign_phase_lower_boundary_one_third_is_early():
    df = _fourstep_frame()
    phase = assign_phase(df)
    # Step 1 has frac = (1-0)/(3-0) = 1/3 exactly. It must be 'early'.
    assert phase[df["checkpoint_step"] == 1].iloc[0] == "early"


def test_assign_phase_upper_boundary_two_thirds_is_middle():
    df = _fourstep_frame()
    phase = assign_phase(df)
    # Step 2 has frac = 2/3 exactly. It must be 'middle', not 'late'.
    assert phase[df["checkpoint_step"] == 2].iloc[0] == "middle"


# =============================================================================
# 3) _apply_canonical_fills — registry-driven semantic dispatch
# =============================================================================
"""
Claim:
    The fill respects four-valued missingness:
      APPLICABLE_ABSENT_SO_FAR    NaN -> 0
      UNKNOWN_DUE_TO_MISSING_ARTIFACT  NaN -> NaN (preserved)
      NOT_APPLICABLE_TO_SOURCE    NaN -> NaN (preserved)
    The dispatch is per-row source: a feature applicable to source A
    but not B fills only A's NaN cells; B's stay NaN.

Plausible wrong implementations:
    - Blanket `df.fillna(0)` for all numeric features.
    - Reads `missingness_semantic` directly without consulting
      `populated_on`, treating every source uniformly.
    - Coerces string columns through fillna and corrupts identity
      fields like `source` or `run_id`.
"""


def test_canonical_fills_zeros_applicable_absent_so_far():
    df = pd.DataFrame([
        {"run_id": "r0", "source": "swe_agent_pilot", "checkpoint_id": "r0::0",
         "num_reopens_so_far": float("nan")},
    ])
    out = _apply_canonical_fills(df)
    assert out["num_reopens_so_far"].iloc[0] == 0


def test_canonical_fills_preserves_nan_for_unknown_due_to_missing_artifact():
    # `initial_files_count` has UNKNOWN_DUE_TO_MISSING_ARTIFACT semantic;
    # the contract is that the fill MUST be None, so NaN survives.
    df = pd.DataFrame([
        {"run_id": "r0", "source": "swe_agent_pilot", "checkpoint_id": "r0::0",
         "initial_files_count": float("nan")},
    ])
    out = _apply_canonical_fills(df)
    assert pd.isna(out["initial_files_count"].iloc[0])


def test_canonical_fills_skips_sources_not_in_populated_on():
    # Same APPLICABLE_ABSENT_SO_FAR feature, two sources: one declared
    # in the registry (`tb_live`) and one fictional (`imaginary_src`).
    # tb_live -> 0 (in populated_on); imaginary_src -> NaN preserved
    # (NOT_APPLICABLE_TO_SOURCE → fill is None).
    df = pd.DataFrame([
        {"run_id": "r0", "source": "tb_live", "checkpoint_id": "r0::0",
         "num_reopens_so_far": float("nan")},
        {"run_id": "r1", "source": "imaginary_src", "checkpoint_id": "r1::0",
         "num_reopens_so_far": float("nan")},
    ])
    out = _apply_canonical_fills(df)
    tb = out.loc[out["source"] == "tb_live", "num_reopens_so_far"].iloc[0]
    img = out.loc[out["source"] == "imaginary_src", "num_reopens_so_far"].iloc[0]
    assert tb == 0
    assert pd.isna(img), (
        "imaginary_src is not in num_reopens_so_far.populated_on, "
        "so its NaN must be preserved (NOT_APPLICABLE_TO_SOURCE)"
    )


def test_canonical_fills_does_not_touch_string_columns():
    df = pd.DataFrame([
        {"run_id": "r0", "source": "swe_agent_pilot", "checkpoint_id": "r0::0",
         "num_reopens_so_far": float("nan")},
    ])
    out = _apply_canonical_fills(df)
    assert out["run_id"].iloc[0] == "r0"
    assert out["source"].iloc[0] == "swe_agent_pilot"
    assert out["checkpoint_id"].iloc[0] == "r0::0"


# =============================================================================
# 4) build_split('temporal') — propagation of the synthetic-timestamps warning
# =============================================================================
"""
Claim:
    `build_split('temporal', df)` calls the protocol's `temporal()`,
    which warns when `timestamp_quality` contains any non-'real' value.
    The builder must not silence that warning.

Plausible wrong implementations:
    - Wraps the call in `warnings.catch_warnings()` to suppress noise
      during JSON emission, accidentally eating the synthetic warning.
    - Filters by message and drops the synthetic-timestamps warning.
"""


def test_build_split_temporal_propagates_synthetic_timestamps_warning():
    df = pd.DataFrame([
        {"run_id": "r0", "source": "src", "checkpoint_id": "r0::0",
         "checkpoint_step": 0,
         "checkpoint_wall_time": pd.Timestamp("2026-01-01"),
         "timestamp_quality": "synthetic"},
        {"run_id": "r1", "source": "src", "checkpoint_id": "r1::0",
         "checkpoint_step": 0,
         "checkpoint_wall_time": pd.Timestamp("2026-01-02"),
         "timestamp_quality": "synthetic"},
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = build_split("temporal", df)
    assert s is not None
    msgs = [str(w.message) for w in caught]
    assert any("synthetic" in m for m in msgs), (
        f"expected the temporal() synthetic-timestamps warning to surface "
        f"through build_split; got {msgs}"
    )
