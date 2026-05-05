"""
Claim:
The Workstream G baseline ladder fits constant / time-only / ledger-basic
predictors and evaluates them under cross-validated splits. Bootstrap CIs
are taken at the RUN level (not the row level). Predicted probabilities
are clipped to (0.001, 0.999) to keep parity with upstream
`q_baselines.py`. ECE uses 10-bin equal-width binning. AUROC averages
ranks within tie groups.

Plausible wrong implementations:
- bootstrap resamples rows instead of runs (CIs too tight)
- AUROC assigns naive ranks within ties (wrong on common-tie cases)
- ECE uses equal-frequency bins instead of equal-width
- OUTPUT_CLIP set to a different constant (e.g. 1e-6) breaking parity
- ledger-basic baseline silently includes dynamics/validation features
- _wide_targets quietly takes the first of a duplicated label row
- feasible=False path falls through and produces zeros instead of None
- time-only baseline always uses wallclock cols even on multi-source train
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from coding_estimator.baselines import (
    CONSTANT,
    LEDGER_BASIC,
    TIME_ONLY,
    fit_binary,
)
from coding_estimator.checkpoints.features.registry import GROUPS
from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.harness import evaluate_cell
from coding_estimator.eval.metrics import (
    LOG_LOSS_CLIP,
    OUTPUT_CLIP,
    auroc,
    brier,
    ece,
)
from coding_estimator.splits.protocol import Fold, Split, loro
from scripts.run_baselines import _wide_targets, run


# --- (1) bootstrap really resamples runs, not rows -------------------------


def test_bootstrap_brier_resamples_at_run_level_not_row_level():
    """Two runs of 100 rows each. Run A: y=0,p=0 (Brier=0). Run B: y=1,p=0
    (Brier=1). Run-level bootstrap of 2 runs gives 25% all-A, 50% mixed,
    25% all-B → CI must reach into 0 and 1. Row-level bootstrap of 200
    rows would concentrate near 0.5 with tight CI.
    """
    n = 100
    bundle = {
        "A": (np.zeros(n, dtype=int), np.zeros(n)),
        "B": (np.ones(n, dtype=int), np.zeros(n)),
    }
    lo, hi = bootstrap_brier_ci(bundle, b=4000, seed=0)
    assert lo < 0.05, f"lo={lo} too high; bootstrap must hit the all-A run"
    assert hi > 0.95, f"hi={hi} too low; bootstrap must hit the all-B run"


def test_bootstrap_single_run_gives_degenerate_ci():
    """With only one run the resample is always that same run; CI must
    collapse to the single-run Brier."""
    bundle = {"only": (np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))}
    lo, hi = bootstrap_brier_ci(bundle, b=200, seed=0)
    assert lo == hi == pytest.approx(0.25)


# --- (2) AUROC tie-handling matches upstream q_baselines convention --------


def test_auroc_average_rank_on_tied_predictions():
    """Hand-worked: y=[0,1,0,1,1], p=[0.1,0.5,0.5,0.5,0.9].
    Ordered: ranks 1,2,3,4,5. Tie group covers ranks 2..4 (avg=3).
    rank_sum = 3*(1+0+1) + 5*1 = 11. pos=3, neg=2.
    AUROC = (11 - 3*4/2)/(3*2) = 5/6.
    """
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.5, 0.5, 0.5, 0.9])
    assert auroc(y, p) == pytest.approx(5.0 / 6.0)


def test_auroc_returns_none_when_one_class_only():
    assert auroc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None
    assert auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_auroc_perfect_ordering_is_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, p) == pytest.approx(1.0)


# --- (3) ECE 10-bin equal-width on a hand-worked case ---------------------


def test_ece_equal_width_distinguishes_from_equal_frequency():
    """5 samples at (p=0.05, y=avg 0.2) → bin 0, contribution 0.5*0.15.
    5 samples at (p=0.95, y=avg 0.8) → bin 9, contribution 0.5*0.15.
    Equal-width ECE = 0.15. Equal-frequency 10-bin ECE on the same data
    would put each row in its own bin and yield 0.23.
    """
    p = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
    y = np.array([0,    0,    0,    0,    1,    1,    1,    1,    1,    0])
    assert ece(y, p, n_bins=10) == pytest.approx(0.15, abs=1e-12)


def test_ece_perfect_calibration_is_zero():
    """Bin 0 (p=0.05): empirical pos rate 5/100 = 0.05 → matches.
    Bin 9 (p=0.95): empirical pos rate 95/100 = 0.95 → matches.
    Both bins are perfectly calibrated, so ECE = 0."""
    p = np.array([0.05] * 100 + [0.95] * 100)
    y = np.array([0] * 95 + [1] * 5 + [0] * 5 + [1] * 95)
    assert ece(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-12)


# --- (4) constant baseline returns training base rate exactly -------------


def test_constant_baseline_returns_training_positive_rate():
    """fit_binary with empty feature set must return the train-set
    positive rate as the prediction for every row."""
    X = pd.DataFrame({"feat": [1, 2, 3, 4, 5]})
    y = np.array([0, 0, 1, 1, 1])  # base rate = 3/5
    fitted = fit_binary(CONSTANT, X, y, ("tb_live",))
    probs = fitted.predict_proba(pd.DataFrame({"feat": [9, 9]}))
    assert probs.shape == (2,)
    assert np.all(probs == pytest.approx(0.6))
    assert fitted.model is None


def test_constant_baseline_handles_single_class_train_set():
    X = pd.DataFrame({"feat": [1, 2, 3]})
    y = np.array([0, 0, 0])
    fitted = fit_binary(CONSTANT, X, y, ("tb_live",))
    probs = fitted.predict_proba(pd.DataFrame({"feat": [9]}))
    assert probs[0] == pytest.approx(0.0)


# --- (5) time-only feature columns vary correctly with source tuple -------


def test_time_only_adds_wallclock_cols_only_for_tb_live_alone():
    # `fraction_timeout_consumed` is reserved but not yet populated by the
    # tb_live producer; baseline currently includes only the wall-time
    # column on top of `elapsed_steps`. Re-add the timeout column here once
    # the producer fills it.
    assert TIME_ONLY.feature_cols_for(("tb_live",)) == (
        "elapsed_steps",
        "elapsed_wall_time",
    )


def test_time_only_drops_wallclock_when_train_spans_multiple_sources():
    cols = TIME_ONLY.feature_cols_for(("tb_live", "swe_agent_pilot"))
    assert cols == ("elapsed_steps",)
    cols_other = TIME_ONLY.feature_cols_for(("swe_agent_pilot",))
    assert cols_other == ("elapsed_steps",)


# --- (6) ledger-basic features come from exactly four feature groups ------


def test_ledger_basic_uses_only_closure_frontier_instability_discovery():
    cols = set(LEDGER_BASIC.feature_cols_for(("tb_live",)))
    expected = {
        f.column_name
        for g in ("closure", "frontier", "instability", "discovery")
        for f in GROUPS[g]
        if f.dtype in ("int", "float", "bool")
    }
    assert cols == expected
    forbidden_groups = ("stalling", "validation", "evidence", "time_budget", "source_task")
    forbidden_cols = {
        f.column_name for g in forbidden_groups for f in GROUPS[g]
    }
    assert cols.isdisjoint(forbidden_cols), (
        f"ledger_basic must not include {sorted(cols & forbidden_cols)}"
    )


# --- (7) OUTPUT_CLIP parity with upstream q_baselines --------------------


def test_output_clip_matches_upstream_q_baselines_constant():
    assert OUTPUT_CLIP == (0.001, 0.999)
    assert LOG_LOSS_CLIP < OUTPUT_CLIP[0]


def test_evaluate_cell_clips_predicted_positive_rate_at_lower_bound():
    """Construct a degenerate case where fit_binary's base_rate is
    below 0.001 (1 positive in 2000 train rows). After clipping every
    prediction to 0.001, predicted_positive_rate must equal 0.001 — not
    the unclipped base rate of 1/2000 = 0.0005."""
    rows = []
    labs = []
    # Two large training runs (no positives) and one small training run
    # carrying the lone positive — keeps each LORO fold's training data
    # below the 0.001 threshold.
    for rid, n_neg in [("big_a", 1000), ("big_b", 999)]:
        for t in range(n_neg):
            cid = f"{rid}::{t}"
            rows.append({
                "run_id": rid, "source": "tb_live",
                "checkpoint_id": cid, "elapsed_steps": t,
            })
            labs.append({
                "run_id": rid, "source": "tb_live",
                "checkpoint_id": cid, "target_name": "y",
                "label_value": 0.0, "is_masked": False,
            })
    rows.append({
        "run_id": "carrier", "source": "tb_live",
        "checkpoint_id": "carrier::0", "elapsed_steps": 0,
    })
    labs.append({
        "run_id": "carrier", "source": "tb_live",
        "checkpoint_id": "carrier::0", "target_name": "y",
        "label_value": 1.0, "is_masked": False,
    })
    ck = pd.DataFrame(rows)
    lab = pd.DataFrame(labs)
    split = loro(ck)
    cell = evaluate_cell(
        checkpoints_df=ck, labels_df=lab,
        target="y", spec=CONSTANT, split=split,
        source_slice="tb_live", sources_in_train=("tb_live",),
        feasible=True, bootstrap_b=20,
    )
    # On the LORO folds that hold out big_a or big_b, the carrier run
    # is in training, so train base_rate = 1/2000 = 0.0005, clipped to
    # 0.001. The held-out big_a / big_b runs each get p=0.001 for every
    # checkpoint. The fold holding out 'carrier' has zero positives in
    # train → base_rate=0 → clipped to 0.001. So every predicted
    # probability in the concatenated test set is exactly 0.001.
    assert cell.predicted_positive_rate == pytest.approx(0.001, abs=1e-12)


# --- (8) _wide_targets hard-fails on duplicate label rows -----------------


def test_wide_targets_rejects_duplicate_unmasked_label_rows():
    labs = pd.DataFrame([
        {"run_id": "r1", "source": "tb_live", "checkpoint_id": "r1::1",
         "target_name": "y_success_eventual", "label_value": 1.0,
         "is_masked": False},
        {"run_id": "r1", "source": "tb_live", "checkpoint_id": "r1::1",
         "target_name": "y_success_eventual", "label_value": 0.0,
         "is_masked": False},
    ])
    with pytest.raises(ValueError, match="duplicated"):
        _wide_targets(labs, ("y_success_eventual",))


# --- (9) feasible=False path produces all-None metrics --------------------


def test_evaluate_cell_returns_na_metrics_when_not_feasible():
    cell = evaluate_cell(
        checkpoints_df=pd.DataFrame(),
        labels_df=pd.DataFrame(),
        target="y_success_eventual",
        spec=CONSTANT,
        split=Split("loro", 0, ()),
        source_slice="tb_live",
        sources_in_train=("tb_live",),
        feasible=False,
    )
    assert cell.feasible is False
    assert cell.note == "insufficient data"
    for field in (
        "n_runs_train", "n_runs_test", "n_checkpoints_test",
        "positive_rate_data", "predicted_positive_rate",
        "auroc", "brier", "log_loss", "ece",
        "brier_ci_low", "brier_ci_high",
    ):
        assert getattr(cell, field) is None, f"{field} must be None"


# --- (10) end-to-end smoke: run() over a synthetic 3-run frame -------------


def _synthetic_frames(n_runs: int = 3, n_steps: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    labs: list[dict] = []
    for k in range(n_runs):
        rid = f"r{k}"
        succ = float(k % 2 == 0)
        for t in range(1, n_steps + 1):
            cid = f"{rid}::{t}"
            base = {"run_id": rid, "source": "tb_live",
                    "checkpoint_id": cid, "checkpoint_step": t}
            feats = {
                "elapsed_steps": t,
                "elapsed_wall_time": float(t * 60),
                "fraction_timeout_consumed": t / float(n_steps),
            }
            for f in GROUPS["closure"] + GROUPS["frontier"] + GROUPS["instability"] + GROUPS["discovery"]:
                if f.dtype in ("int", "float", "bool"):
                    feats[f.column_name] = float(rng.uniform(0.0, 1.0))
            rows.append({**base, **feats})
            for tgt, val in [
                ("y_success_eventual", succ),
                ("y_future_progress_drop_h5", float(rng.integers(0, 2))),
                ("y_validation_new_work_h5", 0.0),
                ("y_submit_without_validation", succ),
                ("y_timeout", 1.0 - succ),
            ]:
                labs.append({
                    "run_id": rid, "source": "tb_live",
                    "checkpoint_id": cid, "target_name": tgt,
                    "label_value": val, "is_masked": False,
                })
    return pd.DataFrame(rows), pd.DataFrame(labs)


def test_run_baselines_smoke_emits_one_row_per_target_model_cell(tmp_path):
    ck, lab = _synthetic_frames()
    ck_path = tmp_path / "ck.parquet"
    lab_path = tmp_path / "lab.parquet"
    ck.to_parquet(ck_path)
    lab.to_parquet(lab_path)
    out_dir = tmp_path / "reports"
    csv_path = run(checkpoints_path=ck_path, labels_path=lab_path, out_dir=out_dir)
    out = pd.read_csv(csv_path)
    # One source × 5 targets × 3 baselines per scheme; both `loro` and
    # `ltfo` cells are emitted (ltfo is infeasible — tb_live has no
    # task_family — so its rows carry feasible=False). LOSO is skipped
    # because there is only one source.
    expected_rows = 5 * 3 * 2
    assert len(out) == expected_rows
    assert set(out["model"]) == {"constant", "time_only", "ledger_basic"}
    assert set(out["scheme"]) == {"loro", "ltfo"}
    assert (~out[out["scheme"] == "ltfo"]["feasible"]).all()
    assert (out_dir / "baseline_results.md").exists()
    assert (out_dir / "baseline_calibration.md").exists()


# --- bonus: parity check against direct sklearn on a tiny case ------------


def test_time_only_matches_direct_sklearn_logreg_on_loro_fold():
    """Train sklearn LogReg(elapsed_steps) on 3 of 4 runs, predict on the
    held-out run, clip to OUTPUT_CLIP. The harness must produce the
    same predicted_positive_rate when run as a single-fold split."""
    rows: list[dict] = []
    labs: list[dict] = []
    for rid, succ in [("r1", 1), ("r2", 0), ("r3", 1), ("r4", 0)]:
        for t in range(1, 11):
            cid = f"{rid}::{t}"
            rows.append({"run_id": rid, "source": "swe_agent_pilot",
                         "checkpoint_id": cid, "elapsed_steps": t})
            labs.append({"run_id": rid, "source": "swe_agent_pilot",
                         "checkpoint_id": cid, "target_name": "y",
                         "label_value": float(succ), "is_masked": False})
    ck = pd.DataFrame(rows)
    lab = pd.DataFrame(labs)
    train_ids = ("r1", "r2", "r3")
    test_ids = ("r4",)
    split = Split(scheme="holdout", seed=0, folds=(
        Fold("h", train_ids, test_ids),
    ))
    # Multi-source train tuple ensures TIME_ONLY requests only
    # elapsed_steps (matches the upstream `q_baselines::elapsed_only`
    # feature set) so the comparison below is direct.
    cell = evaluate_cell(
        checkpoints_df=ck, labels_df=lab,
        target="y", spec=TIME_ONLY, split=split,
        source_slice="swe_agent_pilot",
        sources_in_train=("swe_agent_pilot", "hermes_pilot_h5_v2"),
        feasible=True, bootstrap_b=20,
    )

    train_mask = ck["run_id"].isin(train_ids)
    test_mask = ck["run_id"].isin(test_ids)
    X_train = ck.loc[train_mask, ["elapsed_steps"]].to_numpy(dtype=float)
    y_train = (
        lab[lab["run_id"].isin(train_ids)]
        .merge(ck[train_mask][["run_id", "checkpoint_id"]],
               on=["run_id", "checkpoint_id"])
    )["label_value"].astype(int).to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)
    X_test = ck.loc[test_mask, ["elapsed_steps"]].to_numpy(dtype=float)
    expected = np.clip(clf.predict_proba(X_test)[:, 1], 0.001, 0.999)
    assert cell.predicted_positive_rate == pytest.approx(
        float(expected.mean()), abs=1e-9
    )
    y_test = np.zeros(len(X_test), dtype=int)
    assert cell.brier == pytest.approx(brier(y_test, expected), abs=1e-9)


# --- brier_per_run sanity: rejects mismatched run sets ---------------------


def test_brier_per_run_requires_matching_run_keys():
    with pytest.raises(ValueError):
        brier_per_run(
            {"a": np.array([0])},
            {"b": np.array([0.0])},
        )
