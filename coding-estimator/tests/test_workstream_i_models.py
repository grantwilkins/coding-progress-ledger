"""
Claim:
Workstream I ships two binary model families and a shared evaluation /
bundle layer.

1. `EmpiricalBinModel` predicts the empirical training-fold positive
   rate of the `coding_progress quartile x elapsed quartile` cell and
   falls back to the global base rate for unseen cells.

2. `fit_logreg` consumes exactly the G4 feature set
   (closure/frontier/instability/discovery), and on a clean synthetic
   problem its coefficient signs follow the feature direction.

3. `train_logreg_bundle` writes the required bundle artifacts and
   evaluates only the four v0 headline binary targets, not the broader
   binary set used elsewhere in the repo.

4. `predict_proba` outputs are clipped to OUTPUT_CLIP=(0.001, 0.999)
   on BOTH models so the pickled artifact behaves identically to the
   in-eval pipeline.

5. `evaluate_model_cell` annotates `EvalCell.note` with
   `run_constant_target` for run-constant headline targets (per
   V0_TARGETS[name].run_constant_flag) and with a `single_class`
   substring whenever any train fold collapses to a single class.

6. `save_model_bundle` keeps `calibration.json["targets"]` a subset of
   the pickled model dict and propagates each cell's `note`.
   `calibration_source` is `constant` iff some holdout cell carries a
   `single_class` note; infeasibility notes do not flip it.

Plausible wrong implementations:
- empirical-bin averages over progress alone, elapsed alone, or all rows
  instead of the 2D cell
- empirical-bin uses the wrong seam convention at exact bucket edges
- logreg trains on the wrong columns or returns coefficients under the
  wrong feature names (column-shift: correct sign at wrong index)
- logreg bundle training accidentally includes `y_timeout` because it is
  present in baseline code paths
- `predict_proba` returns 0.0 / 1.0 from a degenerate base rate, leaking
  unclipped values to downstream consumers
- elapsed-signal pipeline raises on NaN `elapsed_steps` even when
  `checkpoint_fraction_timeout` is fully finite (priority inversion)
- NaN `coding_progress` at predict time silently maps to `base_rate`
  rather than hard-failing
- `calibration_source` returns `constant` whenever any cell has a note,
  conflating eval-infeasibility with constant-predictor fallback
- `calibration.json` lists targets with no fitted model in `model.pkl`
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from coding_estimator.eval.harness import EvalCell
from coding_estimator.eval.metrics import OUTPUT_CLIP
from coding_estimator.labels.registry import V0_TARGETS
from coding_estimator.models.common import (
    HEADLINE_BINARY_TARGETS,
    RUN_CONSTANT_TARGETS,
    calibration_source_for,
    evaluate_headline_model_suite,
)
from coding_estimator.models.empirical_bin import (
    EmpiricalBinModel,
    fit_empirical_bin,
)
from coding_estimator.models.logreg import (
    fit_logreg,
    g4_feature_columns,
    train_logreg_bundle,
)


def _headline_labels(
    checkpoints_df: pd.DataFrame,
    *,
    include_timeout: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for run_id, sub in checkpoints_df.groupby("run_id", sort=True):
        success = float(run_id in {"r0", "r2"})
        for _, row in sub.iterrows():
            cid = row["checkpoint_id"]
            target_values = {
                "y_success_eventual": success,
                "y_future_progress_drop_h5": float(row["coding_progress"] < 0.5),
                "y_validation_new_work_h5": float(row["coding_progress"] > 0.75),
                "y_submit_without_validation": float(run_id in {"r1", "r3"}),
            }
            if include_timeout:
                target_values["y_timeout"] = 1.0 - success
            for target, value in target_values.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "source": row["source"],
                        "checkpoint_id": cid,
                        "target_name": target,
                        "label_value": value,
                        "is_masked": False,
                    }
                )
    return pd.DataFrame(rows)


def _g4_frame(n_runs: int = 4, n_steps: int = 8) -> pd.DataFrame:
    rows: list[dict] = []
    for run_idx in range(n_runs):
        run_id = f"r{run_idx}"
        source = "tb_live" if run_idx < 2 else "swe_agent_pilot"
        for step in range(n_steps):
            progress = step / float(n_steps - 1)
            rows.append(
                {
                    "run_id": run_id,
                    "source": source,
                    "checkpoint_id": f"{run_id}::{step}",
                    "checkpoint_step": step,
                    "coding_progress": progress,
                    "completed_leaf_count": float(step),
                    "validation_progress": 0.0,
                    "product_progress": 0.0,
                    "investigation_progress": 0.0,
                    "active_leaf_count": float(n_steps - step),
                    "active_coding_leaf_count": float(n_steps - step),
                    "active_validation_leaf_count": 0.0,
                    "num_reopens_so_far": 0.0,
                    "num_invalidations_so_far": 0.0,
                    "num_deletes_so_far": 0.0,
                    "largest_progress_drop_so_far": 0.0,
                    "num_progress_drops_so_far": float(run_idx % 2),
                    "steps_since_last_drop": float(step),
                    "num_adds_so_far": float(step),
                    "num_splits_so_far": 0.0,
                    "denominator_growth_so_far": 0.0,
                    "steps_since_new_subtask": float(step),
                    "new_leaf_count_last_1_steps": 1.0,
                    "new_leaf_count_last_3_steps": 1.0,
                    "new_leaf_count_last_5_steps": 1.0,
                    "elapsed_steps": float(step),
                    "checkpoint_fraction_timeout": progress,
                }
            )
    return pd.DataFrame(rows)


def test_empirical_bin_uses_2d_cell_rate_and_global_fallback() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "checkpoint_id": "r0::0",
                "coding_progress": 0.10,
                "elapsed_steps": 0.0,
                "checkpoint_fraction_timeout": 0.10,
            },
            {
                "run_id": "r1",
                "checkpoint_id": "r1::0",
                "coding_progress": 0.20,
                "elapsed_steps": 1.0,
                "checkpoint_fraction_timeout": 0.20,
            },
            {
                "run_id": "r2",
                "checkpoint_id": "r2::0",
                "coding_progress": 0.80,
                "elapsed_steps": 2.0,
                "checkpoint_fraction_timeout": 0.80,
            },
            {
                "run_id": "r3",
                "checkpoint_id": "r3::0",
                "coding_progress": 0.90,
                "elapsed_steps": 3.0,
                "checkpoint_fraction_timeout": 0.90,
            },
        ]
    )
    y = np.array([0, 0, 1, 1], dtype=int)
    model = fit_empirical_bin(train, y, ("tb_live",))

    test = pd.DataFrame(
        [
            {
                "run_id": "known_low",
                "checkpoint_id": "known_low::0",
                "coding_progress": 0.15,
                "elapsed_steps": 0.5,
                "checkpoint_fraction_timeout": 0.15,
            },
            {
                "run_id": "known_high",
                "checkpoint_id": "known_high::0",
                "coding_progress": 0.85,
                "elapsed_steps": 2.5,
                "checkpoint_fraction_timeout": 0.85,
            },
            {
                "run_id": "unseen_mix",
                "checkpoint_id": "unseen_mix::0",
                "coding_progress": 0.15,
                "elapsed_steps": 2.5,
                "checkpoint_fraction_timeout": 0.85,
            },
        ]
    )
    probs = model.predict_proba(test)
    assert probs[0] == 0.001
    assert probs[1] == 0.999
    assert probs[2] == 0.5


def test_empirical_bin_exact_edge_uses_upper_bucket() -> None:
    model = EmpiricalBinModel(
        progress_edges=(0.25, 0.5, 0.75),
        elapsed_edges=(0.25, 0.5, 0.75),
        elapsed_min=0.0,
        elapsed_max=10.0,
        base_rate=0.1,
        bin_rates={(2, 2): 0.9},
    )
    X = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "checkpoint_id": "r0::0",
                "coding_progress": 0.5,
                "elapsed_steps": 5.0,
                "checkpoint_fraction_timeout": 0.5,
            }
        ]
    )
    assert model.predict_proba(X)[0] == 0.9


def test_logreg_coefficients_follow_feature_direction_on_clean_problem() -> None:
    frame = _g4_frame()
    target = (
        (frame["coding_progress"] > 0.6)
        & (frame["num_progress_drops_so_far"] < 0.5)
    ).astype(int)
    fitted = fit_logreg(frame, target.to_numpy(), ("tb_live", "swe_agent_pilot"))
    coefs = fitted.coefficient_map()
    assert fitted.feature_columns == g4_feature_columns(("tb_live", "swe_agent_pilot"))
    assert coefs["coding_progress"] > 0
    assert coefs["num_progress_drops_so_far"] < 0


def test_train_logreg_bundle_writes_required_files_and_headline_targets_only(
    tmp_path: Path,
) -> None:
    checkpoints_df = _g4_frame()
    labels_df = _headline_labels(checkpoints_df, include_timeout=True)
    audit = tmp_path / "audit.md"
    audit.write_text("# Audit\n\n---\nOverall: PASS\n", encoding="utf-8")

    out_dir = tmp_path / "models" / "logreg_v0"
    cells = train_logreg_bundle(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        out_dir=out_dir,
        audit_path=audit,
    )

    assert (out_dir / "model.pkl").is_file()
    assert (out_dir / "model_card.md").is_file()
    assert (out_dir / "calibration.json").is_file()
    assert (out_dir / "metrics.csv").is_file()

    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert set(calibration["targets"]) == {
        "y_success_eventual",
        "y_future_progress_drop_h5",
        "y_validation_new_work_h5",
        "y_submit_without_validation",
    }
    assert "y_timeout" not in calibration["targets"]
    assert {cell.model for cell in cells} == {"logreg_v0"}


# ---------------------------------------------------------------------------
# Clipping invariants — pickled artifacts must behave like the eval pipeline.
# ---------------------------------------------------------------------------


def test_empirical_bin_clips_zero_baserate_to_lower_bound() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": f"r{i}",
                "checkpoint_id": f"r{i}::0",
                "coding_progress": 0.1 * i,
                "elapsed_steps": float(i),
                "checkpoint_fraction_timeout": 0.1 * i,
            }
            for i in range(4)
        ]
    )
    model = fit_empirical_bin(train, np.zeros(4, dtype=int), ("tb_live",))
    probs = model.predict_proba(train)
    assert float(probs.min()) == OUTPUT_CLIP[0]
    assert float(probs.max()) == OUTPUT_CLIP[0]


def test_empirical_bin_clips_one_baserate_to_upper_bound() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": f"r{i}",
                "checkpoint_id": f"r{i}::0",
                "coding_progress": 0.1 * i,
                "elapsed_steps": float(i),
                "checkpoint_fraction_timeout": 0.1 * i,
            }
            for i in range(4)
        ]
    )
    model = fit_empirical_bin(train, np.ones(4, dtype=int), ("tb_live",))
    probs = model.predict_proba(train)
    assert float(probs.min()) == OUTPUT_CLIP[1]
    assert float(probs.max()) == OUTPUT_CLIP[1]


def test_logreg_single_class_returns_clipped_base_rate() -> None:
    frame = _g4_frame(n_runs=2, n_steps=4)
    target = np.zeros(len(frame), dtype=int)
    fitted = fit_logreg(frame, target, ("tb_live", "swe_agent_pilot"))
    assert fitted.model is None
    probs = fitted.predict_proba(frame)
    assert float(probs.min()) == OUTPUT_CLIP[0]
    assert float(probs.max()) == OUTPUT_CLIP[0]


# ---------------------------------------------------------------------------
# Empirical-bin elapsed signal: timeout fraction takes priority when finite.
# ---------------------------------------------------------------------------


def test_empirical_bin_predict_succeeds_when_elapsed_steps_nan_but_timeout_frac_finite() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "checkpoint_id": "r0::0",
                "coding_progress": 0.1,
                "elapsed_steps": 0.0,
                "checkpoint_fraction_timeout": 0.1,
            },
            {
                "run_id": "r1",
                "checkpoint_id": "r1::0",
                "coding_progress": 0.9,
                "elapsed_steps": 1.0,
                "checkpoint_fraction_timeout": 0.9,
            },
        ]
    )
    model = fit_empirical_bin(train, np.array([0, 1], dtype=int), ("tb_live",))
    test = pd.DataFrame(
        [
            {
                "run_id": "t0",
                "checkpoint_id": "t0::0",
                "coding_progress": 0.95,
                "elapsed_steps": float("nan"),
                "checkpoint_fraction_timeout": 0.95,
            }
        ]
    )
    probs = model.predict_proba(test)
    assert float(probs[0]) == OUTPUT_CLIP[1]


def test_empirical_bin_predict_falls_back_to_elapsed_when_timeout_frac_partial_nan() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "checkpoint_id": "r0::0",
                "coding_progress": 0.1,
                "elapsed_steps": 0.0,
                "checkpoint_fraction_timeout": 0.1,
            },
            {
                "run_id": "r1",
                "checkpoint_id": "r1::0",
                "coding_progress": 0.9,
                "elapsed_steps": 10.0,
                "checkpoint_fraction_timeout": 0.9,
            },
        ]
    )
    model = fit_empirical_bin(train, np.array([0, 1], dtype=int), ("tb_live",))
    test = pd.DataFrame(
        [
            {
                "run_id": "t0",
                "checkpoint_id": "t0::0",
                "coding_progress": 0.95,
                "elapsed_steps": 10.0,
                "checkpoint_fraction_timeout": float("nan"),
            }
        ]
    )
    probs = model.predict_proba(test)
    assert float(probs[0]) == OUTPUT_CLIP[1]


# ---------------------------------------------------------------------------
# Hard-fail policy: NaN coding_progress must raise, not silently base-rate.
# ---------------------------------------------------------------------------


def test_empirical_bin_predict_hard_fails_on_nan_coding_progress() -> None:
    train = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "checkpoint_id": "r0::0",
                "coding_progress": 0.2,
                "elapsed_steps": 0.0,
                "checkpoint_fraction_timeout": 0.2,
            },
            {
                "run_id": "r1",
                "checkpoint_id": "r1::0",
                "coding_progress": 0.8,
                "elapsed_steps": 1.0,
                "checkpoint_fraction_timeout": 0.8,
            },
        ]
    )
    model = fit_empirical_bin(train, np.array([0, 1], dtype=int), ("tb_live",))
    test = pd.DataFrame(
        [
            {
                "run_id": "t0",
                "checkpoint_id": "t0::0",
                "coding_progress": float("nan"),
                "elapsed_steps": 0.5,
                "checkpoint_fraction_timeout": 0.5,
            }
        ]
    )
    raised = False
    try:
        model.predict_proba(test)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# EvalCell notes: registry-driven run-constant flag + single-class detection.
# ---------------------------------------------------------------------------


def test_run_constant_targets_are_sourced_from_v0_registry() -> None:
    expected = frozenset(
        name
        for name in HEADLINE_BINARY_TARGETS
        if V0_TARGETS[name].run_constant_flag
    )
    assert expected.issubset(RUN_CONSTANT_TARGETS)
    assert "y_future_progress_drop_h5" not in RUN_CONSTANT_TARGETS
    assert "y_validation_new_work_h5" not in RUN_CONSTANT_TARGETS


def test_evaluate_headline_suite_annotates_run_constant_targets() -> None:
    frame = _g4_frame()
    labels = _headline_labels(frame, include_timeout=False)
    cells = evaluate_headline_model_suite(
        checkpoints_df=frame,
        labels_df=labels,
        model_name="logreg_v0",
        fit_model=fit_logreg,
    )
    holdout = [c for c in cells if c.scheme == "holdout"]
    by_target = {c.target: c for c in holdout}
    for run_const in HEADLINE_BINARY_TARGETS:
        if not V0_TARGETS[run_const].run_constant_flag:
            continue
        cell = by_target[run_const]
        assert cell.note is not None
        assert "run_constant_target" in cell.note
    for time_varying in HEADLINE_BINARY_TARGETS:
        if V0_TARGETS[time_varying].run_constant_flag:
            continue
        cell = by_target[time_varying]
        assert cell.note is None or "run_constant_target" not in cell.note


def test_evaluate_cell_marks_single_class_train_folds() -> None:
    frame = _g4_frame(n_runs=4, n_steps=4)
    rows = []
    for run_id in frame["run_id"].unique():
        for cid in frame[frame["run_id"] == run_id]["checkpoint_id"]:
            rows.append(
                {
                    "run_id": run_id,
                    "source": "tb_live" if run_id in {"r0", "r1"} else "swe_agent_pilot",
                    "checkpoint_id": cid,
                    "target_name": "y_future_progress_drop_h5",
                    "label_value": 0.0,
                    "is_masked": False,
                }
            )
    labels = pd.DataFrame(rows)
    cells = evaluate_headline_model_suite(
        checkpoints_df=frame,
        labels_df=labels,
        model_name="logreg_v0",
        fit_model=fit_logreg,
    )
    cell = next(
        c
        for c in cells
        if c.target == "y_future_progress_drop_h5" and c.scheme == "holdout"
    )
    assert cell.note is not None
    assert "single_class" in cell.note


# ---------------------------------------------------------------------------
# Bundle: calibration.json keys ⊆ pickle keys, notes propagate.
# ---------------------------------------------------------------------------


def test_calibration_keys_are_subset_of_pickled_models(tmp_path: Path) -> None:
    frame = _g4_frame()
    labels = _headline_labels(frame, include_timeout=True)
    labels = labels[labels["target_name"] != "y_validation_new_work_h5"]
    audit = tmp_path / "audit.md"
    audit.write_text("# Audit\n\n---\nOverall: PASS\n", encoding="utf-8")

    out_dir = tmp_path / "models" / "logreg_v0"
    train_logreg_bundle(
        checkpoints_df=frame,
        labels_df=labels,
        out_dir=out_dir,
        audit_path=audit,
    )
    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    fitted = pickle.loads((out_dir / "model.pkl").read_bytes())
    assert set(calibration["targets"]).issubset(set(fitted))
    assert "y_validation_new_work_h5" not in calibration["targets"]


def test_calibration_json_carries_run_constant_note(tmp_path: Path) -> None:
    frame = _g4_frame()
    labels = _headline_labels(frame, include_timeout=False)
    audit = tmp_path / "audit.md"
    audit.write_text("# Audit\n\n---\nOverall: PASS\n", encoding="utf-8")

    out_dir = tmp_path / "models" / "logreg_v0"
    train_logreg_bundle(
        checkpoints_df=frame,
        labels_df=labels,
        out_dir=out_dir,
        audit_path=audit,
    )
    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    note = calibration["targets"]["y_success_eventual"]["note"]
    assert note is not None and "run_constant_target" in note


# ---------------------------------------------------------------------------
# calibration_source: constant only on single_class, never on infeasibility.
# ---------------------------------------------------------------------------


def _holdout_cell(target: str, *, note: str | None) -> EvalCell:
    return EvalCell(
        target=target,
        model="logreg_v0",
        scheme="holdout",
        source_slice="holdout->all",
        feasible=True,
        n_runs_train=4,
        n_runs_test=4,
        n_checkpoints_test=16,
        positive_rate_data=0.5,
        predicted_positive_rate=0.5,
        auroc=0.7,
        brier=0.2,
        log_loss=0.5,
        ece=0.05,
        brier_ci_low=0.1,
        brier_ci_high=0.3,
        note=note,
    )


def test_calibration_source_raw_when_no_single_class_notes() -> None:
    cells = [_holdout_cell(t, note=None) for t in HEADLINE_BINARY_TARGETS]
    assert calibration_source_for(cells) == "raw"


def test_calibration_source_constant_when_any_single_class_note() -> None:
    cells = [_holdout_cell(t, note=None) for t in HEADLINE_BINARY_TARGETS[:-1]]
    cells.append(
        _holdout_cell(HEADLINE_BINARY_TARGETS[-1], note="all_train_folds_single_class")
    )
    assert calibration_source_for(cells) == "constant"


def test_calibration_source_raw_when_only_infeasibility_notes() -> None:
    cells = [_holdout_cell(t, note="run_constant_target") for t in HEADLINE_BINARY_TARGETS]
    assert calibration_source_for(cells) == "raw"


# ---------------------------------------------------------------------------
# Logreg coefficient under a single-driver problem must dominate by |coef|.
# ---------------------------------------------------------------------------


def test_logreg_coefficient_dominates_under_single_driver_problem() -> None:
    rng = np.random.default_rng(0)
    n_runs = 6
    n_steps = 12
    frame = _g4_frame(n_runs=n_runs, n_steps=n_steps)
    # The label is a deterministic step function of `coding_progress` only;
    # all other G4 features carry no signal.
    target = (frame["coding_progress"] > 0.55).astype(int).to_numpy()
    # Make every other feature random so they cannot be informative.
    for col in frame.columns:
        if col in {"run_id", "source", "checkpoint_id", "checkpoint_step", "coding_progress"}:
            continue
        if frame[col].dtype.kind in "fi":
            frame[col] = rng.normal(size=len(frame))
    fitted = fit_logreg(frame, target, ("tb_live", "swe_agent_pilot"))
    coefs = fitted.coefficient_map()
    driver = coefs["coding_progress"]
    others = {k: abs(v) for k, v in coefs.items() if k != "coding_progress"}
    assert driver > 0
    assert abs(driver) == max([abs(driver), *others.values()])


# ---------------------------------------------------------------------------
# Headline suite never emits y_timeout cells, even when its labels exist.
# ---------------------------------------------------------------------------


def test_evaluate_headline_suite_excludes_y_timeout_when_labels_present() -> None:
    frame = _g4_frame()
    labels = _headline_labels(frame, include_timeout=True)
    assert (labels["target_name"] == "y_timeout").any()
    cells = evaluate_headline_model_suite(
        checkpoints_df=frame,
        labels_df=labels,
        model_name="logreg_v0",
        fit_model=fit_logreg,
    )
    assert all(cell.target != "y_timeout" for cell in cells)
    assert {cell.target for cell in cells if cell.scheme == "holdout"} == set(
        HEADLINE_BINARY_TARGETS
    )
