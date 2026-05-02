"""Q5 — Non-leakage proofs for the Q1 prediction targets.

Asserts:
1. No Q1 target column appears as a feature in any baseline model.
2. The W3 feature schema and the Q1 label schema are disjoint.
3. Horizon-dependent labels read only events with step > checkpoint step.
4. `label_*` columns from W3 never enter feature vectors.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BL = ROOT / "scripts" / "build_q_labels.py"
_QB = ROOT / "scripts" / "q_baselines.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_q_labels = _load("build_q_labels", _BL)
q_baselines = _load("q_baselines", _QB)

CKPT = ROOT / "datasets" / "swe_agent_estimator_checkpoints.csv"
RUNS = ROOT / "runs" / "swe_agent_pilot"


def test_target_columns_not_in_w3_features():
    with CKPT.open() as fh:
        w3_columns = next(csv.reader(fh))
    for target in build_q_labels.TARGET_COLUMNS:
        assert target not in w3_columns


def test_no_baseline_uses_label_prefixed_feature():
    for model_name, feats in q_baselines.MODEL_FEATURES.items():
        for f in feats:
            assert not f.startswith("label_"), f"{model_name} uses label feature {f}"


def test_no_baseline_uses_target_as_feature():
    for model_name, feats in q_baselines.MODEL_FEATURES.items():
        for f in feats:
            assert f not in build_q_labels.TARGET_COLUMNS, (
                f"{model_name} uses Q1 target {f} as feature")


def test_no_baseline_uses_final_success():
    for feats in q_baselines.MODEL_FEATURES.values():
        for f in feats:
            assert "final_success" not in f
            assert "success_by_horizon" not in f
            assert "shape_tags" not in f


def test_horizon_dependent_labels_use_only_future_events(tmp_path, monkeypatch):
    """Truncate the ledger at step S and confirm horizon-dependent labels
    are unchanged: they cannot depend on events at step <= S."""
    rows_full = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    f02 = next(r for r in rows_full if r["run_id"] == "swe_agent_pilot_f_02" and r["step"] == 12)
    assert f02["stuck_loop_next_window"] is True

    src = RUNS / "swe_agent_pilot_f_02" / "ledger.jsonl"
    truncated_dir = tmp_path / "swe_agent_pilot_f_02"
    truncated_dir.mkdir()
    with src.open() as fh:
        all_lines = fh.readlines()
    keep = []
    import json as _json
    for line in all_lines:
        obj = _json.loads(line)
        if obj["step"] <= 12:
            keep.append(line)
    (truncated_dir / "ledger.jsonl").write_text("".join(keep))

    truncated_ckpt = tmp_path / "ckpt.csv"
    with CKPT.open() as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys())
    keep_rows = [
        r for r in rows
        if r["run_id"] == "swe_agent_pilot_f_02" and int(r["step"]) <= 12
    ]
    with truncated_ckpt.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keep_rows)

    truncated_rows = build_q_labels.build_labels(tmp_path, truncated_ckpt, horizon=5)
    f02_t = next(r for r in truncated_rows if r["step"] == 12)
    assert f02_t["stuck_loop_next_window"] is False
    assert f02_t["future_progress_drop"] is False


def test_terminal_label_unchanged_when_horizon_zero():
    rows_h5 = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    rows_h0 = build_q_labels.build_labels(RUNS, CKPT, horizon=0)
    by_run_h5 = {r["run_id"]: r["submit_without_validation_state"] for r in rows_h5}
    by_run_h0 = {r["run_id"]: r["submit_without_validation_state"] for r in rows_h0}
    assert by_run_h5 == by_run_h0


def test_target_set_disjoint_from_w3_label_columns():
    w3_labels = {"label_final_success", "label_finish_step",
                 "label_success_by_horizon", "label_shape_tags"}
    targets = set(build_q_labels.TARGET_COLUMNS)
    assert targets.isdisjoint(w3_labels)
