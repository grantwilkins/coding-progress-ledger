from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BL_PATH = ROOT / "scripts" / "build_q_labels.py"
_spec = importlib.util.spec_from_file_location("build_q_labels", _BL_PATH)
build_q_labels = importlib.util.module_from_spec(_spec)
sys.modules["build_q_labels"] = build_q_labels
_spec.loader.exec_module(build_q_labels)

CKPT = ROOT / "datasets" / "swe_agent_estimator_checkpoints.csv"
RUNS = ROOT / "runs" / "swe_agent_pilot"


def _row(rows: list[dict], run_id: str, step: int) -> dict:
    for r in rows:
        if r["run_id"] == run_id and r["step"] == step:
            return r
    raise KeyError(f"{run_id}@{step}")


def test_one_row_per_checkpoint():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    with CKPT.open() as fh:
        ckpt_count = sum(1 for _ in csv.DictReader(fh))
    assert len(rows) == ckpt_count == 191


def test_horizon_column_recorded():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    assert all(r["horizon_steps"] == 5 for r in rows)


def test_f_02_stuck_loop_in_window_step_12():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    r = _row(rows, "swe_agent_pilot_f_02", 12)
    assert r["stuck_loop_next_window"] is True


def test_f_02_stuck_loop_outside_window_step_11():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    r = _row(rows, "swe_agent_pilot_f_02", 11)
    assert r["stuck_loop_next_window"] is False


def test_f_02_stuck_loop_masked_when_already_seen_step_17():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    r = _row(rows, "swe_agent_pilot_f_02", 17)
    assert r["stuck_loop_next_window"] is False


def test_s_03_product_reopen_in_window_step_19():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    r = _row(rows, "swe_agent_pilot_s_03", 19)
    assert r["product_reopened_after_completion"] is True
    assert r["future_progress_drop"] is True


def test_s_03_no_reopen_outside_window_step_15():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    r = _row(rows, "swe_agent_pilot_s_03", 15)
    assert r["product_reopened_after_completion"] is False


def test_f_06_no_reopens_anywhere():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    f06 = [r for r in rows if r["run_id"] == "swe_agent_pilot_f_06"]
    assert f06
    assert all(r["product_reopened_after_completion"] is False for r in f06)


def test_submit_without_validation_constant_per_run():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    by_run: dict[str, set[bool]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], set()).add(r["submit_without_validation_state"])
    for run_id, vals in by_run.items():
        assert len(vals) == 1, f"{run_id}: terminal label varies across rows"


def test_f_01_terminal_swv_true():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    f01 = [r for r in rows if r["run_id"] == "swe_agent_pilot_f_01"]
    assert all(r["submit_without_validation_state"] is True for r in f01)


def test_f_02_terminal_swv_false():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    f02 = [r for r in rows if r["run_id"] == "swe_agent_pilot_f_02"]
    assert all(r["submit_without_validation_state"] is False for r in f02)


def test_horizon_zero_yields_all_false_horizon_dependent():
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=0)
    for r in rows:
        assert r["future_progress_drop"] is False
        assert r["product_reopened_after_completion"] is False
        assert r["validation_exposes_new_work"] is False
        assert r["stuck_loop_next_window"] is False


def test_csv_round_trip(tmp_path):
    out = tmp_path / "q_labels.csv"
    rows = build_q_labels.build_labels(RUNS, CKPT, horizon=5)
    build_q_labels.write_csv(rows, out)
    with out.open() as fh:
        reader = csv.DictReader(fh)
        out_rows = list(reader)
    assert len(out_rows) == len(rows)
    assert reader.fieldnames == list(build_q_labels.OUTPUT_COLUMNS)
    swv_set = {r["submit_without_validation_state"] for r in out_rows}
    assert swv_set <= {"true", "false"}


def test_label_columns_disjoint_from_w3_features():
    with CKPT.open() as fh:
        w3_columns = next(csv.reader(fh))
    label_cols = set(build_q_labels.TARGET_COLUMNS)
    assert label_cols.isdisjoint(set(w3_columns))
