import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bldr = _load("build_estimator_checkpoints")
PILOT = ROOT / "runs" / "swe_agent_pilot"
STEP_CSV = ROOT / "datasets" / "swe_agent_pilot_observations_step.csv"
SHAPE_CSV = ROOT / "datasets" / "swe_agent_pilot_shape_labels.csv"
CKPT_CSV = ROOT / "datasets" / "swe_agent_estimator_checkpoints.csv"


def _read(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _build():
    return bldr.build_checkpoints(PILOT, STEP_CSV, horizon=30, shape_labels_csv=SHAPE_CSV)


def test_one_row_per_retained_step():
    rows = _build()
    step_rows = _read(STEP_CSV)
    keyed_step = {(r["run_id"], int(r["step"])) for r in step_rows}
    keyed_ckpt = {(r["run_id"], r["step"]) for r in rows}
    assert keyed_ckpt == keyed_step


def test_label_columns_are_prefixed_and_features_are_not():
    feature_cols = set(bldr.FEATURE_COLUMNS)
    label_cols = set(bldr.LABEL_COLUMNS)
    assert all(c.startswith("label_") for c in label_cols)
    assert all(not c.startswith("label_") for c in feature_cols)
    assert feature_cols.isdisjoint(label_cols)


def test_no_future_features_at_step_zero():
    rows = _build()
    step_zero = [r for r in rows if r["step"] == 0]
    assert step_zero
    for r in step_zero:
        # Init event: no subtasks, no completions, no progress. Features must
        # only reflect at-or-before state — all zero/false at step 0.
        assert r["active_leaf_count"] == 0
        assert r["completed_leaf_count"] == 0
        assert r["coding_progress"] == 0
        assert r["num_reopens_so_far"] == 0
        assert r["strong_completion_count"] == 0
        assert r["validation_complete"] is False
        assert r["submit_without_validation"] is False


def test_coding_progress_matches_step_csv_at_final_row():
    rows = _build()
    step_rows = _read(STEP_CSV)
    by_run_step = {(r["run_id"], int(r["step"])): r for r in step_rows}
    by_run = {}
    for r in rows:
        prev = by_run.get(r["run_id"])
        if prev is None or r["step"] > prev["step"]:
            by_run[r["run_id"]] = r
    for run_id, ckpt in by_run.items():
        ref = by_run_step[(run_id, ckpt["step"])]
        assert abs(float(ckpt["coding_progress"]) - float(ref["coding_progress"])) < 1e-6, run_id


def test_s_04_submit_without_validation_at_final_step():
    rows = _build()
    final_s_04 = max(
        (r for r in rows if r["run_id"] == "swe_agent_pilot_s_04"),
        key=lambda r: r["step"],
    )
    assert final_s_04["submit_without_validation"] is True
    assert final_s_04["validation_complete"] is False


def test_f_06_validation_complete_but_failure_label():
    rows = _build()
    final = max(
        (r for r in rows if r["run_id"] == "swe_agent_pilot_f_06"),
        key=lambda r: r["step"],
    )
    assert final["validation_complete"] is True
    assert final["coding_progress"] == 1
    assert final["label_final_success"] is False
    assert "hidden_work_gap" in final["label_shape_tags"]
    assert "high_progress_failure" in final["label_shape_tags"]


def test_blocked_leaf_count_fires_on_stuck_loop_pilots():
    rows = _build()
    for run_id in ("swe_agent_pilot_f_02", "swe_agent_pilot_f_03"):
        final = max((r for r in rows if r["run_id"] == run_id), key=lambda r: r["step"])
        assert final["blocked_leaf_count"] >= 1
        assert final["repeated_observation_loop_flag"] is True


def test_no_future_leakage_in_reopens_count():
    """At any step S, num_reopens_so_far must equal the count of REOPEN
    events with step <= S in the source ledger."""
    import json
    from ledger_progress.serialization import event_from_dict
    from ledger_progress import EventType
    rows = _build()
    for run_id in {"swe_agent_pilot_s_03", "swe_agent_pilot_s_05", "swe_agent_pilot_f_09"}:
        events = []
        with (PILOT / run_id / "ledger.jsonl").open() as fh:
            for line in fh:
                events.append(event_from_dict(json.loads(line)))
        for r in [x for x in rows if x["run_id"] == run_id]:
            expected = sum(1 for e in events
                           if e.step <= r["step"] and e.event_type is EventType.REOPEN_SUBTASK)
            assert r["num_reopens_so_far"] == expected, (run_id, r["step"])


def test_largest_drop_is_monotone_nondecreasing():
    rows = _build()
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    for run_id, run_rows in by_run.items():
        run_rows.sort(key=lambda x: x["step"])
        prev = 0.0
        for r in run_rows:
            v = float(r["largest_progress_drop_so_far"])
            assert v >= prev - 1e-9, (run_id, r["step"], v, prev)
            prev = v


def test_all_feature_groups_present_in_columns():
    cols = set(bldr.FEATURE_COLUMNS)
    required = {
        "active_leaf_count", "active_coding_leaf_count", "active_validation_leaf_count",
        "completed_leaf_count", "coding_progress", "validation_progress",
        "num_reopens_so_far", "num_invalidations_so_far", "largest_progress_drop_so_far",
        "num_splits_so_far", "steps_since_new_subtask", "denominator_growth_so_far",
        "steps_since_completion", "blocked_leaf_count", "repeated_observation_loop_flag",
        "validation_started", "validation_complete", "validation_failed",
        "submit_without_validation",
        "strong_completion_count", "manual_only_completion_count",
        "weak_product_completion_count",
    }
    assert required.issubset(cols)


def test_success_by_horizon_label_uses_finish_step():
    rows = _build()
    for r in rows:
        success = r["label_final_success"]
        finish = r["label_finish_step"]
        sbh = r["label_success_by_horizon"]
        if success is True and finish <= 30:
            assert sbh is True
        else:
            assert sbh is False


def test_csv_artifact_matches_in_memory_build():
    assert CKPT_CSV.is_file()
    on_disk = _read(CKPT_CSV)
    rows = _build()
    assert len(on_disk) == len(rows)
    assert on_disk[0].keys() == set(bldr.ALL_COLUMNS) or set(on_disk[0].keys()) == set(bldr.ALL_COLUMNS)


def test_legacy_retrospective_runs_supported():
    """Legacy retrospective ledgers (no timestamps) should produce checkpoints
    without error. The pilot is the canonical legacy retrospective dataset."""
    rows = _build()
    assert any(r["run_id"].startswith("swe_agent_pilot_") for r in rows)
    assert all(isinstance(r["step"], int) for r in rows)
