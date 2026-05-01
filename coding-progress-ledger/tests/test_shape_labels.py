import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


labeler = _load("label_observation_shapes")
PILOT = ROOT / "runs" / "swe_agent_pilot"
LIVE = ROOT / "runs" / "swe_agent_live"


def _label(run_id: str, root: Path = PILOT):
    return labeler.label_run(root / run_id)


def test_f_06_high_progress_failure_and_hidden_work_gap():
    r = _label("swe_agent_pilot_f_06")
    assert r.final_success is False
    assert r.final_coding_progress >= labeler.HIGH_PROGRESS_THRESHOLD
    assert "high_progress_failure" in r.tags
    assert "hidden_work_gap" in r.tags


def test_s_04_low_progress_success_and_submit_without_validation():
    r = _label("swe_agent_pilot_s_04")
    assert r.final_success is True
    assert r.final_coding_progress < labeler.HIGH_PROGRESS_THRESHOLD
    assert "low_progress_success" in r.tags
    assert "submit_without_validation" in r.tags


@pytest.mark.parametrize("run_id", ["swe_agent_pilot_f_02", "swe_agent_pilot_f_03"])
def test_stuck_loop_pilots(run_id):
    r = _label(run_id)
    assert "stuck_loop" in r.tags


def test_s_03_nonmonotone_recovery():
    r = _label("swe_agent_pilot_s_03")
    assert "nonmonotone_recovery" in r.tags
    assert r.final_success is True


def test_clean_success_excludes_failure_tags():
    r = _label("swe_agent_pilot_s_01")
    assert r.clean_success is True
    assert "high_progress_failure" not in r.tags
    assert "submit_without_validation" not in r.tags
    assert "no_validation_frontier" not in r.tags


def test_threshold_anchor_is_70_per_w4():
    # f_08 sits at 0.7143 in the current spec; W4 anchored the boundary at 0.70.
    r = _label("swe_agent_pilot_f_08")
    assert r.final_success is False
    assert r.final_coding_progress >= 0.70
    assert "high_progress_failure" in r.tags


def test_no_validation_frontier_for_runs_without_val_subtask():
    # f_02 / f_03 are stuck loops that never opened a validation leaf.
    for rid in ("swe_agent_pilot_f_02", "swe_agent_pilot_f_03"):
        r = _label(rid)
        assert "no_validation_frontier" in r.tags


def test_live_progress_one_runs_are_definitively_classified():
    """Every live N=20 run with progress >= 1.0 must carry one of:
    no_validation_frontier, clean success, or high_progress_failure.
    No silent ambiguous classification allowed (W2 acceptance, with the
    third bucket added so failures-with-validation aren't silently
    misclassified as clean success)."""
    rows = labeler.label_runs(LIVE)
    assert rows, "expected live runs at runs/swe_agent_live"
    high = [r for r in rows if r.final_coding_progress >= 1.0 - 1e-9]
    assert high, "expected at least one progress=1.0 live run"
    for r in high:
        classifications = {
            "no_validation_frontier" in r.tags,
            r.clean_success,
            "high_progress_failure" in r.tags,
        }
        assert any(classifications), (
            f"{r.run_id}: progress=1.0 with no definitive classification; "
            f"tags={sorted(r.tags)} clean_success={r.clean_success}"
        )


def test_csv_and_report_artifacts_exist():
    csv_path = ROOT / "datasets" / "swe_agent_pilot_shape_labels.csv"
    report = ROOT / "datasets" / "swe_agent_pilot_shape_report.md"
    assert csv_path.is_file()
    assert report.is_file()
    header = csv_path.read_text().splitlines()[0]
    for tag in labeler.SHAPE_TAGS:
        assert tag in header
    body = report.read_text()
    assert "audit labels" in body
    assert "high_progress_failure" in body


def test_label_runs_is_deterministic():
    a = labeler.label_runs(PILOT)
    b = labeler.label_runs(PILOT)
    assert [(r.run_id, sorted(r.tags), r.clean_success) for r in a] == [
        (r.run_id, sorted(r.tags), r.clean_success) for r in b
    ]
