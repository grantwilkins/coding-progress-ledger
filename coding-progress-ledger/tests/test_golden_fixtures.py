"""
Claim:
Hand-written event logs define stable progress curves and summary semantics for
canonical ledger scenarios.

Plausible wrong implementations:
- Score after every event instead of after each ledger step.
- Fail to replay grouped same-step discoveries before scoring.
- Drift from the intended non-monotonic progress curve.
- Infer success from progress instead of explicit metadata.
- Report leaf counts that cannot explain weighted progress.
"""

import importlib.util
import json
from pathlib import Path

from conftest import replay_progress_curve
from ledger_progress import load_events_jsonl


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
RESCORE_PATH = ROOT / "scripts" / "rescore_suite_by_category.py"
spec = importlib.util.spec_from_file_location("rescore_suite_by_category", RESCORE_PATH)
rescore = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rescore)


def test_reverse_progress_fixture_matches_expected_curve_and_summary(tmp_path):
    events = load_events_jsonl(str(FIXTURES / "reverse_progress.jsonl"))
    curve = replay_progress_curve(events)

    assert curve == [
        (0, 0.0, 0.0, 0.0),
        (1, 0.0, 4.0, 0.0),
        (3, 2.0, 4.0, 0.5),
        (4, 2.0, 8.0, 0.25),
        (5, 1.0, 8.0, 0.125),
        (8, 8.0, 8.0, 1.0),
    ]

    summary = _rescore_fixture(tmp_path, "reverse_progress", final_success=True)
    assert summary["coding_largest_drop"] == 0.25
    assert summary["overall_largest_drop"] == 0.25
    assert summary["largest_coding_drop_source"] == "mixed"
    assert summary["largest_overall_drop_source"] == "mixed"
    assert summary["final_success"] is True


def test_coding_complete_artifact_incomplete_fixture(tmp_path):
    summary = _rescore_fixture(tmp_path, "coding_complete_artifact_incomplete", final_success=True)

    assert summary["final_coding_progress"] == 1.0
    assert summary["final_overall_progress"] == 2 / 3
    assert summary["excluded_active_weight_final"] == 1.0
    assert summary["excluded_completed_weight_final"] == 0
    assert summary["excluded_categories_final"]["artifact"] == {"active": 1.0, "complete": 0}
    assert summary["active_coding_leaves_final"] == 2
    assert summary["active_overall_leaves_final"] == 3


def test_high_progress_wrong_solution_fixture_keeps_success_separate_from_progress(tmp_path):
    summary = _rescore_fixture(tmp_path, "high_progress_wrong_solution", final_success=False)

    assert summary["final_coding_progress"] == 7 / 8
    assert summary["final_success"] is False
    assert summary["final_success_source"] == "summary.final_success"
    assert summary["completed_coding_leaves_final"] == 3
    assert summary["active_coding_leaves_final"] == 4


def test_monotonic_incomplete_failure_fixture(tmp_path):
    summary = _rescore_fixture(tmp_path, "monotonic_incomplete_failure", final_success=False)

    assert summary["final_coding_progress"] == 3 / 5
    assert summary["coding_largest_drop"] == 0.0
    assert summary["overall_largest_drop"] == 0.0
    assert summary["largest_coding_drop_source"] == "none"
    assert summary["final_success"] is False


def test_weighted_leaves_fixture_explains_scalar_progress_by_weights(tmp_path):
    summary = _rescore_fixture(tmp_path, "weighted_leaves", final_success=True)

    assert summary["final_coding_complete_weight"] == 6.0
    assert summary["final_coding_active_weight"] == 7.0
    assert summary["final_coding_progress"] == 6 / 7
    assert summary["completed_coding_leaves_final"] == 3
    assert summary["active_coding_leaves_final"] == 4


def test_mixed_drop_source_fixture_records_mixed_attribution(tmp_path):
    summary = _rescore_fixture(tmp_path, "mixed_drop_source", final_success=True)

    assert summary["coding_largest_drop"] == 0.5
    assert summary["overall_largest_drop"] == 0.5
    assert summary["largest_coding_drop_source"] == "mixed"
    assert summary["largest_overall_drop_source"] == "mixed"
    assert set(summary["largest_coding_drop_category_contributions"]) == {"product", "validation"}
    assert set(summary["largest_overall_drop_category_contributions"]) == {"product", "validation"}


def _rescore_fixture(tmp_path, fixture_name, final_success):
    run_dir = tmp_path / fixture_name
    run_dir.mkdir()
    (run_dir / "ledger.jsonl").write_text((FIXTURES / f"{fixture_name}.jsonl").read_text())
    (run_dir / "summary.json").write_text(json.dumps({
        "task_id": fixture_name,
        "final_success": final_success,
    }))
    return rescore.rescore_run(run_dir)
