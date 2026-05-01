from __future__ import annotations

import json

from ledger_progress import LedgerSession, SubtaskCategory, to_jsonl
from scripts.build_live_parity_report import compare_pair, policy_adjusted_parity, render_report, shape_class


def test_shape_class_distinguishes_validation_gap_from_absent_validation():
    assert shape_class(_summary(1.0, validation_active=0, validation_done=0)) == "complete_visible_frontier+no_validation_frontier"
    assert shape_class(_summary(2 / 3, validation_active=1, validation_done=0)) == "partial_visible_frontier+validation_gap"
    assert shape_class(_summary(1.0, validation_active=1, validation_done=1)) == "complete_visible_frontier+validation_complete"


def test_compare_pair_reports_validation_category_divergence(tmp_path):
    live = tmp_path / "live"
    retro = tmp_path / "retro"
    _write_live_run(live, source=retro, with_validation=False)
    _write_retro_run(retro, with_validation=True)

    item = compare_pair(live, retro)

    assert item["live_shape"] == "complete_visible_frontier+no_validation_frontier"
    assert item["retro_shape"] == "partial_visible_frontier+validation_gap"
    assert item["live_categories"] == {"product": 1}
    assert item["retro_categories"] == {"product": 1, "validation": 1}


def test_policy_adjusted_parity_accepts_submit_without_validation_frontier(tmp_path):
    live = tmp_path / "live"
    retro = tmp_path / "retro"
    _write_live_run(live, source=retro, with_validation=False)
    _write_retro_run(retro, with_validation=True)

    item = compare_pair(live, retro)
    report = render_report([item])

    assert policy_adjusted_parity(item) is True
    assert "policy-adjusted parity gate passes" in report
    assert "Baseline failing test output before edits" in report
    assert "Retrospective `WIPACrepo__iceprod-339` includes an unstarted validation leaf" in report
    assert "no-validation-frontier policy" in report


def _summary(progress, *, validation_active, validation_done):
    return {
        "final_coding_progress": progress,
        "category_active_weight_final": {"validation": validation_active},
        "category_completed_weight_final": {"validation": validation_done},
    }


def _write_live_run(path, *, source, with_validation):
    path.mkdir()
    session = LedgerSession("Live", clock=lambda: "2026-04-30T12:00:00+00:00")
    product = session.add("Patch", step=1, category=SubtaskCategory.PRODUCT)
    session.complete(product, "step 1: edit observed", step=1)
    if with_validation:
        validation = session.add("Validate", step=2, category=SubtaskCategory.VALIDATION)
        session.complete(validation, "step 2: pytest observed", step=2)
    to_jsonl(session.ledger, str(path / "ledger.jsonl"))
    (path / "summary_by_category.json").write_text(json.dumps(_summary(1.0, validation_active=1 if with_validation else 0, validation_done=1 if with_validation else 0)))
    (path / "live_instrumentation.json").write_text(json.dumps({
        "instance_id": "WIPACrepo__iceprod-339",
        "final_success": False,
        "source_run_dir": str(source),
        "wire_event_count": 1,
    }))


def _write_retro_run(path, *, with_validation):
    path.mkdir()
    session = LedgerSession("Retro", clock=lambda: None)
    product = session.add("Patch", step=1, category=SubtaskCategory.PRODUCT)
    session.complete(product, "step 1: edit observed", step=1)
    if with_validation:
        session.add("Validate", step=2, category=SubtaskCategory.VALIDATION)
    to_jsonl(session.ledger, str(path / "ledger.jsonl"))
    (path / "summary_by_category.json").write_text(json.dumps(_summary(2 / 3 if with_validation else 1.0, validation_active=1 if with_validation else 0, validation_done=0)))
