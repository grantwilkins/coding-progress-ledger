"""
Claim:
Suite-level rescoring is a diagnostic reporting layer: it computes final leaf
counts, drop sources, final success, and evidence audit status without changing
ledger scoring semantics or rewriting historical ledger logs.

Plausible wrong implementations:
- Treat historical subtask counts as the active scoring denominator.
- Attribute any overall drop to "mixed" without checking changed leaf categories.
- Infer final success from progress instead of explicit run metadata.
- Treat narrative/manual evidence as strong product or validation evidence.
- Rewrite legacy ledger JSONL while adding categories or audit metadata.
"""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESCORE_PATH = ROOT / "scripts" / "rescore_suite_by_category.py"
spec = importlib.util.spec_from_file_location("rescore_suite_by_category", RESCORE_PATH)
rescore = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rescore)


def event(step, event_type, subtask_id, payload, reason=None):
    return {
        "step": step,
        "event_type": event_type,
        "subtask_id": subtask_id,
        "payload": payload,
        "reason": reason,
    }


def write_run(tmp_path, run_id, events, final_success=True):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({
        "task_id": run_id,
        "final_success": final_success,
        "test_status": "passed" if final_success else "failed",
    }))
    return run_dir


def test_final_leaf_counts_are_not_historical_subtask_counts(tmp_path):
    run_dir = write_run(tmp_path, "leaf_count_fixture", [
        event(0, "init", None, {"root_task": "Leaf counts"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        event(3, "split_subtask", "P", {"children": [
            {"id": "P.1", "description": "Patch visible branch", "category": "product"},
            {"id": "P.2", "description": "Patch edge branch", "category": "product"},
        ]}),
        event(4, "update_status", "P.1", {"status": "complete", "evidence": ["solver.py modified"]}),
        event(5, "add_subtask", "E", {"description": "Install environment dependency", "category": "environment"}),
        event(6, "invalidate_subtask", "E", {"reason": "Not needed"}),
    ])

    summary = rescore.rescore_run(run_dir)

    assert summary["historical_subtasks_created"] == 5
    assert summary["active_coding_leaves_final"] == 3
    assert summary["completed_coding_leaves_final"] == 1
    assert summary["active_overall_leaves_final"] == 3
    assert summary["completed_overall_leaves_final"] == 1
    assert summary["final_coding_complete_weight"] == 1.0
    assert summary["final_coding_active_weight"] == 3.0
    assert summary["final_overall_complete_weight"] == 1.0
    assert summary["final_overall_active_weight"] == 3.0


def test_drop_source_reports_excluded_category_for_bookkeeping_drop(tmp_path):
    run_dir = write_run(tmp_path, "artifact_drop_fixture", [
        event(0, "init", None, {"root_task": "Artifact drop"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(1, "add_subtask", "A", {"description": "Export artifact bundle", "category": "artifact"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        event(2, "update_status", "A", {"status": "complete", "evidence": ["artifact present"]}),
        event(3, "reopen_subtask", "A", {"reason": "Artifact bundle missing summary"}),
    ])

    summary = rescore.rescore_run(run_dir)

    assert summary["overall_largest_drop"] == 0.5
    assert summary["largest_overall_drop_source"] == "artifact"
    assert summary["largest_overall_drop_category_contributions"] == {"artifact": 0.5}
    assert summary["largest_coding_drop_category_contributions"] == {}
    assert summary["coding_largest_drop"] == 0.0
    assert summary["largest_coding_drop_source"] == "none"


def test_suite_summary_table_includes_weight_and_success_source_columns():
    lines = rescore.suite_summary_table([
        {
            "task_id": "summary_fixture",
            "final_coding_progress": 0.5,
            "final_overall_progress": 0.4,
            "final_coding_complete_weight": 2.0,
            "final_coding_active_weight": 4.0,
            "final_overall_complete_weight": 2.0,
            "final_overall_active_weight": 5.0,
            "active_coding_leaves_final": 4,
            "completed_coding_leaves_final": 2,
            "active_overall_leaves_final": 5,
            "completed_overall_leaves_final": 2,
            "historical_subtasks_created": 6,
            "coding_largest_drop": 0.25,
            "overall_largest_drop": 0.2,
            "largest_coding_drop_source": "validation",
            "largest_overall_drop_source": "artifact",
            "excluded_active_weight_final": 1.0,
            "excluded_completed_weight_final": 0.0,
            "coding_nonmonotonic": True,
            "final_success": False,
            "final_success_source": "summary.final_success",
            "evidence_audit_status": "strong",
            "weak_completion_evidence_count": 0,
        }
    ])

    assert "coding_complete_weight_final" in lines[0]
    assert "overall_active_weight_final" in lines[0]
    assert "final_success_source" in lines[0]
    assert "| `summary_fixture` | 0.500 | 0.400 | 2.000 | 4.000 | 2.000 | 5.000 |" in lines[2]
    assert "summary.final_success" in lines[2]


def test_drop_source_reports_mixed_when_multiple_categories_are_material(tmp_path):
    run_dir = write_run(tmp_path, "mixed_drop_fixture", [
        event(0, "init", None, {"root_task": "Mixed drop"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        event(2, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
        event(3, "reopen_subtask", "P", {"reason": "Product behavior still wrong"}),
        event(3, "reopen_subtask", "V", {"reason": "Validation was incomplete"}),
    ])

    summary = rescore.rescore_run(run_dir)

    assert summary["coding_largest_drop"] == 1.0
    assert summary["largest_coding_drop_source"] == "mixed"
    assert summary["largest_overall_drop_source"] == "mixed"
    assert summary["largest_coding_drop_category_contributions"] == {"product": 0.5, "validation": 0.5}
    assert summary["largest_overall_drop_category_contributions"] == {"product": 0.5, "validation": 0.5}


def test_control_coding_complete_artifacts_incomplete_fixture():
    summary = json.loads((ROOT / "runs/control_coding_complete_artifacts_incomplete/summary_by_category.json").read_text())

    assert summary["final_success"] is True
    assert summary["final_coding_progress"] == 1.0
    assert summary["final_overall_progress"] < 1.0
    assert summary["excluded_categories_final"]["artifact"] == {"active": 1.0, "complete": 0}


def test_control_monotonic_incomplete_failure_fixture():
    summary = json.loads((ROOT / "runs/control_monotonic_incomplete_failure/summary_by_category.json").read_text())

    assert summary["final_success"] is False
    assert 0.4 <= summary["final_coding_progress"] <= 0.8
    assert summary["coding_nonmonotonic"] is False
    assert summary["coding_largest_drop"] == 0.0


def test_control_high_progress_wrong_solution_fixture():
    summary = json.loads((ROOT / "runs/control_high_progress_wrong_solution/summary_by_category.json").read_text())

    assert summary["final_success"] is False
    assert summary["final_coding_progress"] >= 0.8
    assert summary["completed_coding_leaves_final"] < summary["active_coding_leaves_final"]


def test_evidence_audit_marks_test_output_and_diff_evidence_strong():
    audit = rescore.audit_completion_evidence([
        event(0, "init", None, {"root_task": "Evidence"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows solver.py modified"]}),
        event(3, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
    ])

    assert audit["status"] == "strong"
    assert audit["weak_completion_evidence_count"] == 0


def test_evidence_audit_marks_manual_only_product_completion_weak():
    audit = rescore.audit_completion_evidence([
        event(0, "init", None, {"root_task": "Evidence"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["Looks good"]}),
    ])

    assert audit["status"] == "weak"
    assert audit["weak_completion_evidence_count"] == 1
    assert audit["weak_completion_evidence"][0]["subtask_id"] == "P"
    assert audit["weak_categories"] == ["product"]
    assert audit["by_category"]["product"] == {
        "status": "weak",
        "audited_completion_count": 1,
        "weak_completion_evidence_count": 1,
        "weak_subtask_ids": ["P"],
    }
    assert audit["by_category"]["validation"]["status"] == "not_applicable"


def test_evidence_audit_reports_category_severity_independently():
    audit = rescore.audit_completion_evidence([
        event(0, "init", None, {"root_task": "Evidence"}),
        event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
        event(1, "add_subtask", "I", {"description": "Understand task contract", "category": "investigation"}),
        event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows solver.py modified"]}),
        event(3, "update_status", "V", {"status": "complete", "evidence": ["Looks fine"]}),
        event(4, "update_status", "I", {"status": "complete", "evidence": ["task.md describes expected behavior"]}),
    ])

    assert audit["status"] == "weak"
    assert audit["weak_categories"] == ["validation"]
    assert audit["by_category"]["product"]["status"] == "strong"
    assert audit["by_category"]["validation"]["status"] == "weak"
    assert audit["by_category"]["validation"]["weak_subtask_ids"] == ["V"]
    assert audit["by_category"]["investigation"]["status"] == "strong"


def test_legacy_run_is_audited_without_rewriting_ledger_jsonl(tmp_path):
    source = ROOT / "runs/task_1_parser_timezone_offset"
    run_dir = tmp_path / source.name
    run_dir.mkdir()
    original_ledger = (source / "ledger.jsonl").read_text()
    (run_dir / "ledger.jsonl").write_text(original_ledger)
    (run_dir / "summary.json").write_text((source / "summary.json").read_text())

    summary = rescore.rescore_run(run_dir)

    assert (run_dir / "ledger.jsonl").read_text() == original_ledger
    assert summary["evidence_audit_status"] in {"strong", "weak"}
    assert summary["weak_completion_evidence_count"] >= 0


def test_rescore_preserves_explicit_categories_for_new_ledgers(tmp_path):
    run_dir = write_run(tmp_path, "explicit_category_fixture", [
        event(0, "init", None, {"root_task": "Explicit categories"}),
        event(1, "add_subtask", "I", {
            "description": "Inspect ambiguous task contract",
            "category": "investigation",
        }),
        event(2, "update_status", "I", {
            "status": "complete",
            "evidence": ["task.md describes the expected contract"],
        }),
    ])

    summary = rescore.rescore_run(run_dir)

    assert summary["subtask_categories"]["I"] == "investigation"
    assert summary["evidence_audit_by_category"]["investigation"]["status"] == "strong"
    assert summary["evidence_audit_by_category"]["product"]["status"] == "not_applicable"
