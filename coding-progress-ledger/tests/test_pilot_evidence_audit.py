import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rescore = _load("rescore_suite_by_category")
audit_pilot = _load("audit_pilot_evidence")
classify_evidence = rescore.classify_evidence
audit_completion_evidence = rescore.audit_completion_evidence
STRONG_EVIDENCE_TYPES = rescore.STRONG_EVIDENCE_TYPES

AUDIT_JSON = ROOT / "runs" / "swe_agent_pilot" / "EVIDENCE_AUDIT.json"


def _audit_data() -> dict:
    assert AUDIT_JSON.exists(), f"missing K1 output: {AUDIT_JSON} (re-run scripts/audit_pilot_evidence.py)"
    return json.loads(AUDIT_JSON.read_text())


def test_per_pilot_completion_counts_decompose():
    # If classifier ever emits manual_note alongside another type, manual_only + with_strong could exceed completion_events.
    data = _audit_data()
    for row in data["per_pilot"]:
        assert row["completion_events"] >= row["completions_with_strong"] + row["completions_manual_only"], row["pilot_id"]


def test_totals_equal_sum_over_pilots():
    # Catches an aggregator bug that double-counts or drops pilots.
    data = _audit_data()
    rows = data["per_pilot"]
    totals = data["totals"]
    for key in ("completion_events", "completions_with_strong", "completions_manual_only"):
        assert totals[key] == sum(r[key] for r in rows), key


def test_classify_evidence_recognizes_test_output():
    # If pytest output stops being recognized, it'd silently demote to manual_note.
    types = classify_evidence(["pytest output: 5 passed"])
    assert "test_output" in types and "manual_note" not in types


def test_classify_evidence_falls_back_to_manual_note():
    # Locks in manual_note as the no-pattern fallback.
    assert classify_evidence(["random unstructured note"]) == {"manual_note"}


def test_classify_evidence_path_pattern_yields_diff():
    # Source-path mention should classify as diff, not manual_note.
    types = classify_evidence(["step 14: edit utils/foo.py:88 acknowledged"])
    assert "diff" in types and "manual_note" not in types


def test_strong_evidence_types_excludes_manual_note():
    # A PR that promotes manual_note to strong would erase the K1 weak/strong signal.
    known = {"test_output", "diff", "file_exists", "command_output", "contract_text", "manual_note"}
    assert STRONG_EVIDENCE_TYPES and STRONG_EVIDENCE_TYPES.issubset(known)
    assert "manual_note" not in STRONG_EVIDENCE_TYPES


def test_audit_skips_artifact_and_documentation_categories():
    # ARTIFACT/DOCUMENTATION/ENVIRONMENT completions are out of scope per CODING_CATEGORIES.
    events = [
        {"step": 1, "event_type": "add_subtask", "subtask_id": "A1",
         "payload": {"description": "ship artifact", "category": "artifact"}},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "D1",
         "payload": {"description": "write doc", "category": "documentation"}},
        {"step": 3, "event_type": "update_status", "subtask_id": "A1",
         "payload": {"status": "complete", "evidence": ["manual"]}},
        {"step": 4, "event_type": "update_status", "subtask_id": "D1",
         "payload": {"status": "complete", "evidence": ["manual"]}},
    ]
    result = audit_completion_evidence(events)
    assert result["status"] == "not_applicable"
    assert sum(c["audited_completion_count"] for c in result["by_category"].values()) == 0


def test_k1_smoke_pilots_match_swe_agent_pilot_glob():
    # Catches a path-glob regression that picks up runs/task_* or runs/control_*.
    data = _audit_data()
    assert data["totals"]["pilots"] == len(data["per_pilot"])
    for row in data["per_pilot"]:
        assert row["pilot_id"].startswith("swe_agent_pilot_"), row["pilot_id"]
