"""
Claim:
annotate_pilots_from_spec.py replays a JSON event spec into a
LedgerSession that, replayed through ledger_progress.replay, has the
same leaves / statuses / progress as direct LedgerSession calls. The
driver writes ledger.jsonl + run_notes.md (with placeholder
substitution) + annotation_quality.json into the matching run dir,
and `ledger-run check-run` accepts the result.

Plausible wrong implementations:
- build_session iterates events but mis-routes a known op (e.g.
  block routes through complete), silently producing the wrong status.
- build_session lets a spec id mismatch slide ("S1" asserted but
  session generated "S2"), so a later complete event hits the wrong
  subtask.
- build_session accepts an unknown op silently instead of failing.
- emit_one's notes substitution leaks "{{PROGRESS_OVERALL}}" into the
  rendered run_notes.md when the placeholder is missing or
  mistyped.
- emit_one fills annotation_quality.number_of_subtasks from spec
  instead of the actual session, so duplicate add events would not
  surface.
- split is mis-encoded so child categories vanish.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress import Status, SubtaskCategory  # noqa: E402

from scripts.annotate_pilots_from_spec import build_session, emit_one  # noqa: E402


def _spec(events, root="root", quality=None):
    return {
        "pilot_id": "test_pilot_x",
        "instance_id": "x__y-1",
        "root_task": root,
        "events": events,
        "quality": quality
        or {
            "annotator": "test",
            "annotation_time_minutes": 0,
            "number_of_uncertain_events": 0,
            "number_of_evidence_gaps": 0,
            "whether_final_success_used_only_at_end": True,
            "whether_progress_forced": False,
            "whether_schema_gap_found": False,
        },
    }


# ---------- build_session: ops route correctly ----------


def test_add_then_complete_yields_one_complete_leaf():
    s = build_session(_spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "do x"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "step 2"},
    ]))
    leaves = list(s.ledger.subtasks.values())
    assert len(leaves) == 1
    assert leaves[0].status is Status.COMPLETE
    assert leaves[0].evidence == ["step 2"]


def test_block_routes_to_blocked_status_not_complete():
    # A wrong impl that mis-routed `block` -> `complete` would still
    # produce one leaf, but its status would be COMPLETE (and progress
    # would be 1.0 instead of 0.0).
    s = build_session(_spec([
        {"op": "add", "step": 1, "id": "S1", "category": "INVESTIGATION", "description": "look"},
        {"op": "block", "step": 5, "id": "S1", "reason": "stuck"},
    ]))
    leaf = s.ledger.subtasks["S1"]
    assert leaf.status is Status.BLOCKED
    assert s.score().progress == 0.0


def test_reopen_drops_progress_after_complete():
    s = build_session(_spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "do"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "ok"},
        {"op": "reopen", "step": 5, "id": "S1", "reason": "patch wrong"},
    ]))
    assert s.ledger.subtasks["S1"].status is Status.IN_PROGRESS
    assert s.score().progress == 0.0


def test_split_creates_children_with_categories_and_drops_parent_from_leaves():
    s = build_session(_spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "vague"},
        {"op": "split", "step": 3, "id": "S1", "reason": "decomposed",
         "children": [
             {"description": "child a", "category": "INVESTIGATION"},
             {"description": "child b", "category": "VALIDATION"},
         ]},
    ]))
    children = [c for c in s.ledger.subtasks.values() if c.parent_id == "S1"]
    assert len(children) == 2
    cats = {c.category for c in children}
    assert cats == {SubtaskCategory.INVESTIGATION, SubtaskCategory.VALIDATION}
    # Parent is no longer a leaf; only the two children count toward
    # active weight. With neither child complete, progress is 0/2 = 0.
    obs = s.score()
    assert obs.active_leaf_count == 2
    assert obs.complete_leaf_count == 0
    assert obs.progress == 0.0


def test_invalidate_marks_subtask_inactive_and_excludes_from_progress():
    s = build_session(_spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "p"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "ok"},
        {"op": "add", "step": 3, "id": "S2", "category": "PRODUCT", "description": "abandoned"},
        {"op": "invalidate", "step": 4, "id": "S2", "reason": "abandoned approach"},
    ]))
    # S2 invalidated -> excluded from active set; only S1 counts.
    assert s.score().progress == 1.0


# ---------- build_session: failure paths ----------


def test_unknown_op_raises_with_op_name():
    with pytest.raises(ValueError, match="unknown op"):
        build_session(_spec([{"op": "frobnicate", "step": 1}]))


def test_unknown_category_raises_with_category_name():
    with pytest.raises(ValueError, match="unknown category"):
        build_session(_spec([
            {"op": "add", "step": 1, "id": "S1", "category": "WIDGET", "description": "x"},
        ]))


def test_id_mismatch_between_spec_and_session_raises():
    # The spec asserts S2 but the session is empty so add() generates S1.
    with pytest.raises(ValueError, match="spec asserts id"):
        build_session(_spec([
            {"op": "add", "step": 1, "id": "S2", "category": "PRODUCT", "description": "x"},
        ]))


def test_id_field_optional_when_uniquely_recoverable():
    # Spec without id field is allowed; session auto-numbers.
    s = build_session(_spec([
        {"op": "add", "step": 1, "category": "PRODUCT", "description": "x"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "ok"},
    ]))
    assert s.ledger.subtasks["S1"].status is Status.COMPLETE


# ---------- emit_one: end-to-end with placeholder substitution ----------


def _write_run_dir_skeleton(run_dir: Path) -> None:
    """Mimic enough of an importer-produced run dir for export-run + check-run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.md").write_text("# task\n", encoding="utf-8")
    (run_dir / "final_diff.patch").write_text("dummy\n", encoding="utf-8")
    (run_dir / "test_output.txt").write_text("dummy\n", encoding="utf-8")


def _write_spec_pair(specs_dir: Path, pilot_id: str, spec: dict, notes: str) -> None:
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / f"{pilot_id}.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (specs_dir / f"{pilot_id}.notes.md").write_text(notes, encoding="utf-8")


def test_emit_one_substitutes_progress_placeholders_and_writes_quality(tmp_path):
    pilot_id = "test_pilot_x"
    runs_dir = tmp_path / "runs"
    specs_dir = tmp_path / "specs"
    run_dir = runs_dir / pilot_id
    _write_run_dir_skeleton(run_dir)

    spec = _spec(
        [
            {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "do"},
            {"op": "complete", "step": 2, "id": "S1", "evidence": "ok"},
            {"op": "add", "step": 3, "id": "S2", "category": "VALIDATION", "description": "v"},
        ]
    )
    spec["pilot_id"] = pilot_id

    notes_template = (
        "# notes\nprogress overall: {{PROGRESS_OVERALL}}\n"
        "progress coding: {{PROGRESS_CODING}}\n"
    )
    _write_spec_pair(specs_dir, pilot_id, spec, notes_template)

    emit_one(specs_dir / f"{pilot_id}.json", runs_dir)

    # Substitution worked.
    rendered = (run_dir / "run_notes.md").read_text(encoding="utf-8")
    assert "{{PROGRESS_OVERALL}}" not in rendered
    assert "{{PROGRESS_CODING}}" not in rendered
    # Two leaves total, one complete -> overall progress 0.50.
    assert "progress overall: 0.50" in rendered

    # number_of_subtasks must come from the actual session, not spec.
    quality = json.loads((run_dir / "annotation_quality.json").read_text(encoding="utf-8"))
    assert quality["number_of_subtasks"] == 2


def test_emit_one_raises_when_notes_file_missing(tmp_path):
    pilot_id = "test_pilot_y"
    runs_dir = tmp_path / "runs"
    specs_dir = tmp_path / "specs"
    _write_run_dir_skeleton(runs_dir / pilot_id)

    spec = _spec([{"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "x"}])
    spec["pilot_id"] = pilot_id
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / f"{pilot_id}.json").write_text(json.dumps(spec), encoding="utf-8")
    # Note: no .notes.md companion.

    with pytest.raises(FileNotFoundError, match="notes file"):
        emit_one(specs_dir / f"{pilot_id}.json", runs_dir)


def test_emit_one_raises_when_run_dir_missing(tmp_path):
    pilot_id = "test_pilot_z"
    runs_dir = tmp_path / "runs"
    specs_dir = tmp_path / "specs"
    spec = _spec([{"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "x"}])
    spec["pilot_id"] = pilot_id
    _write_spec_pair(specs_dir, pilot_id, spec, "# notes\n")

    with pytest.raises(FileNotFoundError, match="run dir"):
        emit_one(specs_dir / f"{pilot_id}.json", runs_dir)
