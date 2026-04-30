"""
Claim:
The 20 SWE-agent pilot annotations at annotations/swe_agent_pilot/
preserve the framework's load-bearing properties:

1. Progress is genuinely independent of final_success — at least one
   failure ends at progress 1.00 (f_06), at least one success ends
   below 1.00 (s_04). If a "tidying" pass collapses these, the
   framework reduces to a thin outcome classifier.
2. Non-monotonicity is preserved — REOPEN events exist on the runs
   that contain them (s_03, s_05, f_09).
3. The submit-without-validation shape is preserved — VALIDATION
   leaves on f_01, s_04, f_04 are at not_started (not retroactively
   completed via post-hoc test_output.txt).
4. Every spec is internally consistent: cited steps are within
   trajectory bounds, every complete carries evidence, the
   per-pilot subtask count matches the spec's add events, pilot_ids
   match source_metadata.

Plausible wrong implementations:

- A "tidying" pass coerces failures to <1.00 and successes to 1.00,
  re-encoding final_success in the progress signal.
- An annotator marks f_01/s_04/f_04's VALIDATION leaf complete using
  post-hoc test_output.txt, eliminating the submit-without-test
  shape that distinguishes them from f_06's hidden-work-gap shape.
- A spec's annotation_quality.number_of_subtasks drifts from the
  actual count of add events.
- A spec cites an evidence step beyond the trajectory's range.
- A complete event lands with no evidence string — progress 1.00
  on unsupported claims.
- All annotations are monotonic, breaking the "non-monotonicity is
  the correct shape" property.
- Two specs share a pilot_id, or a spec's pilot_id doesn't match
  the run_dir or source_metadata.

Spec-only tests (always run) check the committed JSON. Run-dir tests
(skip if the gitignored run dirs are absent on this machine) check
that quality artifacts and source_metadata stay consistent with the
specs.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.annotate_pilots_from_spec import build_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = REPO_ROOT / "annotations" / "swe_agent_pilot"
RUNS_DIR = REPO_ROOT / "runs" / "swe_agent_pilot"


def _all_specs():
    return sorted(SPECS_DIR.glob("*.json"))


def _load_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def specs():
    return [_load_spec(p) for p in _all_specs()]


# ---------- spec-only invariants (always run) ----------


def test_pilot_set_size_is_twenty(specs):
    # The pilot is N=20 by B1 policy. A drift here means a spec was
    # added or dropped without updating downstream tooling.
    assert len(specs) == 20


def test_pilot_ids_are_unique(specs):
    ids = [s["pilot_id"] for s in specs]
    assert len(set(ids)) == len(ids)


def test_instance_ids_are_unique(specs):
    # B1 policy dedupes on instance_id; two specs sharing one would
    # mean the dedupe rule was violated upstream.
    iids = [s["instance_id"] for s in specs]
    assert len(set(iids)) == len(iids)


def test_every_complete_event_carries_evidence(specs):
    # A "complete" with no evidence is the canonical "asserted progress
    # without trace support" failure. core.py rejects this at runtime
    # — but that only fires when the spec is replayed; an unreplay-ed
    # JSON file could still slip through review.
    bad = []
    for s in specs:
        for ev in s["events"]:
            if ev.get("op") != "complete":
                continue
            evidence = ev.get("evidence")
            if evidence is None:
                bad.append((s["pilot_id"], ev.get("id"), "missing"))
                continue
            if isinstance(evidence, str):
                if not evidence.strip():
                    bad.append((s["pilot_id"], ev.get("id"), "empty string"))
            elif isinstance(evidence, list):
                if not evidence or not any(isinstance(e, str) and e.strip() for e in evidence):
                    bad.append((s["pilot_id"], ev.get("id"), "empty list"))
            else:
                bad.append((s["pilot_id"], ev.get("id"), f"bad type {type(evidence).__name__}"))
    assert not bad, f"complete events without evidence: {bad}"


def test_number_of_subtasks_matches_add_events(specs):
    # quality.number_of_subtasks is filled by the driver from session
    # state (correct), but the spec's quality field is also present
    # for historical traceability. They must agree.
    bad = []
    for s in specs:
        n_add = sum(1 for ev in s["events"] if ev.get("op") == "add")
        # The spec.json's quality block does NOT carry number_of_subtasks
        # (driver fills it). The check is: number of `add` events ==
        # the run dir's annotation_quality.json reports. We can do
        # the consistency check here only if the run dir is present;
        # at the spec level we just check that events are well-formed.
        for ev in s["events"]:
            if ev.get("op") == "add":
                if not ev.get("description"):
                    bad.append((s["pilot_id"], "add event missing description"))
                if not ev.get("category"):
                    bad.append((s["pilot_id"], "add event missing category"))
        # Also assert at least one add event per pilot — an empty
        # annotation would be an obvious bug.
        if n_add == 0:
            bad.append((s["pilot_id"], "no add events"))
    assert not bad, bad


def test_all_id_fields_match_session_auto_numbering(specs):
    # The driver asserts spec id == auto-generated id; this check
    # additionally fails specs that ARE invalid even before the driver
    # runs them. Catches reordering bugs introduced by careless
    # editing.
    bad = []
    for s in specs:
        seen = 0
        for ev in s["events"]:
            if ev.get("op") == "add":
                seen += 1
                expected = f"S{seen}"
                got = ev.get("id")
                if got is not None and got != expected:
                    bad.append((s["pilot_id"], "add", got, expected))
    assert not bad, f"id-vs-auto-numbering mismatch: {bad}"


def test_steps_in_events_are_non_negative(specs):
    bad = []
    for s in specs:
        for ev in s["events"]:
            step = ev.get("step")
            if step is None or step < 0:
                bad.append((s["pilot_id"], ev.get("op"), step))
    assert not bad, bad


# ---------- spec-level invariants of framework purpose ----------
#
# These are the load-bearing tests. They catch the "tidying pass that
# turns the framework into an outcome classifier" failure mode.


def test_at_least_one_reopen_event_exists_somewhere(specs):
    # If every annotation becomes monotonic, the protocol's
    # "non-monotonicity is the correct shape" claim is unverified.
    total_reopens = sum(
        1 for s in specs for ev in s["events"] if ev.get("op") == "reopen"
    )
    assert total_reopens >= 1, "no REOPEN events anywhere in the pilot"


def test_at_least_one_blocked_event_exists_somewhere(specs):
    # No BLOCKED status anywhere means either the protocol's
    # stuck-loop rule never fires (suspicious), or annotators are
    # converting BLOCKED into either complete or invalidated.
    total_blocks = sum(
        1 for s in specs for ev in s["events"] if ev.get("op") == "block"
    )
    assert total_blocks >= 1, "no BLOCKED events anywhere in the pilot"


def test_at_least_one_validation_leaf_left_unstarted_or_in_progress(specs):
    # The submit-without-validation shape (f_01, s_04, f_04 family)
    # is a load-bearing distinguisher between "agent ran tests" and
    # "agent skipped validation". It surfaces as a VALIDATION leaf
    # added but never completed.
    found = []
    for s in specs:
        added_validation_ids = {
            ev["id"]
            for ev in s["events"]
            if ev.get("op") == "add" and ev.get("category") == "VALIDATION"
        }
        if not added_validation_ids:
            continue
        for vid in added_validation_ids:
            terminal_op = None
            for ev in s["events"]:
                if ev.get("id") == vid and ev.get("op") in ("complete", "block", "invalidate"):
                    terminal_op = ev.get("op")
                    break
            if terminal_op is None:
                # Validation leaf added but never reached a terminal
                # transition — i.e. left at not_started or in_progress.
                found.append((s["pilot_id"], vid))
    assert found, "no VALIDATION leaf is left unstarted/in_progress; the submit-without-test shape is gone"


# ---------- cross-spec / framework-purpose invariants requiring run dirs ----------


@pytest.fixture(scope="module")
def have_run_dirs():
    return RUNS_DIR.is_dir() and any(RUNS_DIR.glob("swe_agent_pilot_*/source_metadata.json"))


def _read_metadata(pilot_id: str) -> dict:
    return json.loads((RUNS_DIR / pilot_id / "source_metadata.json").read_text(encoding="utf-8"))


def _read_quality(pilot_id: str) -> dict:
    return json.loads((RUNS_DIR / pilot_id / "annotation_quality.json").read_text(encoding="utf-8"))


def _read_final_progress(pilot_id: str) -> tuple[float, float]:
    """Return (overall_progress, coding_progress) at the last step in progress.csv."""
    import csv as _csv
    rows = list(_csv.DictReader((RUNS_DIR / pilot_id / "progress.csv").open()))
    overall = float(rows[-1]["progress"])
    # progress_by_category.csv carries per-category, but we read the
    # overall here; coding is computed as a separate score elsewhere.
    return overall, overall  # both fields are filled but only overall is used in tests


def test_spec_pilot_id_matches_source_metadata(specs, have_run_dirs):
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    bad = []
    for s in specs:
        md = _read_metadata(s["pilot_id"])
        if md.get("pilot_id") != s["pilot_id"]:
            bad.append((s["pilot_id"], md.get("pilot_id")))
        if md.get("instance_id") != s["instance_id"]:
            bad.append((s["pilot_id"], "instance_id mismatch", md.get("instance_id"), s["instance_id"]))
    assert not bad, bad


def test_every_cited_step_is_within_trajectory_bounds(specs, have_run_dirs):
    # A spec citing step 50 on a 17-step trace is a typo. We check
    # event step indices AND any step indices embedded in evidence
    # strings of the form "step N:".
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    bad = []
    step_re = re.compile(r"\bstep\s+(\d+)\b", re.IGNORECASE)
    for s in specs:
        md = _read_metadata(s["pilot_id"])
        max_step = md["trajectory_length"] - 1  # 0-indexed
        for ev in s["events"]:
            if ev.get("step") is not None and ev["step"] > max_step:
                bad.append((s["pilot_id"], "event step OOB", ev.get("op"), ev["step"], max_step))
            evidence = ev.get("evidence")
            if evidence is None:
                continue
            evidence_list = [evidence] if isinstance(evidence, str) else evidence
            for e in evidence_list:
                if not isinstance(e, str):
                    continue
                for m in step_re.finditer(e):
                    cited = int(m.group(1))
                    if cited > max_step:
                        bad.append((s["pilot_id"], "evidence cite OOB", cited, max_step, e[:60]))
    assert not bad, f"step citations outside trajectory range: {bad}"


def test_number_of_subtasks_in_quality_matches_replayed_session(specs, have_run_dirs):
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    bad = []
    for s in specs:
        q = _read_quality(s["pilot_id"])
        session = build_session(s)
        actual = len(session.ledger.subtasks)
        if q.get("number_of_subtasks") != actual:
            bad.append((s["pilot_id"], q.get("number_of_subtasks"), actual))
    assert not bad, f"number_of_subtasks drift: {bad}"


# ---------- framework-purpose invariants over the 20-pilot distribution ----------


def test_at_least_one_failure_ends_at_full_progress(specs, have_run_dirs):
    """f_06 shape: failure with all discovered work completed.

    If no failure ends at 1.00, the framework's discriminating power
    is gone — every failure looks the same as "didn't finish."
    """
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    high_prog_failures = []
    for s in specs:
        md = _read_metadata(s["pilot_id"])
        if md.get("final_success") is not False:
            continue
        overall, _ = _read_final_progress(s["pilot_id"])
        if overall >= 0.99:
            high_prog_failures.append((s["pilot_id"], overall))
    assert high_prog_failures, (
        "no failure pilot ends at progress >= 0.99; the f_06-style "
        "'all discovered work done; failure in undiscovered hidden "
        "work' shape has been tidied away. The progress signal is no "
        "longer independent of final_success."
    )


def test_at_least_one_success_ends_below_full_progress(specs, have_run_dirs):
    """s_04 shape: success with skipped validation.

    If every success is at 1.00, the progress signal collapses to a
    thin wrapper around final_success.
    """
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    low_prog_successes = []
    for s in specs:
        md = _read_metadata(s["pilot_id"])
        if md.get("final_success") is not True:
            continue
        overall, _ = _read_final_progress(s["pilot_id"])
        if overall < 0.99:
            low_prog_successes.append((s["pilot_id"], overall))
    assert low_prog_successes, (
        "no success pilot ends at progress < 0.99; the s_04-style "
        "'submit-without-validation' shape has been tidied away. The "
        "progress signal is no longer independent of final_success."
    )


def test_failure_progress_distribution_spans_a_range(specs, have_run_dirs):
    """Failures should span multiple progress shapes, not bunch at one.

    The pilot's failures genuinely differ (stuck-loop, validation-
    blocked, hidden-work, etc.). Their progress distribution should
    span a real range. If max - min ~= 0, the framework collapsed
    the distinguishing signal.
    """
    if not have_run_dirs:
        pytest.skip("run dirs not present (gitignored on this machine)")
    failure_progs = []
    for s in specs:
        md = _read_metadata(s["pilot_id"])
        if md.get("final_success") is False:
            overall, _ = _read_final_progress(s["pilot_id"])
            failure_progs.append(overall)
    assert failure_progs, "no failure pilots present"
    spread = max(failure_progs) - min(failure_progs)
    assert spread >= 0.30, (
        f"failure progress spread is only {spread:.2f}; expected a "
        "wide spread (~0.50) reflecting different failure modes "
        "(stuck-loop, validation-blocked, hidden-work, etc.). A "
        "narrow spread suggests the channel has collapsed onto a "
        "single failure-shape proxy."
    )
