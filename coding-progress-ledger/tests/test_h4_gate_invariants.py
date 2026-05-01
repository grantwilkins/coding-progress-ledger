"""H4 gate invariants: lock in the load-bearing claims of
runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress.serialization import from_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "annotations" / "swe_agent_pilot_v3"
RUNS = ROOT / "runs" / "swe_agent_pilot_v3"
PILOTS = ["s_01", "s_03", "f_01", "f_06", "f_03"]
# v1 overall progress from H4_GATE_RESULT.md § 2 (the "v1 overall" column).
V1_OVERALL = {"s_01": 1.00, "s_03": 1.00, "f_01": 0.75, "f_06": 1.00, "f_03": 0.50}
QUADRANT_THRESHOLD = 0.8


def _spec(pilot: str) -> dict:
    return json.loads((SPECS / f"swe_agent_pilot_{pilot}.json").read_text())


def _final_overall_progress(pilot: str) -> float:
    rows = list(csv.DictReader((RUNS / f"swe_agent_pilot_{pilot}" / "progress.csv").open()))
    return float(rows[-1]["progress"])


def _final_coding_progress(pilot: str) -> float:
    rows = list(csv.DictReader((RUNS / f"swe_agent_pilot_{pilot}" / "progress_by_category.csv").open()))
    return float(rows[-1]["coding_progress"])


def test_f_01_coding_progress_within_gate_tolerance():
    # The HIGH-leverage gate condition: v3 must reproduce v1's f_01 conclusion
    # at coding-progress 2/3 within 0.05. A bug that omits the implicit
    # VAL leaf (Pitfall #8) would land at 1.00 and fail; a bug that
    # double-counts would land at 0.50 and fail. 0.05 is the literal
    # tolerance from M1's gate.
    assert abs(_final_coding_progress("f_01") - 2 / 3) <= 0.05


def test_quadrant_agreement_all_five_pilots():
    # Each v3 pilot must end in the same overall-progress quadrant as
    # v1. Threshold 0.8 is the quadrant boundary set by H4. A bug in
    # which any pilot crosses 0.8 in the wrong direction (e.g.
    # spurious BLOCKED on s_01, missing implicit VAL crashing f_06)
    # would surface here.
    for pilot in PILOTS:
        v1_high = V1_OVERALL[pilot] >= QUADRANT_THRESHOLD
        v3_high = _final_overall_progress(pilot) >= QUADRANT_THRESHOLD
        assert v1_high == v3_high, pilot


def test_every_v3_spec_has_validation_leaf():
    # Pitfall #8 (HIGH severity): every bug-fix pilot must contain at
    # least one VALIDATION event. All 5 H pilots are bug-fix. A spec
    # that drops the implicit VAL leaf is the single most likely
    # protocol violation given the v1/v2 history.
    for pilot in PILOTS:
        cats = {e.get("category") for e in _spec(pilot)["events"] if e.get("op") == "add"}
        assert "VALIDATION" in cats, pilot


def test_no_upstream_label_used_as_evidence():
    # The two protocol-violation flags that would invalidate the cold
    # pass: looking at final_success before the end, or backfitting
    # progress to a target. Both must be the safe value on every spec.
    for pilot in PILOTS:
        q = _spec(pilot)["quality"]
        assert q["whether_final_success_used_only_at_end"] is True, pilot
        assert q["whether_progress_forced"] is False, pilot


def test_all_v3_ledgers_replay_cleanly():
    # Replay integrity: every materialized ledger.jsonl must round-trip
    # through from_jsonl (which calls replay). A mis-encoded event
    # (e.g. unknown EventType, missing payload key) would raise here.
    for pilot in PILOTS:
        ledger = from_jsonl(str(RUNS / f"swe_agent_pilot_{pilot}" / "ledger.jsonl"))
        assert ledger.subtasks, pilot
