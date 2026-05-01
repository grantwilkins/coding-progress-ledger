"""Mission features 3, 7, 8, 9, 10 must be first-class columns in the
observation channel — the 'progress' verb of the user's mission.

These are integration-style tests over the rebuilt
datasets/swe_agent_pilot_observations_step.csv. They lock in:
- per-category progress columns exist and are bounded in [0, 1]
- evidence-strength columns are populated (cum_*)
- stalled-interval columns are non-negative and monotone within a run
- step-windowed event counts sum to plausible values across the run
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEP_CSV = ROOT / "datasets" / "swe_agent_pilot_observations_step.csv"

NEW_COLUMNS = (
    "product_progress",
    "validation_progress",
    "investigation_progress",
    "step_added_subtasks",
    "step_split_events",
    "step_reopen_events",
    "step_invalidation_events",
    "step_product_completes",
    "step_validation_completes",
    "step_investigation_completes",
    "step_strong_completions",
    "step_manual_only_completions",
    "steps_since_progress_increase",
    "steps_since_completion",
    "steps_since_subtask_added",
    "cum_strong_completions",
    "cum_manual_only_completions",
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    assert STEP_CSV.exists(), f"missing {STEP_CSV} — rebuild via build_ledger_observation_dataset.py"
    with STEP_CSV.open() as f:
        return list(csv.DictReader(f))


def test_all_new_columns_present(rows):
    header = set(rows[0].keys())
    missing = [c for c in NEW_COLUMNS if c not in header]
    assert not missing, missing


def test_per_category_progress_bounded(rows):
    for r in rows:
        for col in ("product_progress", "validation_progress", "investigation_progress"):
            v = float(r[col])
            assert 0.0 <= v <= 1.0, (r["run_id"], r["step"], col, v)


def test_step_windowed_event_counts_nonnegative(rows):
    for r in rows:
        for col in (
            "step_added_subtasks", "step_split_events", "step_reopen_events",
            "step_invalidation_events", "step_product_completes",
            "step_validation_completes", "step_investigation_completes",
            "step_strong_completions", "step_manual_only_completions",
        ):
            assert int(r[col]) >= 0, (r["run_id"], r["step"], col)


def test_cumulative_completion_counts_monotone(rows):
    by_run: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    for run_id, run_rows in by_run.items():
        prev_strong = 0
        prev_manual = 0
        for r in run_rows:
            strong = int(r["cum_strong_completions"])
            manual = int(r["cum_manual_only_completions"])
            assert strong >= prev_strong, (run_id, r["step"], "cum_strong regressed")
            assert manual >= prev_manual, (run_id, r["step"], "cum_manual regressed")
            prev_strong, prev_manual = strong, manual


def test_stalled_interval_columns_nonnegative(rows):
    for r in rows:
        for col in ("steps_since_progress_increase", "steps_since_completion", "steps_since_subtask_added"):
            assert int(r[col]) >= 0, (r["run_id"], r["step"], col)


def test_evidence_strength_total_matches_k1_audit_order_of_magnitude(rows):
    by_run = {}
    for r in rows:
        by_run[r["run_id"]] = r
    total_strong = sum(int(r["cum_strong_completions"]) for r in by_run.values())
    assert total_strong > 0, "no strong completions recorded; classifier or wiring broken"
    assert total_strong < 1000, total_strong


def test_f_03_stuck_loop_shape(rows):
    f03 = [r for r in rows if r["run_id"] == "swe_agent_pilot_f_03"]
    assert f03, "f_03 missing from corpus"
    final = f03[-1]
    assert float(final["product_progress"]) == 0.0, "f_03 should never reach PRODUCT"
    assert int(final["steps_since_progress_increase"]) > 0, "f_03 stuck-loop should show positive stalled interval"


def test_f_06_high_progress_failure_shape(rows):
    f06 = [r for r in rows if r["run_id"] == "swe_agent_pilot_f_06"]
    assert f06, "f_06 missing from corpus"
    final = f06[-1]
    assert float(final["coding_progress"]) == 1.0, "f_06 is the canonical 1.00-progress failure"
    assert final["final_success"] in {"False", "false", "0"}, final["final_success"]
