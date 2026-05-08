"""Combined long-form label table (Workstream E7).

Claim:
    `build_run_label_rows(run, terminal_labels)` emits exactly 7 rows
    per checkpoint covering the v0 targets:
      y_success_eventual, y_finish_step, y_finish_seconds, y_timeout,
      y_submit_without_validation, y_future_progress_drop_h5,
      y_validation_new_work_h5.
    Run-constant labels (success / submission family) carry the same
    `label_value` at every checkpoint of a run. Each LabelRow's
    `to_schema_dict()` view conforms to `schemas/label_schema.json`,
    which uses a NESTED `target_horizon: {units, value}` object.
    `label_available == not is_masked` for every emitted row.

Plausible wrong implementations:
    - flatten target_horizon in the schema-dict view too -> schema
      validation fails (the bug the critic surfaced).
    - emit a different set of targets (drop one, double up another).
    - run-constant targets diverge across checkpoints (should be the
      bug-surface for any future change to `_terminal_row` that
      sneakily reads checkpoint-local state).
    - swap label_available and is_masked sense -> ALL rows pass an
      identity-style test, but `label_available != not is_masked` would
      flip for masked rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from ledger_progress.serialization import load_events_jsonl

from coding_estimator.checkpoints.policy import p_step_checkpoints
from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.labels.build import LabelRow, build_run_label_rows
from coding_estimator.labels.terminal import TerminalLabels

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_run"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "label_schema.json"
LABEL_SCHEMA = json.loads(SCHEMA_PATH.read_text())

EXPECTED_TARGETS = (
    "y_success_eventual",
    "y_finish_step",
    "y_finish_seconds",
    "y_timeout",
    "y_submit_without_validation",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
)


def _golden_run() -> RunRecord:
    events = tuple(load_events_jsonl(str(FIXTURE_DIR / "ledger.jsonl")))
    return RunRecord(
        run_id="golden",
        source="tb_live",
        ledger_path=FIXTURE_DIR / "ledger.jsonl",
        events=events,
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id="golden",
        task_family="validation_new_work",
        arm="B",
        difficulty="medium",
        agent_scaffold="general-purpose",
        model_name="claude-sonnet-4-6",
        raw_metadata={},
    )


def _terminal_labels() -> TerminalLabels:
    # Hand-constructed so the test does not depend on the FinalLabel
    # loader (which requires a real source_metadata.json side file).
    return TerminalLabels(
        y_success_eventual=True,
        y_finish_step=13,
        y_finish_seconds=None,
        y_timeout=False,
        y_submit_without_validation=False,
    )


def test_emits_exactly_seven_targets_per_checkpoint() -> None:
    """7 targets × n_checkpoints rows. A wrong-arity implementation
    (e.g. duplicated or dropped target) would fail this check."""
    run = _golden_run()
    n_steps = len(p_step_checkpoints(run))
    rows = build_run_label_rows(run, _terminal_labels())
    assert len(rows) == 7 * n_steps, (len(rows), 7 * n_steps)
    by_step: dict[int, set[str]] = {}
    for r in rows:
        by_step.setdefault(r.checkpoint_step, set()).add(r.target_name)
    for step, names in by_step.items():
        assert names == set(EXPECTED_TARGETS), (step, names)


def test_run_constant_targets_have_identical_value_at_every_checkpoint() -> None:
    """y_success_eventual, y_submit_without_validation, y_finish_step,
    y_finish_seconds, y_timeout are run-constant by design (per
    registry V0_TARGETS). They must be replicated unchanged across
    every checkpoint of the same run."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    run_constant = {
        "y_success_eventual",
        "y_finish_step",
        "y_finish_seconds",
        "y_timeout",
        "y_submit_without_validation",
    }
    by_target: dict[str, set] = {t: set() for t in run_constant}
    for r in rows:
        if r.target_name in run_constant:
            by_target[r.target_name].add((r.label_value, r.is_masked))
    for target, values in by_target.items():
        assert len(values) == 1, (target, values)


def test_terminal_step_dynamics_rows_are_masked() -> None:
    """At the terminal checkpoint, the two h5 dynamics targets must
    be masked with reason 'is_terminal_checkpoint'. The terminal
    target rows (success/submission) must NOT be masked because they
    are run-constant and never_mask per the registry."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    terminal_step = max(r.checkpoint_step for r in rows)
    terminal_rows = [r for r in rows if r.checkpoint_step == terminal_step]
    for r in terminal_rows:
        if r.target_name in {"y_future_progress_drop_h5", "y_validation_new_work_h5"}:
            assert r.is_masked is True, r
            assert r.mask_reason == "is_terminal_checkpoint"
            assert r.label_value is None
        else:
            assert r.is_masked is False, r


def test_masked_implies_not_available_and_label_value_none() -> None:
    """Two-direction invariant on the masking semantics:
      - `is_masked=True` => `label_available=False` AND `label_value=None`
      - `label_available=True` <=> `label_value is not None`
    Note the converse `label_available=False => is_masked=True` does
    NOT hold: a never-mask terminal target (e.g. `y_finish_seconds` on
    a no-wallclock source) is unmasked but null. The test pins the
    correct asymmetric implication."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    for r in rows:
        if r.is_masked:
            assert r.label_available is False, r
            assert r.label_value is None, r
        assert r.label_available == (r.label_value is not None), r


def test_to_schema_dict_validates_against_label_schema_for_every_row() -> None:
    """The CRITIC bug: parquet stores horizon flat, but the schema
    requires nested `target_horizon: {units, value}`. `to_schema_dict()`
    must produce the nested view, and every row's view must validate.
    This is the regression test for that bug."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    validator = jsonschema.Draft202012Validator(LABEL_SCHEMA)
    for r in rows:
        d = r.to_schema_dict()
        # Hard-fails with ValidationError on any row that does not
        # match the schema; we don't need a separate assertion.
        validator.validate(d)


def test_to_schema_dict_horizon_for_terminal_target() -> None:
    """Terminal targets carry `target_horizon = {units: 'terminal',
    value: None}`. A bug that emitted units='steps' or value=0 would
    misclassify the target's window kind downstream."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    by_name = {r.target_name: r for r in rows if r.checkpoint_step == 0}
    success_row = by_name["y_success_eventual"].to_schema_dict()
    assert success_row["target_horizon"] == {"units": "terminal", "value": None}


def test_to_schema_dict_horizon_for_h5_target() -> None:
    """h5 dynamics targets carry `target_horizon = {units: 'steps',
    value: 5}`."""
    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    # Pick a non-terminal, non-masked h5 row at t=0.
    by_step_target = {
        (r.checkpoint_step, r.target_name): r for r in rows
    }
    drop_row = by_step_target[(0, "y_future_progress_drop_h5")].to_schema_dict()
    assert drop_row["target_horizon"] == {"units": "steps", "value": 5}


def test_target_names_match_v0_registry() -> None:
    """Emitted targets must EQUAL `V0_TARGETS.keys()`. Any drift
    (registry adds/drops a target without builder following, or vice
    versa) lands here as a hard fail."""
    from coding_estimator.labels.registry import V0_TARGETS

    run = _golden_run()
    rows = build_run_label_rows(run, _terminal_labels())
    emitted = {r.target_name for r in rows}
    assert emitted == set(EXPECTED_TARGETS)
    assert emitted == set(V0_TARGETS.keys()), (
        emitted ^ set(V0_TARGETS.keys())
    )


def test_finish_step_label_value_matches_terminal_labels_finish_step() -> None:
    """y_finish_step must carry the same numeric value at every
    checkpoint, equal to TerminalLabels.y_finish_step coerced to float."""
    run = _golden_run()
    term = _terminal_labels()
    rows = build_run_label_rows(run, term)
    for r in rows:
        if r.target_name == "y_finish_step":
            assert r.label_value == float(term.y_finish_step), r


def test_run_metadata_is_replicated_onto_every_label_row() -> None:
    """Claim:
        `build_run_label_rows()` propagates the run's exact task/model
        metadata onto every emitted label row.

    Plausible wrong implementations:
        - keep only `(run_id, checkpoint_id, target)` and force training
          code to guess task/model context later
        - copy `task_family` but drop exact `task_id` / `arm`
        - fill metadata only on terminal rows
    """
    rows = build_run_label_rows(_golden_run(), _terminal_labels())
    for r in rows:
        assert r.task_id == "golden"
        assert r.task_family == "validation_new_work"
        assert r.arm == "B"
        assert r.difficulty == "medium"
        assert r.agent_scaffold == "general-purpose"
        assert r.model_name == "claude-sonnet-4-6"


def test_label_row_dataclass_is_frozen() -> None:
    """LabelRow being frozen is part of the contract — any future
    mutation in row-postprocessing would produce a hard error rather
    than a silent edit."""
    import dataclasses

    assert dataclasses.is_dataclass(LabelRow)
    # `frozen=True` sets __setattr__ to raise FrozenInstanceError.
    sample = build_run_label_rows(_golden_run(), _terminal_labels())[0]
    try:
        sample.target_name = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("LabelRow is not frozen")
