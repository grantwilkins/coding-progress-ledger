"""Targets registry: every v0 target has a definition, window_kind,
mask_rule, and a compute reference (allowed to be None at this stage,
since Workstream E owns implementations). Deferred targets must have
v0=False."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from coding_estimator.labels.registry import DEFERRED_TARGETS, V0_TARGETS, all_targets

LABEL_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "label_schema.json").read_text()
)


def test_label_schema_is_valid_jsonschema() -> None:
    jsonschema.Draft202012Validator.check_schema(LABEL_SCHEMA)


def test_v0_has_exactly_seven_targets() -> None:
    """v0 ships seven targets per TASKS § E1: four binary headline
    (success/dynamics/validation/submission) + three terminal
    scalars (finish_step, finish_seconds, timeout). The seven
    must match exactly what build.py emits."""
    assert set(V0_TARGETS) == {
        "y_success_eventual",
        "y_future_progress_drop_h5",
        "y_validation_new_work_h5",
        "y_submit_without_validation",
        "y_finish_step",
        "y_finish_seconds",
        "y_timeout",
    }


@pytest.mark.parametrize("name", list(V0_TARGETS))
def test_v0_targets_are_well_formed(name: str) -> None:
    t = V0_TARGETS[name]
    assert t.v0 is True
    assert t.definition and t.window_kind and t.mask_rule
    assert t.window_kind in {"strict-future", "terminal", "regression"}
    assert t.horizon_units in {"steps", "seconds", "terminal", "none"}
    assert isinstance(t.run_constant_flag, bool)
    # The compute hook lands in Workstream E. We just check the slot exists.
    assert hasattr(t, "compute")


def test_run_constant_flagged_correctly() -> None:
    assert V0_TARGETS["y_success_eventual"].run_constant_flag is True
    assert V0_TARGETS["y_submit_without_validation"].run_constant_flag is True
    assert V0_TARGETS["y_future_progress_drop_h5"].run_constant_flag is False


def test_horizon_h5_targets_have_value_5() -> None:
    for n in ("y_future_progress_drop_h5", "y_validation_new_work_h5"):
        assert V0_TARGETS[n].horizon_value == 5
        assert V0_TARGETS[n].horizon_units == "steps"


def test_deferred_targets_have_v0_false() -> None:
    assert DEFERRED_TARGETS, "deferred targets registry must not be empty"
    for t in DEFERRED_TARGETS.values():
        assert t.v0 is False


def test_all_target_names_match_y_pattern() -> None:
    for name in all_targets():
        assert name.startswith("y_") and name.replace("_", "").islower()


def _label_row(name: str, value: float | None, masked: bool, reason: str | None) -> dict:
    t = V0_TARGETS[name]
    return {
        "run_id": "r1",
        "source": "tb_live",
        "checkpoint_id": "r1::5",
        "target_name": name,
        "target_family": t.family,
        "target_horizon": {"units": t.horizon_units, "value": t.horizon_value},
        "label_value": value,
        "is_masked": masked,
        "mask_reason": reason,
        "schema_version": "0.1.0",
    }


@pytest.mark.parametrize("name", list(V0_TARGETS))
def test_label_row_validates_against_schema(name: str) -> None:
    jsonschema.validate(_label_row(name, 1, False, None), LABEL_SCHEMA)
    jsonschema.validate(_label_row(name, None, True, "future_horizon_past_finish"), LABEL_SCHEMA)


def test_label_row_with_bad_target_name_rejected() -> None:
    bad = _label_row("y_future_progress_drop_h5", 0, False, None)
    bad["target_name"] = "Not-A-Label"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, LABEL_SCHEMA)
