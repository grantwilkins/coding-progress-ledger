"""checkpoint_schema.json validates one row per source."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "checkpoint_schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _row(source: str, **overrides: object) -> dict:
    has_live_wallclock = source in {"tb_live", "tb_live_v2"}
    base: dict = {
        "run_id": "run-1",
        "source": source,
        "checkpoint_id": f"run-1::{source}::5",
        "checkpoint_step": 5,
        "checkpoint_event_index": 12,
        "is_terminal_checkpoint": False,
        "timestamp_quality": "real" if has_live_wallclock else "missing",
        "checkpoint_wall_time": "2026-05-04T10:00:00Z" if has_live_wallclock else None,
        "checkpoint_elapsed_seconds": 42.0 if has_live_wallclock else None,
        "checkpoint_fraction_timeout": 0.05 if source == "tb_live" else None,
        "ledger_path": f"runs/{source}/run-1/ledger.jsonl",
        "schema_version": "0.1.0",
        "builder_commit_sha": "deadbee",
        "source_protocol_version": "v1",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "source",
    [
        "swe_agent_pilot",
        "swe_agent_pilot_v3",
        "swe_agent_live",
        "swe_agent_live_wallclock",
        "hermes_pilot",
        "hermes_pilot_h5",
        "hermes_pilot_h5_v2",
        "tb_live",
        "tb_live_v2",
    ],
)
def test_row_validates(source: str) -> None:
    jsonschema.validate(_row(source), _schema())


def test_schema_is_valid_jsonschema() -> None:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_unknown_source_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row("not_a_source"), _schema())


def test_missing_required_rejected() -> None:
    row = _row("tb_live")
    del row["checkpoint_id"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(row, _schema())


def test_negative_step_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row("tb_live", checkpoint_step=-1), _schema())


def test_bad_schema_version_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row("tb_live", schema_version="v1"), _schema())
