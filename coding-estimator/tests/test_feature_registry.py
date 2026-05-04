"""Feature registry: every column belongs to exactly one group, all are
prefix-only, none collide with the forbidden-column list, and each
declaration validates against the JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from coding_estimator.checkpoints.features.registry import (
    GROUPS,
    all_features,
    feature_by_name,
)

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "feature_schema.json").read_text()
)


def test_schema_is_valid_jsonschema() -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_every_column_in_exactly_one_group() -> None:
    seen: dict[str, str] = {}
    for group_name, items in GROUPS.items():
        for f in items:
            assert f.column_name not in seen, (
                f"column {f.column_name} appears in {seen[f.column_name]} and {group_name}"
            )
            seen[f.column_name] = group_name
    assert {f.column_name for f in all_features()} == set(seen)


def test_every_feature_is_prefix_only() -> None:
    for f in all_features():
        assert f.prefix_only is True
        assert f.feature_or_label == "feature"


def test_no_overlap_with_forbidden_columns() -> None:
    # B4 forbidden prefixes/suffixes.
    forbidden_prefixes = ("y_", "label_", "final_", "verifier_", "test_output_", "eval_log_",
                          "final_diff_", "summary_by_category_", "shape_label", "shape_tags",
                          "checkpoint_event_index_at_terminal", "final_artifact_without_validation")
    for f in all_features():
        for fp in forbidden_prefixes:
            assert not f.column_name.startswith(fp), (
                f"feature {f.column_name} clashes with forbidden prefix {fp}"
            )


@pytest.mark.parametrize("f", all_features(), ids=lambda f: f.column_name)
def test_feature_validates_against_schema(f) -> None:
    record = {
        "column_name": f.column_name,
        "dtype": f.dtype,
        "group": f.group,
        "populated_on": list(f.populated_on),
        "upstream_source": f.upstream_source,
        "fill_when_missing": f.fill_when_missing,
        "run_constant_flag": f.run_constant_flag,
        "derivable_from": f.derivable_from,
        "feature_or_label": f.feature_or_label,
        "prefix_only": f.prefix_only,
    }
    jsonschema.validate(record, SCHEMA)


def test_run_constants_live_in_source_task_only() -> None:
    rc = [f for f in all_features() if f.run_constant_flag]
    assert rc, "expected source_task to declare run-constant features"
    for f in rc:
        assert f.group == "source_task"


def test_tb_live_only_features_isolated() -> None:
    for n in ("fraction_timeout_consumed", "remaining_timeout_budget"):
        f = feature_by_name(n)
        assert f.populated_on == ("tb_live",)


def test_wallclock_features_isolated() -> None:
    f = feature_by_name("elapsed_wall_time")
    assert f.populated_on == ("tb_live", "swe_agent_live_wallclock")


def test_lookup_unknown_raises() -> None:
    with pytest.raises(KeyError):
        feature_by_name("nope_nope")
