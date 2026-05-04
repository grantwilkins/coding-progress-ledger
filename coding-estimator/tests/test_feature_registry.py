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
        "missingness_semantic": f.missingness_semantic.value,
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


def test_canonical_fill_for_source_outside_populated_on_returns_null() -> None:
    """Wallclock features must canonically-fill to None on retrospective
    sources, never 0 (silent fabrication of "0 elapsed wall time")."""
    f = feature_by_name("elapsed_wall_time")
    assert "swe_agent_pilot" not in f.populated_on
    assert f.canonical_fill_for("swe_agent_pilot") is None


def test_canonical_fill_distinguishes_not_applicable_from_semantic_fill() -> None:
    """The early-return for `source not in populated_on` is load-bearing.
    Construct a hypothetical feature whose semantic fill is 0 but whose
    populated_on excludes a source — and confirm the source-check wins
    (returns None, not 0). Without the early return, this would
    silently fabricate a 0 for the inapplicable source."""
    from coding_estimator.checkpoints.features.missingness import Missingness
    from coding_estimator.checkpoints.features.registry import Feature

    f = Feature(
        column_name="hypothetical",
        dtype="bool",
        group="validation",
        populated_on=("tb_live",),
        upstream_source=None,
        missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN,
        run_constant_flag=False,
    )
    # On the populated source: declared semantic fires -> 0 (False).
    assert f.canonical_fill_for("tb_live") == 0
    # OUTSIDE the populated set: not_applicable_to_source -> None.
    assert f.canonical_fill_for("swe_agent_pilot") is None


def test_canonical_fill_for_source_in_populated_on_uses_declared_semantic() -> None:
    """A feature whose source IS in populated_on uses the declared
    missingness semantic, not the not-applicable-to-source default."""
    from coding_estimator.checkpoints.features.missingness import Missingness

    f = feature_by_name("validation_failed")
    # validation_failed is APPLICABLE_NEVER_OBSERVED_IN_RUN -> 0/False fill
    assert f.missingness_semantic == Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN
    assert f.canonical_fill_for("tb_live") == 0
    f2 = feature_by_name("num_validation_attempts")
    # count is APPLICABLE_ABSENT_SO_FAR -> 0 fill
    assert f2.canonical_fill_for("tb_live") == 0


def test_unknown_due_to_missing_artifact_fills_null() -> None:
    """When a feature requires a side artifact (live_instrumentation
    timestamps, source_metadata identifiers) and that artifact is
    missing, the cell must be None -- NEVER 0/False."""
    from coding_estimator.checkpoints.features.missingness import Missingness

    f = feature_by_name("fraction_timeout_consumed")
    assert f.missingness_semantic == Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT
    assert f.canonical_fill_for("tb_live") is None


def test_every_feature_declares_a_missingness_semantic() -> None:
    """No feature is allowed to be silently defaulted; the registry
    closes the loop on AGENTS.md invariant 7."""
    for f in all_features():
        # Just confirm it's set; the schema test confirms the value is
        # in the enum.
        assert f.missingness_semantic is not None


def test_no_feature_uses_unknown_semantic_when_no_artifact_dependency() -> None:
    """UNKNOWN_DUE_TO_MISSING_ARTIFACT is reserved for features that
    genuinely depend on a side artifact. Pure aggregate counters
    (num_*, *_count, *_progress) must not use it -- otherwise the
    semantic is being used as a generic 'we didn't compute it' fallback."""
    from coding_estimator.checkpoints.features.missingness import Missingness

    artifact_dependent_prefixes = (
        "elapsed_wall_time",
        "fraction_timeout",
        "remaining_timeout",
        "agent_scaffold",
        "model_name",
        "task_family_hash",
        "repo_family_hash",
        "initial_prompt_length",
        "initial_files_count",
        "latest_completion_evidence_type",
    )
    for f in all_features():
        if f.missingness_semantic == Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT:
            assert any(
                f.column_name.startswith(p) or f.column_name == p
                for p in artifact_dependent_prefixes
            ), (
                f"feature {f.column_name} uses UNKNOWN_DUE_TO_MISSING_ARTIFACT "
                "but has no clear artifact dependency"
            )
