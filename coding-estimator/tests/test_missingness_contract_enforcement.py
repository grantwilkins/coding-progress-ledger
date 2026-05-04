"""Cross-check that the build pipeline's actual null/zero pattern
matches the registry's declared missingness semantics.

Wires the catalogue (registry.canonical_fill_for) to the data
(actual values in the built frame), so a builder that drifts from
its declared semantic is caught.

Claim:
    For every feature column produced by the build pipeline, the
    null pattern across sources matches the registry's
    `canonical_fill_for(source)` semantics:
    - source not in populated_on -> all values null
    - source in populated_on, semantic UNKNOWN_DUE_TO_MISSING_ARTIFACT
      -> may be null OR populated; presence depends on the artifact
    - other semantics on populated sources -> never silently null on
      every row of the source

Plausible wrong implementations:
    - a builder that fills 0 on a source where it should be null
    - a builder that returns null on a source where the semantic says
      0/False is correct
    - the wallclock case: silently fabricating elapsed_wall_time = 0
      on retrospective sources
"""

from __future__ import annotations

import pytest

from coding_estimator.checkpoints.features.missingness import Missingness
from coding_estimator.checkpoints.features.registry import all_features


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_wallclock_features_are_null_on_swe_agent_pilot(real_ledger: None) -> None:
    """`elapsed_wall_time` is `populated_on=tb_live, swe_agent_live_wallclock`;
    every row from `swe_agent_pilot` must have a null value, NEVER 0."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("swe_agent_pilot")
    assert df["elapsed_wall_time"].isna().all()
    # And specifically: not silently filled with 0. (Use isna directly
    # rather than fillna+compare to avoid pandas dtype-coercion warnings.)
    assert (df["elapsed_wall_time"] == 0).sum() == 0


def test_tb_live_features_populated_on_tb_live(real_ledger: None) -> None:
    """elapsed_wall_time on tb_live must be populated for every row past
    the first event; missing-artifact semantics shouldn't fire."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    # At least one row per run must have a non-null elapsed_wall_time.
    by_run = df.groupby("run_id")["elapsed_wall_time"].apply(
        lambda s: s.notna().any()
    )
    assert by_run.all(), "every tb_live run must have at least one populated wallclock row"


def test_count_features_emit_zero_not_null_on_step_zero(real_ledger: None) -> None:
    """Counter features (num_*) at step 0 are APPLICABLE_ABSENT_SO_FAR;
    the canonical fill is 0, NEVER null. A builder that returns null at
    step 0 would violate the contract."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    step_zero = df[df["checkpoint_step"] == 0]
    for col in (
        "num_adds_so_far",
        "num_splits_so_far",
        "num_reopens_so_far",
        "num_invalidations_so_far",
        "num_progress_drops_so_far",
    ):
        assert (step_zero[col] == 0).all(), col
        assert step_zero[col].notna().all(), col


def test_validation_flags_emit_false_not_null_when_no_validation_yet(
    real_ledger: None,
) -> None:
    """Validation bool flags are APPLICABLE_NEVER_OBSERVED_IN_RUN; their
    canonical fill is False, never null. A run with no validation events
    (rare on tb_live but possible) must emit False, not null."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    for col in (
        "validation_failed",
        "validation_blocked",
    ):
        # Whatever the runs look like, no row may be null on these flags.
        assert df[col].notna().all(), col
        assert df[col].dtype == bool, col


def test_unknown_due_to_missing_artifact_can_be_null(real_ledger: None) -> None:
    """fraction_timeout_consumed and remaining_timeout_budget are
    UNKNOWN_DUE_TO_MISSING_ARTIFACT in v0 (timeout artifact not yet
    plumbed). Their values must be uniformly null on tb_live; if
    ever populated, the registry's semantic must be updated."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    assert df["fraction_timeout_consumed"].isna().all()
    assert df["remaining_timeout_budget"].isna().all()


def test_every_built_column_matches_a_registry_entry(real_ledger: None) -> None:
    """Defensive: every feature column in the built frame must appear
    in the registry. Catches drift where a builder emits an undeclared
    column that downstream consumers won't know how to fill."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    registered = {f.column_name for f in all_features()}
    identity_cols = {
        "run_id", "source", "checkpoint_id", "checkpoint_step",
        "checkpoint_event_index", "checkpoint_wall_time",
        "checkpoint_elapsed_seconds", "checkpoint_fraction_timeout",
        "is_terminal_checkpoint", "timestamp_quality", "ledger_path",
        "schema_version", "builder_commit_sha", "source_protocol_version",
    }
    seen = set(df.columns)
    undeclared = seen - registered - identity_cols
    assert not undeclared, undeclared


def test_canonical_fill_for_is_self_consistent_across_sources(real_ledger: None) -> None:
    """For each feature, calling canonical_fill_for on a source NOT in
    its populated_on must always return None. This pins the registry's
    declared semantic against a regression where the early-return
    accidentally evaluates the missingness semantic instead."""
    for feature in all_features():
        unsupported = (
            "_definitely_not_a_real_source_id_"
            if feature.populated_on else "tb_live"
        )
        if unsupported in feature.populated_on:
            continue
        assert feature.canonical_fill_for(unsupported) is None, feature.column_name


def test_count_features_use_applicable_absent_semantic() -> None:
    """The catalogue itself: `num_*_so_far` columns must declare the
    APPLICABLE_ABSENT_SO_FAR semantic. A future contributor who marks
    one as UNKNOWN_DUE_TO_MISSING_ARTIFACT would silently change the
    fill from 0 to None."""
    for f in all_features():
        if f.column_name.startswith("num_") and f.column_name.endswith("_so_far"):
            assert f.missingness_semantic == Missingness.APPLICABLE_ABSENT_SO_FAR, (
                f.column_name
            )
