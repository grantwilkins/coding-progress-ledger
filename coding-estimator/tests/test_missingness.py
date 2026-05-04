"""The four-valued missingness enum is the contract that distinguishes
"not applicable" from "absent so far" from "never observed" from
"unknown due to missing artifact." Mixing these is a real risk; the
enum keeps them apart.

Claim:
    Missingness has exactly four members; canonical fills are None for
    not_applicable_to_source and unknown_due_to_missing_artifact, and
    0/False for the two applicable-but-not-observed semantics.

Plausible wrong implementations:
    - collapse the four values into a 3-valued or 2-valued enum
    - return 0 for unknown_due_to_missing_artifact (silent fabrication)
    - return None for applicable_absent_so_far (loses signal)
"""

from __future__ import annotations

from coding_estimator.checkpoints.features.missingness import (
    CANONICAL_FILL,
    Missingness,
)


def test_enum_has_exactly_four_members() -> None:
    assert {m.value for m in Missingness} == {
        "not_applicable_to_source",
        "applicable_absent_so_far",
        "applicable_never_observed_in_run",
        "unknown_due_to_missing_artifact",
    }


def test_not_applicable_fills_null_not_zero() -> None:
    assert CANONICAL_FILL[Missingness.NOT_APPLICABLE_TO_SOURCE] is None


def test_unknown_due_to_missing_artifact_fills_null_not_zero() -> None:
    """This is the load-bearing semantic: when a source-side artifact is
    missing, we cannot know the value. Filling 0/False here would be
    silent fabrication that mixes 'never happened' with 'we don't know.'"""
    assert CANONICAL_FILL[Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT] is None


def test_applicable_semantics_fill_zero_not_null() -> None:
    """For 'applicable but not observed at this t,' 0/False is the right
    summary -- the value at t is well-defined even if the run later
    surfaces a non-zero value. Filling None here would lose signal."""
    assert CANONICAL_FILL[Missingness.APPLICABLE_ABSENT_SO_FAR] == 0
    assert CANONICAL_FILL[Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN] == 0


def test_canonical_fill_covers_every_enum_member() -> None:
    assert set(CANONICAL_FILL) == set(Missingness)
