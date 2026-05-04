"""Four-valued missingness semantics for feature columns.

Required reading: AGENTS.md cross-cutting invariant 7.

A feature whose value is "missing" at a checkpoint can mean four
distinct things, and the right fill depends on which one.
"""

from __future__ import annotations

from enum import Enum


class Missingness(Enum):
    NOT_APPLICABLE_TO_SOURCE = "not_applicable_to_source"
    APPLICABLE_ABSENT_SO_FAR = "applicable_absent_so_far"
    APPLICABLE_NEVER_OBSERVED_IN_RUN = "applicable_never_observed_in_run"
    UNKNOWN_DUE_TO_MISSING_ARTIFACT = "unknown_due_to_missing_artifact"


# The canonical fill for each semantic. Encoded here so feature builders
# do not invent their own conventions.
CANONICAL_FILL: dict[Missingness, object] = {
    # Concept does not apply -> emit null/NaN; no count, no zero.
    Missingness.NOT_APPLICABLE_TO_SOURCE: None,
    # Event has not been observed yet at t but the value at t is well-
    # defined (e.g. count==0, flag==False). Emit the false/zero variant.
    Missingness.APPLICABLE_ABSENT_SO_FAR: 0,
    # Same as above for the model: at t we cannot distinguish "not yet"
    # from "never," and 0/False is the right summary at this t.
    Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN: 0,
    # Source-side artifact missing -> we cannot know. Emit null and let
    # the leakage audit surface it.
    Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT: None,
}
