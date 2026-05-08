"""Auto-generated caveat blocks fire whenever a report consumes
retrospective sources, and live-source framing fires when a live source
is in the source set. Both checks are mechanical -- absence is a bug.

Claim:
    caveat_block(sources) returns a non-empty caveat fragment whenever
    `sources` contains any swe_agent_* or hermes_* identifier; the
    live-source framing fragment fires when a live source is in the set.
    assert_caveat_present hard-fails when an emitted report omits a
    required fragment.

Plausible wrong implementations:
    - test only retrospective vs not-retrospective and miss live-source framing
    - return empty string when sources is empty (correct) but also when
      sources contains ONLY 'swe_agent_pilot' (incorrect)
    - substring-match on prefix (e.g. 'hermes' in source) -> false negatives
      when a future source is named differently
"""

from __future__ import annotations

import pytest

from coding_estimator.reports.caveats import (
    LIVE_SOURCES,
    assert_caveat_present,
    assert_registry_consistency,
    caveat_block,
)


def test_retrospective_sources_emit_caveat() -> None:
    out = caveat_block(["swe_agent_pilot"])
    assert "Retrospective annotation caveat" in out
    out2 = caveat_block(["hermes_pilot_h5_v2"])
    assert "Retrospective annotation caveat" in out2


def test_only_tb_live_emits_framing_not_retrospective() -> None:
    out = caveat_block(["tb_live"])
    assert "Live-source framing" in out
    assert "Retrospective annotation caveat" not in out


def test_tb_live_v2_also_emits_live_framing() -> None:
    out = caveat_block(["tb_live_v2"])
    assert "Live-source framing" in out
    assert "Retrospective annotation caveat" not in out


def test_mixed_sources_emit_both() -> None:
    out = caveat_block(["tb_live", "swe_agent_pilot"])
    assert "Retrospective annotation caveat" in out
    assert "Live-source framing" in out


def test_empty_source_set_emits_nothing() -> None:
    assert caveat_block([]) == ""


def test_assert_caveat_present_fails_when_caveat_missing() -> None:
    report = "## Section\n\nSome content with no caveat."
    with pytest.raises(AssertionError, match="caveat fragment"):
        assert_caveat_present(report, ["swe_agent_pilot"])


def test_assert_caveat_present_passes_when_caveat_present() -> None:
    report = (
        "# Header\n"
        + caveat_block(["swe_agent_pilot", "tb_live"])
        + "\n## Section"
    )
    assert_caveat_present(report, ["swe_agent_pilot", "tb_live"])


def test_assert_caveat_present_fails_for_partial_caveat() -> None:
    """A report that has the retrospective caveat but is missing the
    live-source framing must still hard-fail when both source families are
    in use."""
    partial = "# H\nRetrospective annotation caveat: ...\n"
    with pytest.raises(AssertionError, match="Live-source framing"):
        assert_caveat_present(partial, ["tb_live", "swe_agent_pilot"])


def test_live_sources_locked() -> None:
    # If LIVE_SOURCES drifts, every report-template consumer drifts
    # with it. Pin.
    assert frozenset({"tb_live", "tb_live_v2"}) == LIVE_SOURCES


def test_caveats_helper_in_sync_with_source_registry() -> None:
    """The retrospective set must be derivable from the source
    registry, not a hand-maintained prefix list. A new retrospective
    source added to sources.py is automatically picked up here; this
    test guards against drift."""
    assert_registry_consistency()


def test_unknown_source_id_not_treated_as_retrospective() -> None:
    """A source ID that isn't declared in the registry must not be
    silently treated as retrospective. The contract is: declare it,
    or it's not in the system."""
    out = caveat_block(["openai_agent_pilot"])
    assert "Retrospective annotation caveat" not in out
