"""D2.5 audit skeleton: structural sections fire on a small frame
even before D3 feature builders exist. The skeleton is a stand-in
for the eventual D5 gate; placeholders are explicit, not silent.

Claim:
    build_audit() produces a CheckpointAudit with all required
    section titles. Sections that depend on D3 are marked
    `placeholder=True` and contribute neither PASS nor FAIL to the
    overall verdict. The forbidden-column section actually inspects
    the frame and returns FAIL when leakage is present.

Plausible wrong implementations:
    - silently drop placeholder sections so the report passes via
      missing checks
    - placeholder sections counted as PASS, hiding D5's still-pending
      work behind a fake green
    - forbidden-column check that uses substring matching, not the
      shared guard
    - retrospective caveat omitted from the rendered audit when
      retrospective sources are present
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.leakage.audit import (
    AUDIT_FILENAME,
    build_audit,
    render_audit,
    write_audit,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"run_id": "r1", "source": "tb_live", "active_leaf_count": 3},
            {"run_id": "r1", "source": "tb_live", "active_leaf_count": 4},
            {"run_id": "r2", "source": "swe_agent_pilot", "active_leaf_count": 2},
        ]
    )


def test_required_sections_present_on_clean_frame() -> None:
    audit = build_audit(_frame(), sources=["tb_live", "swe_agent_pilot"])
    titles = [s.title for s in audit.sections]
    for expected in audit.required_sections_present():
        if expected == "Retrospective-leakage caveat":
            # Caveat is rendered into the report via the report-template
            # helper, not as a separate section.
            continue
        assert expected in titles, f"missing section: {expected}"


def test_forbidden_column_section_fails_on_leakage() -> None:
    df = _frame()
    df["final_success"] = [True, True, False]  # forbidden
    audit = build_audit(df, sources=["tb_live"])
    forbidden = next(s for s in audit.sections if s.title == "Forbidden-column audit")
    assert forbidden.passed is False
    assert "final_success" in forbidden.body


def test_overall_passes_on_a_real_built_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real D4-built frame on a healthy source must produce a
    PASS audit. Failure here means a D5 section is over-strict or a
    builder is silently missing a column."""
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    rendered = render_audit(audit)
    # The Workstream-E-pending placeholder is acceptable.
    assert "Label balance" in rendered
    assert audit.passed is True


def test_renders_caveat_for_retrospective_sources() -> None:
    audit = build_audit(_frame(), sources=["swe_agent_pilot"])
    rendered = render_audit(audit)
    assert "Retrospective annotation caveat" in rendered


def test_renders_tb_framing_for_tb_live() -> None:
    audit = build_audit(_frame(), sources=["tb_live"])
    rendered = render_audit(audit)
    assert "TB-12 framing" in rendered


def test_run_constancy_section_fires_when_supplied(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"run_id": "r1", "source": "tb_live", "y_submit_without_validation": 1},
            {"run_id": "r2", "source": "swe_agent_pilot", "y_submit_without_validation": 0},
        ]
    )
    audit = build_audit(
        df,
        sources=["tb_live", "swe_agent_pilot"],
        feature_columns=["source"],
        target_columns=["y_submit_without_validation"],
    )
    rc = next(s for s in audit.sections if s.title == "Run-constancy audit")
    # `source` is in the run-constant register; y_submit_without_validation
    # is empirically run-constant. Pair must be flagged.
    assert rc.passed is False


def test_write_audit_emits_named_file(tmp_path: Path) -> None:
    audit = build_audit(_frame(), sources=["tb_live"])
    path = write_audit(audit, tmp_path / "reports")
    assert path.name == AUDIT_FILENAME
    assert path.is_file()


def test_failing_section_makes_audit_fail() -> None:
    df = _frame()
    df["final_success"] = [True, True, False]
    audit = build_audit(df, sources=["tb_live"])
    assert audit.passed is False


def test_required_section_list_is_explicit() -> None:
    """The list of required sections is a published contract; pin it
    so D5 can light up each placeholder one at a time without losing
    track of what's still missing."""
    audit = build_audit(_frame(), sources=["tb_live"])
    required = audit.required_sections_present()
    assert "Forbidden-column audit" in required
    assert "Run-constancy audit" in required
    assert "Behavioral prefix-truncation audit" in required
    assert "Missingness by feature and source" in required
    assert len(required) == 10
