"""D5 audit gate: end-to-end behavioral leakage detection on a real
build, plus a synthetic-leakage fixture that injects future state and
confirms the audit catches it.

Claim:
    The behavioral prefix-truncation section detects rows whose
    feature values depend on events past the checkpoint step. The
    forbidden-column section catches y_*/final_*/verifier_* columns.
    Together they constitute the gate that blocks Workstream G.

Plausible wrong implementations:
    - section that compares row counts only, not column values
    - audit emits PASS even when behavioral diffs were found (bool
      collapsed wrong)
    - the gate ignores the missingness section, so a feature that
     silently fills 0 instead of None across a whole source slips
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.leakage.audit import (
    build_audit,
    render_audit,
    write_audit,
)


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_full_audit_on_tb_live_passes(real_ledger: None) -> None:
    """The end-to-end gate must run clean on the canonical tb_live
    source. If a D3 builder regresses, this test fails."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    assert audit.passed is True


def test_audit_renders_caveat_when_retrospective_source_used(real_ledger: None) -> None:
    """The audit on a swe_agent_pilot frame must include the
    retrospective caveat in the rendered text -- AGENTS.md invariant 5
    -- AND the audit must actually pass on this clean source. The
    `passed` assertion is load-bearing: an earlier version of the
    behavioral section produced false-positive LEAKAGE on every
    sparse-event source (events at non-contiguous steps), which
    would have been masked if this test only checked the caveat."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("swe_agent_pilot")
    audit = build_audit(df, sources=["swe_agent_pilot"])
    rendered = render_audit(audit)
    assert "Retrospective annotation caveat" in rendered
    # Critically: a sparse-event source must not produce a behavioral
    # false positive.
    behavioral = next(
        s for s in audit.sections if s.title == "Behavioral prefix-truncation audit"
    )
    assert behavioral.passed is True, behavioral.body
    assert audit.passed is True


def test_full_audit_on_hermes_pilot_h5_v2_passes(real_ledger: None) -> None:
    """Same regression guard for the second canonical retrospective
    source. Both sparse-event sources must clear the audit."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("hermes_pilot_h5_v2")
    audit = build_audit(df, sources=["hermes_pilot_h5_v2"])
    behavioral = next(
        s for s in audit.sections if s.title == "Behavioral prefix-truncation audit"
    )
    assert behavioral.passed is True, behavioral.body


def test_behavioral_section_catches_synthetic_leakage(
    real_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a future-leakage bug into the build path and confirm
    the behavioral prefix-truncation section catches it. We monkey-
    patch build_run_rows to inject a column whose value depends on
    a future event (the run's terminal step), simulating a leakage
    regression."""
    from coding_estimator.checkpoints import build as build_module

    real_build_run_rows = build_module.build_run_rows

    def leaky(run):
        rows = real_build_run_rows(run)
        terminal_step = max(e.step for e in run.events)
        for r in rows:
            r["leaky_feature"] = terminal_step  # depends on the future!
        return rows

    monkeypatch.setattr(build_module, "build_run_rows", leaky)
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    behavioral = next(
        s for s in audit.sections if s.title == "Behavioral prefix-truncation audit"
    )
    # leaky_feature is the SAME terminal step regardless of t, so the
    # mid-step rebuild produces a DIFFERENT terminal value -- the
    # synthetic run truncated at mid_step has a different terminal step
    # than the original.
    assert behavioral.passed is False
    assert "LEAKAGE" in behavioral.body


def test_audit_writes_named_file(real_ledger: None, tmp_path: Path) -> None:
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    path = write_audit(audit, tmp_path / "reports")
    assert path.name == "CHECKPOINT_CONSTRUCTION_AUDIT.md"
    text = path.read_text(encoding="utf-8")
    assert "Behavioral prefix-truncation audit" in text
    assert "PASS" in text


def test_audit_fails_when_forbidden_column_injected(
    real_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injecting a forbidden column into the audited frame must flip
    the audit to FAIL via the forbidden-column section. This is the
    last-line-of-defense check."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    df["final_success"] = True
    audit = build_audit(df, sources=["tb_live"])
    assert audit.passed is False
    forbidden = next(
        s for s in audit.sections if s.title == "Forbidden-column audit"
    )
    assert forbidden.passed is False


def test_overall_passed_collapses_correctly(real_ledger: None) -> None:
    """audit.passed must be the AND of all non-placeholder sections.
    A single FAIL must flip the verdict; placeholders alone never do."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    # Inject a fake FAIL section.
    from coding_estimator.leakage.audit import AuditSection

    audit.sections.append(
        AuditSection("synthetic-fail", "FAIL", passed=False, placeholder=False)
    )
    assert audit.passed is False
    # Now make it a placeholder; passed should not be affected by
    # placeholders.
    audit.sections[-1] = AuditSection(
        "synthetic-placeholder", "_pending_", passed=False, placeholder=True
    )
    assert audit.passed is True


def test_run_constancy_section_runs_when_columns_supplied(
    real_ledger: None,
) -> None:
    """When the caller passes feature/target columns, the run-constancy
    section runs and reports PASS on a clean frame."""
    from coding_estimator.checkpoints.build import build_source_frame

    df = build_source_frame("tb_live")
    df["y_test_target"] = 0  # run-constant by construction
    audit = build_audit(
        df,
        sources=["tb_live"],
        feature_columns=["source"],  # registered run-constant feature
        target_columns=["y_test_target"],
    )
    rc = next(s for s in audit.sections if s.title == "Run-constancy audit")
    # `source` is in run_constant register; y_test_target is run-constant.
    # But also `source` is identical (= 'tb_live') across all runs, so
    # this should fire.
    assert rc.passed is False


def test_synthetic_frame_with_no_features_fails_audit() -> None:
    """A synthetic frame missing every D3 feature column must fail the
    feature-columns section. This guards against accidentally running
    the audit on the wrong dataframe."""
    df = pd.DataFrame(
        [{"run_id": "r", "source": "tb_live", "checkpoint_step": 0}]
    )
    audit = build_audit(df, sources=["tb_live"])
    feat = next(
        s for s in audit.sections if s.title == "Feature columns by group"
    )
    assert feat.passed is False
