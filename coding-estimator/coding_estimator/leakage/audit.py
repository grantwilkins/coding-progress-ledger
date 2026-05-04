"""Checkpoint-construction audit skeleton.

The audit is a structural report that a checkpoint dataset has not
leaked future state, has no forbidden columns, has run-constancy
clean, and lists the missingness profile per source. It is the
PRE-MODELING gate: Workstream G does not start until this report runs
clean.

This module ships the skeleton in D2.5; the full populated audit lands
in D5 after D3 feature builders exist. Until then, the skeleton
asserts on what it CAN check (forbidden columns, run-constancy) and
emits a placeholder for sections that depend on D3.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from coding_estimator.leakage.guard import find_forbidden, load_forbidden_spec
from coding_estimator.leakage.run_constancy import audit as run_constancy_audit
from coding_estimator.reports.caveats import caveat_block

AUDIT_FILENAME = "CHECKPOINT_CONSTRUCTION_AUDIT.md"


@dataclass(frozen=True)
class AuditSection:
    title: str
    body: str
    passed: bool
    placeholder: bool = False


@dataclass
class CheckpointAudit:
    sources: tuple[str, ...]
    sections: list[AuditSection] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.passed or s.placeholder for s in self.sections)

    def required_sections_present(self) -> list[str]:
        return [
            "Run and checkpoint counts per source",
            "Feature columns by group",
            "Forbidden-column audit",
            "Behavioral prefix-truncation audit",
            "Run-constancy audit",
            "Missingness by feature and source",
            "Label balance by target and source",
            "Live-source row examples",
            "Retrospective-source row examples",
            "Retrospective-leakage caveat",
        ]


def _placeholder(title: str, reason: str) -> AuditSection:
    return AuditSection(
        title=title,
        body=f"_PLACEHOLDER — populated by D5 once D3 feature builders ship. ({reason})_",
        passed=False,
        placeholder=True,
    )


def _section_forbidden_columns(df: pd.DataFrame) -> AuditSection:
    bad = find_forbidden(df.columns, spec=load_forbidden_spec())
    if bad:
        body = "FAIL — frame contains forbidden columns:\n\n" + "\n".join(
            f"- `{c}`" for c in bad
        )
        return AuditSection("Forbidden-column audit", body, passed=False)
    return AuditSection(
        "Forbidden-column audit",
        "PASS — no forbidden columns detected on the checkpoint frame.",
        passed=True,
    )


def _section_run_constancy(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_columns: Iterable[str],
) -> AuditSection:
    offenders = run_constancy_audit(
        df, feature_columns=feature_columns, target_columns=target_columns
    )
    if offenders:
        body = "FAIL — joint run-constant pairs detected:\n\n" + "\n".join(
            f"- feature=`{f}` target=`{t}`" for f, t in offenders
        )
        return AuditSection("Run-constancy audit", body, passed=False)
    return AuditSection(
        "Run-constancy audit",
        "PASS — no joint run-constant (feature, target) pairs.",
        passed=True,
    )


def _section_run_counts(df: pd.DataFrame) -> AuditSection:
    if "source" not in df.columns or "run_id" not in df.columns:
        return AuditSection(
            "Run and checkpoint counts per source",
            "FAIL — frame lacks `source` or `run_id`.",
            passed=False,
        )
    rows = []
    for src, sub in df.groupby("source"):
        rows.append(
            f"- `{src}`: {sub['run_id'].nunique()} runs, {len(sub)} checkpoints"
        )
    return AuditSection(
        "Run and checkpoint counts per source",
        "\n".join(sorted(rows)),
        passed=True,
    )


def build_audit(
    df: pd.DataFrame,
    *,
    sources: Iterable[str],
    feature_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] | None = None,
) -> CheckpointAudit:
    """Build the audit object. Sections that depend on D3 builders are
    emitted as placeholders until D3 lands."""
    audit = CheckpointAudit(sources=tuple(sorted(set(sources))))
    audit.sections.append(_section_run_counts(df))
    audit.sections.append(
        _placeholder("Feature columns by group", "needs D3 feature builders")
    )
    audit.sections.append(_section_forbidden_columns(df))
    audit.sections.append(
        _placeholder("Behavioral prefix-truncation audit", "needs D3 + D4 build CLI")
    )
    if feature_columns is not None and target_columns is not None:
        audit.sections.append(
            _section_run_constancy(
                df, feature_columns=feature_columns, target_columns=target_columns
            )
        )
    else:
        audit.sections.append(
            _placeholder("Run-constancy audit", "no feature/target columns supplied")
        )
    audit.sections.append(
        _placeholder("Missingness by feature and source", "needs D3 feature builders")
    )
    audit.sections.append(
        _placeholder("Label balance by target and source", "needs Workstream E labels")
    )
    audit.sections.append(
        _placeholder("Live-source row examples", "needs D3 feature builders")
    )
    audit.sections.append(
        _placeholder("Retrospective-source row examples", "needs D3 feature builders")
    )
    return audit


def render_audit(audit: CheckpointAudit) -> str:
    parts = ["# Checkpoint construction audit", ""]
    parts.append(caveat_block(audit.sources))
    parts.append("")
    for section in audit.sections:
        parts.append(f"## {section.title}")
        parts.append("")
        parts.append(section.body)
        parts.append("")
    parts.append("---")
    parts.append(f"Overall: {'PASS' if audit.passed else 'FAIL'}")
    return "\n".join(parts) + "\n"


def write_audit(audit: CheckpointAudit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / AUDIT_FILENAME
    target.write_text(render_audit(audit), encoding="utf-8", newline="\n")
    return target
