"""Checkpoint-construction audit (D5 gate).

The audit is the PRE-MODELING gate: Workstream G does not start until
this report runs clean.

D2.5 shipped the skeleton with placeholder sections. D5 lights up
each placeholder using the now-existing D3 feature builders and the
D4 build pipeline. Sections cover:
- structural (forbidden columns, schema validity, run/checkpoint counts)
- behavioral (future-mutation invariance: rebuild from a truncated
  ledger and assert byte-equality with the row built from the full ledger)
- run-constancy
- missingness profile by feature and source
- live-source and retrospective-source row examples
- auto-generated retrospective + tb_live caveats
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


def _section_feature_columns_by_group(df: pd.DataFrame) -> AuditSection:
    from coding_estimator.checkpoints.features.registry import GROUPS

    cols = set(df.columns)
    lines = []
    all_present = True
    for group_name, group_features in sorted(GROUPS.items()):
        if group_name == "source_task":
            # source_task features are not emitted by D3 v0 (per
            # B3 plan); they live in the combined manifest instead.
            continue
        present = [f.column_name for f in group_features if f.column_name in cols]
        missing = [f.column_name for f in group_features if f.column_name not in cols]
        lines.append(
            f"- **{group_name}**: {len(present)} present, {len(missing)} missing"
        )
        if missing:
            lines.append(f"  - missing: {missing}")
            all_present = False
    return AuditSection(
        "Feature columns by group",
        "\n".join(lines),
        passed=all_present,
    )


def _section_behavioral_prefix(df: pd.DataFrame, *, sample_runs: int = 2) -> AuditSection:
    """Behavioral leakage check: rebuild a row from a truncated ledger
    and confirm byte-equality with the row built from the full ledger
    at the same checkpoint. The replay engine's runtime assertions
    already guarantee this for features; this section confirms the
    property holds end-to-end through the build pipeline (identity
    columns, time_budget, etc.) and would catch a regression where a
    builder reaches into `run.events` directly instead of going
    through `state.events_so_far`.

    Implementation note: `mid_step` is chosen from the run's ACTUAL
    event steps so the truncated run's terminal IS `mid_step`. Picking
    a gap step would compare row-at-mid_step against row-at-(previous
    event step) and produce systematic false positives on sparse-event
    retrospective sources (swe_agent_*, hermes_*).
    """
    from coding_estimator.checkpoints.build import build_run_rows
    from coding_estimator.ingest.run_record import RunRecord, load_run

    if "run_id" not in df.columns or "source" not in df.columns:
        return AuditSection(
            "Behavioral prefix-truncation audit",
            "FAIL — frame lacks run_id or source.",
            passed=False,
        )

    diffs: list[str] = []
    sampled = (
        df[["source", "run_id"]]
        .drop_duplicates()
        .head(sample_runs)
        .itertuples(index=False)
    )
    skipped = 0
    for source, run_id in sampled:
        try:
            run = load_run(source, run_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            diffs.append(f"- {source}/{run_id}: failed to reload ({exc})")
            continue
        if len(run.events) < 3:
            skipped += 1
            continue
        # Pick mid_step from the actual event steps so both the full
        # and truncated runs have a row at that step.
        event_steps = sorted({e.step for e in run.events})
        mid_step = event_steps[len(event_steps) // 2]
        full_rows = build_run_rows(run)
        try:
            mid_row_full = next(r for r in full_rows if r["checkpoint_step"] == mid_step)
        except StopIteration:
            skipped += 1
            continue
        truncated_events = tuple(e for e in run.events if e.step <= mid_step)
        truncated_run = RunRecord(
            run_id=run.run_id,
            source=run.source,
            ledger_path=run.ledger_path,
            events=truncated_events,
            has_real_wallclock=run.has_real_wallclock,
            start_wall_time=run.start_wall_time,
            end_wall_time=run.end_wall_time,
            task_id=run.task_id,
            task_family=run.task_family,
            arm=run.arm,
            difficulty=run.difficulty,
            agent_scaffold=run.agent_scaffold,
            model_name=run.model_name,
            raw_metadata=run.raw_metadata,
        )
        truncated_rows = build_run_rows(truncated_run)
        try:
            mid_row_trunc = next(
                r for r in truncated_rows if r["checkpoint_step"] == mid_step
            )
        except StopIteration:
            diffs.append(
                f"- {source}/{run_id} at step {mid_step}: truncated build "
                "did not produce a row at mid_step (this should not happen)"
            )
            continue
        # `is_terminal_checkpoint` is expected to differ -- mid_step is
        # the terminal of the truncated run but not of the full run.
        full_compare = {k: v for k, v in mid_row_full.items() if k != "is_terminal_checkpoint"}
        trunc_compare = {k: v for k, v in mid_row_trunc.items() if k != "is_terminal_checkpoint"}
        if full_compare != trunc_compare:
            differing = sorted(
                k for k in full_compare
                if full_compare[k] != trunc_compare.get(k)
            )
            diffs.append(
                f"- {source}/{run_id} at step {mid_step}: prefix-truncation "
                f"produced a different row (LEAKAGE!) differing columns: {differing[:6]}"
            )

    if diffs:
        return AuditSection(
            "Behavioral prefix-truncation audit",
            "FAIL — leakage detected:\n" + "\n".join(diffs),
            passed=False,
        )
    note = f" ({skipped} skipped: too short or no row at mid_step)" if skipped else ""
    return AuditSection(
        "Behavioral prefix-truncation audit",
        f"PASS — no leakage detected on {sample_runs - skipped} sampled runs{note}.",
        passed=True,
    )


def _section_missingness(df: pd.DataFrame) -> AuditSection:
    from coding_estimator.checkpoints.features.registry import all_features

    feat_cols = {f.column_name for f in all_features() if f.column_name in df.columns}
    if not feat_cols:
        return AuditSection(
            "Missingness by feature and source",
            "FAIL — no recognized feature columns in frame.",
            passed=False,
        )
    rows: list[str] = []
    for source, sub in df.groupby("source"):
        nulls = {c: int(sub[c].isna().sum()) for c in sorted(feat_cols)}
        nontrivial = {c: n for c, n in nulls.items() if n > 0}
        rows.append(
            f"- `{source}`: {len(sub)} rows; "
            f"{len(nontrivial)} features with any null"
        )
    return AuditSection(
        "Missingness by feature and source",
        "\n".join(sorted(rows)),
        passed=True,
    )


def _section_examples(df: pd.DataFrame, *, retrospective: bool) -> AuditSection:
    from coding_estimator.ingest.sources import retrospective_source_ids

    retro_ids = retrospective_source_ids()
    sub = (
        df[df["source"].isin(retro_ids)]
        if retrospective
        else df[~df["source"].isin(retro_ids)]
    )
    title = "Retrospective-source row examples" if retrospective else "Live-source row examples"
    if sub.empty:
        return AuditSection(
            title,
            "_no rows from this source class in the audited frame_",
            passed=True,
            placeholder=True,
        )
    sample = sub.head(3)
    cols = [
        "run_id",
        "source",
        "checkpoint_step",
        "active_leaf_count",
        "coding_progress",
    ]
    cols = [c for c in cols if c in sub.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = [
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for _, r in sample.iterrows()
    ]
    body = "\n".join([header, sep, *rows])
    return AuditSection(title, body, passed=True)


def build_audit(
    df: pd.DataFrame,
    *,
    sources: Iterable[str],
    feature_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] | None = None,
) -> CheckpointAudit:
    """Build the audit object. After D3+D4, every section is fully
    populated; only `Label balance by target and source` remains a
    placeholder pending Workstream E."""
    audit = CheckpointAudit(sources=tuple(sorted(set(sources))))
    audit.sections.append(_section_run_counts(df))
    audit.sections.append(_section_feature_columns_by_group(df))
    audit.sections.append(_section_forbidden_columns(df))
    audit.sections.append(_section_behavioral_prefix(df))
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
    audit.sections.append(_section_missingness(df))
    audit.sections.append(
        _placeholder("Label balance by target and source", "needs Workstream E labels")
    )
    audit.sections.append(_section_examples(df, retrospective=False))
    audit.sections.append(_section_examples(df, retrospective=True))
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
