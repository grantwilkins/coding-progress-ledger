"""F11 — Profiling go/no-go gate.

Five binary checks decide whether the data is ready to train baselines
on. The gate consumes already-built artifacts (manifest + checkpoints +
labels frames). It does NOT raise — it returns a structured result and
emits a markdown report. Callers (CI, orchestration scripts) decide
whether to halt; per AGENTS.md hard-fail discipline, do halt on FAIL.

Criteria (TASKS § F11):
  C1  >= 1 source has >= 5 successes AND >= 5 failures for terminal labels.
  C2  >= 1 source has >= 50 checkpoints with valid wall-clock features.
  C3  No feature has > 95% missingness across all sources combined.
  C4  No forbidden column appears in the feature schema.
  C5  Cross-source KS for at least the closure features is < 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

from coding_estimator.checkpoints.features.missingness import Missingness
from coding_estimator.checkpoints.features.registry import GROUPS, all_features
from coding_estimator.leakage.guard import find_forbidden, load_forbidden_spec

WALLCLOCK_FEATURE_COL = "elapsed_wall_time"
MIN_TERMINAL_PER_CLASS = 5
MIN_WALLCLOCK_CHECKPOINTS = 50
MAX_MISSINGNESS = 0.95
MAX_KS_THRESHOLD = 0.5


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateResult:
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> tuple[GateCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)


# ---- individual checks ----


def check_terminal_class_balance(labels_df: pd.DataFrame) -> GateCheck:
    """C1: >= 1 source has >= 5 successes AND >= 5 failures on
    `y_success_eventual`. We use that target specifically because it is
    THE headline binary outcome; the other binary v0 targets can have
    arbitrarily skewed base rates by design."""
    target = "y_success_eventual"
    sub = labels_df[labels_df["target_name"] == target]
    rows: list[str] = []
    qualifying: list[str] = []
    for source, grp in sub.groupby("source"):
        unmasked = grp[~grp["is_masked"].astype(bool)]
        # Run-constant target replicated across checkpoints; collapse
        # to per-run before counting class balance.
        per_run = unmasked.groupby("run_id")["label_value"].first()
        pos = int((per_run >= 0.5).sum())
        neg = int((per_run < 0.5).sum())
        rows.append(f"  - {source}: positives={pos}, negatives={neg}")
        if pos >= MIN_TERMINAL_PER_CLASS and neg >= MIN_TERMINAL_PER_CLASS:
            qualifying.append(str(source))
    detail = (
        f"target={target}; "
        f"qualifying sources (>= {MIN_TERMINAL_PER_CLASS} of each class): "
        f"{qualifying or 'NONE'}\n" + "\n".join(rows)
    )
    return GateCheck(
        name="C1_terminal_class_balance",
        passed=bool(qualifying),
        detail=detail,
    )


def check_wallclock_coverage(checkpoints_df: pd.DataFrame) -> GateCheck:
    """C2: >= 1 source has >= 50 checkpoints with valid wallclock
    features (`elapsed_wall_time` non-null)."""
    rows: list[str] = []
    qualifying: list[str] = []
    for source, grp in checkpoints_df.groupby("source"):
        if WALLCLOCK_FEATURE_COL not in grp.columns:
            n = 0
        else:
            n = int(grp[WALLCLOCK_FEATURE_COL].notna().sum())
        rows.append(f"  - {source}: wallclock_checkpoints={n}")
        if n >= MIN_WALLCLOCK_CHECKPOINTS:
            qualifying.append(str(source))
    detail = (
        f"qualifying sources (>= {MIN_WALLCLOCK_CHECKPOINTS} wallclock "
        f"checkpoints): {qualifying or 'NONE'}\n" + "\n".join(rows)
    )
    return GateCheck(
        name="C2_wallclock_coverage",
        passed=bool(qualifying),
        detail=detail,
    )


def check_global_feature_missingness(checkpoints_df: pd.DataFrame) -> GateCheck:
    """C3: No registered feature has > 95% missingness across all rows
    of the combined frame. Features whose registry `populated_on`
    explicitly excludes most sources (e.g. tb_live-only wallclock cols)
    are exempt: their missingness is structural, not a data quality
    flag. Enforce on features whose `populated_on` covers ALL sources
    in the frame."""
    sources_in_frame = set(checkpoints_df["source"].unique())
    # Features whose missingness is intrinsically tied to upstream
    # artifact availability (agent_scaffold, model_name, etc.) can be
    # legitimately >95% null without that being a data-quality issue;
    # the cell is "unknown" by contract, not "missing-when-it-should-
    # be-present". Exempt them from C3.
    exempt_semantics = {Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT}
    applicable: list = []
    bad: list[tuple[str, float]] = []
    for f in all_features():
        if not sources_in_frame.issubset(set(f.populated_on)):
            continue
        if f.missingness_semantic in exempt_semantics:
            continue
        applicable.append(f)
        if f.column_name not in checkpoints_df.columns:
            bad.append((f.column_name, 1.0))
            continue
        miss = float(checkpoints_df[f.column_name].isna().mean())
        if miss > MAX_MISSINGNESS:
            bad.append((f.column_name, miss))
    detail = (
        f"features over {MAX_MISSINGNESS:.0%} missing: "
        f"{len(bad)} of {len(applicable)} applicable features "
        f"(exempted: {Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT.value})\n"
        + "\n".join(f"  - {name}: {m:.2%} missing" for name, m in bad)
    )
    return GateCheck(
        name="C3_global_feature_missingness",
        passed=not bad,
        detail=detail,
    )


def check_no_forbidden_columns(checkpoints_df: pd.DataFrame) -> GateCheck:
    """C4: No forbidden column appears in the checkpoint frame.
    Mirrors `assert_no_forbidden` but produces a report-style result
    instead of raising."""
    spec = load_forbidden_spec()
    hits = find_forbidden(checkpoints_df.columns, spec)
    return GateCheck(
        name="C4_no_forbidden_columns",
        passed=not hits,
        detail=(
            "no forbidden columns" if not hits
            else f"FORBIDDEN HITS: {sorted(hits)}"
        ),
    )


def check_closure_cross_source_ks(
    checkpoints_df: pd.DataFrame, *, max_ks: float = MAX_KS_THRESHOLD
) -> GateCheck:
    """C5: For closure-group features, at least one feature must have
    pairwise KS < `max_ks` between every pair of sources. The
    interpretation: closure features (progress, completed-leaf counts)
    must be comparable enough across sources to support cross-source
    transfer in Workstream H."""
    closure_cols = [f.column_name for f in GROUPS["closure"]]
    sources = sorted(checkpoints_df["source"].unique())
    if len(sources) < 2:
        return GateCheck(
            name="C5_closure_cross_source_ks",
            passed=True,
            detail=f"only {len(sources)} source(s); KS not applicable",
        )
    qualifying: list[str] = []
    rows: list[str] = []
    for col in closure_cols:
        if col not in checkpoints_df.columns:
            continue
        worst = -1.0  # sentinel: no valid pair compared yet
        n_pairs = 0
        for i, sa in enumerate(sources):
            for sb in sources[i + 1 :]:
                a = checkpoints_df.loc[checkpoints_df["source"] == sa, col].dropna()
                b = checkpoints_df.loc[checkpoints_df["source"] == sb, col].dropna()
                if a.empty or b.empty:
                    continue
                ks = float(stats.ks_2samp(a, b).statistic)
                if ks > worst:
                    worst = ks
                n_pairs += 1
        if n_pairs == 0:
            rows.append(f"  - {col}: skipped (no valid source pairs)")
            continue
        rows.append(f"  - {col}: worst-pair KS = {worst:.3f} ({n_pairs} pairs)")
        if worst < max_ks:
            qualifying.append(col)
    detail = (
        f"closure features with worst-pair KS < {max_ks}: "
        f"{qualifying or 'NONE'}\n" + "\n".join(rows)
    )
    return GateCheck(
        name="C5_closure_cross_source_ks",
        passed=bool(qualifying),
        detail=detail,
    )


# ---- driver ----


def run_gate(
    *,
    labels_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
) -> GateResult:
    return GateResult(
        checks=(
            check_terminal_class_balance(labels_df),
            check_wallclock_coverage(checkpoints_df),
            check_global_feature_missingness(checkpoints_df),
            check_no_forbidden_columns(checkpoints_df),
            check_closure_cross_source_ks(checkpoints_df),
        )
    )


def render_gate_report(result: GateResult) -> str:
    parts: list[str] = ["# Profiling go/no-go gate (F11)\n"]
    verdict = "PASS" if result.passed else "FAIL"
    parts.append(f"## Verdict: **{verdict}**\n")
    if not result.passed:
        parts.append(
            "One or more gate checks failed. Halt before training; the "
            "remediation is in the per-check detail below.\n"
        )
    parts.append("## Checks\n")
    for c in result.checks:
        marker = "PASS" if c.passed else "FAIL"
        parts.append(f"### [{marker}] {c.name}\n")
        parts.append("```")
        parts.append(c.detail)
        parts.append("```")
        parts.append("")
    return "\n".join(parts) + "\n"


class GateFailedError(RuntimeError):
    """Raised by `write_gate_report` when any F11 check fails. The
    report is still written for forensics before the exception fires.
    Mirrors `assert_no_forbidden`'s hard-fail discipline."""


def write_gate_report(result: GateResult, out_dir: Path) -> Path:
    """Write the gate report and HARD-FAIL when any check fails.
    The report is always written before raising, so a CI run that
    halts here still leaves a forensic artifact behind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "F_profiling_go_no_go.md"
    target.write_text(render_gate_report(result), encoding="utf-8", newline="\n")
    if not result.passed:
        names = [c.name for c in result.failed]
        raise GateFailedError(
            f"F11 gate failed on {len(names)} check(s): {names}; "
            f"see {target} for remediation"
        )
    return target
