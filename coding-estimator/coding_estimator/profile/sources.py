"""F1 — Source-level profile.

For each source: run count, label availability, success/failure split,
wallclock coverage, and run-length quantiles. The output is a markdown
artifact consumed by F11 (gate) and the model card (Workstream N).

The profile reads pre-built artifacts; it does NOT re-load runs:
  - `combined_manifest_df`  (from C5 — one row per run with final label)
  - `checkpoints_df`        (optional; for run-length / wallclock cover)

If `checkpoints_df` is None we fall back to `ledger_event_count` from
the manifest as a proxy for run length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

WALLCLOCK_FEATURE_COL = "elapsed_wall_time"


@dataclass(frozen=True)
class SourceProfileRow:
    source: str
    n_runs: int
    n_successful: int
    n_failed: int
    n_label_unresolvable: int
    has_real_wallclock_runs: int
    p25_run_length_steps: float | None
    p50_run_length_steps: float | None
    p75_run_length_steps: float | None
    n_checkpoints: int | None
    n_wallclock_checkpoints: int | None


def _quantile(s: pd.Series, q: float) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.quantile(q))


def _length_from_checkpoints(ck: pd.DataFrame) -> pd.Series:
    """Run length = max checkpoint_step per run."""
    return ck.groupby("run_id")["checkpoint_step"].max()


def source_profile_rows(
    manifest_df: pd.DataFrame,
    *,
    checkpoints_df: pd.DataFrame | None = None,
) -> list[SourceProfileRow]:
    rows: list[SourceProfileRow] = []
    for source, grp in manifest_df.groupby("source"):
        if checkpoints_df is not None and not checkpoints_df.empty:
            ck = checkpoints_df[checkpoints_df["source"] == source]
            lengths = _length_from_checkpoints(ck)
            n_ck = int(len(ck))
            wc = (
                int(ck[WALLCLOCK_FEATURE_COL].notna().sum())
                if WALLCLOCK_FEATURE_COL in ck.columns
                else 0
            )
        else:
            lengths = grp["ledger_event_count"].astype("float")
            n_ck = None
            wc = None
        fs = grp["final_success"]
        # A run is "unresolvable" iff its final_success is null OR its
        # final_success_source is the literal "missing" enum value
        # (set by the C5 manifest builder when load_final_label
        # raises). Set-OR over the two predicates dedup-counts rows
        # that hit BOTH, so a future inconsistency (fs=True paired
        # with source="missing") still surfaces as unresolvable rather
        # than being silently absorbed into n_succ.
        unres_mask = fs.isna() | (grp["final_success_source"] == "missing")
        # Use explicit equality on the underlying object dtype to avoid
        # pandas' object->bool downcasting deprecation; semantics are
        # identical to fs.fillna(...).astype(bool).
        is_true = (fs == True) & ~unres_mask  # noqa: E712
        is_false = (fs == False) & ~unres_mask  # noqa: E712
        n_succ = int(is_true.sum())
        n_fail = int(is_false.sum())
        n_unres = int(unres_mask.sum())
        n_runs = int(len(grp))
        # Sanity invariant: counts must partition the source.
        assert n_succ + n_fail + n_unres == n_runs, (
            source, n_succ, n_fail, n_unres, n_runs
        )
        rows.append(
            SourceProfileRow(
                source=str(source),
                n_runs=n_runs,
                n_successful=n_succ,
                n_failed=n_fail,
                n_label_unresolvable=n_unres,
                has_real_wallclock_runs=int(grp["has_real_wallclock"].sum()),
                p25_run_length_steps=_quantile(lengths, 0.25),
                p50_run_length_steps=_quantile(lengths, 0.5),
                p75_run_length_steps=_quantile(lengths, 0.75),
                n_checkpoints=n_ck,
                n_wallclock_checkpoints=wc,
            )
        )
    return sorted(rows, key=lambda r: r.source)


def render_source_profile(rows: list[SourceProfileRow]) -> str:
    parts: list[str] = []
    parts.append("# Source-level profile (F1)\n")
    parts.append(
        "Per canonical source: run count, label availability, "
        "success/failure split, wallclock coverage, run-length quantiles. "
        "Generated from the combined manifest (C5) + optional checkpoint frame.\n"
    )
    parts.append(
        "| source | n_runs | n_succ | n_fail | n_unres | n_real_wc | "
        "p25_len | p50_len | p75_len | n_ckpts | n_wc_ckpts |"
    )
    parts.append("| --- " * 11 + "|")
    for r in rows:
        parts.append(
            f"| {r.source} | {r.n_runs} | {r.n_successful} | {r.n_failed} | "
            f"{r.n_label_unresolvable} | {r.has_real_wallclock_runs} | "
            f"{_fmt(r.p25_run_length_steps)} | {_fmt(r.p50_run_length_steps)} | "
            f"{_fmt(r.p75_run_length_steps)} | "
            f"{r.n_checkpoints if r.n_checkpoints is not None else '—'} | "
            f"{r.n_wallclock_checkpoints if r.n_wallclock_checkpoints is not None else '—'} |"
        )
    parts.append("")
    return "\n".join(parts) + "\n"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f}"


def write_source_profile(
    manifest_df: pd.DataFrame,
    out_dir: Path,
    *,
    checkpoints_df: pd.DataFrame | None = None,
) -> Path:
    rows = source_profile_rows(manifest_df, checkpoints_df=checkpoints_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "sources.md"
    target.write_text(render_source_profile(rows), encoding="utf-8", newline="\n")
    return target
