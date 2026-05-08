"""F4 — Leakage profile.

Per-feature flags rolled up across the checkpoint frame. Reads the
feature registry, the forbidden-column spec, and the actual checkpoint
parquet to derive:

  derived_only_from_prefix    registry.prefix_only
  available_at_checkpoint     % non-null per source
  contains_forbidden_token    forbidden-column match (exact/prefix/suffix)
  constant_or_near_constant   >= 0.99 same value across all rows
  high_cardinality_id         string col with > 0.5 * n_rows distinct values

This is a profile artifact, NOT a gate — F11 owns the go/no-go. Any new
forbidden-token hit will trip F11 separately via `assert_no_forbidden`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.features.registry import (
    GROUPS,
    Feature,
    all_features,
)
from coding_estimator.leakage.guard import (
    ForbiddenSpec,
    find_forbidden,
    load_forbidden_spec,
)

NEAR_CONSTANT_THRESHOLD = 0.99
HIGH_CARDINALITY_RATIO = 0.5


@dataclass(frozen=True)
class FeatureLeakageRow:
    column_name: str
    group: str
    derived_only_from_prefix: bool
    contains_forbidden_token: bool
    constant_or_near_constant: bool
    high_cardinality_id: bool
    overall_non_null_rate: float
    populated_on_sources: tuple[str, ...]


def _availability(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(df[col].notna().mean())


def _is_near_constant(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns or len(df) == 0:
        return False
    s = df[col].dropna()
    if s.empty:
        return False
    top_freq = s.value_counts(normalize=True).iloc[0]
    return float(top_freq) >= NEAR_CONSTANT_THRESHOLD


def _is_high_cardinality_id(df: pd.DataFrame, col: str, dtype: str) -> bool:
    if col not in df.columns or len(df) == 0:
        return False
    if dtype != "str":
        return False
    n_unique = df[col].nunique(dropna=True)
    return n_unique > HIGH_CARDINALITY_RATIO * len(df)


def feature_leakage_rows(
    df: pd.DataFrame,
    *,
    forbidden: ForbiddenSpec | None = None,
    feature_registry: Iterable[Feature] | None = None,
) -> list[FeatureLeakageRow]:
    spec = forbidden if forbidden is not None else load_forbidden_spec()
    feats = list(feature_registry) if feature_registry is not None else all_features()
    rows: list[FeatureLeakageRow] = []
    for f in feats:
        forbidden_hit = bool(find_forbidden([f.column_name], spec))
        rows.append(
            FeatureLeakageRow(
                column_name=f.column_name,
                group=f.group,
                derived_only_from_prefix=f.prefix_only,
                contains_forbidden_token=forbidden_hit,
                constant_or_near_constant=_is_near_constant(df, f.column_name),
                high_cardinality_id=_is_high_cardinality_id(df, f.column_name, f.dtype),
                overall_non_null_rate=_availability(df, f.column_name),
                populated_on_sources=f.populated_on,
            )
        )
    return rows


def _format_row(r: FeatureLeakageRow) -> str:
    flags = []
    if not r.derived_only_from_prefix:
        flags.append("NOT_PREFIX_ONLY")
    if r.contains_forbidden_token:
        flags.append("FORBIDDEN_TOKEN")
    if r.constant_or_near_constant:
        flags.append("NEAR_CONSTANT")
    if r.high_cardinality_id:
        flags.append("HIGH_CARDINALITY_ID")
    return (
        f"| {r.column_name} | {r.group} | "
        f"{r.overall_non_null_rate:.2f} | "
        f"{','.join(flags) if flags else '—'} |"
    )


def render_leakage_profile(rows: list[FeatureLeakageRow]) -> str:
    parts: list[str] = []
    parts.append("# Feature leakage profile (F4)\n")
    parts.append(
        "Auto-generated. Per-feature audit of leakage indicators. "
        "A feature is **clean** when no FLAG appears in its row.\n"
    )
    parts.append("## Summary\n")
    parts.append(f"- Total features: {len(rows)}")
    parts.append(
        f"- Features flagged FORBIDDEN_TOKEN: "
        f"{sum(1 for r in rows if r.contains_forbidden_token)}"
    )
    parts.append(
        f"- Features flagged NOT_PREFIX_ONLY: "
        f"{sum(1 for r in rows if not r.derived_only_from_prefix)}"
    )
    parts.append(
        f"- Features flagged NEAR_CONSTANT: "
        f"{sum(1 for r in rows if r.constant_or_near_constant)}"
    )
    parts.append(
        f"- Features flagged HIGH_CARDINALITY_ID: "
        f"{sum(1 for r in rows if r.high_cardinality_id)}\n"
    )

    by_group: dict[str, list[FeatureLeakageRow]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r)
    for group in GROUPS:
        gr = by_group.get(group, [])
        if not gr:
            continue
        parts.append(f"## Group: {group}\n")
        parts.append("| feature | group | non_null_rate | flags |")
        parts.append("| --- | --- | --- | --- |")
        for r in gr:
            parts.append(_format_row(r))
        parts.append("")
    return "\n".join(parts) + "\n"


def write_leakage_profile(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    forbidden: ForbiddenSpec | None = None,
) -> Path:
    rows = feature_leakage_rows(df, forbidden=forbidden)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "leakage_audit.md"
    target.write_text(render_leakage_profile(rows), encoding="utf-8", newline="\n")
    return target
