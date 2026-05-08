"""Label-balance audit (Workstream E8).

Emit `datasets/profiles/labels_<source>_balance.md`. Rollups:
  * overall positives / negatives / masked counts and positive rate
  * by source
  * by progress bucket   (requires `checkpoints_df` with coding_progress)
  * by elapsed-fraction  (requires `checkpoints_df` with elapsed-fraction)
  * by shape class       (requires `shapes_df`)

Flag any (target, source) cell where positives < 5 or negatives < 5;
those targets cannot be trained on that source alone.

The function takes already-built dataframes; it does not re-load runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

THIN_THRESHOLD = 5

PROGRESS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("[0.0, 0.25)", 0.0, 0.25),
    ("[0.25, 0.5)", 0.25, 0.5),
    ("[0.5, 0.75)", 0.5, 0.75),
    ("[0.75, 1.0]", 0.75, 1.000001),
)
ELAPSED_BUCKETS = PROGRESS_BUCKETS  # same edges, different signal


@dataclass(frozen=True)
class CellCounts:
    positives: int
    negatives: int
    masked: int

    @property
    def n(self) -> int:
        return self.positives + self.negatives + self.masked

    @property
    def positive_rate(self) -> float | None:
        denom = self.positives + self.negatives
        return None if denom == 0 else self.positives / denom

    @property
    def thin(self) -> bool:
        return self.positives < THIN_THRESHOLD or self.negatives < THIN_THRESHOLD


def _count(df: pd.DataFrame) -> CellCounts:
    masked = int(df["is_masked"].sum())
    unmasked = df[~df["is_masked"].astype(bool)]
    pos = int((unmasked["label_value"] >= 1.0 - 1e-9).sum())
    neg = int((unmasked["label_value"] <= 0.0 + 1e-9).sum())
    return CellCounts(positives=pos, negatives=neg, masked=masked)


def _bucketize(value: float | None, buckets) -> str | None:
    if value is None or pd.isna(value):
        return None
    for label, lo, hi in buckets:
        if lo <= value < hi:
            return label
    return None


def _binary_targets(labels_df: pd.DataFrame) -> list[str]:
    """Targets whose unmasked `label_value` lies entirely in {0.0, 1.0}.
    Derived from data, not a hard-coded name list, so a future regression
    target won't be silently miscounted as binary."""
    out: list[str] = []
    unmasked = labels_df[~labels_df["is_masked"].astype(bool)]
    for target, grp in unmasked.groupby("target_name"):
        vals = grp["label_value"].dropna().unique()
        if len(vals) == 0:
            continue
        if set(float(v) for v in vals).issubset({0.0, 1.0}):
            out.append(str(target))
    return sorted(out)


def _rollup(labels_df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (target, key), grp in labels_df.groupby(["target_name", by], dropna=False):
        c = _count(grp)
        rows.append(
            {
                "target_name": target,
                by: key,
                "positives": c.positives,
                "negatives": c.negatives,
                "masked": c.masked,
                "n_unmasked": c.positives + c.negatives,
                "positive_rate": c.positive_rate,
                "thin": c.thin,
            }
        )
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame) -> str:
    """Hand-rolled markdown pipe table — `pd.DataFrame.to_markdown`
    requires the `tabulate` package which we do not depend on."""
    if df.empty:
        return "_(no rows)_\n"
    out = df.copy()
    if "positive_rate" in out.columns:
        out["positive_rate"] = out["positive_rate"].apply(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
    cols = [str(c) for c in out.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in out.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body_rows]) + "\n"


def build_balance_report(
    source_id: str,
    labels_df: pd.DataFrame,
    *,
    shapes_df: pd.DataFrame | None = None,
    checkpoints_df: pd.DataFrame | None = None,
) -> str:
    targets = _binary_targets(labels_df)
    src_df = labels_df[labels_df["source"] == source_id]
    src_df = src_df[src_df["target_name"].isin(targets)]

    parts: list[str] = []
    parts.append(f"# Label balance: {source_id}\n")
    parts.append(
        f"_n_runs={src_df['run_id'].nunique()}, "
        f"n_checkpoints={src_df['checkpoint_id'].nunique()}_\n"
    )

    parts.append("## Per target (binary targets only)\n")
    overall = _rollup(src_df.assign(_all="all"), "_all").drop(columns=["_all"])
    parts.append(_format_table(overall))

    thin = overall[overall["thin"]]
    if not thin.empty:
        parts.append("### Thin cells (positives<5 or negatives<5)\n")
        parts.append(_format_table(thin[["target_name", "positives", "negatives"]]))

    if shapes_df is not None and not shapes_df.empty:
        shape_cols = [c for c in shapes_df.columns if c.startswith("shape_")]
        run_to_shapes = shapes_df.set_index("run_id")[shape_cols]
        joined = src_df.merge(
            run_to_shapes, left_on="run_id", right_index=True, how="left"
        )
        parts.append("## By shape class\n")
        for col in shape_cols:
            tag = col.removeprefix("shape_")
            sub = joined[joined[col].fillna(False).astype(bool)]
            if sub.empty:
                continue
            tbl = _rollup(sub.assign(_shape=tag), "_shape").drop(columns=["_shape"])
            parts.append(f"### {tag}\n")
            parts.append(_format_table(tbl))

    if checkpoints_df is not None and not checkpoints_df.empty:
        ck = checkpoints_df[checkpoints_df["source"] == source_id]
        bucket_cols: list[tuple[str, str, tuple]] = []
        if "coding_progress" in ck.columns:
            ck = ck.assign(
                progress_bucket=ck["coding_progress"].apply(
                    lambda v: _bucketize(v, PROGRESS_BUCKETS)
                )
            )
            bucket_cols.append(("progress_bucket", "By progress bucket", PROGRESS_BUCKETS))
        if "checkpoint_fraction_timeout" in ck.columns:
            ck = ck.assign(
                elapsed_bucket=ck["checkpoint_fraction_timeout"].apply(
                    lambda v: _bucketize(v, ELAPSED_BUCKETS)
                )
            )
            bucket_cols.append(("elapsed_bucket", "By elapsed-fraction bucket", ELAPSED_BUCKETS))
        if bucket_cols:
            keep = ["checkpoint_id"] + [c for c, _, _ in bucket_cols]
            joined = src_df.merge(ck[keep], on="checkpoint_id", how="left")
            for col, header, _buckets in bucket_cols:
                parts.append(f"## {header}\n")
                parts.append(_format_table(_rollup(joined, col)))

    return "\n".join(parts)


def write_balance_report(
    source_id: str,
    labels_df: pd.DataFrame,
    out_dir: Path,
    *,
    shapes_df: pd.DataFrame | None = None,
    checkpoints_df: pd.DataFrame | None = None,
) -> Path:
    text = build_balance_report(
        source_id,
        labels_df,
        shapes_df=shapes_df,
        checkpoints_df=checkpoints_df,
    )
    out_path = out_dir / f"labels_{source_id}_balance.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
