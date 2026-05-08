"""F3 — Label-balance profile (combined).

Wraps `labels.balance.build_balance_report` over every source present in
the long-form labels frame and emits a single aggregated artifact at
`datasets/profiles/labels_balance.md`. Per-source detail sections still
live under `datasets/profiles/labels_<source>_balance.md` (E8); F3 is
the single rollup F11 reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coding_estimator.labels.balance import (
    _binary_targets,
    _count,
    build_balance_report,
)


def _per_source_summary(labels_df: pd.DataFrame) -> str:
    binary = _binary_targets(labels_df)
    parts: list[str] = []
    parts.append("## Cross-source headline (binary targets only)\n")
    parts.append(
        "| target | source | positives | negatives | masked | n_unmasked | "
        "positive_rate | thin |"
    )
    parts.append("| --- " * 8 + "|")
    for target in binary:
        for source, grp in labels_df[labels_df["target_name"] == target].groupby("source"):
            c = _count(grp)
            rate = "" if c.positive_rate is None else f"{c.positive_rate:.3f}"
            parts.append(
                f"| {target} | {source} | {c.positives} | {c.negatives} | "
                f"{c.masked} | {c.positives + c.negatives} | {rate} | "
                f"{'YES' if c.thin else 'no'} |"
            )
    parts.append("")
    return "\n".join(parts)


def render_combined(labels_df: pd.DataFrame) -> str:
    parts: list[str] = ["# Label-balance profile (F3)\n"]
    parts.append(
        "Cross-source rollup of E8 per-source balance reports. Used by "
        "F11 to enforce the `>= 5 positives AND >= 5 negatives on >= 1 "
        "source for terminal labels` gate.\n"
    )
    parts.append(_per_source_summary(labels_df))
    parts.append("---\n")
    for source in sorted(labels_df["source"].unique()):
        parts.append(f"## Source: {source}\n")
        parts.append(build_balance_report(source, labels_df))
        parts.append("---\n")
    return "\n".join(parts) + "\n"


def write_combined(labels_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "labels_balance.md"
    target.write_text(render_combined(labels_df), encoding="utf-8", newline="\n")
    return target
