"""F2 — Checkpoint-distribution profile.

For each canonical source, summarise how the prefix-only feature space
is populated across its checkpoints. Goal: make catastrophic regimes
(95% of checkpoints sit at progress=0; only one source has wallclock;
validation never starts) impossible to miss before training.

The profile reads a built checkpoint frame; it does NOT recompute
features. Buckets and column names are pinned so the report is
diff-friendly run-over-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROGRESS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("[0.0, 0.25)", 0.0, 0.25),
    ("[0.25, 0.5)", 0.25, 0.5),
    ("[0.5, 0.75)", 0.5, 0.75),
    ("[0.75, 1.0]", 0.75, 1.000001),
)
ELAPSED_BUCKETS = PROGRESS_BUCKETS  # half-open same edges


@dataclass(frozen=True)
class CheckpointDistribution:
    source: str
    n_checkpoints: int
    progress_bucket_counts: dict[str, int]
    elapsed_fraction_bucket_counts: dict[str, int]
    leaf_count_p25: float | None
    leaf_count_p50: float | None
    leaf_count_p75: float | None
    validation_started_rate: float | None
    validation_complete_rate: float | None
    blocked_rate: float | None


def _bucket_counts(s: pd.Series, buckets) -> dict[str, int]:
    out = {label: 0 for label, _, _ in buckets}
    for v in s.dropna():
        for label, lo, hi in buckets:
            if lo <= float(v) < hi:
                out[label] += 1
                break
    return out


def _quantile(s: pd.Series, q: float) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.quantile(q))


def _rate(s: pd.Series) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.astype(bool).mean())


def distributions(checkpoints_df: pd.DataFrame) -> list[CheckpointDistribution]:
    out: list[CheckpointDistribution] = []
    for source, grp in checkpoints_df.groupby("source"):
        active_leaf_count = grp.get("active_leaf_count", pd.Series(dtype=float))
        validation_started = grp.get("validation_started", pd.Series(dtype=bool))
        validation_complete = grp.get("validation_complete", pd.Series(dtype=bool))
        blocked_leaf_count = grp.get("blocked_leaf_count", pd.Series(dtype=float))
        out.append(
            CheckpointDistribution(
                source=str(source),
                n_checkpoints=int(len(grp)),
                progress_bucket_counts=_bucket_counts(
                    grp.get("coding_progress", pd.Series(dtype=float)), PROGRESS_BUCKETS
                ),
                elapsed_fraction_bucket_counts=_bucket_counts(
                    grp.get("checkpoint_fraction_timeout", pd.Series(dtype=float)),
                    ELAPSED_BUCKETS,
                ),
                leaf_count_p25=_quantile(active_leaf_count, 0.25),
                leaf_count_p50=_quantile(active_leaf_count, 0.5),
                leaf_count_p75=_quantile(active_leaf_count, 0.75),
                validation_started_rate=_rate(validation_started),
                validation_complete_rate=_rate(validation_complete),
                blocked_rate=_rate(blocked_leaf_count.fillna(0).gt(0)),
            )
        )
    return sorted(out, key=lambda d: d.source)


def _format_bucket_table(
    title: str,
    dists: list[CheckpointDistribution],
    attr: str,
    buckets,
) -> list[str]:
    parts: list[str] = []
    parts.append(f"### {title}\n")
    header = "| source | n | " + " | ".join(label for label, _, _ in buckets) + " |"
    sep = "| --- " * (2 + len(buckets)) + "|"
    parts.append(header)
    parts.append(sep)
    for d in dists:
        bc = getattr(d, attr)
        parts.append(
            f"| {d.source} | {d.n_checkpoints} | "
            + " | ".join(str(bc.get(label, 0)) for label, _, _ in buckets)
            + " |"
        )
    parts.append("")
    return parts


def render_distribution(dists: list[CheckpointDistribution]) -> str:
    parts: list[str] = []
    parts.append("# Checkpoint-distribution profile (F2)\n")
    parts.append(
        "Per canonical source: how checkpoints are distributed across "
        "progress / elapsed-fraction buckets, leaf-count quantiles, and "
        "validation/blocked state rates. A heavily skewed distribution "
        "(e.g. >80% of checkpoints at progress=0) flags an evaluation "
        "regime where most baselines look good for the wrong reason.\n"
    )
    parts.extend(_format_bucket_table(
        "Coding progress buckets", dists, "progress_bucket_counts", PROGRESS_BUCKETS
    ))
    parts.extend(_format_bucket_table(
        "Elapsed-fraction buckets (tb_live only)",
        dists,
        "elapsed_fraction_bucket_counts",
        ELAPSED_BUCKETS,
    ))
    parts.append("### Leaf-count quantiles\n")
    parts.append("| source | p25 | p50 | p75 |")
    parts.append("| --- | --- | --- | --- |")
    for d in dists:
        parts.append(
            f"| {d.source} | {_fmt(d.leaf_count_p25)} | "
            f"{_fmt(d.leaf_count_p50)} | {_fmt(d.leaf_count_p75)} |"
        )
    parts.append("")
    parts.append("### Validation + blocked rates\n")
    parts.append("| source | validation_started | validation_complete | any_blocked |")
    parts.append("| --- | --- | --- | --- |")
    for d in dists:
        parts.append(
            f"| {d.source} | {_fmt(d.validation_started_rate)} | "
            f"{_fmt(d.validation_complete_rate)} | {_fmt(d.blocked_rate)} |"
        )
    parts.append("")
    return "\n".join(parts) + "\n"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def write_distribution(checkpoints_df: pd.DataFrame, out_dir: Path) -> Path:
    dists = distributions(checkpoints_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "checkpoints_distribution.md"
    target.write_text(render_distribution(dists), encoding="utf-8", newline="\n")
    return target
