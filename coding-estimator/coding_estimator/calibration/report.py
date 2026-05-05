"""J2/J3/J5 — markdown rendering for calibration outputs.

Two artifact families:
    `calibration_<model>_<source>.md`  — one reliability table per
        (model, source, target). Pure markdown so the v0 pipeline has
        no PNG dependency (deferred to a later workstream).
    `calibration_slices.md`            — one wide table covering every
        (model, source, target, slice_kind, slice_value).
    `calibration_v0.md`                — headline rollup with
        `not_safe_for_control` gate (ECE > 0.1).

All callers feed long-form predictions frames with columns
`(run_id, source, checkpoint_id, checkpoint_step, _y, _p)` produced by
`coding_estimator.eval.harness.predict_cell`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from coding_estimator.calibration.metrics import (
    expected_calibration_error,
    reliability_table,
)
from coding_estimator.calibration.recalibrate import (
    IsotonicRecalibrator,
    PlattRecalibrator,
)
from coding_estimator.eval.metrics import brier
from coding_estimator.eval.slices import assign_phase

ECE_GATE: float = 0.1
RECAL_KFOLD: int = 5
RECAL_SEED: int = 0


def kfold_recalibrated_predictions(
    predictions_df: pd.DataFrame,
    method: str = "isotonic",
    k: int = RECAL_KFOLD,
    seed: int = RECAL_SEED,
) -> np.ndarray:
    """Honest recalibration across run_ids: K-fold over runs, fit
    recalibrator on training runs' OOF predictions, transform held-out
    runs. Returns a recalibrated probability per input row, aligned to
    `predictions_df.index` order.

    Falls back to a single-fold fit-and-apply when n_runs < 2 or when
    every fit-fold collapses to a single class.
    """
    if predictions_df.empty:
        return np.array([], dtype=float)
    runs = predictions_df["run_id"].astype(str).to_numpy()
    p_arr = predictions_df["_p"].astype(float).to_numpy()
    y_arr = predictions_df["_y"].astype(int).to_numpy()
    unique_runs = np.array(sorted(set(runs.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_runs))
    ordered = unique_runs[perm]
    folds = np.array_split(ordered, min(k, len(ordered))) if len(ordered) else []
    if len(folds) < 2:
        cls = (
            PlattRecalibrator if method == "platt" else IsotonicRecalibrator
        )
        if len(np.unique(y_arr)) < 2:
            return p_arr.copy()
        return cls().fit(p_arr, y_arr).transform(p_arr)
    out = np.empty_like(p_arr)
    for fold_runs in folds:
        test_set = set(fold_runs.tolist())
        test_mask = np.array([r in test_set for r in runs])
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        y_train = y_arr[train_mask]
        p_train = p_arr[train_mask]
        if len(np.unique(y_train)) < 2:
            out[test_mask] = p_arr[test_mask]
            continue
        cls = (
            PlattRecalibrator if method == "platt" else IsotonicRecalibrator
        )
        cal = cls().fit(p_train, y_train)
        out[test_mask] = cal.transform(p_arr[test_mask])
    return out


def _fmt(v: float | int | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _reliability_table_md(rt: pd.DataFrame) -> list[str]:
    out = [
        "| bin | range | count | avg_predicted | avg_observed | gap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in rt.iterrows():
        rng = f"[{row['bin_lower']:.2f}, {row['bin_upper']:.2f})"
        out.append(
            "| {b} | {rng} | {n} | {ap} | {ay} | {g} |".format(
                b=int(row["bin_index"]),
                rng=rng,
                n=int(row["count"]),
                ap=_fmt(row["avg_predicted"]),
                ay=_fmt(row["avg_observed"]),
                g=_fmt(row["gap"]),
            )
        )
    return out


@dataclass(frozen=True)
class CalibrationStats:
    n: int
    positive_rate: float | None
    brier_raw: float | None
    ece_raw: float | None
    brier_platt: float | None
    ece_platt: float | None
    brier_isotonic: float | None
    ece_isotonic: float | None
    n_bins: int


def _stats_for(
    df: pd.DataFrame, n_bins: int = 10
) -> CalibrationStats:
    """Cross-validated calibration stats. `df` must have columns
    `(run_id, _y, _p)`. Recalibrated metrics use `kfold_recalibrated_predictions`
    so the test row's run is excluded from the fit; in-sample fits are
    used as a single-fold fallback when n_runs < 2."""
    if df.empty:
        return CalibrationStats(
            n=0, positive_rate=None,
            brier_raw=None, ece_raw=None,
            brier_platt=None, ece_platt=None,
            brier_isotonic=None, ece_isotonic=None,
            n_bins=n_bins,
        )
    y = df["_y"].astype(int).to_numpy()
    p = df["_p"].astype(float).to_numpy()
    pos = float(np.mean(y))
    if len(np.unique(y)) < 2:
        return CalibrationStats(
            n=len(y), positive_rate=pos,
            brier_raw=brier(y, p),
            ece_raw=expected_calibration_error(y, p, n_bins=n_bins),
            brier_platt=None, ece_platt=None,
            brier_isotonic=None, ece_isotonic=None,
            n_bins=n_bins,
        )
    p_platt = kfold_recalibrated_predictions(df, method="platt")
    p_iso = kfold_recalibrated_predictions(df, method="isotonic")
    return CalibrationStats(
        n=len(y), positive_rate=pos,
        brier_raw=brier(y, p),
        ece_raw=expected_calibration_error(y, p, n_bins=n_bins),
        brier_platt=brier(y, p_platt),
        ece_platt=expected_calibration_error(y, p_platt, n_bins=n_bins),
        brier_isotonic=brier(y, p_iso),
        ece_isotonic=expected_calibration_error(y, p_iso, n_bins=n_bins),
        n_bins=n_bins,
    )


def _progress_bucket(progress: pd.Series) -> pd.Series:
    """Bucket continuous progress into low/mid/high thirds at [0, 1/3, 2/3, 1]."""
    out = pd.Series(np.full(len(progress), "low", dtype=object), index=progress.index)
    out[(progress > 1 / 3) & (progress <= 2 / 3)] = "mid"
    out[progress > 2 / 3] = "high"
    return out


def _validation_state(df: pd.DataFrame) -> pd.Series:
    """Validation slice tag.

    `none`        — no validation events observed yet
    `started`     — at least one validation attempt, none successful yet
    `succeeded`   — at least one validation success
    """
    state = pd.Series(np.full(len(df), "none", dtype=object), index=df.index)
    started = df.get("num_validation_attempts", pd.Series(0, index=df.index)).fillna(0) > 0
    succeeded = df.get("num_validation_successes", pd.Series(0, index=df.index)).fillna(0) > 0
    state[started.to_numpy()] = "started"
    state[succeeded.to_numpy()] = "succeeded"
    return state


def render_reliability_report(
    *,
    title: str,
    model: str,
    source: str,
    by_target: dict[str, pd.DataFrame],
    n_bins: int = 10,
) -> str:
    """One markdown report covering every target predicted for a
    (model, source) pair. `by_target[target]` is a long-form predictions
    frame with columns `(_y, _p, ...)`."""
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        f"- model: `{model}`",
        f"- source: `{source}`",
        f"- bins: {n_bins} equal-width on [0, 1]",
        "",
    ]
    for target in sorted(by_target):
        df = by_target[target]
        lines.append(f"## Target: `{target}`")
        lines.append("")
        if df.empty:
            lines.append("_no predictions on this slice_")
            lines.append("")
            continue
        y = df["_y"].astype(int).to_numpy()
        p = df["_p"].astype(float).to_numpy()
        stats = _stats_for(df, n_bins=n_bins)
        lines.extend(
            [
                "| metric | raw | platt | isotonic |",
                "|---|---:|---:|---:|",
                "| Brier | {r} | {pl} | {iso} |".format(
                    r=_fmt(stats.brier_raw),
                    pl=_fmt(stats.brier_platt),
                    iso=_fmt(stats.brier_isotonic),
                ),
                "| ECE | {r} | {pl} | {iso} |".format(
                    r=_fmt(stats.ece_raw),
                    pl=_fmt(stats.ece_platt),
                    iso=_fmt(stats.ece_isotonic),
                ),
                "| n | {n} | | |".format(n=stats.n),
                "| positive rate | {pr} | | |".format(pr=_fmt(stats.positive_rate)),
                "",
                "### Reliability table (raw)",
                "",
            ]
        )
        rt = reliability_table(y, p, n_bins=n_bins)
        lines.extend(_reliability_table_md(rt))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_reliability_report(
    path: Path,
    *,
    title: str,
    model: str,
    source: str,
    by_target: dict[str, pd.DataFrame],
    n_bins: int = 10,
) -> Path:
    md = render_reliability_report(
        title=title, model=model, source=source, by_target=by_target, n_bins=n_bins
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


@dataclass(frozen=True)
class SliceCalibrationRow:
    model: str
    source: str
    target: str
    slice_kind: str
    slice_value: str
    n: int
    positives: int
    negatives: int
    brier_raw: float | None
    ece_raw: float | None
    brier_isotonic: float | None
    ece_isotonic: float | None


MIN_SLICE_N: int = 5


def slice_calibration_rows(
    *,
    model: str,
    source: str,
    target: str,
    predictions_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame | None = None,
    shapes_df: pd.DataFrame | None = None,
    target_horizon: str | None = None,
    n_bins: int = 10,
) -> list[SliceCalibrationRow]:
    """Compute calibration metrics across the J3 slice axes:
    source (caller-supplied), target_horizon, phase, shape class,
    progress bucket, validation state.

    `checkpoints_df` is needed for progress and validation features; when
    omitted, only phase / shape / target_horizon slices are produced.
    """
    if predictions_df.empty:
        return []
    rows: list[SliceCalibrationRow] = []
    df = predictions_df.copy()
    df["_phase"] = assign_phase(df)

    join_cols = ["run_id", "checkpoint_id"]
    if checkpoints_df is not None:
        keep = [
            c
            for c in [
                "coding_progress",
                "investigation_progress",
                "validation_progress",
                "num_validation_attempts",
                "num_validation_successes",
            ]
            if c in checkpoints_df.columns
        ]
        df = df.merge(
            checkpoints_df[join_cols + keep], on=join_cols, how="left"
        )

    def _emit(kind: str, value: str, sub: pd.DataFrame) -> None:
        n = len(sub)
        pos = int(sub["_y"].sum()) if n else 0
        neg = n - pos
        if n < MIN_SLICE_N or len(np.unique(sub["_y"])) < 2:
            rows.append(
                SliceCalibrationRow(
                    model=model, source=source, target=target,
                    slice_kind=kind, slice_value=value,
                    n=n, positives=pos, negatives=neg,
                    brier_raw=None, ece_raw=None,
                    brier_isotonic=None, ece_isotonic=None,
                )
            )
            return
        stats = _stats_for(sub, n_bins=n_bins)
        rows.append(
            SliceCalibrationRow(
                model=model, source=source, target=target,
                slice_kind=kind, slice_value=value,
                n=n, positives=pos, negatives=neg,
                brier_raw=stats.brier_raw, ece_raw=stats.ece_raw,
                brier_isotonic=stats.brier_isotonic,
                ece_isotonic=stats.ece_isotonic,
            )
        )

    # source axis: one slice (caller selects single source)
    _emit("source", source, df)

    # target_horizon: one slice (carries the metadata; callers vary)
    if target_horizon is not None:
        _emit("target_horizon", target_horizon, df)

    for phase in ("early", "middle", "late"):
        _emit("phase", phase, df[df["_phase"] == phase])

    if "coding_progress" in df.columns:
        bucket = _progress_bucket(df["coding_progress"].fillna(0.0))
        for tag in ("low", "mid", "high"):
            _emit("progress", tag, df[bucket == tag])

    if "num_validation_attempts" in df.columns:
        vs = _validation_state(df)
        for tag in ("none", "started", "succeeded"):
            _emit("validation", tag, df[vs == tag])

    if shapes_df is not None and not shapes_df.empty:
        shape_cols = [c for c in shapes_df.columns if c.startswith("shape_")]
        sj = df.merge(shapes_df[["run_id", *shape_cols]], on="run_id", how="left")
        for col in shape_cols:
            tag = col[len("shape_"):]
            sub = sj[sj[col].astype("boolean").fillna(False).astype(bool)]
            _emit("shape", tag, sub)

    return rows


def render_slice_table(rows: list[SliceCalibrationRow]) -> str:
    if not rows:
        return "_no slice rows_\n"
    lines = [
        "| model | source | target | slice_kind | slice_value | n | pos | neg | "
        "Brier (raw) | ECE (raw) | Brier (iso) | ECE (iso) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(
        rows,
        key=lambda r: (r.model, r.source, r.target, r.slice_kind, r.slice_value),
    ):
        lines.append(
            "| {m} | {s} | {t} | {k} | {v} | {n} | {p} | {ng} | {br} | {er} | {bi} | {ei} |".format(
                m=r.model, s=r.source, t=r.target, k=r.slice_kind, v=r.slice_value,
                n=r.n, p=r.positives, ng=r.negatives,
                br=_fmt(r.brier_raw), er=_fmt(r.ece_raw),
                bi=_fmt(r.brier_isotonic), ei=_fmt(r.ece_isotonic),
            )
        )
    return "\n".join(lines) + "\n"


def render_slice_report(
    *, title: str, rows: list[SliceCalibrationRow], summary: str | None = None
) -> str:
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    lines.append(
        f"Slices with fewer than {MIN_SLICE_N} checkpoints, or single-class y, emit `n/a`."
    )
    lines.append("")
    lines.append(render_slice_table(rows))
    return "\n".join(lines) + "\n"


def write_slice_report(
    path: Path, *, title: str, rows: list[SliceCalibrationRow], summary: str | None = None
) -> Path:
    md = render_slice_report(title=title, rows=rows, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


@dataclass(frozen=True)
class HeadlineRow:
    model: str
    source: str
    target: str
    n: int
    brier_raw: float | None
    ece_raw: float | None
    brier_isotonic: float | None
    ece_isotonic: float | None
    not_safe_for_control: bool


def headline_rows(
    items: list[tuple[str, str, str, pd.DataFrame]],
    *,
    n_bins: int = 10,
    ece_gate: float = ECE_GATE,
) -> list[HeadlineRow]:
    """`items` is a list of (model, source, target, predictions_df)."""
    out: list[HeadlineRow] = []
    for model, source, target, df in items:
        if df.empty:
            out.append(
                HeadlineRow(
                    model=model, source=source, target=target,
                    n=0, brier_raw=None, ece_raw=None,
                    brier_isotonic=None, ece_isotonic=None,
                    not_safe_for_control=True,
                )
            )
            continue
        stats = _stats_for(df, n_bins=n_bins)
        ece_after = (
            stats.ece_isotonic if stats.ece_isotonic is not None else stats.ece_raw
        )
        unsafe = ece_after is None or ece_after > ece_gate
        out.append(
            HeadlineRow(
                model=model, source=source, target=target,
                n=stats.n, brier_raw=stats.brier_raw, ece_raw=stats.ece_raw,
                brier_isotonic=stats.brier_isotonic, ece_isotonic=stats.ece_isotonic,
                not_safe_for_control=unsafe,
            )
        )
    return out


def render_headline_report(
    *,
    title: str,
    rows: list[HeadlineRow],
    ece_gate: float = ECE_GATE,
    summary: str | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    lines.append(
        f"Gate: any (model, source, target) with `ECE > {ece_gate}` after isotonic "
        "recalibration is **not_safe_for_control** and must carry that annotation in "
        "its model card."
    )
    lines.append("")
    lines.extend(
        [
            "| model | source | target | n | Brier (raw) | ECE (raw) | "
            "Brier (iso) | ECE (iso) | not_safe_for_control |",
            "|---|---|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for r in sorted(rows, key=lambda r: (r.model, r.source, r.target)):
        lines.append(
            "| {m} | {s} | {t} | {n} | {br} | {er} | {bi} | {ei} | {flag} |".format(
                m=r.model, s=r.source, t=r.target, n=r.n,
                br=_fmt(r.brier_raw), er=_fmt(r.ece_raw),
                bi=_fmt(r.brier_isotonic), ei=_fmt(r.ece_isotonic),
                flag="**yes**" if r.not_safe_for_control else "no",
            )
        )
    lines.append("")
    flagged = [r for r in rows if r.not_safe_for_control]
    if flagged:
        lines.append("## Cells flagged not_safe_for_control")
        lines.append("")
        for r in flagged:
            ece_after = r.ece_isotonic if r.ece_isotonic is not None else r.ece_raw
            lines.append(
                f"- `{r.model}` / `{r.source}` / `{r.target}` — ECE_after={_fmt(ece_after)}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_headline_report(
    path: Path,
    *,
    title: str,
    rows: list[HeadlineRow],
    ece_gate: float = ECE_GATE,
    summary: str | None = None,
) -> Path:
    md = render_headline_report(
        title=title, rows=rows, ece_gate=ece_gate, summary=summary
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "ECE_GATE",
    "MIN_SLICE_N",
    "SliceCalibrationRow",
    "HeadlineRow",
    "CalibrationStats",
    "render_reliability_report",
    "write_reliability_report",
    "slice_calibration_rows",
    "render_slice_report",
    "write_slice_report",
    "headline_rows",
    "render_headline_report",
    "write_headline_report",
]
