"""Markdown writers for the baseline ladder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _row(r: pd.Series) -> str:
    if not r["feasible"]:
        return (
            f"| {r['source_slice']} | {r['target']} | {r['model']} | "
            f"n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )
    ci = f"[{_fmt(r['brier_ci_low'])}, {_fmt(r['brier_ci_high'])}]"
    return (
        f"| {r['source_slice']} | {r['target']} | {r['model']} | "
        f"{_fmt(r['n_runs_train'])} | {_fmt(r['n_runs_test'])} | "
        f"{_fmt(r['n_checkpoints_test'])} | {_fmt(r['positive_rate_data'])} | "
        f"{_fmt(r['auroc'])} | {_fmt(r['brier'])} | {ci} | {_fmt(r['log_loss'])} |"
    )


def render_baseline_results_md(df: pd.DataFrame) -> str:
    lines = [
        "# Baseline ladder — metrics",
        "",
        "Per-cell results from `scripts/run_baselines.py`. Bootstrap CIs are",
        "computed by resampling **test runs** with replacement (B=1000, seed=0).",
        "Cells flagged not-feasible by the data-budget gate are emitted as",
        "`n/a (insufficient data)` and never silently zeroed.",
        "",
    ]
    for scheme, sub in df.groupby("scheme", sort=True):
        lines.append(f"## Scheme: `{scheme}`")
        lines.append("")
        lines.append(
            "| source_slice | target | model | n_train | n_test | n_ckpts | "
            "pos_rate | AUROC | Brier | Brier 95% CI | log_loss |"
        )
        lines.append(
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|"
        )
        sub2 = sub.sort_values(["source_slice", "target", "model"])
        for _, r in sub2.iterrows():
            lines.append(_row(r))
        lines.append("")
    return "\n".join(lines) + "\n"


def render_baseline_calibration_md(df: pd.DataFrame) -> str:
    lines = [
        "# Baseline ladder — calibration",
        "",
        "Expected Calibration Error (ECE, 10-bin equal-width) and predicted vs.",
        "observed positive rates per cell. ECE > 0.10 indicates the model is",
        "miscalibrated enough that downstream consumers should not trust raw",
        "probabilities without per-source recalibration (Workstream J).",
        "",
    ]
    for scheme, sub in df.groupby("scheme", sort=True):
        lines.append(f"## Scheme: `{scheme}`")
        lines.append("")
        lines.append(
            "| source_slice | target | model | pos_rate (data) | "
            "pos_rate (predicted) | ECE |"
        )
        lines.append("|---|---|---|---:|---:|---:|")
        sub2 = sub.sort_values(["source_slice", "target", "model"])
        for _, r in sub2.iterrows():
            if not r["feasible"]:
                lines.append(
                    f"| {r['source_slice']} | {r['target']} | {r['model']} | "
                    f"n/a (insufficient data) | n/a | n/a |"
                )
                continue
            lines.append(
                f"| {r['source_slice']} | {r['target']} | {r['model']} | "
                f"{_fmt(r['positive_rate_data'])} | "
                f"{_fmt(r['predicted_positive_rate'])} | "
                f"{_fmt(r['ece'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_baseline_results_md(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_baseline_results_md(df), encoding="utf-8", newline="\n")
    return path


def write_baseline_calibration_md(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_baseline_calibration_md(df), encoding="utf-8", newline="\n")
    return path
