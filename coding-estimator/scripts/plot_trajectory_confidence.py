#!/usr/bin/env python
"""Plot per-step prediction trajectories for held-out LORO runs with a
bootstrap band over training-data resampling.

For each LORO fold, the held-out run's checkpoints get a P(target)
trajectory. The band at step t comes from B bootstrap iterations:
resample the training run-ids with replacement, refit the baseline,
predict on the held-out run. Band = 2.5/97.5 percentiles.

What the plot shows:
- whether predictions sharpen (move toward 0/1) as t grows.
- whether the bootstrap-over-training band shrinks at late t.
- where the model is over- or under-confident relative to the true
  outcome shown as a horizontal dashed line.

Usage:
    uv run python scripts/plot_trajectory_confidence.py \\
        --checkpoints datasets/checkpoints_swe_agent_pilot.parquet \\
        --labels datasets/labels_swe_agent_pilot.parquet \\
        --target y_success_eventual \\
        --source swe_agent_pilot \\
        --out reports/trajectory_confidence_swe_agent_pilot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY, fit_binary
from coding_estimator.checkpoints.features.registry import all_features
from coding_estimator.eval.metrics import OUTPUT_CLIP


def _fill_canonical(df: pd.DataFrame, source_id: str) -> pd.DataFrame:
    """Apply per-feature canonical fills from the registry. Required
    because the parquet stores null where the missingness semantic is
    'applicable absent so far' (count==0, flag==False); sklearn refuses
    NaN, so the boundary is here."""
    out = df.copy()
    for f in all_features():
        if f.column_name not in out.columns:
            continue
        fill = f.canonical_fill_for(source_id)
        if fill is None:
            continue
        out[f.column_name] = out[f.column_name].fillna(fill)
    return out


def _join(ck: pd.DataFrame, lab: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = lab[(lab["target_name"] == target) & (~lab["is_masked"].astype(bool))]
    j = ck.merge(sub[["run_id", "checkpoint_id", "label_value"]],
                 on=["run_id", "checkpoint_id"], how="inner")
    j = j.rename(columns={"label_value": "_y"})
    j["_y"] = j["_y"].astype(int)
    return j


def _bootstrap_predictions(
    train_pool: pd.DataFrame,
    test_run: pd.DataFrame,
    spec,
    sources: tuple[str, ...],
    b: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, P[B, T]) for the test run. Each row of P is a
    refit-on-bootstrap-train-resample prediction trajectory.

    `train_pool` is the joined frame of training runs (already excluding
    the test run, so this also covers LOSO where train is a different
    source entirely)."""
    train_ids = sorted(train_pool["run_id"].unique())
    steps = test_run["checkpoint_step"].to_numpy()
    rng = np.random.default_rng(seed)
    P = np.empty((b, len(test_run)), dtype=float)
    lo, hi = OUTPUT_CLIP
    for i in range(b):
        boot_ids = rng.choice(train_ids, size=len(train_ids), replace=True)
        train = train_pool[train_pool["run_id"].isin(boot_ids)]
        if train.empty:
            P[i] = (lo + hi) / 2
            continue
        y = train["_y"].to_numpy()
        fitted = fit_binary(spec, train, y, sources)
        P[i] = np.clip(fitted.predict_proba(test_run), lo, hi)
    return steps, P


def _load(ck_path: Path, lab_path: Path, source: str, target: str) -> pd.DataFrame:
    ck = pd.read_parquet(ck_path)
    lab = pd.read_parquet(lab_path)
    ck = ck[ck["source"] == source]
    lab = lab[lab["source"] == source]
    ck = _fill_canonical(ck, source)
    return _join(ck, lab, target)


def plot(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    target: str,
    source: str,
    out_path: Path,
    b: int = 100,
    seed: int = 0,
    train_checkpoints_path: Path | None = None,
    train_labels_path: Path | None = None,
    train_source: str | None = None,
) -> Path:
    cross_source = train_source is not None and train_source != source
    j = _load(checkpoints_path, labels_path, source, target)
    if j.empty:
        raise SystemExit(f"no joined rows for {source}/{target}")
    if cross_source:
        train_j = _load(
            train_checkpoints_path or checkpoints_path,
            train_labels_path or labels_path,
            train_source,
            target,
        )
        if train_j.empty:
            raise SystemExit(f"no joined rows for {train_source}/{target}")
        sources_in_train = (train_source,)
    else:
        train_j = j
        sources_in_train = (source,)
    runs = sorted(j["run_id"].unique())
    n = len(runs)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows),
                              sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    specs = [(LEDGER_BASIC, "tab:blue", "G4 ledger-basic"),
             (TIME_ONLY, "tab:orange", "G2 time-only")]
    for k, run_id in enumerate(runs):
        ax = axes[k // cols, k % cols]
        run_rows = j[j["run_id"] == run_id].sort_values("checkpoint_step")
        y_steps = run_rows["checkpoint_step"].to_numpy()
        y_vals = run_rows["_y"].to_numpy()
        run_constant = bool(np.all(y_vals == y_vals[0]))
        train_pool = train_j if cross_source else j[j["run_id"] != run_id]
        for spec, color, label in specs:
            steps, P = _bootstrap_predictions(
                train_pool, run_rows, spec, sources_in_train, b, seed,
            )
            med = np.median(P, axis=0)
            lo_q = np.percentile(P, 2.5, axis=0)
            hi_q = np.percentile(P, 97.5, axis=0)
            ax.fill_between(steps, lo_q, hi_q, color=color, alpha=0.18)
            ax.plot(steps, med, color=color, linewidth=1.4, label=label)
        if run_constant:
            ax.axhline(y_vals[0], color="black", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            title = f"{run_id}\n(y={int(y_vals[0])})"
        else:
            ax.scatter(y_steps, y_vals, color="black", s=10,
                       marker="o", alpha=0.7, zorder=5)
            pos = int(y_vals.sum())
            title = f"{run_id}\n(pos {pos}/{len(y_vals)})"
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        if k % cols == 0:
            ax.set_ylabel(f"P({target})", fontsize=8)
        if k // cols == rows - 1:
            ax.set_xlabel("checkpoint step", fontsize=8)
    # blank unused panels
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    truth_handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.8)
    fig.legend(
        handles + [truth_handle],
        labels + ["true outcome"],
        loc="upper center", ncol=3, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    if cross_source:
        suptitle = (
            f"LOSO trajectory confidence — train: {train_source}, "
            f"test: {source} / {target} (B={b} train-bootstrap refits)"
        )
    else:
        suptitle = (
            f"LORO trajectory confidence — {source} / {target} "
            f"(B={b} train-bootstrap refits)"
        )
    fig.suptitle(suptitle, y=1.005, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True,
                   help="test-side checkpoints parquet")
    p.add_argument("--labels", type=Path, required=True,
                   help="test-side labels parquet")
    p.add_argument("--target", required=True)
    p.add_argument("--source", required=True, help="test source id")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-checkpoints", type=Path, default=None,
                   help="LOSO mode: separate train-side checkpoints parquet")
    p.add_argument("--train-labels", type=Path, default=None,
                   help="LOSO mode: separate train-side labels parquet")
    p.add_argument("--train-source", default=None,
                   help="LOSO mode: train source id (different from --source)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = plot(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        target=args.target,
        source=args.source,
        out_path=args.out,
        b=args.bootstrap,
        seed=args.seed,
        train_checkpoints_path=args.train_checkpoints,
        train_labels_path=args.train_labels,
        train_source=args.train_source,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
