#!/usr/bin/env python
"""Human-baseline driver — prepare and compare.

Two subcommands:

    prepare:  emit reports/human_baseline/{prompts/, samples.json}.
              The human reads each prompt, writes their probabilities
              into reports/human_baseline/human_predictions.csv.

    compare:  read human_predictions.csv, compute G2 / G4 predictions
              at the same checkpoints, render the comparison report.

Usage:
    uv run python scripts/run_human_baseline.py prepare \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --out-dir reports/human_baseline \\
        --n-samples 6

    # ... human fills in reports/human_baseline/human_predictions.csv ...

    uv run python scripts/run_human_baseline.py compare \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports/human_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.human_baseline import (
    HumanBaselineSample,
    compare_to_models,
    render_comparison_report,
    select_samples,
    write_prompts,
    write_sample_manifest,
)


def cmd_prepare(args: argparse.Namespace) -> int:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(args.checkpoints))
    samples = select_samples(
        checkpoints_df=checkpoints_df,
        n_samples=args.n_samples,
        source=args.source,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prompts(samples, out_dir / "prompts")
    write_sample_manifest(samples, out_dir / "samples.json")
    csv_path = out_dir / "human_predictions.csv"
    if not csv_path.exists():
        # Seed an empty CSV so the human has somewhere to fill in.
        rows = ["run_id,target,p_success"]
        for s in samples:
            for tgt in ("y_success_eventual", "y_future_progress_drop_h5"):
                rows.append(f"{s.run_id},{tgt},")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"prompts: {out_dir / 'prompts'}")
    print(f"samples: {out_dir / 'samples.json'}")
    print(f"human:   {csv_path} (fill this in)")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(args.checkpoints))
    labels_df = pd.read_parquet(args.labels)
    samples_path = args.out_dir / "samples.json"
    if not samples_path.exists():
        print(f"missing {samples_path} — run `prepare` first", file=sys.stderr)
        return 1
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = [
        HumanBaselineSample(
            run_id=s["run_id"], source=s["source"],
            midpoint_step=int(s["midpoint_step"]),
            n_events_visible=int(s["n_events_visible"]),
            task_md_path=s.get("task_md_path"),
        )
        for s in payload["samples"]
    ]
    comparison = compare_to_models(
        human_predictions_csv=args.out_dir / "human_predictions.csv",
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        samples=samples,
    )
    md_path = args.out_dir / "comparison.md"
    md_path.write_text(render_comparison_report(comparison), encoding="utf-8")
    if not comparison.empty:
        comparison.to_csv(args.out_dir / "comparison.csv", index=False)
    print(f"comparison: {md_path}")
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--checkpoints", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--source", default="tb_live")
    prep.add_argument("--n-samples", type=int, default=6)
    prep.set_defaults(func=cmd_prepare)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--checkpoints", type=Path, required=True)
    cmp_.add_argument("--labels", type=Path, required=True)
    cmp_.add_argument("--out-dir", type=Path, required=True)
    cmp_.set_defaults(func=cmd_compare)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
