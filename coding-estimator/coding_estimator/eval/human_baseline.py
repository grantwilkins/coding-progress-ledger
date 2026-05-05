"""Human-baseline scaffolding (recommended in feedback round).

Question this answers: is the prefix ledger *readable* to a person as a
belief-state signal? If a human reading the midpoint ledger prefix can
identify likely progress drops or terminal failures and G4 cannot, the
model is weak. If neither can, the observation channel is missing
signal. If G4 matches the human, the channel carries the signal.

This module:
    - exports a fixed, reproducible sample of (run_id, midpoint_step)
      tuples from `tb_live`;
    - emits one prompt per sample to `reports/human_baseline/prompts/`,
      a single text file containing the agent task description and
      the ledger events at steps `<= midpoint_step`;
    - provides a comparison harness that takes a CSV of human
      predictions and renders a human vs. G4 vs. G2 contingency table.

It does NOT run the human — that's a manual step. The artifacts make
that step low-friction and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import auroc, brier
from coding_estimator.ingest.run_record import load_run
from coding_estimator.splits.protocol import loro

TB_LIVE = "tb_live"
HUMAN_BASELINE_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
)


@dataclass(frozen=True)
class HumanBaselineSample:
    run_id: str
    source: str
    midpoint_step: int
    n_events_visible: int
    task_md_path: str | None


def select_samples(
    *,
    checkpoints_df: pd.DataFrame,
    n_samples: int = 6,
    source: str = TB_LIVE,
    seed: int = 0,
) -> list[HumanBaselineSample]:
    """Deterministic selection: first `n_samples` runs in sorted run_id
    order on `source`, midpoint = floor(median checkpoint_step). Seed
    reserved for future stratification but unused at v0."""
    sub = checkpoints_df[checkpoints_df["source"] == source]
    samples: list[HumanBaselineSample] = []
    for run_id, g in sorted(sub.groupby("run_id"), key=lambda kv: kv[0]):
        if len(samples) >= n_samples:
            break
        steps = sorted(g["checkpoint_step"].astype(int).unique())
        if not steps:
            continue
        midpoint = steps[len(steps) // 2]
        samples.append(
            HumanBaselineSample(
                run_id=str(run_id),
                source=source,
                midpoint_step=int(midpoint),
                n_events_visible=int(len(g[g["checkpoint_step"].astype(int) <= midpoint])),
                task_md_path=None,
            )
        )
    return samples


def render_prompt(sample: HumanBaselineSample) -> str:
    """One text prompt the human reads. Includes the task description
    when available and every ledger event at steps `<= midpoint_step`."""
    try:
        run = load_run(sample.source, sample.run_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return f"# {sample.run_id}\n\n_failed to load run: {exc}_\n"
    task_text = ""
    task_md = run.ledger_path.parent / "task.md"
    if task_md.is_file():
        task_text = task_md.read_text(encoding="utf-8")
    visible = [e for e in run.events if e.step <= sample.midpoint_step]
    lines = [
        f"# Human baseline prompt — {sample.source}/{sample.run_id}",
        "",
        f"**Midpoint step:** {sample.midpoint_step}",
        f"**Events visible:** {len(visible)} of {len(run.events)} total",
        "",
        "## Task",
        "",
        task_text or "_no task.md available_",
        "",
        "## Ledger events visible (prefix only)",
        "",
        "```jsonl",
    ]
    for e in visible:
        lines.append(
            json.dumps(
                {
                    "step": e.step,
                    "event_type": e.event_type,
                    "subtask_id": e.subtask_id,
                    "payload": e.payload,
                    "reason": e.reason,
                    "timestamp": (
                        e.timestamp.isoformat()
                        if hasattr(e.timestamp, "isoformat")
                        else (str(e.timestamp) if e.timestamp is not None else None)
                    ),
                },
                sort_keys=True,
                default=str,
            )
        )
    lines.append("```")
    lines.append("")
    lines.append("## Predict")
    lines.append("")
    lines.append(
        "Given only the prefix above, fill in `human_predictions.csv` "
        "with one row per target:"
    )
    lines.append("")
    lines.append("```csv")
    lines.append("run_id,target,p_success")
    for target in HUMAN_BASELINE_TARGETS:
        lines.append(f"{sample.run_id},{target},<your probability in [0, 1]>")
    lines.append("```")
    return "\n".join(lines) + "\n"


def write_prompts(samples: list[HumanBaselineSample], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in samples:
        path = out_dir / f"{s.source}__{s.run_id}__step{s.midpoint_step}.md"
        path.write_text(render_prompt(s), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def write_sample_manifest(
    samples: list[HumanBaselineSample], path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "samples": [asdict(s) for s in samples],
    }
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def compare_to_models(
    *,
    human_predictions_csv: Path,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    samples: list[HumanBaselineSample],
) -> pd.DataFrame:
    """`human_predictions_csv` columns: run_id, target, p_success.
    For each (run_id, target), compute the at-midpoint G2 and G4
    predictions (LORO predict_cell on tb_live), pull the true label,
    and emit a long-form comparison frame."""
    if not human_predictions_csv.exists():
        return pd.DataFrame()
    human = pd.read_csv(human_predictions_csv)
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE]
    rows: list[dict] = []
    for target in HUMAN_BASELINE_TARGETS:
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        g2 = predict_cell(
            checkpoints_df=sub, labels_df=labels_df, target=target,
            spec=TIME_ONLY, split=split, sources_in_train=(TB_LIVE,),
        )
        g4 = predict_cell(
            checkpoints_df=sub, labels_df=labels_df, target=target,
            spec=LEDGER_BASIC, split=split, sources_in_train=(TB_LIVE,),
        )
        for s in samples:
            human_p = human[
                (human["run_id"] == s.run_id) & (human["target"] == target)
            ]["p_success"]
            if human_p.empty:
                continue
            row_g2 = g2[
                (g2["run_id"] == s.run_id) & (g2["checkpoint_step"] == s.midpoint_step)
            ]
            row_g4 = g4[
                (g4["run_id"] == s.run_id) & (g4["checkpoint_step"] == s.midpoint_step)
            ]
            true_y = (
                row_g4["_y"].iloc[0] if not row_g4.empty
                else (row_g2["_y"].iloc[0] if not row_g2.empty else None)
            )
            rows.append(
                {
                    "run_id": s.run_id,
                    "target": target,
                    "step": s.midpoint_step,
                    "human_p": float(human_p.iloc[0]),
                    "g2_p": float(row_g2["_p"].iloc[0]) if not row_g2.empty else None,
                    "g4_p": float(row_g4["_p"].iloc[0]) if not row_g4.empty else None,
                    "true_y": int(true_y) if true_y is not None else None,
                }
            )
    return pd.DataFrame(rows)


def render_comparison_report(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return (
            "# Human baseline — comparison\n\n"
            "_No human predictions found. Run "
            "`scripts/run_human_baseline.py prepare ...` first, "
            "have a human fill in `reports/human_baseline/human_predictions.csv`, "
            "then run `scripts/run_human_baseline.py compare ...`._\n"
        )
    lines = [
        "# Human baseline — comparison",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        "Per-sample human vs G2 vs G4 prediction at the midpoint "
        "checkpoint, alongside the true label.",
        "",
        "| run_id | target | step | human_p | g2_p | g4_p | true_y |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in comparison.sort_values(["target", "run_id"]).iterrows():
        def _fmt(v) -> str:
            return "n/a" if pd.isna(v) else (
                f"{v:.3f}" if isinstance(v, float) else str(v)
            )
        lines.append(
            f"| {r['run_id']} | {r['target']} | {r['step']} | "
            f"{_fmt(r['human_p'])} | {_fmt(r['g2_p'])} | {_fmt(r['g4_p'])} | "
            f"{_fmt(r['true_y'])} |"
        )
    lines.append("")
    # Aggregate by target
    lines.append("## Aggregate Brier per predictor (where label is known)")
    lines.append("")
    lines.append("| target | n | human Brier | g2 Brier | g4 Brier |")
    lines.append("|---|---:|---:|---:|---:|")
    for target, sub in comparison.groupby("target"):
        sub = sub.dropna(subset=["true_y"])
        if sub.empty:
            continue
        y = sub["true_y"].astype(int).to_numpy()
        def _b(col: str) -> str:
            vals = sub[col].dropna()
            if vals.empty:
                return "n/a"
            paired = sub.dropna(subset=[col])
            return f"{brier(paired['true_y'].astype(int).to_numpy(), paired[col].astype(float).to_numpy()):.3f}"
        lines.append(
            f"| {target} | {len(sub)} | {_b('human_p')} | "
            f"{_b('g2_p')} | {_b('g4_p')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "HumanBaselineSample",
    "select_samples",
    "render_prompt",
    "write_prompts",
    "write_sample_manifest",
    "compare_to_models",
    "render_comparison_report",
]
