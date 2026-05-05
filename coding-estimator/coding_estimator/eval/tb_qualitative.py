"""K3 — TB-live qualitative rollup.

For every tb_live run, emit a single block covering:
    - phase/shape distribution
    - stuck-loop precursor signals (no-progress windows, repeated-loop flag)
      and the checkpoint at which they first fired
    - validation timeline (first attempt, first success/failure) correlated
      with prediction-update magnitude

Intended for human eyes; one report — three reports for 12 runs is
overkill (per TASKS.md K3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from coding_estimator.eval.slices import assign_phase

TB_LIVE = "tb_live"
NO_PROGRESS_THRESHOLD: int = 5  # checkpoints — first time `no_progress_window_5` >= this
REPEATED_LOOP_FLAG_COL = "repeated_observation_loop_flag"


@dataclass(frozen=True)
class RunRollup:
    run_id: str
    n_checkpoints: int
    final_progress: float | None
    shape_tags: tuple[str, ...]
    phase_at_first_no_progress: str | None
    step_at_first_no_progress: int | None
    step_at_first_repeated_loop: int | None
    step_at_first_validation: int | None
    step_at_first_validation_success: int | None
    step_at_first_validation_failure: int | None
    max_prediction_jump: float | None
    step_of_max_prediction_jump: int | None
    final_success: int | None


def _first_step_where(s: pd.Series, mask: pd.Series) -> int | None:
    if not mask.any():
        return None
    return int(s[mask].iloc[0])


def _max_prediction_jump(p: np.ndarray) -> tuple[float, int] | tuple[None, None]:
    if len(p) < 2:
        return None, None
    diffs = np.abs(np.diff(p))
    if len(diffs) == 0:
        return None, None
    j = int(np.argmax(diffs))
    return float(diffs[j]), j + 1


def _shape_tags_for(run_id: str, shapes_df: pd.DataFrame) -> tuple[str, ...]:
    if shapes_df is None or shapes_df.empty or "run_id" not in shapes_df.columns:
        return ()
    row = shapes_df[shapes_df["run_id"] == run_id]
    if row.empty:
        return ()
    cols = [c for c in row.columns if c.startswith("shape_")]
    out: list[str] = []
    for c in cols:
        v = row[c].iloc[0]
        if bool(v):
            out.append(c[len("shape_"):])
    return tuple(out)


def build_rollups(
    *,
    checkpoints_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    shapes_df: pd.DataFrame | None,
    final_success: dict[str, int] | None = None,
) -> list[RunRollup]:
    """`predictions_df` is the long-form (run_id, _y, _p, ...) frame for
    one (model, target) — typically from `predict_cell` for the
    `y_success_eventual` headline."""
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["run_id", "checkpoint_step"], kind="mergesort")
    out: list[RunRollup] = []
    preds_by_run: dict[str, pd.DataFrame] = {}
    if not predictions_df.empty:
        for rid, g in predictions_df.sort_values(
            ["run_id", "checkpoint_step"], kind="mergesort"
        ).groupby("run_id", sort=True):
            preds_by_run[str(rid)] = g
    for run_id, g in sub.groupby("run_id", sort=True):
        run_id_s = str(run_id)
        steps = g["checkpoint_step"].astype(int).reset_index(drop=True)
        progress = g["coding_progress"].astype(float).reset_index(drop=True)
        no_prog_5 = g.get(
            "no_progress_window_5",
            pd.Series(np.zeros(len(g)), index=g.index),
        ).fillna(0).astype(int).reset_index(drop=True)
        repeated = g.get(
            REPEATED_LOOP_FLAG_COL,
            pd.Series(np.zeros(len(g), dtype=bool), index=g.index),
        ).fillna(False).astype(bool).reset_index(drop=True)
        v_attempts = g.get(
            "num_validation_attempts",
            pd.Series(np.zeros(len(g)), index=g.index),
        ).fillna(0).astype(int).reset_index(drop=True)
        v_success = g.get(
            "num_validation_successes",
            pd.Series(np.zeros(len(g)), index=g.index),
        ).fillna(0).astype(int).reset_index(drop=True)
        v_failure = g.get(
            "num_validation_failures",
            pd.Series(np.zeros(len(g)), index=g.index),
        ).fillna(0).astype(int).reset_index(drop=True)

        no_prog_step = _first_step_where(steps, no_prog_5 >= NO_PROGRESS_THRESHOLD)
        repeated_step = _first_step_where(steps, repeated)
        validation_step = _first_step_where(steps, v_attempts > 0)
        v_success_step = _first_step_where(steps, v_success > 0)
        v_failure_step = _first_step_where(steps, v_failure > 0)

        phase_at_no_progress = None
        if no_prog_step is not None:
            phase_series = assign_phase(g.assign(checkpoint_step=g["checkpoint_step"]))
            phase_at_no_progress = str(
                phase_series.iloc[(g["checkpoint_step"] == no_prog_step).idxmax()]
                if (g["checkpoint_step"] == no_prog_step).any()
                else "early"
            )

        max_jump: float | None = None
        max_jump_step: int | None = None
        run_preds = preds_by_run.get(run_id_s)
        if run_preds is not None and len(run_preds) >= 2:
            p_arr = run_preds["_p"].to_numpy(dtype=float)
            steps_p = run_preds["checkpoint_step"].to_numpy(dtype=int)
            mj, idx = _max_prediction_jump(p_arr)
            if mj is not None:
                max_jump = mj
                max_jump_step = int(steps_p[idx])

        out.append(
            RunRollup(
                run_id=run_id_s,
                n_checkpoints=int(len(g)),
                final_progress=float(progress.iloc[-1]) if len(progress) else None,
                shape_tags=_shape_tags_for(run_id_s, shapes_df) if shapes_df is not None else (),
                phase_at_first_no_progress=phase_at_no_progress,
                step_at_first_no_progress=no_prog_step,
                step_at_first_repeated_loop=repeated_step,
                step_at_first_validation=validation_step,
                step_at_first_validation_success=v_success_step,
                step_at_first_validation_failure=v_failure_step,
                max_prediction_jump=max_jump,
                step_of_max_prediction_jump=max_jump_step,
                final_success=(final_success or {}).get(run_id_s),
            )
        )
    return out


def render_qualitative_report(rollups: list[RunRollup], summary: str | None = None) -> str:
    lines = [
        "# TB-live qualitative rollup (K3)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    if not rollups:
        lines.append("_No tb_live runs in the current checkpoint frame._")
        return "\n".join(lines) + "\n"

    n = len(rollups)
    n_with_no_progress = sum(1 for r in rollups if r.step_at_first_no_progress is not None)
    n_with_repeated = sum(1 for r in rollups if r.step_at_first_repeated_loop is not None)
    n_with_validation = sum(1 for r in rollups if r.step_at_first_validation is not None)
    n_with_validation_success = sum(
        1 for r in rollups if r.step_at_first_validation_success is not None
    )

    lines.extend(
        [
            "## Cohort summary",
            "",
            f"- runs: {n}",
            f"- runs with `no_progress_window_5 >= {NO_PROGRESS_THRESHOLD}`: "
            f"{n_with_no_progress} ({n_with_no_progress / n:.0%})",
            f"- runs with `{REPEATED_LOOP_FLAG_COL}` ever set: "
            f"{n_with_repeated} ({n_with_repeated / n:.0%})",
            f"- runs with at least one validation attempt: "
            f"{n_with_validation} ({n_with_validation / n:.0%})",
            f"- runs with at least one validation success: "
            f"{n_with_validation_success} ({n_with_validation_success / n:.0%})",
            "",
            "## Shape distribution",
            "",
        ]
    )
    shape_counts: dict[str, int] = {}
    for r in rollups:
        for tag in r.shape_tags:
            shape_counts[tag] = shape_counts.get(tag, 0) + 1
    if shape_counts:
        for tag in sorted(shape_counts):
            lines.append(f"- `{tag}`: {shape_counts[tag]}")
    else:
        lines.append("_no shape labels available for tb_live (live source — expected)_")
    lines.append("")

    lines.extend(
        [
            "## Per-run rollup",
            "",
            "| run_id | n_ckpts | final_prog | first_no_prog (phase) | first_repeated_loop | first_validation | first_v_success | first_v_failure | max Δp (step) | shape tags | final_success |",
            "|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for r in sorted(rollups, key=lambda r: r.run_id):
        nop = (
            f"{r.step_at_first_no_progress} ({r.phase_at_first_no_progress})"
            if r.step_at_first_no_progress is not None
            else "-"
        )
        rep = "-" if r.step_at_first_repeated_loop is None else r.step_at_first_repeated_loop
        v0 = "-" if r.step_at_first_validation is None else r.step_at_first_validation
        vs = "-" if r.step_at_first_validation_success is None else r.step_at_first_validation_success
        vf = "-" if r.step_at_first_validation_failure is None else r.step_at_first_validation_failure
        mj = (
            f"{r.max_prediction_jump:.3f} ({r.step_of_max_prediction_jump})"
            if r.max_prediction_jump is not None
            else "-"
        )
        fp = "n/a" if r.final_progress is None else f"{r.final_progress:.2f}"
        tags = ", ".join(r.shape_tags) if r.shape_tags else "-"
        fs = "?" if r.final_success is None else r.final_success
        lines.append(
            f"| {r.run_id} | {r.n_checkpoints} | {fp} | {nop} | {rep} | {v0} | {vs} | {vf} | {mj} | {tags} | {fs} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_qualitative_report(
    path: Path, rollups: list[RunRollup], summary: str | None = None
) -> Path:
    md = render_qualitative_report(rollups, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "TB_LIVE",
    "RunRollup",
    "build_rollups",
    "render_qualitative_report",
    "write_qualitative_report",
]
