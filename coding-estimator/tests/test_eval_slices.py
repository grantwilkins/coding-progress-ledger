"""H4 — slice-specific evaluation.

Claim:
    `evaluate_phase_slices` partitions test rows into thirds of
    (checkpoint_step within run); cells with <5 pos OR <5 neg in the
    slice emit feasible=False with note 'insufficient data' and no
    metrics. `evaluate_shape_slices` joins runs to their multi-label
    shape booleans and emits one cell per shape tag, with the same
    feasibility rule.

Plausible wrong implementations:
    - phase computed across the whole frame (not per run) → all early
      checkpoints of a run leak into 'middle' once another run has more
      steps
    - <5 cutoff applied to total rows rather than positives+negatives
    - shape slicing flips truthy mask (drops 'in shape' instead of
      keeping it)
    - single-checkpoint runs become 'late' via float-edge bugs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.eval.slices import (
    MIN_PER_SLICE,
    assign_phase,
    evaluate_phase_slices,
    evaluate_shape_slices,
)


def _preds(n_runs: int, steps_per_run: int, p_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(p_seed)
    rows = []
    for r in range(n_runs):
        for s in range(steps_per_run):
            rows.append({
                "run_id": f"r{r}",
                "source": "src",
                "checkpoint_id": f"r{r}::{s}",
                "checkpoint_step": s,
                "_y": int((r + s) % 2 == 0),
                "_p": float(rng.uniform()),
            })
    return pd.DataFrame(rows)


def test_assign_phase_is_per_run() -> None:
    # run with 9 steps; thirds at steps 0..2 / 3..5 / 6..8
    df = _preds(n_runs=1, steps_per_run=9)
    phase = assign_phase(df)
    assert (phase[df["checkpoint_step"] <= 2] == "early").all()
    assert (phase[(df["checkpoint_step"] >= 3) & (df["checkpoint_step"] <= 5)] == "middle").all()
    assert (phase[df["checkpoint_step"] >= 7] == "late").all()


def test_assign_phase_short_run_is_early_only() -> None:
    df = pd.DataFrame([
        {"run_id": "r0", "checkpoint_step": 0, "_y": 0, "_p": 0.1, "source": "s",
         "checkpoint_id": "r0::0"},
    ])
    assert (assign_phase(df) == "early").all()


def test_assign_phase_uses_per_run_max_not_global() -> None:
    # Long run (50 steps) and short run (3 steps). Step 2 of the short
    # run is its 'late' phase; if we used the global max the short run's
    # rows would all collapse to 'early'.
    rows = [{"run_id": "long", "checkpoint_step": s, "_y": 0, "_p": 0.5,
             "source": "s", "checkpoint_id": f"long::{s}"} for s in range(50)]
    rows += [{"run_id": "short", "checkpoint_step": s, "_y": 0, "_p": 0.5,
              "source": "s", "checkpoint_id": f"short::{s}"} for s in range(3)]
    df = pd.DataFrame(rows)
    phase = assign_phase(df)
    short_phases = set(phase[df["run_id"] == "short"])
    assert "late" in short_phases or "middle" in short_phases  # not all 'early'


def test_phase_slice_marks_insufficient_when_few_positives() -> None:
    # 6 runs × 3 steps = 18 rows; mostly negative so positives < 5
    rows = []
    for r in range(6):
        for s in range(3):
            rows.append({
                "run_id": f"r{r}", "source": "s", "checkpoint_id": f"r{r}::{s}",
                "checkpoint_step": s,
                "_y": 1 if (r == 0 and s == 0) else 0,
                "_p": 0.1,
            })
    df = pd.DataFrame(rows)
    cells = evaluate_phase_slices(
        df, target="t", model="m", scheme="loro", source_slice="src",
    )
    # No phase should be feasible (insufficient positives).
    assert all(not c.feasible for c in cells)
    assert all(c.note == "insufficient data" for c in cells if c.n_checkpoints)


def test_phase_slice_feasible_when_balanced() -> None:
    # 4 runs × 30 steps = 120 rows, alternating labels → ~60 pos, ~60 neg
    rows = []
    for r in range(4):
        for s in range(30):
            rows.append({
                "run_id": f"r{r}", "source": "s", "checkpoint_id": f"r{r}::{s}",
                "checkpoint_step": s,
                "_y": s % 2,
                "_p": 0.5,
            })
    df = pd.DataFrame(rows)
    cells = evaluate_phase_slices(
        df, target="t", model="m", scheme="loro", source_slice="src",
    )
    by_value = {c.slice_value: c for c in cells}
    for v in ("early", "middle", "late"):
        assert by_value[v].feasible
        assert by_value[v].positives >= MIN_PER_SLICE
        assert by_value[v].negatives >= MIN_PER_SLICE


def test_shape_slice_keeps_only_runs_in_shape() -> None:
    # 6 runs total; 3 are tagged shape_foo. Predictions cover all 6.
    pred_rows = []
    for r in range(6):
        for s in range(20):
            pred_rows.append({
                "run_id": f"r{r}", "source": "s", "checkpoint_id": f"r{r}::{s}",
                "checkpoint_step": s, "_y": s % 2, "_p": 0.5,
            })
    preds = pd.DataFrame(pred_rows)
    shapes = pd.DataFrame([
        {"run_id": f"r{r}", "shape_foo": (r < 3), "shape_bar": False}
        for r in range(6)
    ])
    cells = evaluate_shape_slices(
        preds, shapes, target="t", model="m", scheme="loro", source_slice="src",
    )
    foo = next(c for c in cells if c.slice_value == "foo")
    bar = next(c for c in cells if c.slice_value == "bar")
    assert foo.n_runs == 3
    assert bar.n_runs == 0
    assert not bar.feasible
