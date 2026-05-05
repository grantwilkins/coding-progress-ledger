"""
Claim:
- build_rollups returns one RunRollup per tb_live run; runs from other
  sources are filtered out.
- step_at_first_no_progress is the FIRST checkpoint_step at which
  `no_progress_window_5 >= 5`, not the largest such step.
- max_prediction_jump uses absolute consecutive differences; positive
  AND negative jumps both count.
- step_of_max_prediction_jump points to the END step of the maximal
  diff (consistent with `np.diff`'s convention: diff[i] = p[i+1] -
  p[i], reported at p[i+1]'s step).

Plausible wrong implementations:
- `first_step_where(...)` returns the LAST matching step (uses .max
  instead of .iloc[0]).
- max_prediction_jump uses signed diff; large negative jumps (drop
  in P(success)) get missed.
- Includes non-tb_live runs (filter wrong).
- `step_of_max_prediction_jump` reports the BEFORE step instead of the
  AFTER step (off-by-one).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.eval.tb_qualitative import (
    NO_PROGRESS_THRESHOLD,
    TB_LIVE,
    build_rollups,
    render_qualitative_report,
)


def _ck(run_id: str, source: str, n: int, **kw) -> pd.DataFrame:
    base = {
        "run_id": [run_id] * n,
        "source": [source] * n,
        "checkpoint_id": [f"{run_id}_c{i}" for i in range(n)],
        "checkpoint_step": list(range(n)),
        "coding_progress": [0.1 * i for i in range(n)],
    }
    for col, default in (
        ("no_progress_window_5", 0),
        ("repeated_observation_loop_flag", False),
        ("num_validation_attempts", 0),
        ("num_validation_successes", 0),
        ("num_validation_failures", 0),
    ):
        base.setdefault(col, [default] * n)
    base.update({k: list(v) for k, v in kw.items()})
    return pd.DataFrame(base)


def test_build_rollups_filters_to_tb_live():
    df = pd.concat(
        [
            _ck("r_tb", TB_LIVE, 5),
            _ck("r_other", "swe_agent_pilot", 5),
        ],
        ignore_index=True,
    )
    out = build_rollups(checkpoints_df=df, predictions_df=pd.DataFrame(), shapes_df=None)
    assert {r.run_id for r in out} == {"r_tb"}


def test_first_no_progress_step_is_first_not_last():
    """no_progress_window_5 jumps to threshold at step 2 and stays there.
    A wrong impl that returns max(step | mask) would report step 4."""
    df = _ck(
        "r_tb",
        TB_LIVE,
        5,
        no_progress_window_5=[0, 0, NO_PROGRESS_THRESHOLD, NO_PROGRESS_THRESHOLD, NO_PROGRESS_THRESHOLD],
    )
    out = build_rollups(checkpoints_df=df, predictions_df=pd.DataFrame(), shapes_df=None)
    assert out[0].step_at_first_no_progress == 2


def test_first_validation_steps_are_recorded_independently():
    """Validation attempt at step 1, success at step 3, failure at step 4."""
    df = _ck(
        "r_tb",
        TB_LIVE,
        5,
        num_validation_attempts=[0, 1, 1, 1, 1],
        num_validation_successes=[0, 0, 0, 1, 1],
        num_validation_failures=[0, 0, 0, 0, 1],
    )
    out = build_rollups(checkpoints_df=df, predictions_df=pd.DataFrame(), shapes_df=None)
    r = out[0]
    assert r.step_at_first_validation == 1
    assert r.step_at_first_validation_success == 3
    assert r.step_at_first_validation_failure == 4


def test_max_prediction_jump_is_absolute_value():
    """Predictions: 0.1, 0.9, 0.2. Diffs: +0.8 (step 1), -0.7 (step 2).
    Max abs jump is 0.8 at step 1. A signed-diff impl would still pick
    step 1 here, so we additionally test a negative-only sequence:
    0.9, 0.7, 0.1 — diffs -0.2, -0.6 — abs-max is 0.6 at step 2."""
    df = _ck("r_tb", TB_LIVE, 3)
    preds = pd.DataFrame(
        {
            "run_id": ["r_tb"] * 3,
            "source": [TB_LIVE] * 3,
            "checkpoint_id": [f"r_tb_c{i}" for i in range(3)],
            "checkpoint_step": [0, 1, 2],
            "_y": [1, 1, 1],
            "_p": [0.9, 0.7, 0.1],
        }
    )
    out = build_rollups(checkpoints_df=df, predictions_df=preds, shapes_df=None)
    assert out[0].max_prediction_jump is not None
    assert abs(out[0].max_prediction_jump - 0.6) < 1e-9
    assert out[0].step_of_max_prediction_jump == 2


def test_max_prediction_jump_none_when_lt_two_predictions():
    df = _ck("r_tb", TB_LIVE, 1)
    preds = pd.DataFrame(
        {
            "run_id": ["r_tb"],
            "source": [TB_LIVE],
            "checkpoint_id": ["r_tb_c0"],
            "checkpoint_step": [0],
            "_y": [1],
            "_p": [0.5],
        }
    )
    out = build_rollups(checkpoints_df=df, predictions_df=preds, shapes_df=None)
    assert out[0].max_prediction_jump is None
    assert out[0].step_of_max_prediction_jump is None


def test_render_qualitative_report_handles_empty_input():
    md = render_qualitative_report([])
    assert "TB-live qualitative rollup" in md
    assert "No tb_live runs" in md
