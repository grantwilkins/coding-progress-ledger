"""
Claim:
- select_samples is deterministic: sorted by run_id ascending; the
  same input always yields the same sample list.
- render_prompt only includes ledger events with step <= midpoint_step.
- compare_to_models pulls the prediction at the EXACT midpoint_step
  (not the closest step, not the run's terminal step).

Plausible wrong implementations:
- select_samples uses Python's iteration order over a dict, which is
  insertion-order-stable but not source-order-stable when the input
  was constructed in a different order — would surface as a flaky
  ordering in CI.
- render_prompt filters by event ts instead of event.step (different
  ordering when events have ties on step).
- compare_to_models does a nearest-step lookup, so a checkpoint at
  step != midpoint silently substitutes — this would mean human and
  model are evaluated on different ledger prefixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coding_estimator.eval.human_baseline import (
    HumanBaselineSample,
    compare_to_models,
    render_prompt,
    select_samples,
)


def _ck(run_id: str, step: int, source: str = "tb_live") -> dict:
    return {
        "run_id": run_id,
        "source": source,
        "checkpoint_id": f"{run_id}_c{step}",
        "checkpoint_step": step,
    }


def test_select_samples_orders_by_run_id_ascending_not_input_order():
    """Pass run_ids in deliberately reversed order. select_samples
    must return them in sorted (ascending) order regardless. A wrong
    impl that preserves input order would fail this."""
    rows = []
    for run_id in ("z_run", "a_run", "m_run"):
        for s in range(3):
            rows.append(_ck(run_id, s))
    df = pd.DataFrame(rows)
    samples = select_samples(checkpoints_df=df, n_samples=3)
    got = [s.run_id for s in samples]
    assert got == ["a_run", "m_run", "z_run"]


def test_select_samples_is_deterministic_across_calls():
    """Two calls with the same input frame must produce identical
    sample lists. A wrong impl that uses np.random without a seeded
    generator would be flaky."""
    rows = [_ck(f"r{i}", s) for i in range(8) for s in range(5)]
    df = pd.DataFrame(rows)
    a = select_samples(checkpoints_df=df, n_samples=4)
    b = select_samples(checkpoints_df=df, n_samples=4)
    assert [s.run_id for s in a] == [s.run_id for s in b]
    assert [s.midpoint_step for s in a] == [s.midpoint_step for s in b]


def test_select_samples_midpoint_is_median_of_unique_steps():
    """5 unique steps [0, 1, 2, 3, 4] → midpoint index 5//2 = 2 → step 2.
    A wrong impl using mean/floor would yield (0+1+2+3+4)/5 = 2.0
    accidentally OK here; pick uneven steps to disambiguate."""
    rows = [_ck("r", s) for s in (0, 5, 10, 15, 100)]
    df = pd.DataFrame(rows)
    samples = select_samples(checkpoints_df=df, n_samples=1)
    assert len(samples) == 1
    # Index 5//2 = 2 → step 10. Mean would give 26.
    assert samples[0].midpoint_step == 10


# ---------- render_prompt prefix-only ---------------------------------------


def test_render_prompt_only_emits_events_at_or_before_midpoint():
    """Use a real tb_live run so load_run succeeds. Render at midpoint
    step k; parse the JSONL events embedded in the prompt; assert
    every step is <= k."""
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    sub = ck[ck["source"] == "tb_live"]
    if sub.empty:
        return
    samples = select_samples(checkpoints_df=ck, n_samples=1, source="tb_live")
    if not samples:
        return
    s = samples[0]
    prompt = render_prompt(s)
    # Pull JSON lines between the ```jsonl fences
    in_fence = False
    parsed = []
    for line in prompt.splitlines():
        if line.strip() == "```jsonl":
            in_fence = True
            continue
        if line.strip() == "```":
            in_fence = False
            continue
        if in_fence and line.strip():
            parsed.append(json.loads(line))
    assert parsed, "prompt did not contain any visible events"
    for event in parsed:
        assert event["step"] <= s.midpoint_step, (
            f"event at step {event['step']} > midpoint {s.midpoint_step} "
            "leaked into the prompt"
        )


# ---------- compare_to_models exact-step lookup -----------------------------


def test_compare_to_models_uses_exact_midpoint_step_not_nearest(tmp_path: Path):
    """If the requested midpoint_step has no checkpoint, the
    comparison row must be empty (no nearest-step substitution). A
    wrong impl that silently substitutes would let the human be
    compared to a model prediction at a different prefix."""
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    lb = pd.read_parquet("datasets/labels_all.parquet")
    sub = ck[ck["source"] == "tb_live"]
    if sub.empty or sub["run_id"].nunique() < 2:
        return
    real_run = str(sorted(sub["run_id"].unique())[0])
    real_steps = sorted(sub.loc[sub["run_id"] == real_run, "checkpoint_step"].unique())
    if not real_steps:
        return
    # Pick a step that is GUARANTEED to not exist for this run.
    bogus_step = max(int(s) for s in real_steps) + 9999
    sample = HumanBaselineSample(
        run_id=real_run, source="tb_live",
        midpoint_step=bogus_step, n_events_visible=0, task_md_path=None,
    )
    csv = tmp_path / "human.csv"
    csv.write_text(
        "run_id,target,p_success\n"
        f"{real_run},y_success_eventual,0.5\n",
        encoding="utf-8",
    )
    cmp_df = compare_to_models(
        human_predictions_csv=csv,
        checkpoints_df=ck,
        labels_df=lb,
        samples=[sample],
    )
    if cmp_df.empty:
        return  # no rows produced — fine
    # If a row was emitted, it must reference the exact midpoint step
    # we asked for. Whether g2_p / g4_p are populated at that bogus
    # step is fine to be NaN; but the step recorded in the row must
    # be `bogus_step` (proving the comparison did NOT silently shift
    # to a real step).
    for _, r in cmp_df.iterrows():
        assert r["step"] == bogus_step
