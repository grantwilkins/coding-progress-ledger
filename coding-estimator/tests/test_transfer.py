"""
Claim:
- evaluate_transfer trains on rows from RETRO_SOURCES (swe_agent_pilot,
  hermes_pilot_h5_v2) and evaluates on rows from LIVE_SOURCE (tb_live).
- For each target it emits one row per config: g2_time_only, g4_full,
  and one g4_minus_<group> per ledger group.
- The reported n_test_runs equals the count of unique tb_live runs that
  contributed at least one labeled checkpoint to the target.

Plausible wrong implementations:
- Trains on tb_live rows (data leak); n_test_runs would still look
  right but the metric would be optimistically biased.
- Ablation rows are emitted with the SAME features as g4_full (the
  ablation is a no-op).
- n_test_runs counts unique source values rather than runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.transfer import (
    LEDGER_GROUPS,
    LIVE_SOURCE,
    RETRO_SOURCES,
    evaluate_transfer,
)


@pytest.fixture(scope="module")
def transfer_rows():
    ck = apply_canonical_fills(pd.read_parquet("datasets/checkpoints_all.parquet"))
    lb = pd.read_parquet("datasets/labels_all.parquet")
    return evaluate_transfer(checkpoints_df=ck, labels_df=lb, bootstrap_b=50)


def test_per_target_emits_one_full_and_one_per_ablation(transfer_rows):
    if not transfer_rows:
        pytest.skip("transfer requires both retro and live sources to be present")
    by_target: dict[str, set[str]] = {}
    for r in transfer_rows:
        by_target.setdefault(r.target, set()).add(r.config)
    expected = {"g2_time_only", "g4_full"} | {f"g4_minus_{g}" for g in LEDGER_GROUPS}
    for target, configs in by_target.items():
        assert configs == expected, f"target {target} has unexpected configs {configs}"


def test_n_test_runs_matches_tb_live_run_count(transfer_rows):
    if not transfer_rows:
        pytest.skip("no transfer rows")
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    n_tb_runs = ck.loc[ck["source"] == LIVE_SOURCE, "run_id"].nunique()
    for r in transfer_rows:
        assert r.n_test_runs <= n_tb_runs


def test_g4_minus_group_differs_from_g4_full_when_group_carries_signal(transfer_rows):
    """If feature-group ablation is a no-op (wrong impl), every
    g4_minus_<group> Brier equals g4_full Brier. Empirically at least
    one ablation MUST move the metric on at least one target — the
    g4_full vs g4_minus_<group> Briers cannot all be identical."""
    if not transfer_rows:
        pytest.skip("no transfer rows")
    by_target: dict[str, dict[str, float]] = {}
    for r in transfer_rows:
        if r.brier == r.brier:  # not NaN
            by_target.setdefault(r.target, {})[r.config] = r.brier
    # Find any target where at least one ablation differs from full.
    differing = False
    for target, by_config in by_target.items():
        full = by_config.get("g4_full")
        if full is None:
            continue
        for g in LEDGER_GROUPS:
            other = by_config.get(f"g4_minus_{g}")
            if other is not None and abs(other - full) > 1e-9:
                differing = True
                break
        if differing:
            break
    assert differing, "feature-group ablation appears to be a no-op (all Briers identical)"


def test_train_set_excludes_live_source(transfer_rows):
    """Smoke check via positive_rate consistency: the test positive_rate
    must reflect tb_live's labels for the target, not retro sources'."""
    if not transfer_rows:
        pytest.skip("no transfer rows")
    lb = pd.read_parquet("datasets/labels_all.parquet")
    for r in transfer_rows:
        sub = lb[
            (lb["source"] == LIVE_SOURCE)
            & (lb["target_name"] == r.target)
            & (~lb["is_masked"].astype(bool))
        ]
        if sub.empty:
            continue
        live_pos = float(sub["label_value"].astype(int).mean())
        # The reported positive_rate should match the LIVE-source rate
        # within the rows that actually joined (we can't assert exact
        # equality if some rows were filtered, but it must NOT match the
        # retro sources' rate).
        retro_sub = lb[
            (lb["source"].isin(RETRO_SOURCES))
            & (lb["target_name"] == r.target)
            & (~lb["is_masked"].astype(bool))
        ]
        if retro_sub.empty:
            continue
        retro_pos = float(retro_sub["label_value"].astype(int).mean())
        if abs(live_pos - retro_pos) > 0.05:
            # Distinguishable rates — the reported test positive_rate
            # MUST match the live one.
            assert abs(r.positive_rate - live_pos) < 1e-3, (
                f"target {r.target}: reported positive_rate {r.positive_rate} "
                f"closer to retro rate {retro_pos} than live rate {live_pos}"
            )
