"""Label-balance audit (Workstream E8).

Claim:
    `_binary_targets(labels_df)` identifies binary targets by
    inspecting the unmasked `label_value` distribution: a target whose
    unmasked values lie entirely in {0.0, 1.0} is binary; everything
    else (continuous regression targets like `y_finish_step`) is
    excluded. `build_balance_report` then computes positives /
    negatives / masked counts ONLY on binary targets.

    Counts are exact: positives = (label_value >= 1 - 1e-9), negatives
    = (label_value <= 0 + 1e-9), masked = sum(is_masked). The thin
    flag fires when positives < 5 OR negatives < 5.

Plausible wrong implementations:
    - hard-code a name-list skip (e.g. {"y_finish_step",
      "y_finish_seconds"}) -> a future regression target with a
      different name silently slips through and gets miscounted as
      "all positives" (every value >= 1 - 1e-9 for steps >= 1).
    - use `pd.api.types.is_numeric_dtype` to detect binary -> matches
      both binary and continuous numeric columns.
    - count masked rows as positives or negatives -> inflates totals
      and breaks the `n_unmasked = positives + negatives` invariant.
    - use `>` instead of `>=` for the positive boundary (or vice
      versa) -> miscounts exact 1.0 values.
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.labels.balance import (
    THIN_THRESHOLD,
    _binary_targets,
    _count,
    build_balance_report,
)


def _row(
    target_name: str,
    label_value: float | None,
    is_masked: bool,
    *,
    source: str = "tb_live",
    run_id: str = "r1",
    checkpoint_step: int = 0,
) -> dict:
    return {
        "run_id": run_id,
        "source": source,
        "checkpoint_id": f"{run_id}::{checkpoint_step}",
        "checkpoint_step": checkpoint_step,
        "is_terminal_checkpoint": False,
        "target_name": target_name,
        "target_family": "success",
        "target_horizon_units": "terminal",
        "target_horizon_value": None,
        "label_value": label_value,
        "is_masked": is_masked,
        "mask_reason": None,
        "label_available": (not is_masked) and (label_value is not None),
        "schema_version": "0.1.0",
    }


def test_binary_detection_rejects_continuous_target_even_with_unfamiliar_name() -> None:
    """Regression: a future regression target named e.g.
    `y_remaining_steps` (NOT in the previously hard-coded skip list)
    must NOT be classified as binary. The data-driven detector reads
    its values and rejects it because they are not all in {0,1}."""
    rows = [
        _row("y_success_eventual", 1.0, False, run_id="a"),
        _row("y_success_eventual", 0.0, False, run_id="b"),
        # y_remaining_steps (a hypothetical future target) carries
        # continuous values 23.0, 1.0, 47.0. The 1.0 alone is in {0,1},
        # but the set must lie ENTIRELY in {0,1} to qualify -> rejected.
        _row("y_remaining_steps", 23.0, False, run_id="a"),
        _row("y_remaining_steps", 1.0, False, run_id="b"),
        _row("y_remaining_steps", 47.0, False, run_id="c"),
    ]
    df = pd.DataFrame(rows)
    binary = _binary_targets(df)
    assert "y_success_eventual" in binary
    assert "y_remaining_steps" not in binary, binary


def test_binary_detection_accepts_target_with_only_zeros_or_only_ones() -> None:
    """A target with all-zero or all-one unmasked values still has its
    domain entirely in {0,1}. The detector should classify it as
    binary even though it is degenerate. The downstream `thin` flag
    then correctly fires (one of pos/neg is zero)."""
    rows = [
        _row("y_all_pos", 1.0, False, run_id=f"r{i}") for i in range(3)
    ] + [
        _row("y_all_neg", 0.0, False, run_id=f"r{i}") for i in range(3)
    ]
    df = pd.DataFrame(rows)
    binary = _binary_targets(df)
    assert "y_all_pos" in binary
    assert "y_all_neg" in binary


def test_binary_detection_ignores_values_inside_masked_rows() -> None:
    """Masked rows can have label_value=None OR a stale numeric value.
    The detector must read only UNMASKED values (the contract). A bug
    that leaks masked values into the binary check could (mis)classify
    a target as non-binary."""
    rows = [
        _row("y_target", 1.0, False, run_id="a"),
        _row("y_target", 0.0, False, run_id="b"),
        # Masked rows with arbitrary numeric label_value that is NOT in
        # {0,1}. A leaky implementation would see 99.0 and reject.
        _row("y_target", 99.0, True, run_id="c"),
    ]
    df = pd.DataFrame(rows)
    assert "y_target" in _binary_targets(df)


def test_count_obeys_pos_neg_masked_partition() -> None:
    """Invariant: positives + negatives + masked == total row count for
    a single target. A double-count or undercount in `_count` lands
    here as an arithmetic violation."""
    rows = [
        _row("y_t", 1.0, False, run_id="a"),
        _row("y_t", 1.0, False, run_id="b"),
        _row("y_t", 0.0, False, run_id="c"),
        _row("y_t", None, True, run_id="d"),
        _row("y_t", None, True, run_id="e"),
    ]
    df = pd.DataFrame(rows)
    c = _count(df)
    assert c.positives == 2
    assert c.negatives == 1
    assert c.masked == 2
    assert c.positives + c.negatives + c.masked == len(rows)


def test_thin_flag_fires_below_threshold_only() -> None:
    """The thin flag must fire iff positives < THIN_THRESHOLD OR
    negatives < THIN_THRESHOLD (= 5). A boundary check at exactly 5
    must NOT fire the flag."""
    # Build a target with exactly 5 positives and 5 negatives -> NOT thin.
    rows = []
    for i in range(5):
        rows.append(_row("y_ok", 1.0, False, run_id=f"p{i}"))
        rows.append(_row("y_ok", 0.0, False, run_id=f"n{i}"))
    # Target with 4 positives, 100 negatives -> THIN (positives < 5).
    for i in range(4):
        rows.append(_row("y_thin", 1.0, False, run_id=f"tp{i}"))
    for i in range(100):
        rows.append(_row("y_thin", 0.0, False, run_id=f"tn{i}"))
    df = pd.DataFrame(rows)
    ok = _count(df[df["target_name"] == "y_ok"])
    thin = _count(df[df["target_name"] == "y_thin"])
    assert ok.thin is False
    assert thin.thin is True
    assert THIN_THRESHOLD == 5


def test_positive_rate_is_pos_over_pos_plus_neg_excluding_masked() -> None:
    """positive_rate = positives / (positives + negatives). Masked rows
    must NOT be in the denominator. A regression that includes masked
    rows would deflate the rate."""
    rows = [
        _row("y_t", 1.0, False, run_id="a"),
        _row("y_t", 1.0, False, run_id="b"),
        _row("y_t", 1.0, False, run_id="c"),
        _row("y_t", 0.0, False, run_id="d"),
        _row("y_t", None, True, run_id="e"),
        _row("y_t", None, True, run_id="f"),
    ]
    df = pd.DataFrame(rows)
    c = _count(df)
    assert c.positive_rate == 0.75  # 3 / (3 + 1), NOT 3 / 6


def test_build_balance_report_excludes_continuous_targets_from_per_target_table() -> None:
    """End-to-end: the markdown report's per-target table must include
    binary target names and exclude continuous ones. A bug in
    _binary_targets propagates straight into the report."""
    rows = []
    for i in range(8):
        rows.append(_row("y_success_eventual", float(i % 2), False, run_id=f"r{i}"))
        rows.append(_row("y_finish_step", float(13 + i), False, run_id=f"r{i}"))
    df = pd.DataFrame(rows)
    text = build_balance_report("tb_live", df)
    assert "y_success_eventual" in text
    assert "y_finish_step" not in text


def test_build_balance_report_only_uses_rows_for_the_named_source() -> None:
    """If the same target appears in multiple sources, the
    per-source report must aggregate ONLY rows of the named source.
    A cross-source leak would fabricate counts from unrelated runs."""
    rows = []
    for i in range(6):
        rows.append(
            _row("y_t", 1.0 if i < 5 else 0.0, False,
                 source="tb_live", run_id=f"a{i}")
        )
    for i in range(20):
        rows.append(
            _row("y_t", 0.0, False,
                 source="swe_agent_pilot", run_id=f"b{i}")
        )
    df = pd.DataFrame(rows)
    text = build_balance_report("tb_live", df)
    # tb_live has 5 positives, 1 negative, total unmasked 6.
    # If the report leaked swe_agent_pilot rows, n_unmasked would be 26.
    assert "y_t" in text
    # Search for an explicit n_unmasked = 6 cell. The markdown table
    # has columns including n_unmasked, with a row for "all".
    assert " 6 " in text or "|6|" in text or "| 6 " in text, text
    assert "26" not in text
