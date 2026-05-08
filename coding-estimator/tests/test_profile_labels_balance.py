"""F3 combined label-balance profile (Workstream F3).

Claim:
    `render_combined(labels_df)` produces a single markdown artifact
    with two layers:
      1. A cross-source headline table that lists ONE row per
         (binary_target, source) combination present in the frame.
      2. A per-source detail section delegated to E8's
         `build_balance_report`.

    Continuous targets (`y_finish_step`, `y_finish_seconds`) must NOT
    appear in the headline (they have no positive-rate concept).

Plausible wrong implementations:
    - render_combined includes continuous targets in the headline
      (because someone hard-coded a target list instead of calling
      `_binary_targets`).
    - The per-source detail loop misses sources whose target column
      values are all-masked.
    - The cross-source headline aggregates across sources (single
      row per target instead of one row per (target, source)),
      losing per-source visibility.
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.profile.labels_balance import _per_source_summary, render_combined


def _label_row(
    target: str, value: float | None, masked: bool, *, source="tb_live", run_id="r1"
) -> dict:
    return {
        "run_id": run_id,
        "source": source,
        "checkpoint_id": f"{run_id}::0",
        "checkpoint_step": 0,
        "is_terminal_checkpoint": False,
        "target_name": target,
        "target_family": "success",
        "target_horizon_units": "terminal",
        "target_horizon_value": None,
        "label_value": value,
        "is_masked": masked,
        "mask_reason": None,
        "label_available": (not masked) and (value is not None),
        "schema_version": "0.1.0",
    }


def test_continuous_targets_excluded_from_headline() -> None:
    """y_finish_step values like 13.0, 24.0 are continuous and must
    not appear in the cross-source headline (no positive-rate concept).
    Catches a future regression that hard-codes the target list."""
    rows = []
    for i in range(8):
        rows.append(_label_row("y_success_eventual", float(i % 2), False, run_id=f"r{i}"))
        rows.append(_label_row("y_finish_step", float(13 + i), False, run_id=f"r{i}"))
    df = pd.DataFrame(rows)
    summary = _per_source_summary(df)
    assert "y_success_eventual" in summary
    assert "y_finish_step" not in summary


def test_headline_emits_one_row_per_target_source_pair() -> None:
    """If a binary target appears in two sources, the headline must
    show two rows (one per source) — NOT a single aggregated row.
    A bug that aggregates would lose the per-source split."""
    rows = []
    for i in range(6):
        rows.append(_label_row(
            "y_success_eventual", float(i < 3), False,
            source="src_a", run_id=f"a{i}",
        ))
        rows.append(_label_row(
            "y_success_eventual", float(i < 4), False,
            source="src_b", run_id=f"b{i}",
        ))
    df = pd.DataFrame(rows)
    summary = _per_source_summary(df)
    assert "src_a" in summary
    assert "src_b" in summary
    # Both sources show up under the SAME target row.
    src_a_lines = [line for line in summary.splitlines() if "src_a" in line]
    src_b_lines = [line for line in summary.splitlines() if "src_b" in line]
    assert len(src_a_lines) == 1
    assert len(src_b_lines) == 1


def test_render_combined_includes_per_source_detail_for_each_source() -> None:
    """Per-source detail sections must appear once per source. A bug
    that iterates over `binary_targets` instead of sources (or vice
    versa) would mismatch the section headers."""
    rows = [
        _label_row("y_success_eventual", 1.0, False, source="src_a", run_id="a1"),
        _label_row("y_success_eventual", 0.0, False, source="src_b", run_id="b1"),
    ]
    df = pd.DataFrame(rows)
    text = render_combined(df)
    assert "## Source: src_a" in text
    assert "## Source: src_b" in text


def test_render_combined_does_not_crash_on_empty_frame() -> None:
    """An empty labels frame (no runs ingested yet) must produce
    valid markdown rather than raising. F11 will surface the data
    deficit elsewhere."""
    empty_df = pd.DataFrame(columns=[
        "run_id", "source", "target_name", "label_value",
        "is_masked", "checkpoint_id",
    ])
    text = render_combined(empty_df)
    assert text.startswith("# Label-balance profile (F3)")


def test_per_source_summary_counts_match_underlying_data() -> None:
    """Sanity: 5 successes + 5 failures on the same source produce
    a row showing positives=5, negatives=5. A bug in `_count` would
    diverge here."""
    rows = []
    for i in range(5):
        rows.append(_label_row(
            "y_success_eventual", 1.0, False, run_id=f"p{i}"
        ))
    for i in range(5):
        rows.append(_label_row(
            "y_success_eventual", 0.0, False, run_id=f"n{i}"
        ))
    df = pd.DataFrame(rows)
    summary = _per_source_summary(df)
    # Row format: | target | source | positives | negatives | ...
    target_line = next(
        line for line in summary.splitlines()
        if "y_success_eventual" in line and "tb_live" in line
    )
    parts = [p.strip() for p in target_line.split("|")]
    # parts: ['', 'y_success_eventual', 'tb_live', '5', '5', ...]
    assert parts[3] == "5"  # positives
    assert parts[4] == "5"  # negatives
