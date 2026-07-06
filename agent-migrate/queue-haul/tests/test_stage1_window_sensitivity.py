from __future__ import annotations

"""
Claim:
Stage 1a window sensitivity recomputes raw-window power/load arrays for each
window size and reports comparable concave-fit diagnostics.

Plausible wrong implementations:
- Reuse the same 5s windows for every window size.
- Aggregate prefill, decode, or power at the wrong field level.
- Accept invalid or duplicated window sizes.
- Emit a fitted curve whose ell axis is unsorted or not tied to the analyzed range.
"""

import numpy as np
import pytest

import stage1_window_sensitivity as ws


def test_validate_windows_sorts_but_rejects_nonpositive_or_duplicate():
    assert ws.validate_windows([5, 1, 2]) == (1.0, 2.0, 5.0)

    with pytest.raises(ValueError, match="unique positive"):
        ws.validate_windows([1, 1])
    with pytest.raises(ValueError, match="unique positive"):
        ws.validate_windows([0, 1])


def test_concat_runs_preserves_field_level_and_order():
    runs = [
        {
            "power_w": np.array([10.0]),
            "f_tps": np.array([1.0]),
            "g_tps": np.array([2.0]),
            "itl_ms": np.array([3.0]),
            "ttft_per_tok_ms": np.array([4.0]),
        },
        {
            "power_w": np.array([20.0, 30.0]),
            "f_tps": np.array([5.0, 6.0]),
            "g_tps": np.array([7.0, 8.0]),
            "itl_ms": np.array([9.0, 10.0]),
            "ttft_per_tok_ms": np.array([11.0, 12.0]),
        },
    ]

    out = ws.concat_runs(runs)

    assert out["P"] == pytest.approx([10.0, 20.0, 30.0])
    assert out["f"] == pytest.approx([1.0, 5.0, 6.0])
    assert out["g"] == pytest.approx([2.0, 7.0, 8.0])
    assert out["itl"] == pytest.approx([3.0, 9.0, 10.0])
    assert out["ttft_pt"] == pytest.approx([4.0, 11.0, 12.0])


def test_summary_row_reports_window_and_ell_range():
    summary = {
        "n_windows": 3,
        "F_prefill_tps": 4.0,
        "G_decode_tps": 5.0,
        "P0_w": 10.0,
        "P_max_w": 30.0,
        "r2_linear": 0.2,
        "r2_saturating": 0.9,
        "ell_power_knee": 0.3,
        "rho_star": 0.8,
    }

    row = ws.summary_row(2.0, summary, {"ell": np.array([0.0, 1.0, 2.0])})

    assert row["window_s"] == 2.0
    assert row["n_windows"] == 3
    assert row["ell_p95"] == pytest.approx(1.9)
    assert row["ell_max"] == pytest.approx(2.0)


def test_fit_rows_covers_analyzed_ell_range_in_order():
    rows = ws.fit_rows(5.0, {"ell": np.array([0.0, 2.0]), "P_of_ell": lambda x: 10 + x}, 3)

    assert [r["window_s"] for r in rows] == [5.0, 5.0, 5.0]
    assert [r["ell"] for r in rows] == pytest.approx([0.0, 1.0, 2.0])
    assert [r["power_w"] for r in rows] == pytest.approx([10.0, 11.0, 12.0])
