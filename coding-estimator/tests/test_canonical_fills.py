"""
Claim:
- apply_canonical_fills(df) replaces NaN cells in numeric/bool feature
  columns with the registry's per-source canonical fill.
- When canonical_fill_for(source) returns None — either because the
  source is NOT in `populated_on` for that feature, or because the
  feature's missingness semantic itself resolves to None — the cell
  is left as NaN. Downstream `_features` is expected to hard-fail on
  it. This is the contract.
- Per-source independence: row's source determines its fill.

Plausible wrong implementations:
- Fills None-fills with 0 anyway, silently masking missing data.
- Fills using a global default (loses per-source distinction).
- Cross-source contamination: applies source A's fill to source B's row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.checkpoints.features.registry import feature_by_name
from coding_estimator.checkpoints.fills import apply_canonical_fills


def test_canonical_fill_none_leaves_nan():
    """`fraction_timeout_consumed` is populated only on tb_live and uses
    UNKNOWN_DUE_TO_MISSING_ARTIFACT (fill=None) even there. A NaN cell
    on swe_agent_pilot must remain NaN."""
    f = feature_by_name("fraction_timeout_consumed")
    assert f.canonical_fill_for("swe_agent_pilot") is None
    df = pd.DataFrame(
        {
            "run_id": ["r0"],
            "source": ["swe_agent_pilot"],
            "fraction_timeout_consumed": [np.nan],
        }
    )
    out = apply_canonical_fills(df)
    assert pd.isna(out["fraction_timeout_consumed"].iloc[0])


def test_canonical_fill_concrete_fills_nan():
    """`elapsed_steps` defaults to APPLICABLE_ABSENT_SO_FAR -> fill 0
    on every populated source. NaN must become 0."""
    f = feature_by_name("elapsed_steps")
    declared = f.canonical_fill_for("swe_agent_pilot")
    assert declared == 0
    df = pd.DataFrame(
        {
            "run_id": ["r0"],
            "source": ["swe_agent_pilot"],
            "elapsed_steps": [np.nan],
        }
    )
    out = apply_canonical_fills(df)
    assert out["elapsed_steps"].iloc[0] == declared


def test_canonical_fill_does_not_overwrite_existing_values():
    """A non-NaN cell must not be overwritten."""
    df = pd.DataFrame(
        {
            "run_id": ["r0"],
            "source": ["swe_agent_pilot"],
            "elapsed_steps": [42],
        }
    )
    out = apply_canonical_fills(df)
    assert out["elapsed_steps"].iloc[0] == 42


def test_canonical_fill_respects_per_source_routing():
    """For `elapsed_steps`, both sources should get a concrete fill (0)
    so the test is uninformative; for `fraction_timeout_consumed`,
    swe_agent_pilot should be left NaN (None fill) and tb_live should
    also be left NaN (UNKNOWN semantic). This exercises that NO source
    silently defaults to a global 0 when the registry says None."""
    df = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "source": ["swe_agent_pilot", "tb_live"],
            "fraction_timeout_consumed": [np.nan, np.nan],
        }
    )
    out = apply_canonical_fills(df).set_index("run_id")
    assert pd.isna(out.loc["a", "fraction_timeout_consumed"])
    assert pd.isna(out.loc["b", "fraction_timeout_consumed"])
