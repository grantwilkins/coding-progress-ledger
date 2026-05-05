"""Per-source canonical-fill application for prepared checkpoint frames.

`apply_canonical_fills(df)` rewrites every numeric/bool feature column
to the registry's per-source canonical fill when the cell is missing,
matching the contract from AGENTS.md invariant 7. Cells whose declared
fill is `None` (NOT_APPLICABLE_TO_SOURCE / UNKNOWN_DUE_TO_MISSING_ARTIFACT)
are left as NaN so the downstream `_features` strict-NaN guard still
fires on those.
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.checkpoints.features.registry import all_features


def apply_canonical_fills(checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    df = checkpoints_df.copy()
    feats = {f.column_name: f for f in all_features()}
    for source, sub in df.groupby("source", sort=True):
        idx = sub.index
        for col in df.columns:
            f = feats.get(col)
            if f is None or f.dtype not in ("int", "float", "bool"):
                continue
            fill = f.canonical_fill_for(str(source))
            if fill is None:
                continue
            df.loc[idx, col] = sub[col].fillna(fill)
    return df


__all__ = ["apply_canonical_fills"]
