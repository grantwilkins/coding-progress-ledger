"""G2 — Time-only baseline.

`elapsed_steps` everywhere; `elapsed_wall_time` and
`fraction_timeout_consumed` are only added when the slice is exclusively
`tb_live` (the only canonical source where both are populated).
"""

from __future__ import annotations

from coding_estimator.baselines.base import BaselineSpec


def _cols(sources: tuple[str, ...]) -> tuple[str, ...]:
    base = ("elapsed_steps",)
    if sources and all(s == "tb_live" for s in sources):
        return base + ("elapsed_wall_time", "fraction_timeout_consumed")
    return base


TIME_ONLY = BaselineSpec(name="time_only", feature_cols_for=_cols)
