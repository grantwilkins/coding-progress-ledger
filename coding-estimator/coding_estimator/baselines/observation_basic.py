"""Observation-rich baseline for tb_live_v2/tb_live_v3 style sources."""

from __future__ import annotations

from coding_estimator.baselines.base import BaselineSpec
from coding_estimator.checkpoints.features.registry import GROUPS

_GROUPS = ("closure", "frontier", "instability", "discovery", "validation", "observation")
_OBS_SOURCES = frozenset({"tb_live_v2", "tb_live_v3"})


def _cols(sources: tuple[str, ...]) -> tuple[str, ...]:
    if not sources or not all(source in _OBS_SOURCES for source in sources):
        return tuple()
    out: list[str] = []
    for group_name in _GROUPS:
        for feature in GROUPS[group_name]:
            if feature.dtype in ("int", "float", "bool"):
                out.append(feature.column_name)
    return tuple(out)


OBSERVATION_BASIC = BaselineSpec(name="observation_basic", feature_cols_for=_cols)
