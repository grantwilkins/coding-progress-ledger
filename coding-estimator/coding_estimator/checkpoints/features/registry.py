"""Feature registry: every v0 feature column with its group, dtype,
missingness semantics, and per-source availability.

Computation lands in Workstream D (build_estimator_checkpoints local
re-implementation). This file is the *catalogue*, not the builder.

The `missingness_semantic` field carries the four-valued contract from
AGENTS.md invariant 7. The `populated_on` tuple already covers
`not_applicable_to_source` implicitly (a source not in `populated_on`
gets `None` per the canonical fill); `missingness_semantic` is the
meaning when the source IS in `populated_on` but the cell is still
missing (count not accumulated, event never fired, side artifact
absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from coding_estimator.checkpoints.features.missingness import (
    CANONICAL_FILL,
    Missingness,
)

Group = Literal[
    "frontier",
    "closure",
    "discovery",
    "instability",
    "stalling",
    "validation",
    "evidence",
    "time_budget",
    "source_task",
]


@dataclass(frozen=True)
class Feature:
    column_name: str
    dtype: Literal["int", "float", "bool", "str"]
    group: Group
    populated_on: tuple[str, ...]
    upstream_source: str | None
    missingness_semantic: Missingness
    run_constant_flag: bool
    derivable_from: Literal["yes", "no", "requires_transcript", "requires_source_trace"] = "yes"
    feature_or_label: Literal["feature"] = "feature"
    prefix_only: bool = True

    def canonical_fill_for(self, source_id: str) -> Any:
        """Canonical fill value for this feature when the cell is empty.

        - source not in populated_on -> not_applicable_to_source -> None
        - source in populated_on -> the feature's declared semantic's fill
        """
        if source_id not in self.populated_on:
            return CANONICAL_FILL[Missingness.NOT_APPLICABLE_TO_SOURCE]
        return CANONICAL_FILL[self.missingness_semantic]


_ALL_SOURCES: tuple[str, ...] = (
    "swe_agent_pilot",
    "swe_agent_pilot_v3",
    "swe_agent_live",
    "swe_agent_live_wallclock",
    "hermes_pilot",
    "hermes_pilot_h5",
    "hermes_pilot_h5_v2",
    "tb_live",
)
_TB_ONLY: tuple[str, ...] = ("tb_live",)
_WALLCLOCK_SOURCES: tuple[str, ...] = ("tb_live", "swe_agent_live_wallclock")


def _f(name: str, dtype, group, **kw) -> Feature:
    kw.setdefault("populated_on", _ALL_SOURCES)
    kw.setdefault("upstream_source", "build_estimator_checkpoints.py")
    kw.setdefault("missingness_semantic", Missingness.APPLICABLE_ABSENT_SO_FAR)
    kw.setdefault("run_constant_flag", False)
    return Feature(column_name=name, dtype=dtype, group=group, **kw)


FRONTIER: list[Feature] = [
    _f("active_leaf_count", "int", "frontier"),
    _f("active_coding_leaf_count", "int", "frontier"),
    _f("active_validation_leaf_count", "int", "frontier"),
]

CLOSURE: list[Feature] = [
    _f("completed_leaf_count", "int", "closure"),
    _f("coding_progress", "float", "closure"),
    _f("validation_progress", "float", "closure"),
    _f("product_progress", "float", "closure"),
    _f("investigation_progress", "float", "closure"),
]

DISCOVERY: list[Feature] = [
    _f("num_adds_so_far", "int", "discovery"),
    _f("num_splits_so_far", "int", "discovery"),
    _f("denominator_growth_so_far", "int", "discovery"),
    _f("steps_since_new_subtask", "int", "discovery"),
    _f("new_leaf_count_last_1_steps", "int", "discovery", upstream_source=None),
    _f("new_leaf_count_last_3_steps", "int", "discovery", upstream_source=None),
    _f("new_leaf_count_last_5_steps", "int", "discovery", upstream_source=None),
]

INSTABILITY: list[Feature] = [
    _f("num_reopens_so_far", "int", "instability"),
    _f("num_invalidations_so_far", "int", "instability"),
    _f("num_deletes_so_far", "int", "instability"),
    _f("largest_progress_drop_so_far", "float", "instability"),
    _f("num_progress_drops_so_far", "int", "instability"),
    _f("steps_since_last_drop", "int", "instability"),
]

STALLING: list[Feature] = [
    _f("blocked_leaf_count", "int", "stalling"),
    _f("blocked_coding_leaf_count", "int", "stalling"),
    _f("blocked_validation_leaf_count", "int", "stalling"),
    _f("steps_since_completion", "int", "stalling"),
    _f("steps_since_progress_increase", "int", "stalling"),
    _f("steps_since_status_change", "int", "stalling"),
    _f("steps_since_evidence", "int", "stalling"),
    _f("repeated_observation_loop_flag", "bool", "stalling",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("no_progress_window_5", "bool", "stalling",
       upstream_source=None,
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("no_progress_window_10", "bool", "stalling",
       upstream_source=None,
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
]

VALIDATION: list[Feature] = [
    _f("validation_leaf_exists", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("validation_started", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("validation_complete", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("validation_failed", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("validation_blocked", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("validation_in_progress", "bool", "validation",
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
    _f("num_validation_attempts", "int", "validation"),
    _f("num_validation_failures", "int", "validation"),
    _f("num_validation_successes", "int", "validation"),
    _f("steps_since_last_validation", "int", "validation"),
    _f("submit_without_validation_so_far", "bool", "validation",
       upstream_source=None,
       missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN),
]

EVIDENCE: list[Feature] = [
    _f("strong_completion_count", "int", "evidence"),
    _f("manual_only_completion_count", "int", "evidence"),
    _f("weak_product_completion_count", "int", "evidence"),
    _f("strong_evidence_fraction", "float", "evidence"),
    _f("manual_only_evidence_fraction", "float", "evidence"),
    _f("latest_completion_evidence_type", "str", "evidence",
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
]

TIME_BUDGET: list[Feature] = [
    _f("elapsed_steps", "int", "time_budget"),
    _f("elapsed_wall_time", "float", "time_budget", populated_on=_WALLCLOCK_SOURCES,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("fraction_timeout_consumed", "float", "time_budget", populated_on=_TB_ONLY,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("remaining_timeout_budget", "float", "time_budget", populated_on=_TB_ONLY,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("completion_rate_recent_steps", "float", "time_budget", upstream_source=None),
]

SOURCE_TASK: list[Feature] = [
    _f("source", "str", "source_task", run_constant_flag=True),
    _f("agent_scaffold", "str", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("model_name", "str", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("task_family_hash", "str", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("repo_family_hash", "str", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("initial_prompt_length", "int", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
    _f("initial_files_count", "int", "source_task", run_constant_flag=True,
       missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT),
]


GROUPS: dict[str, list[Feature]] = {
    "frontier": FRONTIER,
    "closure": CLOSURE,
    "discovery": DISCOVERY,
    "instability": INSTABILITY,
    "stalling": STALLING,
    "validation": VALIDATION,
    "evidence": EVIDENCE,
    "time_budget": TIME_BUDGET,
    "source_task": SOURCE_TASK,
}


def all_features() -> list[Feature]:
    out: list[Feature] = []
    for group in GROUPS.values():
        out.extend(group)
    return out


def feature_by_name(name: str) -> Feature:
    for f in all_features():
        if f.column_name == name:
            return f
    raise KeyError(f"no such feature: {name}")
