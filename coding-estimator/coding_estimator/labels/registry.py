"""v0 prediction targets and their (declared) computation hooks.

The computation functions are not implemented in this workstream; the
registry records *what* each target is and *where* its compute will live.
Workstream E wires them up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

WindowKind = Literal["strict-future", "terminal", "regression"]
HorizonUnits = Literal["steps", "seconds", "terminal", "none"]


@dataclass(frozen=True)
class Target:
    name: str
    family: Literal["success", "progress_dynamics", "validation", "submission"]
    horizon_units: HorizonUnits
    horizon_value: int | None
    definition: str
    window_kind: WindowKind
    source_signal: str
    mask_rule: str
    upstream_q_target_id: str | None
    run_constant_flag: bool
    base_rate_estimate: float | None
    compute: Callable[[pd.DataFrame], pd.Series] | None = None
    v0: bool = True


_NOT_IMPL = None  # implementations land in Workstream E.


V0_TARGETS: dict[str, Target] = {
    "y_success_eventual": Target(
        name="y_success_eventual",
        family="success",
        horizon_units="terminal",
        horizon_value=None,
        definition=(
            "1 iff the run's final verdict is success at terminal step. "
            "Run-constant within a run; replicated across all checkpoints."
        ),
        window_kind="terminal",
        source_signal="run_manifest.json::final_success",
        mask_rule="never_mask",
        upstream_q_target_id=None,
        run_constant_flag=True,
        base_rate_estimate=0.50,
        compute=_NOT_IMPL,
    ),
    "y_future_progress_drop_h5": Target(
        name="y_future_progress_drop_h5",
        family="progress_dynamics",
        horizon_units="steps",
        horizon_value=5,
        definition=(
            "1 iff there exists a step s in (t, t+5] where overall progress "
            "decreases relative to its value at s-1 (drop = strict decrease "
            "of the upstream W3 progress series)."
        ),
        window_kind="strict-future",
        source_signal="ledger.jsonl progress trajectory",
        mask_rule=(
            "mask if t + 5 > finish_step or is_terminal_checkpoint==True"
        ),
        upstream_q_target_id="future_progress_drop",
        run_constant_flag=False,
        base_rate_estimate=0.30,
        compute=_NOT_IMPL,
    ),
    "y_validation_new_work_h5": Target(
        name="y_validation_new_work_h5",
        family="validation",
        horizon_units="steps",
        horizon_value=5,
        definition=(
            "1 iff a validation event in (t, t+5] introduces a new product "
            "leaf or reopens a completed one (==upstream Q1 "
            "validation_exposes_new_work)."
        ),
        window_kind="strict-future",
        source_signal="ledger.jsonl validation+add/reopen events",
        mask_rule=(
            "mask if t + 5 > finish_step or is_terminal_checkpoint==True"
        ),
        upstream_q_target_id="validation_exposes_new_work",
        run_constant_flag=False,
        base_rate_estimate=0.02,
        compute=_NOT_IMPL,
    ),
    "y_submit_without_validation": Target(
        name="y_submit_without_validation",
        family="submission",
        horizon_units="terminal",
        horizon_value=None,
        definition=(
            "1 iff the run terminated with a submitted artifact and no "
            "validation events ever occurred (==upstream Q1 "
            "submit_without_validation_state). RUN-CONSTANT within a run; "
            "any non-trivial AUROC at non-terminal t is a data property, "
            "not skill."
        ),
        window_kind="terminal",
        source_signal="ledger.jsonl + run_manifest.json",
        mask_rule="never_mask",
        upstream_q_target_id="submit_without_validation_state",
        run_constant_flag=True,
        base_rate_estimate=0.10,
        compute=_NOT_IMPL,
    ),
    "y_finish_step": Target(
        name="y_finish_step",
        family="success",
        horizon_units="terminal",
        horizon_value=None,
        definition=(
            "Terminal step index of the run (max ledger event step). "
            "Continuous regression target. Run-constant; replicated to "
            "every checkpoint. Used by the time-only baseline G2 and as "
            "a sanity probe for elapsed-fraction features."
        ),
        window_kind="terminal",
        source_signal="ledger.jsonl::final event step",
        mask_rule="never_mask",
        upstream_q_target_id=None,
        run_constant_flag=True,
        base_rate_estimate=None,
        compute=_NOT_IMPL,
    ),
    "y_finish_seconds": Target(
        name="y_finish_seconds",
        family="success",
        horizon_units="terminal",
        horizon_value=None,
        definition=(
            "Terminal wall-clock duration in seconds (end_wall_time "
            "- start_wall_time). NULL on sources without real wallclock "
            "(label_available=false there). Run-constant when present."
        ),
        window_kind="terminal",
        source_signal="run_record.start/end_wall_time",
        mask_rule="null_when_no_wallclock",
        upstream_q_target_id=None,
        run_constant_flag=True,
        base_rate_estimate=None,
        compute=_NOT_IMPL,
    ),
    "y_timeout": Target(
        name="y_timeout",
        family="success",
        horizon_units="terminal",
        horizon_value=None,
        definition=(
            "1 iff the run terminated by hitting the source's "
            "step/wall-clock budget rather than verifying. Sourced from "
            "FinalLabel.timeout. Run-constant."
        ),
        window_kind="terminal",
        source_signal="FinalLabel.timeout",
        mask_rule="never_mask",
        upstream_q_target_id=None,
        run_constant_flag=True,
        base_rate_estimate=None,
        compute=_NOT_IMPL,
    ),
}


# Deferred targets — recorded so they aren't lost. Re-evaluate at N > 100 runs.
DEFERRED_TARGETS: dict[str, Target] = {
    name: Target(
        name=name,
        family=family,
        horizon_units=units,
        horizon_value=horizon,
        definition="DEFERRED: not implemented in v0; see TASKS.md § B2.bis.",
        window_kind=window,
        source_signal="n/a (deferred)",
        mask_rule="n/a (deferred)",
        upstream_q_target_id=q_id,
        run_constant_flag=False,
        base_rate_estimate=None,
        compute=None,
        v0=False,
    )
    for (name, family, units, horizon, window, q_id) in [
        ("y_success_by_h_steps_5", "success", "steps", 5, "strict-future", None),
        ("y_success_by_h_steps_10", "success", "steps", 10, "strict-future", None),
        ("y_success_by_h_steps_25", "success", "steps", 25, "strict-future", None),
        ("y_success_by_h_steps_50", "success", "steps", 50, "strict-future", None),
        ("y_success_by_timeout", "success", "steps", None, "terminal", None),
        ("y_success_by_h_seconds_300", "success", "seconds", 300, "strict-future", None),
        ("y_success_by_h_seconds_900", "success", "seconds", 900, "strict-future", None),
        ("y_success_by_h_seconds_1800", "success", "seconds", 1800, "strict-future", None),
        ("y_success_by_h_seconds_runtimeout", "success", "seconds", None, "terminal", None),
        ("y_remaining_steps_if_success", "success", "steps", None, "regression", None),
        ("y_remaining_seconds_if_success", "success", "seconds", None, "regression", None),
        ("y_product_reopen_h5", "progress_dynamics", "steps", 5, "strict-future",
         "product_reopened_after_completion"),
        ("y_stuck_loop_h5", "progress_dynamics", "steps", 5, "strict-future",
         "stuck_loop_next_window"),
        ("y_blocked_within_h5", "progress_dynamics", "steps", 5, "strict-future", None),
        ("y_new_scope_within_h5", "progress_dynamics", "steps", 5, "strict-future", None),
        ("y_validation_failure_within_h5", "validation", "steps", 5, "strict-future", None),
    ]
}


def all_targets() -> dict[str, Target]:
    return {**V0_TARGETS, **DEFERRED_TARGETS}
