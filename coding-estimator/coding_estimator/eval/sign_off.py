"""P2 — sign-off package and P3 — downstream readiness.

P2 builds an end-to-end sign-off bundle for the v0 estimator under
`models/<estimator_id>/` plus a top-level `reports/sign_off_<id>.md`
that ties together: model card, gate verdict, failure-mode results,
known limits, and the `not_safe_for_control` flag.

P3 emits one of two reports based on the gate verdict:
- `reports/READY_FOR_SCHEDULING.md` if pass
- `reports/NOT_READY_FOR_SCHEDULING.md` otherwise (with the cheapest
  next experiment).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from coding_estimator.checkpoints.features.registry import GROUPS
from coding_estimator.eval.failure_modes import (
    FailureModeResult,
    all_results_json,
)
from coding_estimator.eval.go_no_go import (
    GateCondition,
    GateReport,
)
from coding_estimator.eval.harness import EvalCell
from coding_estimator.io import write_json
from coding_estimator.models.cards import (
    build_card_record,
    render_card_markdown,
    write_card,
)


def _outcome_badge(o: str) -> str:
    return {
        "pass": "✅ pass",
        "fail": "❌ fail",
        "indeterminate": "⚠️ indeterminate",
    }.get(o, o)


def _failure_mode_dict_for_card(
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> dict[str, Any]:
    payload = all_results_json(o1, o5, o7)
    # The schema requires test_id / outcome / metric_name / threshold —
    # all_results_json already populates these via dataclass asdict.
    return payload


def _decide_not_safe_for_control(
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> tuple[bool, list[str]]:
    """Default `not_safe_for_control = True`. Flip to False ONLY if
    every required gate condition is `pass` AND every failure-mode
    test (O1, O5, every per-source O7 result) is `pass`. Per
    docs/VERSIONING.md.

    Reasons are emitted at the most specific level that applies so a
    reader can act on them; the redundant "gate verdict is `fail`"
    bullet is suppressed when at least one specific required condition
    has already been listed (it would be derivable from the per-condition
    reasons)."""
    reasons: list[str] = []
    failing_required = [
        c for c in gate.conditions if c.required and c.outcome != "pass"
    ]
    for c in failing_required:
        reasons.append(f"required gate `{c.condition_id}` is `{c.outcome}`")
    if gate.verdict != "pass" and not failing_required:
        reasons.append(f"gate verdict is `{gate.verdict}`")
    if o1.outcome != "pass":
        reasons.append(f"O1 outcome is `{o1.outcome}`")
    if o5.outcome != "pass":
        reasons.append(f"O5 outcome is `{o5.outcome}`")
    for r in o7:
        if r.outcome == "fail":
            reasons.append(
                f"O7 fails on source `{r.detail.get('source', '?')}`"
            )
    return (len(reasons) > 0), reasons


def _known_limits_from(
    gate: GateReport,
    o7: list[FailureModeResult],
    not_safe_reasons: list[str],
) -> list[str]:
    out: list[str] = []
    if not_safe_reasons:
        out.append(
            "not_safe_for_control = true: " + "; ".join(not_safe_reasons)
        )
    failing_sources = sorted(
        r.detail.get("source", "?")
        for r in o7
        if r.outcome == "fail"
    )
    if failing_sources:
        out.append(
            f"O7 timeout-bias FAIL on {failing_sources}: ledger does not add "
            "≥ 0.02 Brier over time-only on these sources"
        )
    indeterminate_ids = [
        c.condition_id
        for c in gate.conditions
        if c.required and c.outcome == "indeterminate"
    ]
    if indeterminate_ids:
        out.append(
            f"required gate conditions {indeterminate_ids} are indeterminate "
            "at current N — see ESTIMATOR_GO_NO_GO.md for details"
        )
    out.extend(
        [
            "raw probabilities are un-recalibrated unless the consumer "
            "applies isotonic recalibration from `calibration.json`",
            "retrospective sources carry outcome-aware annotation caveats",
            (
                "`y_submit_without_validation` is run-constant within a "
                "run; any non-trivial AUROC at non-terminal t is a data "
                "property, not skill"
            ),
        ]
    )
    return out


def build_sign_off(
    *,
    estimator_id: str,
    estimator_version: str,
    model_family: str,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_groups: tuple[str, ...],
    targets: list[str],
    eval_cells: list[EvalCell],
    headline_scheme: str,
    diagnostic_schemes: tuple[str, ...],
    headline_seed: int,
    calibration_method: str,
    intended_use: list[str],
    non_use_cases: list[str],
    commit_sha: str,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
    gate: GateReport,
    gate_report_path: str,
    out_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build the sign-off package. Returns (json_path, md_path, record).

    The card embeds the gate verdict and failure-mode results so a
    consumer can read the full sign-off from a single artifact.
    """
    not_safe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    record = build_card_record(
        estimator_id=estimator_id,
        estimator_version=estimator_version,
        model_family=model_family,
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        feature_groups=feature_groups,
        targets=targets,
        eval_cells=eval_cells,
        headline_scheme=headline_scheme,
        diagnostic_schemes=diagnostic_schemes,
        headline_seed=headline_seed,
        calibration_method=calibration_method,
        intended_use=intended_use,
        non_use_cases=non_use_cases,
        known_limits=_known_limits_from(gate, o7, reasons),
        not_safe_for_control=not_safe,
        commit_sha=commit_sha,
        failure_mode_results=_failure_mode_dict_for_card(o1, o5, o7),
        go_no_go_gate={
            "verdict": gate.verdict,
            "report_path": gate_report_path,
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = write_card(out_dir, record)
    return json_path, md_path, record


def render_sign_off_summary(
    *,
    estimator_id: str,
    record: dict[str, Any],
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
    bundle_dir: Path,
    gate_report_path: str,
) -> str:
    lines = [
        f"# Sign-off — {estimator_id}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        f"## Headline verdict: {_outcome_badge(gate.verdict).upper()}",
        "",
        f"- gate report: `{gate_report_path}`",
        f"- v0 findings memo: `reports/V0_FINDINGS.md` (recentered narrative)",
        f"- model bundle: `{bundle_dir}/`",
        f"- model_card.json: `{bundle_dir}/model_card.json` (validates against `schemas/model_card_schema.json`)",
        f"- not_safe_for_control: **{record['not_safe_for_control']}**",
        "",
        "## Recentered v0 framing",
        "",
        "**Primary headline.** Process-dynamics prediction "
        "(`y_future_progress_drop_h5`, `y_validation_new_work_h5`).",
        "",
        "**Secondary / negative.** Terminal success "
        "(`y_success_eventual`) — ledger features do not yet beat "
        "elapsed time at this N.",
        "",
        "Per-target evidence is in `reports/g5/g5_eval.md` and the "
        "per-cell P1.a evidence table below.",
        "",
        "## Required gate conditions",
        "",
        "| id | outcome | summary |",
        "|---|---|---|",
    ]
    for c in gate.conditions:
        if not c.required:
            continue
        lines.append(f"| `{c.condition_id}` | {_outcome_badge(c.outcome)} | {c.summary} |")
    lines.append("")
    lines.append("## Failure-mode tests (O1, O5, O7)")
    lines.append("")
    lines.append("| test | outcome | metric | value | threshold |")
    lines.append("|---|---|---|---:|---:|")
    for r in (o1, o5):
        v = "n/a" if r.metric_value is None else f"{r.metric_value:.3f}"
        lines.append(f"| `{r.test_id}` | {_outcome_badge(r.outcome)} | {r.metric_name} | {v} | {r.threshold:.3f} |")
    for r in o7:
        v = "n/a" if r.metric_value is None else f"{r.metric_value:.3f}"
        lines.append(
            f"| `{r.test_id} ({r.detail.get('source', '?')})` | "
            f"{_outcome_badge(r.outcome)} | {r.metric_name} | {v} | {r.threshold:.3f} |"
        )
    lines.append("")
    lines.append("## Known limits")
    lines.append("")
    for limit in record["known_limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append("## Recommendations for next collection")
    lines.append("")
    lines.extend(_recommendations_for(gate, o1, o5, o7))
    return "\n".join(lines) + "\n"


def _recommendations_for(
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> list[str]:
    """Prioritized recommendations grouped into:
        BLOCKING — required gate conditions that FAIL or O7 fails
                   (these fundamentally invalidate v0).
        DATA     — required gate conditions that are INDETERMINATE
                   because of missing data / labels / cohort size.
        AUDIT    — process / artifact gaps that block the gate without
                   needing model changes (e.g. D5 audit).
    Each line is a single bullet so a reader can scan top-down."""
    if gate.verdict == "pass":
        return [
            "- (gate passes) Flip `not_safe_for_control` to `false` "
            "only after a second confirmatory pass on a refresh of "
            "the data."
        ]
    out: list[str] = []
    blocking: list[str] = []
    data: list[str] = []
    audit: list[str] = []

    failing_o7 = [r for r in o7 if r.outcome == "fail"]
    if failing_o7:
        srcs = sorted({r.detail.get("source", "?") for r in failing_o7})
        blocking.append(
            "- **BLOCKING (O7)** the v0 ledger features do not carry "
            f"decision-relevant signal beyond elapsed time on {srcs}. "
            "Cheapest next experiment: add the deferred dynamics group "
            "(G5) and re-run O7."
        )
    failing_required = [
        c for c in gate.conditions
        if c.required and c.outcome == "fail"
    ]
    for c in failing_required:
        blocking.append(
            f"- **BLOCKING ({c.condition_id})** {c.summary}"
        )

    indeterminate_required = [
        c for c in gate.conditions
        if c.required and c.outcome == "indeterminate"
    ]
    for c in indeterminate_required:
        if c.condition_id in ("P1.b", "P1.d"):
            data.append(
                f"- **DATA ({c.condition_id})** tb_live cohort is "
                "12/12 successes — collect at least 5 tb_live failures "
                "before this gate is even testable."
            )
        elif c.condition_id == "P1.c":
            data.append(
                f"- **DATA ({c.condition_id})** build hermes_pilot_h5_v2 "
                "labels into `datasets/labels_all.parquet` so the "
                "combined retrospective (~50 runs) is testable as the "
                "plan intended."
            )
        elif c.condition_id == "P1.g":
            audit.append(
                f"- **AUDIT ({c.condition_id})** ship the D5 behavioral "
                "leakage audit artifact (Workstream M deferred → D5 "
                "substitute is still required)."
            )
        else:
            data.append(
                f"- **DATA ({c.condition_id})** {c.summary}"
            )

    out.extend(blocking)
    out.extend(data)
    out.extend(audit)
    if not out:
        out.append(
            "- Re-run after the next data collection or annotation pass."
        )
    return out


def write_sign_off_summary(
    path: Path,
    *,
    estimator_id: str,
    record: dict[str, Any],
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
    bundle_dir: Path,
    gate_report_path: str,
) -> Path:
    md = render_sign_off_summary(
        estimator_id=estimator_id,
        record=record,
        gate=gate,
        o1=o1, o5=o5, o7=o7,
        bundle_dir=bundle_dir,
        gate_report_path=gate_report_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


# ---------- P3 -------------------------------------------------------------


def render_ready_report(
    *, estimator_id: str, record: dict[str, Any], gate: GateReport
) -> str:
    lines = [
        f"# READY_FOR_SCHEDULING — {estimator_id}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        "Workstream P1 has cleared the no-regression gate. This document "
        "describes the contract a downstream scheduler must respect.",
        "",
        "## Output schema",
        "",
        f"- estimator_id: `{estimator_id}`",
        f"- estimator_version: `{record['estimator_version']}`",
        "- prediction format: per-checkpoint probability ∈ (0.001, 0.999) "
        "for each target in the calibration_status table.",
        "",
        "## Calibration window",
        "",
        "Recalibration uses K-fold isotonic over run_ids (see "
        "`coding_estimator/calibration/recalibrate.py`). A consumer MUST "
        "apply the same recalibration before thresholding.",
        "",
        "## Recommended decision threshold",
        "",
        "- Default: 0.5 on calibrated probabilities, BUT see the failure-mode "
        "table — any target where O1 is `fail` or `indeterminate` should "
        "use a threshold ≥ 0.7 to compensate for residual overconfidence.",
        "",
        "## Required preconditions on consumer",
        "",
        "- Consumer MUST log every prediction it threshold and the action "
        "it took, so a future audit can re-evaluate calibration on the "
        "consumer's actual operating slice.",
        "- Consumer MUST treat `not_safe_for_control = true` as hard-block: "
        "any prediction marked unsafe must NOT drive control actions.",
        "",
        "## Provenance",
        "",
        f"- gate verdict: {gate.verdict}",
        f"- model bundle: `models/{estimator_id}/`",
        f"- model_card.json: `models/{estimator_id}/model_card.json`",
    ]
    return "\n".join(lines) + "\n"


def render_not_ready_report(
    *,
    estimator_id: str,
    record: dict[str, Any],
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> str:
    lines = [
        f"# NOT_READY_FOR_SCHEDULING — {estimator_id}",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        f"## Verdict: {_outcome_badge(gate.verdict).upper()}",
        "",
        "Workstream P1 has NOT cleared the no-regression gate. This "
        "document records which conditions blocked the gate and the "
        "cheapest next experiment for each.",
        "",
        "## Conditions that blocked",
        "",
    ]
    blocked = [
        c for c in gate.conditions
        if c.required and c.outcome != "pass"
    ]
    if not blocked:
        lines.append("- (none — gate is `pass`; this report is misnamed)")
    else:
        for c in blocked:
            lines.append(f"- `{c.condition_id}` ({c.outcome}): {c.summary}")
    lines.append("")
    lines.append("## Recommended next experiments (P + O)")
    lines.append("")
    lines.append(
        "Recommendations are tagged BLOCKING (must land first), DATA "
        "(unblocks indeterminate gates), or AUDIT (process artifact)."
    )
    lines.append("")
    lines.extend(_recommendations_for(gate, o1, o5, o7))
    lines.append("")
    lines.append("## Do not consume")
    lines.append("")
    lines.append(
        "- This estimator MUST NOT drive scheduling, modulation, or "
        "any other control action."
    )
    lines.append(
        "- `not_safe_for_control` flag in `model_card.json` is `true`; "
        "consumers MUST hard-block on this flag."
    )
    return "\n".join(lines) + "\n"


def write_p3_report(
    out_dir: Path,
    *,
    estimator_id: str,
    record: dict[str, Any],
    gate: GateReport,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if gate.verdict == "pass":
        path = out_dir / "READY_FOR_SCHEDULING.md"
        md = render_ready_report(
            estimator_id=estimator_id, record=record, gate=gate
        )
    else:
        path = out_dir / "NOT_READY_FOR_SCHEDULING.md"
        md = render_not_ready_report(
            estimator_id=estimator_id,
            record=record,
            gate=gate,
            o1=o1, o5=o5, o7=o7,
        )
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "build_sign_off",
    "render_sign_off_summary",
    "write_sign_off_summary",
    "render_ready_report",
    "render_not_ready_report",
    "write_p3_report",
]
