from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from .audits import corpus_hardening_report, prefix_safety_report
from .estimator_artifacts import validate_estimator_artifacts
from .observation import read_jsonl
from .observation_quality import corpus_observation_quality_report


def pilot_gate_report(
    run_dirs: list[Path],
    *,
    estimator_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    all_run_metrics = [_run_gate_metrics(run_dir) for run_dir in run_dirs]
    eligible_run_dirs = [run_dir for run_dir, metrics in zip(run_dirs, all_run_metrics) if metrics["eligible_for_L_gate"]]
    observation_quality = corpus_observation_quality_report(eligible_run_dirs)
    hardening = corpus_hardening_report(eligible_run_dirs)
    estimator = (
        validate_estimator_artifacts(estimator_artifact_dir)
        if estimator_artifact_dir is not None
        else {"passed": False, "prefix_provenance_complete": False, "issues": ["estimator artifact dir not provided"]}
    )
    estimator_alignment = (
        _estimator_alignment_report(eligible_run_dirs, estimator_artifact_dir)
        if estimator_artifact_dir is not None
        else {"passed": False, "issues": ["estimator artifact dir not provided"]}
    )
    run_metrics = [metric for metric in all_run_metrics if metric["eligible_for_L_gate"]]
    run_count = len(run_metrics)
    terminal_labels = [
        metric["final_success"]
        for metric in run_metrics
        if isinstance(metric["final_success"], bool)
    ]
    failure_rate = (
        sum(label is False for label in terminal_labels) / len(terminal_labels)
        if terminal_labels
        else 0.0
    )
    transcript_steps = [metric["max_transcript_step"] for metric in run_metrics]
    validation_attempt_runs = sum(metric["validation_attempt_count"] > 0 for metric in run_metrics)
    validation_fail_runs = sum(metric["validation_fail_observed_count"] > 0 for metric in run_metrics)
    validation_disagreement_runs = sum(metric["validation_disagreement"] for metric in run_metrics)
    progress_drop_runs = sum(metric["has_progress_drop"] for metric in run_metrics)
    high_progress_failures = sum(metric["is_high_progress_failure"] for metric in run_metrics)
    verifier_disagreements = sum(metric["verifier_disagreement_count"] > 0 for metric in run_metrics)
    determinism = _determinism_gate(eligible_run_dirs)

    gate_inputs = {
        "total_run_count": len(all_run_metrics),
        "eligible_run_count": run_count,
        "excluded_protocol_smoke_run_count": sum(not metric["eligible_for_L_gate"] for metric in all_run_metrics),
        "run_count": run_count,
        "median_transcript_steps": median(transcript_steps) if transcript_steps else 0,
        "validation_attempt_run_fraction": _ratio(validation_attempt_runs, run_count),
        "validation_fail_observed_run_fraction": _ratio(validation_fail_runs, run_count),
        "validation_disagreement_run_fraction": _ratio(validation_disagreement_runs, run_count),
        "progress_drop_run_fraction": _ratio(progress_drop_runs, run_count),
        "terminal_failure_rate": failure_rate,
        "high_progress_failure_or_disagreement_count": high_progress_failures + verifier_disagreements,
        "median_observation_events_per_run": observation_quality["median_observation_events_per_run"],
        "shell_exit_code_coverage": observation_quality["shell_exit_code_coverage"],
        "shell_stdout_snippet_coverage": observation_quality["shell_stdout_snippet_coverage"],
        "shell_stderr_snippet_coverage": observation_quality["shell_stderr_snippet_coverage"],
        "prefix_provenance_present": bool(estimator.get("prefix_provenance_complete")),
        "leakage_incidents": hardening["redaction"]["leakage_incidents"],
        "verifier_determinism_passed": determinism["passed"],
    }
    gates = {
        "real_agent_pilot_runs_present": gate_inputs["eligible_run_count"] > 0,
        "median_transcript_steps": gate_inputs["median_transcript_steps"] >= 15,
        "validation_attempt_coverage": gate_inputs["validation_attempt_run_fraction"] >= 0.50,
        "validation_fail_observed_coverage": gate_inputs["validation_fail_observed_run_fraction"] >= 0.25,
        "validation_disagreement_coverage": gate_inputs["validation_disagreement_run_fraction"] >= 0.25,
        "progress_drop_coverage": gate_inputs["progress_drop_run_fraction"] >= 0.20,
        "terminal_failure_rate": 0.25 <= gate_inputs["terminal_failure_rate"] <= 0.70,
        "high_progress_failures_or_disagreements": gate_inputs["high_progress_failure_or_disagreement_count"] >= 5,
        "median_observation_events_per_run": gate_inputs["median_observation_events_per_run"] >= 10,
        "shell_exit_code_coverage": gate_inputs["shell_exit_code_coverage"] >= 0.95,
        "shell_stdout_stderr_snippet_coverage": (
            gate_inputs["shell_stdout_snippet_coverage"] >= 0.80
            and gate_inputs["shell_stderr_snippet_coverage"] >= 0.80
        ),
        "prefix_provenance_present": gate_inputs["prefix_provenance_present"],
        "zero_leakage_incidents": gate_inputs["leakage_incidents"] == 0,
        "verifier_outcomes_reproducible": gate_inputs["verifier_determinism_passed"],
        "artifact_hardening": hardening["passed"],
        "estimator_artifacts": bool(estimator.get("passed")),
        "estimator_prefix_safety_and_alignment": estimator_alignment["passed"],
    }
    return {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "eligible_run_dirs": [str(run_dir) for run_dir in eligible_run_dirs],
        "gate_inputs": gate_inputs,
        "gates": gates,
        "passed": all(gates.values()),
        "run_metrics": run_metrics,
        "all_run_metrics": all_run_metrics,
        "observation_quality": observation_quality,
        "hardening": hardening,
        "estimator_artifacts": estimator,
        "estimator_alignment": estimator_alignment,
        "verifier_determinism": determinism,
    }


def failure_analysis(report: dict[str, Any]) -> str:
    failed = [name for name, passed in report["gates"].items() if not passed]
    lines = [
        "# Pilot Failure Analysis",
        "",
        "Scale batch is blocked because at least one pilot gate failed.",
        "",
        "## Failed Gates",
        "",
    ]
    for name in failed:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Recommended Action", ""])
    if "real_agent_pilot_runs_present" in failed:
        lines.append("- Run typed `model_tool_loop` arms; protocol-smoke shell runs are excluded from L metrics.")
    if "median_transcript_steps" in failed or "median_observation_events_per_run" in failed:
        lines.append("- Resample toward longer tasks or improve live transcript capture before K.")
    if "validation_attempt_coverage" in failed or "validation_fail_observed_coverage" in failed:
        lines.append("- Prefer tasks with visible test/validation loops and richer verifier feedback.")
    if "validation_disagreement_coverage" in failed:
        lines.append("- Add tasks where apparent closure can be contrasted with terminal verifier failure.")
    if "progress_drop_coverage" in failed or "high_progress_failures_or_disagreements" in failed:
        lines.append("- Add tasks likely to expose late failures after apparent progress.")
    if "zero_leakage_incidents" in failed:
        lines.append("- Quarantine leaking runs and fix task extraction/redaction before training.")
    if (
        "prefix_provenance_present" in failed
        or "estimator_artifacts" in failed
        or "estimator_prefix_safety_and_alignment" in failed
    ):
        lines.append("- Rebuild estimator artifacts and require complete prefix provenance/alignment.")
    if "verifier_outcomes_reproducible" in failed:
        lines.append("- Rerun verifier determinism on a broader sample before scaling.")
    lines.extend(["", "Do not run Workstream M until all gates pass.", ""])
    return "\n".join(lines)


def write_pilot_gate_outputs(
    *,
    report: dict[str, Any],
    report_path: Path,
    failure_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(failure_analysis(report), encoding="utf-8")


def _run_gate_metrics(run_dir: Path) -> dict[str, Any]:
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    manifest = _read_json(run_dir / "run_manifest.json")
    progress = _read_progress(run_dir / "progress_by_category.csv")
    event_counts = {
        event_type: sum(event.get("event_type") == event_type for event in observations)
        for event_type in {
            "validation_attempt",
            "validation_fail_observed",
            "agent_claims_done",
            "verifier_disagreement",
        }
    }
    max_progress = max(progress, default=0.0)
    has_progress_drop = any(curr < prev for prev, curr in zip(progress, progress[1:]))
    final_success = manifest.get("final_success")
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    eligible_for_l_gate = bool(metrics.get("eligible_for_L_gate"))
    return {
        "run_dir": str(run_dir),
        "agent_backend": metrics.get("agent_backend", "unknown"),
        "pilot_type": metrics.get("pilot_type", "unknown"),
        "eligible_for_L_gate": eligible_for_l_gate,
        "max_transcript_step": max((int(row.get("step", 0)) for row in transcript), default=0),
        "observation_event_count": len(observations),
        "validation_attempt_count": event_counts["validation_attempt"],
        "validation_fail_observed_count": event_counts["validation_fail_observed"],
        "agent_claims_done_count": event_counts["agent_claims_done"],
        "verifier_disagreement_count": event_counts["verifier_disagreement"],
        "validation_disagreement": (
            final_success is False
            and (event_counts["validation_attempt"] > 0 or event_counts["agent_claims_done"] > 0)
        ),
        "has_progress_drop": has_progress_drop,
        "max_coding_progress": max_progress,
        "final_success": final_success,
        "is_high_progress_failure": final_success is False and max_progress >= 0.5,
    }


def _determinism_gate(run_dirs: list[Path]) -> dict[str, Any]:
    reports = []
    for run_dir in run_dirs:
        path = run_dir / "verifier_determinism_report.json"
        if path.is_file():
            payload = _read_json(path)
            reports.append(
                {
                    "run_dir": str(run_dir),
                    "path": str(path),
                    "passed": bool(payload.get("passed") or payload.get("deterministic")),
                }
            )
    return {
        "sample_count": len(reports),
        "passed": bool(reports) and all(report["passed"] for report in reports),
        "reports": reports,
    }


def _estimator_alignment_report(run_dirs: list[Path], artifact_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    checkpoint_path = artifact_dir / "checkpoints.parquet"
    if not checkpoint_path.is_file():
        return {"passed": False, "issues": ["checkpoints.parquet: missing"]}
    checkpoints = pd.read_parquet(checkpoint_path)
    required = {"run_id", "checkpoint_id", "checkpoint_step", "max_observation_step_used", "max_ledger_step_used"}
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        return {"passed": False, "issues": [f"checkpoints.parquet: missing columns {missing}"]}

    run_by_id = {run_dir.name: run_dir for run_dir in run_dirs}
    prefix_reports = []
    for run_id, group in checkpoints.groupby("run_id", sort=True):
        run_dir = run_by_id.get(str(run_id))
        if run_dir is None:
            issues.append(f"checkpoints.parquet: run_id {run_id} has no matching run directory")
            continue
        report = prefix_safety_report(
            group.to_dict(orient="records"),
            read_jsonl(run_dir / "observation_events.jsonl"),
        )
        prefix_reports.append({"run_id": str(run_id), **report})
        if not report["passed"]:
            issues.append(f"checkpoints.parquet: prefix safety failed for {run_id}")

    checkpoint_ids = set(checkpoints["checkpoint_id"].astype(str))
    for artifact in ("labels.parquet", "estimator_predictions.parquet"):
        path = artifact_dir / artifact
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if "checkpoint_id" not in frame.columns:
            issues.append(f"{artifact}: missing checkpoint_id")
            continue
        unknown = sorted(set(frame["checkpoint_id"].astype(str)) - checkpoint_ids)
        if unknown:
            issues.append(f"{artifact}: {len(unknown)} checkpoint_id values are not in checkpoints.parquet")

    return {
        "passed": not issues,
        "issues": issues,
        "prefix_reports": prefix_reports,
    }


def _read_progress(path: Path) -> list[float]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [
            float(row.get("coding_progress") or row.get("progress") or 0.0)
            for row in csv.DictReader(file)
        ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
