from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from coding_data_collection.observation import read_jsonl


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    task_id: str
    arm: str
    run_dir: Path
    status: str
    final_success: bool | None
    eligible: bool
    transcript_steps: int
    observation_events: int
    validation_attempts: int
    validation_failures: int
    verifier_disagreements: int
    product_events: int
    progress_drop: bool
    max_coding_progress: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare V10 targeted pilot audits and candidate scoring from V9.")
    parser.add_argument("--v9-root", type=Path, default=Path("runs/terminal_bench_real_pilot_gpt54_vs_mini_v9"))
    parser.add_argument("--v9-gate-report", type=Path, default=Path("reports/TERMINAL_BENCH_REAL_PILOT_V9_GATE_REPORT.json"))
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("manifests/pilots/terminal_bench_candidate_scores.csv"),
    )
    parser.add_argument(
        "--out-candidates",
        type=Path,
        default=Path("manifests/pilots/terminal_bench_v10_candidate_scores.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    summaries = load_run_summaries(args.v9_root, args.v9_gate_report)
    candidates = load_candidate_rows(args.candidate_scores)
    task_groups = group_by_task(summaries)

    validation_report = validation_attempt_audit(summaries)
    ledger_report = ledger_discovery_audit(summaries, args.v9_gate_report)
    setup_report = setup_failure_triage(task_groups)
    scored = score_candidates(candidates, task_groups)
    selection_report = task_selection_report(scored)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_text(args.reports_dir / "V9_VALIDATION_ATTEMPT_AUDIT.md", validation_report)
    write_text(args.reports_dir / "V9_LEDGER_DISCOVERY_AUDIT.md", ledger_report)
    write_text(args.reports_dir / "V9_SETUP_FAILURE_TRIAGE.md", setup_report)
    write_candidate_csv(args.out_candidates, scored)
    write_text(args.reports_dir / "TERMINAL_BENCH_V10_TASK_SELECTION.md", selection_report)
    return 0


def load_run_summaries(v9_root: Path, gate_report: Path) -> list[RunSummary]:
    gate = json.loads(gate_report.read_text(encoding="utf-8")) if gate_report.is_file() else {}
    gate_metrics = {
        Path(row["run_dir"]).name: row
        for row in gate.get("all_run_metrics", [])
        if isinstance(row, dict) and row.get("run_dir")
    }
    summaries: list[RunSummary] = []
    for run_dir in sorted(v9_root.iterdir()):
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = manifest.get("metrics", {})
        transcript = read_jsonl(run_dir / "transcript.jsonl")
        observations = read_jsonl(run_dir / "observation_events.jsonl")
        event_counts = Counter(event.get("event_type") for event in observations)
        gate_row = gate_metrics.get(run_dir.name, {})
        task_id, arm = split_run_id(run_dir.name)
        summaries.append(
            RunSummary(
                run_id=run_dir.name,
                task_id=task_id,
                arm=arm,
                run_dir=run_dir,
                status=str(manifest.get("run_status")),
                final_success=manifest.get("final_success"),
                eligible=bool(metrics.get("eligible_for_L_gate")),
                transcript_steps=max((int(row.get("step", 0)) for row in transcript), default=0),
                observation_events=len(observations),
                validation_attempts=int(event_counts["validation_attempt"]),
                validation_failures=int(event_counts["validation_fail_observed"]),
                verifier_disagreements=int(event_counts["verifier_disagreement"]),
                product_events=sum(
                    event_counts[name]
                    for name in ("product_file_written", "product_file_edited")
                ),
                progress_drop=bool(gate_row.get("has_progress_drop")),
                max_coding_progress=float(gate_row.get("max_coding_progress") or 0.0),
            )
        )
    return summaries


def split_run_id(run_id: str) -> tuple[str, str]:
    if "__" not in run_id:
        return run_id, ""
    task_id, arm = run_id.rsplit("__", 1)
    return task_id, arm


def group_by_task(summaries: list[RunSummary]) -> dict[str, list[RunSummary]]:
    groups: dict[str, list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        groups[summary.task_id].append(summary)
    return dict(groups)


def load_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validation_attempt_audit(summaries: list[RunSummary]) -> str:
    eligible = [summary for summary in summaries if summary.eligible]
    validation_fail_runs = sum(s.validation_failures > 0 for s in eligible)
    lines = [
        "# V9 Validation Attempt Audit",
        "",
        "## Summary",
        "",
        f"L-eligible runs audited: {len(eligible)}",
        f"Runs with validation_attempt: {sum(s.validation_attempts > 0 for s in eligible)}",
        f"Runs with validation_fail_observed: {validation_fail_runs}",
        "",
        (
            "Finding: visible validation failure remains too sparse in V9. "
            "A small number of failed visible checks is now recognized, but most terminal "
            "failures are still hidden-verifier failures, blocked data/dependency cases, "
            "or semantic mismatches after visible smoke checks passed."
            if validation_fail_runs
            else "Finding: failed visible validation is absent in V9, not merely unrecognized. "
            "The recognized validation attempts all have successful shell exits; remaining "
            "terminal failures are hidden-verifier failures, blocked data/dependency cases, "
            "or semantic mismatches after visible smoke checks passed."
        ),
        "",
        "## Runs",
        "",
        "| run_id | status | final_success | validation_attempts | validation_failures | verifier_disagreements | note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in eligible:
        if summary.validation_attempts and not summary.validation_failures and summary.final_success is False:
            note = "visible validation passed or was non-failing; hidden verifier failed"
        elif summary.validation_attempts:
            note = "visible validation attempt observed"
        elif summary.final_success is False:
            note = "no visible validation attempt; terminal failure came from verifier"
        else:
            note = "no visible validation failure signal"
        lines.append(
            "| {run_id} | {status} | {success} | {attempts} | {failures} | {disagreements} | {note} |".format(
                run_id=summary.run_id,
                status=summary.status,
                success=summary.final_success,
                attempts=summary.validation_attempts,
                failures=summary.validation_failures,
                disagreements=summary.verifier_disagreements,
                note=note,
            )
        )
    lines.extend(
        [
            "",
            "## Implication For V10",
            "",
            "V10 needs tasks with visible tests or smoke checks that can fail before the hidden verifier. "
            "If a task only allows hidden-verifier disagreement, it helps the high-progress/disagreement gate "
            "but will not satisfy `validation_fail_observed`.",
            "",
        ]
    )
    return "\n".join(lines)


def ledger_discovery_audit(summaries: list[RunSummary], gate_report: Path) -> str:
    gate = json.loads(gate_report.read_text(encoding="utf-8")) if gate_report.is_file() else {}
    eligible = [summary for summary in summaries if summary.eligible]
    progress_drops = sum(summary.progress_drop for summary in eligible)
    max_progress = sorted((summary.max_coding_progress for summary in eligible), reverse=True)[:5]
    progress_drop_fraction = gate.get("gate_inputs", {}).get("progress_drop_run_fraction")
    if progress_drops:
        finding = (
            "Finding: after the V10 ledger bridge hardening, the sidecar now observes progress drops. "
            "The bridge completes concrete successful tool rows, blocks failed tool/controller rows, "
            "and reopens prior visible work when the final verifier shows it was incomplete. "
            "This preserves ledger-side scoring semantics while exposing process dynamics that V9 "
            "previously hid."
        )
        requirement = (
            "Keep the explicit ledger bridge behavior. The remaining V10 selection issue is not progress-drop "
            "coverage; it is selecting enough tasks with visible validation failures without counting setup "
            "failures as model failures."
        )
    else:
        finding = (
            "Finding: `progress_drop_run_fraction` is zero because the current transcript-to-ledger bridge "
            "mostly creates one inferred subtask per transcript row and leaves those subtasks `in_progress`. "
            "Only explicit `done` boundaries or explicit row-level `ledger_ops` complete work. The sidecar "
            "therefore sees sparse, mostly monotonic completion with no later regression surface."
        )
        requirement = (
            "Do not weaken the progress-drop gate. V10 should either use tasks with natural visible "
            "validation failures after partial fixes or add explicit agent/controller ledger events for "
            "`hypothesis_started`, `visible_check_failed_after_edit`, `rework_started`, and "
            "`previous_fix_invalidated`. That preserves ledger semantics while exposing process dynamics."
        )
    lines = [
        "# V9 Ledger Discovery Audit",
        "",
        "## Summary",
        "",
        f"L-eligible runs audited: {len(eligible)}",
        f"progress_drop runs: {progress_drops}",
        f"progress_drop_run_fraction: {progress_drop_fraction}",
        f"largest max_coding_progress values: {max_progress}",
        "",
        finding,
        "",
        "## Evidence",
        "",
        "| run_id | final_success | transcript_steps | max_coding_progress | progress_drop | observation_events |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for summary in eligible:
        lines.append(
            f"| {summary.run_id} | {summary.final_success} | {summary.transcript_steps} | "
            f"{summary.max_coding_progress:.4f} | {summary.progress_drop} | {summary.observation_events} |"
        )
    lines.extend(
        [
            "",
            "## V10 Requirement",
            "",
            requirement,
            "",
        ]
    )
    return "\n".join(lines)


def setup_failure_triage(task_groups: dict[str, list[RunSummary]]) -> str:
    rows: list[tuple[str, str, str, str]] = []
    for task_id, runs in sorted(task_groups.items()):
        failures = [run for run in runs if run.status == "environment_setup_failure"]
        if not failures:
            continue
        failed_checks = set()
        snippets = []
        for run in failures:
            report_path = run.run_dir / "agent_readiness_preflight.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            failed_checks.update(report.get("failed_checks", []))
            for result in report.get("results", []):
                if not result.get("passed", True):
                    text = " ".join(
                        str(result.get(key) or "")
                        for key in ("stdout_snippet", "stderr_snippet", "reason")
                    ).strip()
                    if text:
                        snippets.append(text.replace("\n", " ")[:160])
        classification, recommendation = classify_setup_failure(failed_checks, snippets)
        rows.append((task_id, ", ".join(sorted(failed_checks)), classification, recommendation))
    lines = [
        "# V9 Setup Failure Triage",
        "",
        "Environment setup failures are excluded from model outcome metrics and should not be counted as model failures.",
        "",
        "| task_id | failed_checks | class | V10 decision |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "V10 excludes tasks with readable hidden image artifacts until the image is rebuilt or the "
            "hidden path is proven inaccessible from the agent container. Dependency-only failures may be "
            "reintroduced after an explicit image/dependency prebuild smoke passes.",
            "",
        ]
    )
    return "\n".join(lines)


def classify_setup_failure(failed_checks: set[str], snippets: list[str]) -> tuple[str, str]:
    if "hidden_image_artifacts_unreadable" in failed_checks:
        return (
            "hidden_artifact_leakage_risk",
            "exclude from V10 until image rebuild removes or protects hidden paths",
        )
    if "nginx_available" in failed_checks:
        return (
            "dependency_prebuild_required",
            "fixable by image prebuild, but exclude until nginx preflight passes",
        )
    if "python_imports_available" in failed_checks:
        return (
            "dependency_prebuild_required",
            "fixable by image prebuild, but exclude until required imports pass",
        )
    if "r_runtime_available" in failed_checks:
        return (
            "runtime_prebuild_required",
            "fixable only after R/runtime image prebuild and hidden path audit",
        )
    return ("unclear_setup_failure", "exclude pending manual harness review")


def score_candidates(candidates: list[dict[str, str]], task_groups: dict[str, list[RunSummary]]) -> list[dict[str, Any]]:
    scored = []
    for row in candidates:
        task_id = row["task_id"]
        runs = task_groups.get(task_id, [])
        env_score = environment_compatibility_score(row, runs)
        process_score = process_richness_score(row, runs)
        visible_fail = visible_validation_failure_likelihood(row, runs)
        hidden_disagreement = hidden_verifier_disagreement_likelihood(row, runs)
        base_priority = int(float(row.get("pilot_priority") or 0))
        v10_priority = base_priority + env_score + process_score + visible_fail + hidden_disagreement
        setup_status = setup_status_for_task(runs)
        selected = should_select_v10(task_id, setup_status, env_score, process_score, hidden_disagreement, visible_fail)
        prepilot = task_id in {"csv-to-parquet", "grid-pattern-transform", "attention-mil", "broken-python"}
        scored.append(
            {
                **row,
                "environment_compatibility_score": env_score,
                "process_richness_score": process_score,
                "visible_validation_failure_likelihood": visible_fail,
                "hidden_verifier_disagreement_likelihood": hidden_disagreement,
                "v10_priority": v10_priority,
                "v10_setup_status": setup_status,
                "v10_selected": selected,
                "v10_prepilot_selected": prepilot and selected,
                "v10_selection_reason": selection_reason(task_id, setup_status, process_score, visible_fail, hidden_disagreement, selected, prepilot),
            }
        )
    return sorted(scored, key=lambda row: (not row["v10_selected"], -int(row["v10_priority"]), row["task_id"]))


def environment_compatibility_score(row: dict[str, str], runs: list[RunSummary]) -> int:
    if any(run.status == "environment_setup_failure" for run in runs):
        return 0
    if runs and all(run.eligible for run in runs):
        return 5
    if truthy(row.get("requires_internet")) or truthy(row.get("large_download_or_build")):
        return 2
    if int(float(row.get("docker_feasibility") or 0)) >= 5:
        return 4
    return 3


def process_richness_score(row: dict[str, str], runs: list[RunSummary]) -> int:
    eligible = [run for run in runs if run.eligible]
    if eligible:
        avg_steps = mean(run.transcript_steps for run in eligible)
        avg_obs = mean(run.observation_events for run in eligible)
        product_rate = mean(1 if run.product_events > 0 else 0 for run in eligible)
        score = 0
        if avg_steps >= 25:
            score += 3
        elif avg_steps >= 15:
            score += 2
        elif avg_steps >= 8:
            score += 1
        if avg_obs >= 10:
            score += 3
        elif avg_obs >= 8:
            score += 2
        elif avg_obs >= 5:
            score += 1
        if product_rate >= 0.5:
            score += 1
        return min(score, 5)
    richness = int(float(row.get("trajectory_richness") or 0))
    if richness >= 24:
        return 4
    if richness >= 18:
        return 3
    return 2


def visible_validation_failure_likelihood(row: dict[str, str], runs: list[RunSummary]) -> int:
    if any(run.validation_failures > 0 for run in runs):
        return 5
    tags = row.get("tags", "").lower()
    notes = row.get("calibration_notes", "").lower()
    if any(token in tags for token in ("pytest", "test", "lint")) or "direct validation" in notes:
        return 4
    if any(run.validation_attempts > 0 and run.final_success is False for run in runs):
        return 3
    if any(run.validation_attempts > 0 for run in runs):
        return 2
    return 1


def hidden_verifier_disagreement_likelihood(row: dict[str, str], runs: list[RunSummary]) -> int:
    eligible = [run for run in runs if run.eligible]
    if eligible and any(run.verifier_disagreements > 0 for run in eligible):
        return 5
    if eligible and any(run.final_success is False and run.validation_attempts > 0 for run in eligible):
        return 4
    if eligible and any(run.final_success is False for run in eligible):
        return 3
    if row.get("difficulty", "").lower() in {"medium", "hard"}:
        return 2
    return 1


def setup_status_for_task(runs: list[RunSummary]) -> str:
    if any(run.status == "environment_setup_failure" for run in runs):
        checks = set()
        for run in runs:
            report_path = run.run_dir / "agent_readiness_preflight.json"
            if report_path.is_file():
                checks.update(json.loads(report_path.read_text(encoding="utf-8")).get("failed_checks", []))
        if "hidden_image_artifacts_unreadable" in checks:
            return "exclude_hidden_artifact_risk"
        return "exclude_prebuild_required"
    if runs:
        return "v9_compatible"
    return "reserve_unproven"


def should_select_v10(
    task_id: str,
    setup_status: str,
    env_score: int,
    process_score: int,
    hidden_disagreement: int,
    visible_fail: int,
) -> bool:
    if setup_status.startswith("exclude_"):
        return False
    if task_id.startswith("terminal-bench/"):
        return False
    if task_id == "count-dataset-tokens":
        return False
    return env_score >= 4 and (process_score >= 2 or hidden_disagreement >= 3 or visible_fail >= 3)


def selection_reason(
    task_id: str,
    setup_status: str,
    process_score: int,
    visible_fail: int,
    hidden_disagreement: int,
    selected: bool,
    prepilot: bool,
) -> str:
    if not selected:
        if setup_status.startswith("exclude_"):
            return setup_status
        if task_id == "count-dataset-tokens":
            return "exclude_solve_time_data_or_network_mismatch"
        return "reserve_not_targeted_for_v10"
    reasons = []
    if prepilot:
        reasons.append("prepilot")
    if process_score >= 3:
        reasons.append("process_rich")
    if visible_fail >= 3:
        reasons.append("visible_validation_candidate")
    if hidden_disagreement >= 3:
        reasons.append("hidden_disagreement_candidate")
    return "|".join(reasons) or "selected"


def task_selection_report(scored: list[dict[str, Any]]) -> str:
    selected = [row for row in scored if row["v10_selected"]]
    prepilot = [row for row in scored if row["v10_prepilot_selected"]]
    excluded = [row for row in scored if str(row["v10_setup_status"]).startswith("exclude_")]
    lines = [
        "# Terminal-Bench V10 Task Selection",
        "",
        "## Decision",
        "",
        f"Selected V10 targeted sample: {len(selected)} tasks x 2 arms.",
        f"Selected V10 pre-pilot: {len(prepilot)} tasks x 2 arms.",
        "",
        "V10 keeps L gates unchanged and does not count environment setup failures as model failures.",
        "",
        "## Pre-Pilot Tasks",
        "",
        "| task_id | reason | env | process | visible_fail | hidden_disagreement |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in prepilot:
        lines.append(score_row(row))
    lines.extend(
        [
            "",
            "## Full Targeted Sample",
            "",
            "| task_id | reason | env | process | visible_fail | hidden_disagreement |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected:
        lines.append(score_row(row))
    lines.extend(
        [
            "",
            "## Excluded Setup-Incompatible Tasks",
            "",
            "| task_id | setup_status | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in excluded:
        lines.append(f"| {row['task_id']} | {row['v10_setup_status']} | {row['v10_selection_reason']} |")
    lines.extend(
        [
            "",
            "## V10 Pre-Pilot Criteria",
            "",
            "Run the 4-task pre-pilot first. Continue to 8-12 tasks only if it improves at least "
            "two of the four failed V9 gate signals: observation density, validation-fail coverage, "
            "progress-drop coverage, and high-progress/disagreement count. Do not run Workstream M.",
            "",
        ]
    )
    return "\n".join(lines)


def score_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['task_id']} | {row['v10_selection_reason']} | "
        f"{row['environment_compatibility_score']} | {row['process_richness_score']} | "
        f"{row['visible_validation_failure_likelihood']} | {row['hidden_verifier_disagreement_likelihood']} |"
    )


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
