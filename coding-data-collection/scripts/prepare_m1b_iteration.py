from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from coding_data_collection.observation import FAIL_RE, VALIDATION_RE, read_jsonl


ARMS = ("gpt54", "gpt53codex", "gpt54mini")
ZERO_ELIGIBLE_EXCLUDE = {
    "classifier-debug",
    "adaptive-rejection-sampler",
    "blind-maze-explorer-algorithm",
    "nginx-request-logging",
}
PREFLIGHT_TARGETS = {
    "broken-python",
    "grid-pattern-transform",
    "attention-mil",
    "csv-to-parquet",
}
CONTROL_TASKS = {
    "extract-safely",
    "fix-permissions",
    "aimo-airline-departures",
}


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    task_id: str
    arm: str
    run_dir: Path
    terminal_success: bool | None
    run_status: str
    termination_reason: str
    eligible: bool
    validation_attempt_count: int
    validation_fail_observed_count: int
    validation_pass_observed_count: int
    agent_claims_done_count: int
    verifier_disagreement_count: int
    visible_check_exists: bool
    validation_like_shell_count: int
    nonzero_validation_like_shell_count: int
    missed_validation_failure_like_shell_count: int
    hidden_verifier_fail_type: str
    no_validation_fail_reason: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit M1 validation signals and prepare M1b task selection.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/terminal_bench_v10_openai_m1"))
    parser.add_argument("--accepted", type=Path, default=Path("reports/terminal_bench_v10_openai_m1_accepted.json"))
    parser.add_argument("--rejected", type=Path, default=Path("reports/terminal_bench_v10_openai_m1_rejected.json"))
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("manifests/pilots/terminal_bench_v10_candidate_scores.csv"),
    )
    parser.add_argument("--audit-out", type=Path, default=Path("reports/M1_VALIDATION_SIGNAL_AUDIT.md"))
    parser.add_argument("--profile-out", type=Path, default=Path("reports/M1_TASK_OUTCOME_PROFILE.md"))
    parser.add_argument(
        "--scores-out",
        type=Path,
        default=Path("manifests/pilots/terminal_bench_m1b_candidate_scores.csv"),
    )
    parser.add_argument("--plan-out", type=Path, default=Path("reports/TERMINAL_BENCH_M1B_TASK_PLAN.md"))
    args = parser.parse_args(argv)

    candidates = _read_candidates(args.candidate_scores)
    accepted = json.loads(args.accepted.read_text(encoding="utf-8")).get("runs", [])
    rejected = json.loads(args.rejected.read_text(encoding="utf-8")).get("runs", [])
    audits = [_audit_run(Path(row["run_dir"]), candidates.get(row["task_id"], {})) for row in accepted]
    task_rows = _task_rows(audits, rejected, candidates)
    scored_candidates = _score_candidates(candidates, task_rows)

    _write_validation_signal_audit(args.audit_out, audits)
    _write_task_profile(args.profile_out, task_rows)
    _write_candidate_scores(args.scores_out, scored_candidates)
    _write_m1b_plan(args.plan_out, scored_candidates, task_rows)
    return 0


def _read_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def _audit_run(run_dir: Path, candidate: dict[str, str]) -> RunAudit:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = manifest.get("metrics", {})
    run_id = str(manifest.get("run_id") or run_dir.name)
    task_id, arm = _split_run_id(run_id)
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    event_counts = Counter(row.get("event_type") for row in observations)
    validation_fail_steps = {
        int(row.get("step", -1))
        for row in observations
        if row.get("event_type") == "validation_fail_observed"
    }
    shell_rows = [row for row in transcript if row.get("kind") == "shell"]
    validation_like_shells = [_shell_summary(row) for row in shell_rows if VALIDATION_RE.search(_shell_text(row))]
    nonzero_validation_like = [
        row
        for row in shell_rows
        if VALIDATION_RE.search(_shell_text(row)) and row.get("exit_code") not in (None, 0)
    ]
    missed_validation_failure_like = [
        row
        for row in nonzero_validation_like
        if int(row.get("step", -1)) not in validation_fail_steps
    ]
    visible_check_exists = _visible_check_exists(run_dir, candidate, bool(validation_like_shells))
    terminal_success = manifest.get("final_success")
    val_attempts = int(event_counts["validation_attempt"])
    val_fails = int(event_counts["validation_fail_observed"])
    val_passes = int(event_counts["validation_pass_observed"])
    disagreements = int(event_counts["verifier_disagreement"])
    hidden_type = _hidden_verifier_fail_type(
        terminal_success=terminal_success,
        validation_attempt_count=val_attempts,
        validation_fail_observed_count=val_fails,
        validation_pass_observed_count=val_passes,
        verifier_disagreement_count=disagreements,
    )
    no_fail_reason = _no_validation_fail_reason(
        terminal_success=terminal_success,
        validation_attempt_count=val_attempts,
        validation_fail_observed_count=val_fails,
        visible_check_exists=visible_check_exists,
        missed_validation_failure_like_shell_count=len(missed_validation_failure_like),
        validation_pass_observed_count=val_passes,
    )
    return RunAudit(
        run_id=run_id,
        task_id=task_id,
        arm=arm,
        run_dir=run_dir,
        terminal_success=terminal_success,
        run_status=str(manifest.get("run_status")),
        termination_reason=str(manifest.get("termination_reason")),
        eligible=bool(metrics.get("eligible_for_L_gate")),
        validation_attempt_count=val_attempts,
        validation_fail_observed_count=val_fails,
        validation_pass_observed_count=val_passes,
        agent_claims_done_count=int(event_counts["agent_claims_done"]),
        verifier_disagreement_count=disagreements,
        visible_check_exists=visible_check_exists,
        validation_like_shell_count=len(validation_like_shells),
        nonzero_validation_like_shell_count=len(nonzero_validation_like),
        missed_validation_failure_like_shell_count=len(missed_validation_failure_like),
        hidden_verifier_fail_type=hidden_type,
        no_validation_fail_reason=no_fail_reason,
    )


def _split_run_id(run_id: str) -> tuple[str, str]:
    if "__" not in run_id:
        return run_id, ""
    return run_id.rsplit("__", 1)


def _shell_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(key) or "")
        for key in ("summary", "command", "stdout_snippet", "stderr_snippet", "obs_snippet")
    )


def _shell_summary(row: dict[str, Any]) -> str:
    command = str(row.get("command") or row.get("summary") or "")
    return f"step={row.get('step')} exit={row.get('exit_code')} command={command[:160]}"


def _visible_check_exists(run_dir: Path, candidate: dict[str, str], agent_ran_check: bool) -> bool:
    if agent_ran_check:
        return True
    if _int(candidate.get("expected_validation_visibility")) >= 4:
        return True
    if _int(candidate.get("visible_validation_failure_likelihood")) >= 4:
        return True
    task_text = ""
    for path in (run_dir / "task.md", run_dir / "agent_workspace" / "task.md"):
        if path.is_file():
            task_text += "\n" + path.read_text(encoding="utf-8", errors="ignore")
    lower = task_text.lower()
    return any(term in lower for term in ("test", "pytest", "check", "validate", "run-tests", "imported by the test"))


def _hidden_verifier_fail_type(
    *,
    terminal_success: bool | None,
    validation_attempt_count: int,
    validation_fail_observed_count: int,
    validation_pass_observed_count: int,
    verifier_disagreement_count: int,
) -> str:
    if terminal_success is not False:
        return "not_terminal_failure"
    if validation_fail_observed_count:
        return "visible_validation_failure"
    if validation_pass_observed_count:
        return "visible_validation_pass_then_hidden_fail"
    if verifier_disagreement_count:
        return "agent_claimed_done_then_hidden_fail"
    if validation_attempt_count:
        return "validation_attempt_then_terminal_fail"
    return "no_visible_validation_attempt"


def _no_validation_fail_reason(
    *,
    terminal_success: bool | None,
    validation_attempt_count: int,
    validation_fail_observed_count: int,
    visible_check_exists: bool,
    missed_validation_failure_like_shell_count: int,
    validation_pass_observed_count: int,
) -> str:
    if validation_fail_observed_count:
        return "visible_validation_failure"
    if missed_validation_failure_like_shell_count:
        return "validation_failure_not_detected"
    if terminal_success is True:
        return "visible_check_passed_terminal_passed" if validation_attempt_count else "terminal_passed_without_visible_check"
    if validation_attempt_count and terminal_success is False:
        return "visible_check_passed_hidden_failed" if validation_pass_observed_count else "validation_attempt_terminal_failed"
    if validation_attempt_count:
        return "visible_check_passed_terminal_passed"
    if terminal_success is False and visible_check_exists:
        return "agent_did_not_run_check"
    if terminal_success is False:
        return "no_visible_validation_route"
    if visible_check_exists:
        return "agent_did_not_run_check"
    return "no_visible_validation_route"


def _task_rows(
    audits: list[RunAudit],
    rejected: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[RunAudit]] = defaultdict(list)
    for audit in audits:
        by_task[audit.task_id].append(audit)
    rejected_by_task = Counter(str(row.get("task_id")) for row in rejected)
    attempted_by_task = Counter()
    for audit in audits:
        attempted_by_task[audit.task_id] += 1
    for row in rejected:
        attempted_by_task[str(row.get("task_id"))] += 1

    for task_id, candidate in candidates.items():
        rows = by_task.get(task_id, [])
        outcomes = Counter(a.no_validation_fail_reason for a in rows)
        hidden_types = Counter(a.hidden_verifier_fail_type for a in rows)
        attempts = [a.validation_attempt_count for a in rows]
        out[task_id] = {
            "task_id": task_id,
            "title": candidate.get("title", ""),
            "category": candidate.get("category", ""),
            "difficulty": candidate.get("difficulty", ""),
            "attempted": attempted_by_task[task_id],
            "eligible": len(rows),
            "rejected": rejected_by_task[task_id],
            "success": sum(a.terminal_success is True for a in rows),
            "terminal_failure": sum(a.terminal_success is False for a in rows),
            "validation_attempt_runs": sum(a.validation_attempt_count > 0 for a in rows),
            "validation_fail_runs": sum(a.validation_fail_observed_count > 0 for a in rows),
            "validation_pass_hidden_fail_runs": sum(
                a.no_validation_fail_reason == "visible_check_passed_hidden_failed" for a in rows
            ),
            "agent_did_not_run_check_failures": sum(
                a.no_validation_fail_reason == "agent_did_not_run_check" for a in rows
            ),
            "verifier_disagreement_runs": sum(a.verifier_disagreement_count > 0 for a in rows),
            "missed_failure_like_shell_runs": sum(a.missed_validation_failure_like_shell_count > 0 for a in rows),
            "median_validation_attempts": median(attempts) if attempts else 0,
            "outcome_reasons": "; ".join(f"{key}:{value}" for key, value in sorted(outcomes.items())),
            "hidden_failure_types": "; ".join(f"{key}:{value}" for key, value in sorted(hidden_types.items())),
        }
    return out


def _score_candidates(
    candidates: dict[str, dict[str, str]],
    task_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = []
    for task_id, candidate in candidates.items():
        row = task_rows[task_id]
        score, reasons = _visible_validation_loop_score(candidate, row)
        setup_risk = _setup_failure_risk(row)
        disagreement = _hidden_disagreement_score(candidate, row)
        selected, role, selection_reason = _m1b_selection(task_id, score, setup_risk, row)
        scored.append(
            {
                **candidate,
                "m1_attempted": row["attempted"],
                "m1_eligible": row["eligible"],
                "m1_rejected": row["rejected"],
                "m1_terminal_success": row["success"],
                "m1_terminal_failure": row["terminal_failure"],
                "m1_validation_attempt_runs": row["validation_attempt_runs"],
                "m1_validation_fail_runs": row["validation_fail_runs"],
                "m1_validation_pass_hidden_fail_runs": row["validation_pass_hidden_fail_runs"],
                "m1_agent_did_not_run_check_failures": row["agent_did_not_run_check_failures"],
                "m1_verifier_disagreement_runs": row["verifier_disagreement_runs"],
                "m1_missed_failure_like_shell_runs": row["missed_failure_like_shell_runs"],
                "visible_validation_loop_score": score,
                "visible_validation_loop_score_reasons": "|".join(reasons),
                "hidden_verifier_disagreement_likelihood_m1": disagreement,
                "setup_failure_risk_m1": setup_risk,
                "m1b_selected": selected,
                "m1b_role": role,
                "m1b_selection_reason": selection_reason,
            }
        )
    return sorted(scored, key=lambda r: (-int(r["m1b_selected"]), -int(r["visible_validation_loop_score"]), r["task_id"]))


def _visible_validation_loop_score(candidate: dict[str, str], row: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if _int(candidate.get("expected_validation_visibility")) >= 4 or _int(candidate.get("visible_validation_failure_likelihood")) >= 4:
        score += 2
        reasons.append("+2_visible_check_obvious")
    if row["validation_fail_runs"] > 0:
        score += 2
        reasons.append("+2_m1_visible_check_failed")
    elif _int(candidate.get("visible_validation_failure_likelihood")) >= 4:
        score += 1
        reasons.append("+1_prior_visible_failure_likely")
    if row["validation_fail_runs"] > 0 or candidate.get("category") in {"python", "debugging", "systems", "model-training"}:
        score += 2
        reasons.append("+2_actionable_failure_likely")
    if row["validation_pass_hidden_fail_runs"] > 0:
        score += 1
        reasons.append("+1_visible_pass_hidden_fail_observed")
    if row["median_validation_attempts"] >= 2 or (row["success"] > 0 and row["terminal_failure"] > 0):
        score += 1
        reasons.append("+1_multiple_attempt_or_mixed_outcome")
    if row["validation_attempt_runs"] == 0 and row["terminal_failure"] > 0:
        score -= 3
        reasons.append("-3_no_visible_validation_route_observed")
    if row["rejected"] >= 3 and row["eligible"] == 0:
        score -= 3
        reasons.append("-3_setup_preflight_failures")
    if row["terminal_failure"] > 0 and row["validation_fail_runs"] == 0 and row["verifier_disagreement_runs"] > 0:
        score -= 2
        reasons.append("-2_hidden_verifier_only_pattern")
    return score, reasons


def _setup_failure_risk(row: dict[str, Any]) -> int:
    if row["attempted"] == 0:
        return 1
    if row["eligible"] == 0 and row["rejected"] > 0:
        return 5
    if row["rejected"] > 0:
        return 3
    return 0


def _hidden_disagreement_score(candidate: dict[str, str], row: dict[str, Any]) -> int:
    base = _int(candidate.get("hidden_verifier_disagreement_likelihood"))
    return max(base, min(5, row["verifier_disagreement_runs"] + row["validation_pass_hidden_fail_runs"]))


def _m1b_selection(task_id: str, score: int, setup_risk: int, row: dict[str, Any]) -> tuple[bool, str, str]:
    if task_id in ZERO_ELIGIBLE_EXCLUDE:
        return False, "excluded", "exclude_no_eligible_m1_setup_failures"
    if setup_risk >= 5:
        return False, "excluded", "exclude_setup_failure_risk"
    if task_id in PREFLIGHT_TARGETS:
        return True, "m1b_preflight_target", "target_visible_validation_loop_or_mixed_failure_signal"
    if task_id in CONTROL_TASKS:
        return True, "m1b_control_or_reserve", "retain_as_clean_or_mixed_control_after_preflight_targets"
    if score >= 4 and row["eligible"] > 0:
        return True, "m1b_candidate", "high_visible_validation_loop_score"
    return False, "not_selected", "not_prioritized_for_m1b_visible_validation_loop"


def _write_validation_signal_audit(path: Path, audits: list[RunAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M1 Validation Signal Audit",
        "",
        "## Summary",
        "",
        f"Eligible runs audited: {len(audits)}",
        f"Runs with validation attempts: {sum(a.validation_attempt_count > 0 for a in audits)}",
        f"Runs with observed validation failures: {sum(a.validation_fail_observed_count > 0 for a in audits)}",
        f"Terminal failures: {sum(a.terminal_success is False for a in audits)}",
        f"Verifier disagreements: {sum(a.verifier_disagreement_count > 0 for a in audits)}",
        f"Runs with validation-like nonzero shells not classified as validation failure: {sum(a.missed_validation_failure_like_shell_count > 0 for a in audits)}",
        "",
        "Finding: M1 mostly lacks visible validation-loop failures. The audit shows this is primarily a task/agent behavior issue rather than a broad extractor miss: terminal failures usually had no visible failed validation, or had visible checks that passed before hidden verifier failure.",
        "",
        "## Run Audit",
        "",
        "| run_id | task_id | arm | terminal_success | validation_attempt_count | validation_fail_observed_count | visible_validation_pass_count | agent_claims_done | verifier_disagreement | hidden_verifier_fail_type | did_visible_check_exist | did_agent_run_visible_check | did_visible_check_fail | if_no_validation_fail_why | missed_failure_like_shells |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for audit in sorted(audits, key=lambda a: (a.task_id, a.arm)):
        lines.append(
            "| {run_id} | {task_id} | {arm} | {terminal_success} | {attempts} | {fails} | {passes} | {done} | {disagree} | {hidden_type} | {check_exists} | {ran_check} | {check_failed} | {why} | {missed} |".format(
                run_id=audit.run_id,
                task_id=audit.task_id,
                arm=audit.arm,
                terminal_success=audit.terminal_success,
                attempts=audit.validation_attempt_count,
                fails=audit.validation_fail_observed_count,
                passes=audit.validation_pass_observed_count,
                done=audit.agent_claims_done_count,
                disagree=audit.verifier_disagreement_count,
                hidden_type=audit.hidden_verifier_fail_type,
                check_exists=str(audit.visible_check_exists).lower(),
                ran_check=str(audit.validation_attempt_count > 0).lower(),
                check_failed=str(audit.validation_fail_observed_count > 0).lower(),
                why=audit.no_validation_fail_reason,
                missed=audit.missed_validation_failure_like_shell_count,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_task_profile(path: Path, task_rows: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# M1 Task Outcome Profile",
        "",
        "| task_id | attempted | eligible | rejected | success | terminal_failure | validation_attempt_runs | validation_fail_runs | validation_pass_hidden_fail_runs | verifier_disagreement_runs | profile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for task_id, row in sorted(task_rows.items()):
        if row["attempted"] == 0:
            continue
        profile = _task_profile_label(row)
        lines.append(
            "| {task_id} | {attempted} | {eligible} | {rejected} | {success} | {terminal_failure} | {attempt_runs} | {fail_runs} | {pass_hidden} | {disagree} | {profile} |".format(
                task_id=task_id,
                attempted=row["attempted"],
                eligible=row["eligible"],
                rejected=row["rejected"],
                success=row["success"],
                terminal_failure=row["terminal_failure"],
                attempt_runs=row["validation_attempt_runs"],
                fail_runs=row["validation_fail_runs"],
                pass_hidden=row["validation_pass_hidden_fail_runs"],
                disagree=row["verifier_disagreement_runs"],
                profile=profile,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`classifier-debug`, `adaptive-rejection-sampler`, `blind-maze-explorer-algorithm`, and `nginx-request-logging` should be excluded from M1b unless their setup path is fixed.",
            "",
            "`broken-python`, `grid-pattern-transform`, `attention-mil`, and `csv-to-parquet` are the best small M1b preflight targets because they combine visible-check potential with either observed validation activity or mixed/failed terminal outcomes.",
            "",
            "`extract-safely`, `fix-permissions`, and `aimo-airline-departures` are useful controls or reserves, but should not dominate M1b because they are not the missing validation-loop signal.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_profile_label(row: dict[str, Any]) -> str:
    if row["eligible"] == 0 and row["rejected"] > 0:
        return "setup_failure_exclude"
    if row["validation_fail_runs"] > 0:
        return "visible_validation_failure_present"
    if row["terminal_failure"] > 0 and row["validation_pass_hidden_fail_runs"] > 0:
        return "visible_pass_then_hidden_fail"
    if row["terminal_failure"] > 0 and row["validation_attempt_runs"] == 0:
        return "hidden_or_terminal_failure_without_visible_check"
    if row["success"] > 0 and row["terminal_failure"] > 0:
        return "mixed_terminal_outcome"
    if row["success"] > 0:
        return "clean_success_control"
    return "unclear"


def _write_candidate_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_m1b_plan(path: Path, rows: list[dict[str, Any]], task_rows: dict[str, dict[str, Any]]) -> None:
    selected = [row for row in rows if str(row["m1b_selected"]) == "True"]
    preflight = [row for row in selected if row["m1b_role"] == "m1b_preflight_target"]
    controls = [row for row in selected if row["m1b_role"] == "m1b_control_or_reserve"]
    excluded = [row for row in rows if row["m1b_role"] == "excluded"]
    lines = [
        "# Terminal-Bench M1b Task Plan",
        "",
        "## Intent",
        "",
        "M1b should test whether the same OpenAI adaptive path can collect visible validation-loop trajectories. It should not scale the M1 task mix.",
        "",
        "Model arms:",
        "",
        "- `gpt-5.4`",
        "- `gpt-5.3-codex`",
        "- `gpt-5.4-mini`",
        "",
        "## Small Preflight",
        "",
        "Run 4 tasks x 3 arms = 12 attempted runs first. Continue only if `validation_fail_observed` coverage improves materially.",
        "",
        "| task_id | visible_validation_loop_score | role | reason |",
        "| --- | ---: | --- | --- |",
    ]
    for row in preflight:
        lines.append(
            f"| {row['task_id']} | {row['visible_validation_loop_score']} | {row['m1b_role']} | {row['m1b_selection_reason']} |"
        )
    lines.extend(["", "## Controls / Reserves", "", "| task_id | score | reason |", "| --- | ---: | --- |"])
    for row in controls:
        lines.append(f"| {row['task_id']} | {row['visible_validation_loop_score']} | {row['m1b_selection_reason']} |")
    lines.extend(["", "## Excluded", "", "| task_id | reason |", "| --- | --- |"])
    for row in excluded:
        lines.append(f"| {row['task_id']} | {row['m1b_selection_reason']} |")
    lines.extend(
        [
            "",
            "## Gate Additions",
            "",
            "Keep `validation_fail_observed_coverage`. Add a separate `validation_disagreement_coverage` metric instead of weakening the visible-failure gate.",
            "",
            "Definition: a run has `validation_disagreement` when it has a visible validation attempt or `agent_claims_done`, and the terminal verifier fails.",
            "",
            "## Cost Guard",
            "",
            "- per-run warning at `$0.75`",
            "- per-run hard stop at `$1.25` unless explicitly overridden",
            "- batch warning at `$10`",
            "",
            "## Do Not Run Yet",
            "",
            "This plan is a preparation artifact. Before launching M1b, inspect `reports/M1_VALIDATION_SIGNAL_AUDIT.md` and confirm the 4-task preflight target list.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
