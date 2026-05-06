"""C3 — static resume_package validator.

`validate_package(package, events, base_repo_path=None)` runs deterministic
checks over a `ResumePackage` against the trace it claims to resume from. It
does NOT run a model, execute a tool, run tests, or spin up a real harness.
The only subprocess invoked is `git` for `apply --check` and `status
--porcelain` on diff_bearing packages — both syntactic checks that do not
mutate the worktree.

Checks (per TASKS.md C3 spec):

  1. transcript prefix hash matches `events[0:cut.event_index]`
  2. content_hash on every `state_entry` matches what was declared in the trace
     (accumulates ALL mismatches, not just the first)
  3. every state_id consumed by the next llm_call (looking forward in `events`)
     is covered by the package with `materialization == "included"`. Looking
     forward stops at the next `add_subtask` whose `node_type == "llm_call"`,
     so post-`update_status complete` reads are still counted. If no
     `state_read` carries `consumer_node_id` in the window, fall back to
     attributing every `state_read` in the window.
  4. for `transcript_plus_diff`: `git status --porcelain` is empty (clean
     base repo — `dirty_base_repo` reason) and `git apply --check` succeeds
     against `base_commit` in `base_repo_path`
  5. harness config has the required keys (`HARNESS_REQUIRED_KEYS`) for any
     package type that requires it

Failures accumulate (deduped); the validator returns ALL failures plus
`checks_run` so callers (C4) can distinguish "not validated" from
"validated and passed."
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .resume_packages import (
    HARNESS_REQUIRED_KEYS,
    ResumePackage,
    transcript_prefix_hash,
)


VALIDATION_REASONS: tuple[str, ...] = (
    "transcript_prefix_mismatch",
    "missing_state_object",
    "content_hash_mismatch",
    "diff_does_not_apply",
    "missing_diff_for_transcript_plus_diff",
    "missing_base_commit",
    "missing_base_repo_path",
    "dirty_base_repo",
    "harness_config_schema_violation",
    "harness_config_missing",
)

CHECKS: tuple[str, ...] = (
    "transcript_prefix",
    "content_hashes",
    "state_coverage",
    "harness_schema",
    "diff_apply",
)

_PACKAGES_REQUIRING_HARNESS: frozenset[str] = frozenset({
    "transcript_plus_harness_state",
    "transcript_plus_diff",
    "full_workspace_snapshot",
    "agent_migrate_minimal",
})


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...]
    checks_run: tuple[str, ...]


def validate_package(
    package: ResumePackage,
    events: list[dict],
    *,
    base_repo_path: str | Path | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    checks_run: list[str] = []
    cut = package.cut_point

    checks_run.append("transcript_prefix")
    if cut.event_index > len(events):
        reasons.append("transcript_prefix_mismatch")
    elif transcript_prefix_hash(events, cut.event_index) != package.transcript_prefix_hash:
        reasons.append("transcript_prefix_mismatch")

    declared_hashes = _declared_hashes_at(events, cut.event_index)

    checks_run.append("content_hashes")
    for entry in package.state_entries:
        if entry.validator != "digest":
            continue
        expected = declared_hashes.get(entry.state_id)
        if expected is None:
            continue
        if expected != entry.content_hash:
            reasons.append("content_hash_mismatch")

    checks_run.append("state_coverage")
    needed = required_state_ids(package, events)
    included = {e.state_id for e in package.state_entries if e.materialization == "included"}
    if not needed.issubset(included):
        reasons.append("missing_state_object")

    if package.package_type in _PACKAGES_REQUIRING_HARNESS:
        checks_run.append("harness_schema")
        if package.harness_config is None:
            reasons.append("harness_config_missing")
        elif not all(k in package.harness_config for k in HARNESS_REQUIRED_KEYS):
            reasons.append("harness_config_schema_violation")

    if package.package_type == "transcript_plus_diff":
        checks_run.append("diff_apply")
        if not package.diff_blob:
            reasons.append("missing_diff_for_transcript_plus_diff")
        elif not package.base_commit:
            reasons.append("missing_base_commit")
        elif base_repo_path is None:
            reasons.append("missing_base_repo_path")
        else:
            repo = Path(base_repo_path)
            if not _worktree_clean(repo):
                reasons.append("dirty_base_repo")
            elif not _git_apply_check(repo, package.base_commit, package.diff_blob):
                reasons.append("diff_does_not_apply")

    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return ValidationResult(valid=not deduped,
                            reasons=tuple(deduped),
                            checks_run=tuple(checks_run))


def required_state_ids(package: ResumePackage, events: list[dict]) -> set[str]:
    """Public helper: state_ids the next llm_call after the cut would consume.

    Useful for C4 ablations: `if I drop tool_output T, which packages still validate?`
    """
    return _next_llm_call_state_reads(events, package.cut_point.event_index)


def _declared_hashes_at(events: list[dict], cut_index: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in events[:cut_index]:
        if e.get("event_type") != "state_declare":
            continue
        p = e.get("payload") or {}
        sid = p.get("state_id")
        h = p.get("content_hash")
        if sid and h:
            out[sid] = h
    return out


def _next_llm_call_state_reads(events: list[dict], cut_index: int) -> set[str]:
    """Collect state_ids read by the next llm_call following the cut.

    The cut is at events[cut_index] (an add_subtask llm_call). The window for
    that call's reads runs from cut_index+1 to the next `add_subtask` of
    `node_type=="llm_call"` (we deliberately do NOT stop at the call's own
    `update_status complete` so adapters that emit reads after completion
    still get attributed correctly).

    If any `state_read` in the window carries `consumer_node_id == sid`, only
    those are attributed to this call. Otherwise (adapters that don't set
    `consumer_node_id`), every `state_read` in the window is attributed —
    a permissive fallback that keeps the coverage check meaningful.
    """
    if cut_index >= len(events):
        return set()
    head = events[cut_index]
    if head.get("event_type") != "add_subtask":
        return set()
    sid = head.get("subtask_id")

    window: list[dict] = []
    for e in events[cut_index + 1:]:
        if e.get("event_type") == "add_subtask" \
           and (e.get("payload") or {}).get("node_type") == "llm_call":
            break
        window.append(e)

    matched_by_consumer: set[str] = set()
    all_reads: set[str] = set()
    for e in window:
        if e.get("event_type") != "state_read":
            continue
        p = e.get("payload") or {}
        state_id = p.get("state_id")
        if not state_id:
            continue
        all_reads.add(state_id)
        if p.get("consumer_node_id") == sid:
            matched_by_consumer.add(state_id)

    return matched_by_consumer or all_reads


def _worktree_clean(base_repo_path: Path) -> bool:
    if not base_repo_path.exists():
        return False
    proc = subprocess.run(
        ["git", "-C", str(base_repo_path), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == ""


def _git_apply_check(base_repo_path: Path, base_commit: str, diff_blob: str) -> bool:
    """Return True iff `git apply --check` accepts `diff_blob` at `base_commit`.

    Uses `--check` only — the diff is not applied. No model, no tool, no
    test execution. Returns False if git is unavailable, the repo is
    missing, the commit is unknown, or the diff fails to apply.
    """
    if not base_repo_path.exists():
        return False
    rev = subprocess.run(
        ["git", "-C", str(base_repo_path), "rev-parse", "--verify", base_commit + "^{commit}"],
        capture_output=True, text=True,
    )
    if rev.returncode != 0:
        return False
    proc = subprocess.run(
        ["git", "-C", str(base_repo_path), "apply", "--check", "-"],
        input=diff_blob, capture_output=True, text=True,
    )
    return proc.returncode == 0
