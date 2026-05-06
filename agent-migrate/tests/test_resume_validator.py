import shutil
import subprocess
from pathlib import Path

import pytest

from agent_migrate_agent.adapters.swe_agent import swe_agent_to_trace
from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl
from agent_migrate_agent.resume_packages import (
    StateEntry,
    WorkspaceFileEntry,
    build_agent_migrate_minimal,
    build_full_workspace_snapshot,
    build_prompt_only,
    build_transcript_plus_diff,
    build_transcript_plus_harness_state,
    transcript_prefix_hash,
)
from agent_migrate_agent.resume_validator import (
    VALIDATION_REASONS,
    required_state_ids,
    validate_package,
)

FIXTURE = Path(__file__).parent / "fixtures" / "swe_agent_pilot_s_07.json"


def _git_available() -> bool:
    return shutil.which("git") is not None


def _events_and_cut(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    events = load_trace_jsonl(out)
    cps = find_cut_points(events, trace_id="s_07")
    return events, cps


def _harness() -> dict:
    return {"cwd": "/changelog_cli", "open_file": "src/changelog/utils.py", "env": {}}


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Init a git repo with one tracked file. Returns (repo, base_commit, valid_diff)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], repo)
    _run(["git", "config", "user.email", "t@t"], repo)
    _run(["git", "config", "user.name", "t"], repo)
    (repo / "x.py").write_text("x = 1\n")
    _run(["git", "add", "x.py"], repo)
    _run(["git", "commit", "-q", "-m", "init"], repo)
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()

    (repo / "x.py").write_text("x = 2\n")
    diff = subprocess.run(["git", "-C", str(repo), "diff"],
                          capture_output=True, text=True, check=True).stdout
    _run(["git", "checkout", "-q", "x.py"], repo)
    return repo, base, diff


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_validate_prompt_only_passes_on_clean_build(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    res = validate_package(pkg, events)
    assert res.valid, res.reasons


def test_validate_transcript_plus_harness_passes(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    res = validate_package(pkg, events)
    assert res.valid, res.reasons


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_validate_transcript_plus_diff_passes_with_real_repo(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    repo, base, diff = _make_repo(tmp_path)
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit=base, diff_blob=diff)
    res = validate_package(pkg, events, base_repo_path=repo)
    assert res.valid, res.reasons


# ---------------------------------------------------------------------------
# Failure paths (the falsifications)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_dropping_diff_from_transcript_plus_diff_fails_with_diff_does_not_apply(tmp_path: Path):
    """C3 falsification target per TASKS.md: dropping the diff at an edit cut point
    must fail with reason `diff_does_not_apply` (or, if blob empties out entirely,
    `missing_diff_for_transcript_plus_diff`)."""
    events, cps = _events_and_cut(tmp_path)
    repo, base, _good_diff = _make_repo(tmp_path)

    unrelated = (
        "diff --git a/y.py b/y.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/y.py\n"
        "+++ b/y.py\n"
        "@@\n_orig\n+new\n"
    )
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit=base, diff_blob=unrelated)
    res = validate_package(pkg, events, base_repo_path=repo)
    assert not res.valid
    assert "diff_does_not_apply" in res.reasons


def test_empty_diff_fails_with_missing_diff_reason(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit="abc",
                                     diff_blob="")
    res = validate_package(pkg, events)
    assert not res.valid
    assert "missing_diff_for_transcript_plus_diff" in res.reasons


def test_missing_base_commit_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit="",
                                     diff_blob="--- a/x\n+++ b/x\n@@\n_a\n+b\n")
    res = validate_package(pkg, events)
    assert not res.valid
    assert "missing_base_commit" in res.reasons


def test_missing_base_repo_path_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit="abc123",
                                     diff_blob="--- a/x\n+++ b/x\n@@\n_a\n+b\n")
    res = validate_package(pkg, events)
    assert not res.valid
    assert "missing_base_repo_path" in res.reasons


def test_corrupted_transcript_prefix_hash_fails(tmp_path: Path):
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    bad = replace(pkg, transcript_prefix_hash="0" * 64)
    res = validate_package(bad, events)
    assert not res.valid
    assert "transcript_prefix_mismatch" in res.reasons


def test_corrupted_content_hash_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    from dataclasses import replace
    sp_idx = next(i for i, e in enumerate(pkg.state_entries) if e.state_id == "system_prompt")
    new_entries = list(pkg.state_entries)
    new_entries[sp_idx] = replace(new_entries[sp_idx], content_hash="h_TAMPERED")
    bad = replace(pkg, state_entries=tuple(new_entries))
    res = validate_package(bad, events)
    assert not res.valid
    assert "content_hash_mismatch" in res.reasons


def test_missing_state_object_fails(tmp_path: Path):
    """If the next llm_call needs a tool_output that the package omits, fail."""
    events, cps = _events_and_cut(tmp_path)
    cp = cps[5]  # mid_trajectory cut: next llm_call reads many tool outputs
    pkg = build_prompt_only(events, cp)
    from dataclasses import replace
    sparse = tuple(e for e in pkg.state_entries
                   if e.state_id in ("system_prompt", "issue_text"))
    bad = replace(pkg, state_entries=sparse)
    res = validate_package(bad, events)
    assert not res.valid
    assert "missing_state_object" in res.reasons


def test_harness_config_missing_for_required_package(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_full_workspace_snapshot(events, cps[0],
                                        harness_config=_harness(),
                                        workspace_files=())
    from dataclasses import replace
    bad = replace(pkg, harness_config=None)
    res = validate_package(bad, events)
    assert not res.valid
    assert "harness_config_missing" in res.reasons


def test_harness_config_schema_violation(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_harness_state(events, cps[0],
                                              harness_config={"cwd": "/x"})
    res = validate_package(pkg, events)
    assert not res.valid
    assert "harness_config_schema_violation" in res.reasons


def test_validation_reasons_are_documented():
    expected = {
        "transcript_prefix_mismatch", "missing_state_object", "content_hash_mismatch",
        "unknown_state_entry", "workspace_digest_mismatch", "base_commit_mismatch",
        "diff_does_not_apply", "missing_diff_for_transcript_plus_diff",
        "missing_base_commit", "missing_base_repo_path", "dirty_base_repo",
        "harness_config_schema_violation", "harness_config_missing",
    }
    assert set(VALIDATION_REASONS) == expected


def test_validation_result_carries_checks_run(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    res = validate_package(pkg, events)
    assert "transcript_prefix" in res.checks_run
    assert "state_coverage" in res.checks_run
    assert "diff_apply" not in res.checks_run  # no diff = check skipped


def test_diff_check_runs_only_when_package_has_diff(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    res = validate_package(pkg, events)
    assert res.valid
    assert "diff_apply" not in res.checks_run


def test_content_hash_mismatches_accumulate(tmp_path: Path):
    """Two corrupted hashes should both surface — not stop at the first."""
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[3])
    new_entries = list(pkg.state_entries)
    for i, e in enumerate(new_entries[:2]):
        new_entries[i] = replace(e, content_hash="h_TAMPERED" + str(i))
    bad = replace(pkg, state_entries=tuple(new_entries))
    res = validate_package(bad, events)
    assert "content_hash_mismatch" in res.reasons
    # Reason is deduped to one occurrence even with two underlying mismatches.
    assert res.reasons.count("content_hash_mismatch") == 1


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_mutated_real_diff_fails_diff_does_not_apply(tmp_path: Path):
    """Stricter falsification: build with the recorded diff, mutate one byte
    in a hunk body so the patch no longer applies."""
    events, cps = _events_and_cut(tmp_path)
    repo, base, diff = _make_repo(tmp_path)
    mutated = diff.replace("-x = 1\n", "-x = 99\n")
    assert mutated != diff
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit=base, diff_blob=mutated)
    res = validate_package(pkg, events, base_repo_path=repo)
    assert not res.valid
    assert "diff_does_not_apply" in res.reasons


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_dirty_worktree_refused(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    repo, base, diff = _make_repo(tmp_path)
    # Leave the worktree dirty by re_applying the in_memory diff via filesystem write.
    (repo / "x.py").write_text("x = 99\n")
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit=base, diff_blob=diff)
    res = validate_package(pkg, events, base_repo_path=repo)
    assert not res.valid
    assert "dirty_base_repo" in res.reasons


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_diff_apply_requires_worktree_at_base_commit(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    repo, base, diff = _make_repo(tmp_path)
    (repo / "x.py").write_text("x = 3\n")
    _run(["git", "add", "x.py"], repo)
    _run(["git", "commit", "-q", "-m", "second"], repo)
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit=base, diff_blob=diff)
    res = validate_package(pkg, events, base_repo_path=repo)
    assert not res.valid
    assert "base_commit_mismatch" in res.reasons


def test_workspace_file_hash_mismatch_fails(tmp_path: Path):
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_full_workspace_snapshot(
        events, cps[0],
        harness_config=_harness(),
        workspace_files=(WorkspaceFileEntry("src/x.py", 10, "h_good"),),
    )
    bad = replace(
        pkg,
        workspace_files=(WorkspaceFileEntry("src/x.py", 10, "h_bad"),),
    )
    res = validate_package(bad, events)
    assert not res.valid
    assert "workspace_digest_mismatch" in res.reasons


def test_workspace_digest_state_entry_checked(tmp_path: Path):
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_agent_migrate_minimal(
        events, cps[0],
        harness_config=_harness(),
        workspace_files=(WorkspaceFileEntry("uncommitted_diff.patch", 10, "h_diff"),),
        workspace_layer_for_file={"uncommitted_diff.patch": "uncommitted_diff"},
    )
    entries = list(pkg.state_entries)
    idx = next(i for i, e in enumerate(entries) if e.state_id == "workspace_layer:uncommitted_diff")
    entries[idx] = replace(entries[idx], content_hash="h_tampered")
    bad = replace(pkg, state_entries=tuple(entries))
    res = validate_package(bad, events)
    assert not res.valid
    assert "workspace_digest_mismatch" in res.reasons


def test_unknown_state_entry_fails_validation(tmp_path: Path):
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    bad = replace(pkg, state_entries=pkg.state_entries + (
        StateEntry(
            state_id="not_declared",
            layer="prompt_context",
            bytes=1,
            content_hash="h_unknown",
            materialization="included",
            validator="digest",
        ),
    ))
    res = validate_package(bad, events)
    assert not res.valid
    assert "unknown_state_entry" in res.reasons


def test_lazy_rehydrate_does_not_satisfy_required_state(tmp_path: Path):
    """A state_id consumed by the next llm_call must be `included`, not `lazy_rehydrate`."""
    from dataclasses import replace
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    # Force the system_prompt entry to lazy_rehydrate (which it should not be).
    new_entries = []
    for e in pkg.state_entries:
        if e.state_id == "system_prompt":
            new_entries.append(replace(e, materialization="lazy_rehydrate"))
        else:
            new_entries.append(e)
    bad = replace(pkg, state_entries=tuple(new_entries))
    res = validate_package(bad, events)
    assert not res.valid
    assert "missing_state_object" in res.reasons


def test_required_state_ids_helper_matches_lookahead(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    needed = required_state_ids(pkg, events)
    # F2 fixture: every llm_call reads system_prompt and issue_text.
    assert "system_prompt" in needed
    assert "issue_text" in needed


def test_consumer_node_id_fallback_attributes_window_reads():
    """Adapters without consumer_node_id should still get a meaningful coverage check."""
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1, "layer": "prompt_context",
                     "lifetime": "persistent", "bytes": None, "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x", "workflow_id": "w"}, "reason": None},
        {"step": 3, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x", "workflow_id": "w"}, "reason": None},
        # state_read without consumer_node_id — fallback should attribute it to S2.
        {"step": 5, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1}, "reason": None},
        {"step": 6, "event_type": "update_status", "subtask_id": "S2",
         "payload": {"status": "complete"}, "reason": None},
    ]
    from agent_migrate_agent.cut_points import find_cut_points
    cps = find_cut_points(events, trace_id="t")
    assert len(cps) == 1
    pkg = build_prompt_only(events, cps[0])
    assert "p" in required_state_ids(pkg, events)


def test_post_complete_state_read_still_counted():
    """state_reads emitted after update_status complete must still attribute to the prior llm_call."""
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1, "layer": "prompt_context",
                     "lifetime": "persistent", "bytes": None, "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x", "workflow_id": "w"}, "reason": None},
        {"step": 3, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x", "workflow_id": "w"}, "reason": None},
        {"step": 5, "event_type": "update_status", "subtask_id": "S2",
         "payload": {"status": "complete"}, "reason": None},
        # state_read emitted AFTER complete — must still count toward S2.
        {"step": 6, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1,
                     "consumer_node_id": "S2"}, "reason": None},
    ]
    from agent_migrate_agent.cut_points import find_cut_points
    cps = find_cut_points(events, trace_id="t")
    assert len(cps) == 1
    pkg = build_prompt_only(events, cps[0])
    assert "p" in required_state_ids(pkg, events)


def test_required_state_ids_interleaved_sessions_are_session_scoped():
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1, "layer": "prompt_context",
                     "lifetime": "persistent", "bytes": None, "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "A1",
         "payload": {"node_type": "llm_call", "session_id": "A", "workflow_id": "w"}, "reason": None},
        {"step": 3, "event_type": "update_status", "subtask_id": "A1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "B1",
         "payload": {"node_type": "llm_call", "session_id": "B", "workflow_id": "w"}, "reason": None},
        {"step": 5, "event_type": "update_status", "subtask_id": "B1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 6, "event_type": "add_subtask", "subtask_id": "A2",
         "payload": {"node_type": "llm_call", "session_id": "A", "workflow_id": "w"}, "reason": None},
        {"step": 7, "event_type": "add_subtask", "subtask_id": "B2",
         "payload": {"node_type": "llm_call", "session_id": "B", "workflow_id": "w"}, "reason": None},
        {"step": 8, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1,
                     "consumer_node_id": "A2"}, "reason": None},
        {"step": 9, "event_type": "add_subtask", "subtask_id": "A3",
         "payload": {"node_type": "llm_call", "session_id": "A", "workflow_id": "w"}, "reason": None},
    ]
    from agent_migrate_agent.cut_points import find_cut_points
    cp = next(cp for cp in find_cut_points(events, trace_id="t") if cp.next_llm_call_id == "A2")
    pkg = build_prompt_only(events, cp)
    assert "p" in required_state_ids(pkg, events)


def test_wrong_cut_point_against_truncated_events_fails(tmp_path: Path):
    """Build at cps[0]; validate against a window that ends earlier — prefix hash differs."""
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[3])
    # Truncate so the package's claimed event_index points past the new end_of_events.
    truncated = events[: cps[1].event_index]
    res = validate_package(pkg, truncated)
    assert not res.valid
    assert "transcript_prefix_mismatch" in res.reasons


def test_two_cuts_validate_with_distinct_prefix_hashes(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    p0 = build_prompt_only(events, cps[0])
    p2 = build_prompt_only(events, cps[2])
    assert p0.transcript_prefix_hash != p2.transcript_prefix_hash
    assert validate_package(p0, events).valid
    assert validate_package(p2, events).valid


@pytest.mark.skipif(not _git_available(), reason="git binary required")
def test_all_five_package_types_validate_green(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    repo, base, diff = _make_repo(tmp_path)
    p1 = build_prompt_only(events, cps[0])
    p2 = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    p3 = build_transcript_plus_diff(events, cps[0], harness_config=_harness(),
                                    base_commit=base, diff_blob=diff)
    p4 = build_full_workspace_snapshot(events, cps[0], harness_config=_harness(),
                                       workspace_files=())
    from agent_migrate_agent.resume_packages import build_agent_migrate_minimal
    p5 = build_agent_migrate_minimal(events, cps[0], harness_config=_harness())
    for pkg in (p1, p2, p3, p4, p5):
        res = validate_package(pkg, events, base_repo_path=repo)
        assert res.valid, (pkg.package_type, res.reasons)


def test_no_subprocess_for_non_diff_packages(tmp_path: Path, monkeypatch):
    """Ensure validator never shells out for prompt_only / transcript_plus_harness."""
    events, cps = _events_and_cut(tmp_path)
    called = {"n": 0}
    real_run = subprocess.run

    def spy(*args, **kwargs):
        called["n"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr("agent_migrate_agent.resume_validator.subprocess.run", spy)

    pkg1 = build_prompt_only(events, cps[0])
    pkg2 = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    validate_package(pkg1, events)
    validate_package(pkg2, events)
    assert called["n"] == 0
