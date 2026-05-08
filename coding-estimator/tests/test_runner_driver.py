"""
Claim:
The runner driver's prepare/finalize pair turns a task directory plus a
subagent transcript into a verifier-graded run with a wire-format
events.jsonl, a run_manifest.json, and a final_success that comes from
the pytest verifier on the held-back tests/ directory.

Plausible wrong implementations:
- prepare() copies tests/ into the workspace (the agent must not see the
  verifier; § 0.4 trace-only contract).
- finalize() trusts the subagent's `done.summary` for final_success
  instead of the verifier exit (would let self-claim drive the label).
- finalize() writes final_success=True when verifier exit != 0.
- prepare() writes into the user's runs dir on top of an existing run.
- A missing transcript silently produces an empty events.jsonl and a
  null verdict instead of infrastructure_failure.
- The same workspace gets reused across runs (state leakage).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from coding_estimator.runner.driver import ARM_BUDGETS, finalize, prepare


def _make_minimal_task(parent: Path, name: str = "demo_task",
                       passing: bool = True) -> Path:
    """A trivial task: agent writes /app/answer.txt = '42'.
    Verifier asserts it. We control whether the verifier passes by
    flipping `passing`."""
    d = parent / name
    (d / "tests").mkdir(parents=True)
    (d / "task.yaml").write_text(textwrap.dedent("""\
        descriptions:
          - key: base
            description: |
              Write the string 42 to ./answer.txt
        author_email: t@x
        difficulty: easy
        tags: [test_fixture]
        max_agent_timeout_sec: 60
        max_test_timeout_sec: 10
        test_scripts: [run-uv-pytest.sh]
        run_tests_in_same_shell: true
    """))
    expected = "42" if passing else "7"
    (d / "tests" / "test_outputs.py").write_text(textwrap.dedent(f"""\
        from pathlib import Path
        def test_answer():
            assert (Path.cwd() / "answer.txt").read_text().strip() == "{expected}"
    """))
    (d / "shape.yaml").write_text("target_shape: low_progress_success\n"
                                  "expected_pass_rate: 0.9\nnotes: |\n  fixture\n")
    (d / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    return d


def _simulate_subagent(workspace: Path, run_dir: Path,
                       answer_text: str = "42",
                       include_done: bool = True,
                       include_action: bool = True) -> None:
    """Stand-in for the Agent-tool subagent. Writes the answer file in
    the workspace and emits a transcript that the runner can replay."""
    if include_action:
        (workspace / "answer.txt").write_text(answer_text)
    transcript: list[dict] = []
    if include_action:
        transcript.append({
            "step": 1, "ts": "2026-05-05T00:00:01Z", "kind": "write_file",
            "path": str(workspace / "answer.txt"),
            "summary": "write answer.txt",
        })
    if include_done:
        transcript.append({
            "step": 2 if include_action else 1, "ts": "2026-05-05T00:00:02Z",
            "kind": "done", "summary": "answered",
        })
    (run_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in transcript)
    )


def test_prepare_does_not_leak_tests_into_workspace(tmp_path):
    """The verifier (tests/) must NOT appear in the agent workspace."""
    task = _make_minimal_task(tmp_path, "t1")
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    assert not (prep.workspace / "tests").exists(), \
        "tests/ leaked into workspace; the agent could read the verifier"
    # Seed file shape preserved, sans tests:
    assert (prep.workspace / "task.md").is_file()
    assert (prep.workspace / "Dockerfile").is_file()


def test_prepare_writes_prompt_and_workspace_pointer(tmp_path):
    task = _make_minimal_task(tmp_path, "t1")
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    assert (prep.run_dir / "prompt.txt").is_file()
    assert prep.workspace.is_dir()
    assert (prep.run_dir / "workspace_path.txt").read_text().strip() == str(prep.workspace)


def test_prepare_rejects_unknown_arm(tmp_path):
    task = _make_minimal_task(tmp_path, "t1")
    with pytest.raises(ValueError):
        prepare(task, "Z", runs_root=tmp_path / "runs")


def test_prepare_refuses_to_overwrite_existing_run(tmp_path):
    """Running into an existing run_dir would clobber prior artifacts."""
    task = _make_minimal_task(tmp_path, "t1")
    runs_root = tmp_path / "runs"
    prep1 = prepare(task, "A", runs_root=runs_root)
    # forge a second prep that targets the SAME run dir
    (runs_root / prep1.run_id).mkdir(exist_ok=True)
    # mkdtemp gives a different run_id so this is fine; assert no two
    # preps land in the same dir.
    prep2 = prepare(task, "A", runs_root=runs_root)
    assert prep2.run_dir != prep1.run_dir


def test_finalize_pass_path_runs_verifier_and_marks_success(tmp_path):
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    _simulate_subagent(prep.workspace, prep.run_dir, answer_text="42")
    result = finalize(prep, task, skip_sidecar=True)
    assert result.final_success is True
    assert result.termination_reason == "verifier_pass"
    assert result.verifier_exit == 0
    assert result.num_events == 2  # one (add, complete) pair
    manifest = json.loads((prep.run_dir / "run_manifest.json").read_text())
    assert manifest["final_success"] is True
    assert manifest["final_success_source"] == "internal_verifier"
    assert manifest["arm"] == "A"
    assert manifest["budget_lines"] == ARM_BUDGETS["A"]


def test_finalize_fail_path_marks_failure_even_if_done_was_logged(tmp_path):
    """Self-claim must NOT drive final_success. Verifier wins."""
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    _simulate_subagent(prep.workspace, prep.run_dir, answer_text="WRONG")
    result = finalize(prep, task, skip_sidecar=True)
    assert result.final_success is False
    assert result.termination_reason == "verifier_fail"  # done was logged
    assert result.verifier_exit != 0


def test_finalize_no_done_record_is_distinct_termination(tmp_path):
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    _simulate_subagent(prep.workspace, prep.run_dir, answer_text="WRONG",
                       include_done=False)
    result = finalize(prep, task, skip_sidecar=True)
    assert result.final_success is False
    assert result.termination_reason == "no_done_record"


def test_finalize_missing_transcript_is_infrastructure_failure(tmp_path):
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    # Do not emit a transcript at all.
    result = finalize(prep, task, skip_sidecar=True)
    assert result.final_success is None
    assert result.termination_reason == "infrastructure_failure"
    assert result.num_events == 0
    # Manifest must still be written so the run is auditable.
    manifest = json.loads((prep.run_dir / "run_manifest.json").read_text())
    assert manifest["termination_reason"] == "infrastructure_failure"


def test_finalize_thought_only_transcript_is_infrastructure_failure(tmp_path):
    """A subagent that thinks but never acts produces no leaves; that
    is operationally indistinguishable from a missing transcript."""
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    (prep.run_dir / "transcript.jsonl").write_text(
        json.dumps({"step": 1, "ts": "2026-05-05T00:00:00Z",
                    "kind": "thought", "summary": "thinking"}) + "\n"
        + json.dumps({"step": 2, "ts": "2026-05-05T00:00:01Z",
                      "kind": "done", "summary": "gave up"}) + "\n"
    )
    result = finalize(prep, task, skip_sidecar=True)
    assert result.final_success is None
    assert result.termination_reason == "infrastructure_failure"


def test_events_jsonl_emitted_in_wire_format(tmp_path):
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    _simulate_subagent(prep.workspace, prep.run_dir, answer_text="42")
    finalize(prep, task, skip_sidecar=True)
    events_path = prep.run_dir / "events.jsonl"
    assert events_path.is_file()
    lines = events_path.read_text().splitlines()
    assert len(lines) >= 1
    e = json.loads(lines[0])
    assert e["schema_version"] == "1.0"
    assert e["run_id"] == prep.run_id
    assert isinstance(e["step"], int)
    assert e["timestamp"].endswith("Z")
    assert isinstance(e["ledger_ops"], list)


def test_finalize_writes_observation_events_jsonl(tmp_path):
    task = _make_minimal_task(tmp_path, "t1", passing=True)
    prep = prepare(task, "A", runs_root=tmp_path / "runs")
    _simulate_subagent(prep.workspace, prep.run_dir, answer_text="42")
    finalize(prep, task, skip_sidecar=True)
    rows = [
        json.loads(line)
        for line in (prep.run_dir / "observation_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "expected at least one observation event"
    assert any(row["event_type"] == "product_file_written" for row in rows)
    assert any(row["event_type"] == "agent_claims_done" for row in rows)
    assert any(row["event_type"] == "verifier_pass" for row in rows)
