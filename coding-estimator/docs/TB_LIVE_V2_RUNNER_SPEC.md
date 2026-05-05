# tb_live_v2 runner spec

_Generated 2026-05-05. Replaces the upstream `tb run` harness assumption
in earlier U-plan drafts._

## Why no harness

We do not need:
- Upstream Terminal-Bench `tb run` harness (Docker orchestration is
  overkill; the schema bridge to our ledger would be brittle).
- Docker per task (the harness's main job is process isolation; we get
  enough isolation from a per-run workspace + fresh venv for tasks
  that depend on Python state).
- Anthropic API direct calls (Claude Code's Agent tool already spawns
  Sonnet subagents with full tool use).

The runner is a ~150-line Python driver that:
1. Materializes the task workspace.
2. Spawns one subagent (Sonnet by default; Opus 4.7 for Arm A) with the
   task statement and an instruction to log structured actions.
3. Runs the verifier on the workspace after the subagent returns.
4. Emits ledger artifacts via the upstream sidecar.

## Per-run flow

```text
input:  task_id, arm (A | B), seed
output: runs/tb_live_v2/<run_id>/{ledger.jsonl,
                                  events.jsonl,
                                  run_manifest.json,
                                  transcript.jsonl,
                                  verifier_output.txt,
                                  run_notes.md}

1. workspace = mkdtemp(prefix="tb_live_v2_<run_id>_")
2. copy tasks/tb_live_v2/<task_id>/* into workspace EXCEPT tests/
   (tests/ is held back; the agent must not read it)
3. arm A: model = claude-opus-4-7, budget_lines = 30
   arm B: model = claude-sonnet-4-6, budget_lines = 20
4. spawn subagent (Agent tool):
     subagent_type = general-purpose
     model         = arm.model
     prompt        = render_prompt(task.yaml::descriptions[base], workspace, run_dir, budget_lines)
5. subagent works against the workspace (Bash/Read/Write/Edit tools).
   per the prompt, it appends one JSONL line per action to
   <workspace>/.transcript.jsonl. when done, it calls a final emit.
6. on subagent return:
     - parse <workspace>/.transcript.jsonl into events.jsonl
       (wire format matches upstream tb_emit.py one-event-per-line)
     - copy task's tests/ into <workspace>/tests/
     - run pytest --tb=short tests/ within <workspace>; capture exit
     - final_success = (pytest exit == 0)
     - final_success_source = "internal_verifier"
7. write run_manifest.json with:
     task_id, arm, model, max_lines, start_time, end_time,
     final_success, final_success_source, termination_reason,
     num_events, has_real_wallclock = true
8. invoke upstream replayer:
     ledger_progress.serialization.replay_events_to_ledger(
         events_path=<run_dir>/events.jsonl,
         out_path=<run_dir>/ledger.jsonl,
     )
   (this produces ledger.jsonl, progress.csv, progress_by_category.csv,
   summary_by_category.json — same artifacts as Hermes)
9. write run_notes.md summarizing arm config + outcome.
```

## Subagent prompt contract

The subagent must:

1. Read `task.md` (rendered from `task.yaml::descriptions[base].description`)
   in the workspace root.
2. Solve the task by editing files and running shell commands inside
   the workspace.
3. **Append one JSONL line to `.transcript.jsonl` before each action.**
   Each line is:
   ```json
   {"step": <int>,
    "ts": "<ISO-8601 UTC>",
    "kind": "shell" | "read_file" | "write_file" | "edit_file" | "thought" | "done",
    "summary": "<short imperative description>",
    "command": "<shell command, if kind == shell>",
    "path": "<path, if file op>",
    "exit_code": <int, optional>,
    "obs_snippet": "<≤500 chars of stdout/stderr or file content snippet>"}
   ```
4. When the subagent decides the task is complete, it appends a `done`
   record to `.transcript.jsonl` and stops.
5. The subagent must NOT read anything under `tests/` (the directory
   is held back from the workspace; if the subagent looks for it,
   that's a no-op).

The subagent's `budget_lines` is a soft cap on action count. The driver
hard-stops the subagent only by virtue of the Agent tool's own runtime
limits (we do not need to enforce a wall-clock from outside).

## Mapping transcript → ledger events

The transcript JSON-Lines are converted to upstream ledger ops by a
small classifier (lives in
`coding_estimator/runner/transcript_to_events.py`):

| transcript.kind | ledger op | category |
|---|---|---|
| read_file, list, grep, find | add+complete | INVESTIGATION |
| write_file, edit_file, patch | add+complete | PRODUCT |
| shell with `pip install` / `apt` | add+complete | ENVIRONMENT |
| shell with `pytest` / `python -m unittest` | add+complete | VALIDATION |
| shell with redirection / write side effect | add+complete | PRODUCT |
| shell other | add+complete | INVESTIGATION |
| thought (no action) | (skipped) | — |
| done | (close any open leaves) | — |

This mirrors the upstream `auto_annotate_hermes.py` classifier exactly,
which is why the resulting `ledger.jsonl` is interoperable with the
existing Hermes-shaped pipeline.

## Workspace isolation

Per run:

```text
- /tmp/tb_live_v2_<run_id>_<rand>/      # workspace (fresh tempdir)
  ├── task.md                            # rendered from task.yaml
  ├── <task-specific seed files>         # copied from task dir
  ├── .transcript.jsonl                  # subagent emits this
  └── tests/                             # added AFTER subagent returns
```

Process isolation:

- Each run gets a **fresh `python -m venv .venv`** in the workspace
  for tasks that declare a Python environment (a `requirements.txt`
  or `pyproject.toml` in the task dir signals this). The subagent is
  told to `source .venv/bin/activate` before any pip/pytest.
- Tasks that depend on a *missing* package (e.g.,
  `stuck_blocked_01_missing_dep_loop`) verify the package is absent
  in the fresh venv, not on the host.
- Tasks that need OS-level state (apt-get, system services) are
  marked `requires_docker: true` in their `shape.yaml` and **skipped
  by the host runner**. Those tasks queue for a future Docker arm if
  we ever decide to add one.

For our 5 shipped scaffolds:

| task | needs |
|---|---|
| progress_drop_01_lint_then_runtime_failure | tempdir + Python stdlib |
| validation_new_work_01_test_reveals_edge_case | tempdir + Python stdlib |
| stuck_blocked_01_missing_dep_loop | tempdir + fresh venv (without bs4) |
| high_progress_failure_01_subtasks_done_verifier_strict | tempdir + Python stdlib + free port |
| low_progress_success_01_oneline_fix | tempdir + Python stdlib |

None require Docker. The Dockerfile in each scaffold is preserved as
optional documentation for a future Docker arm but is not invoked.

## Termination

The subagent stops when one of:
- it appends a `done` record to `.transcript.jsonl` and returns;
- it returns without a `done` record (the driver records
  `termination_reason: "no_done_record"`);
- the Agent tool runtime caps it (recorded as
  `termination_reason: "subagent_limit"`).

`final_success` is **always** computed from the verifier exit code
after the subagent returns, regardless of how it terminated. This is
the Terminal-Bench-style ground-truth contract.

## What this gives us that `tb run` did not

- Direct integration with the existing ledger sidecar (no schema
  bridge to maintain).
- No Docker daemon dependency, no harness version pinning.
- Trivial parallelism (spawn N subagents concurrently from the driver).
- Free model swapping via the Agent tool's `model` override.
- The task pool is anything we have on disk shaped per
  `tasks/tb_live_v2/<id>/`. We can ingest TB2 tasks by cloning the
  upstream repo and translating their task.yaml verbatim — the
  verifier (`pytest tests/test_outputs.py`) is identical.

## What we lose

- TB2 tasks with non-Python verifiers, GPU dependencies, or
  multi-container setups cannot be run on the host. These get the
  `requires_docker: true` flag and remain out of v2.
- The leaderboard-comparable "Terminal-Bench score" is no longer
  meaningful — we are measuring a different thing (estimator dataset
  collection), so this is fine.

## Files we still need

- `coding_estimator/runner/run_internal_task.py` — driver (~150 lines).
- `coding_estimator/runner/transcript_to_events.py` — JSONL→ledger op
  classifier (~80 lines, mirrors upstream auto-annotator).
- `coding_estimator/runner/prompts.py` — subagent prompt template
  (the contract above as a Jinja-style render).
- `tests/test_runner_transcript_to_events.py` — semantic tests on the
  classifier so the ledger interop stays honest.
- `scripts/run_tb_live_v2_batch.py` — thin wrapper that runs the
  driver across a batch's task list.

These ship in U4 implementation, not U4 plan.
