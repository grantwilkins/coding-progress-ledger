# TB-live task format

Source spec for every Workstream N_TB task. See `WORKSTREAM_N_TB_PLAN.md`
for the workstream context.

## Layout

```
tasks/tb_live/<task_id>/
├── task.md                # the only file the agent reads. Self-contained spec.
├── verifier.sh            # exit 0 ⇔ task complete. Hidden from the agent.
├── verifier_tests/        # hidden tests + fixtures invoked by verifier.sh.
└── solution_reference/    # hidden reference impl. Used to (a) prove the verifier
                           #   is achievable and (b) seed retro re-annotation.
```

## Verifier contract

`verifier.sh <agent_repo_dir>` runs the hidden tests against the agent's
implementation and exits 0 on pass, non-zero on fail. The verifier
must be deterministic (same agent_repo → same exit code).

Agent never sees `verifier.sh`, `verifier_tests/`, or `solution_reference/`.

## Run-time layout

```
runs/tb_live/<task_id>/
├── task.md                # copied from tasks/tb_live/<task_id>/task.md
├── repo/                  # agent's working dir (gitignored)
├── events.jsonl           # agent appends one wire-format line per action
├── ledger.jsonl           # produced by the sidecar at end-of-run
├── progress.csv           # produced by the sidecar
├── summary_by_category.json   # produced by the sidecar
├── final_diff.patch       # captured by validate_tb_run.py
├── test_output.txt        # captured by validate_tb_run.py
└── live_instrumentation.json  # written by validate_tb_run.py
```

## Wire-format event (one line per agent action)

The agent emits these via `scripts/tb_emit.py`. The sidecar consumes them.

```json
{
  "schema_version": "1.0",
  "run_id": "<task_id>",
  "step": 1,
  "timestamp": "2026-05-03T17:42:11Z",
  "ledger_ops": [
    {"op": "add", "id": "s1", "description": "Set up package skeleton", "category": "product"}
  ]
}
```

Allowed `op` values: `add`, `start`, `complete`, `block`, `reopen`,
`invalidate`, `split`, `add_evidence`. Allowed `category` values:
`product`, `validation`, `investigation`. See `ledger_progress/sidecar.py`
for the schema.

## Acceptance (per-task)

Every TB-12 task must satisfy:

1. `verifier.sh tasks/tb_live/<id>/solution_reference/` exits 0.
2. `tests/test_tb_live_verifiers.py` exercises (1) for each task in CI.
3. `task.md` is 200–1500 words and names the public API the verifier checks.

## Running a task end-to-end

Three steps. Prepare is two shell commands; dispatch is one Agent call;
validate is one script.

```bash
# 1. Prepare the run dir.
mkdir -p runs/tb_live/<task_id>/repo
cp tasks/tb_live/<task_id>/task.md runs/tb_live/<task_id>/task.md

# 2. Dispatch the subagent (use tasks/tb_live/_prompt_template.md as the prompt;
#    substitute <task_id>). One task at a time, sonnet model, isolation: none —
#    the agent works directly in runs/tb_live/<task_id>/repo/.

# 3. Post-run validation.
uv run python scripts/validate_tb_run.py <task_id>
```

`validate_tb_run.py` runs the sidecar over `events.jsonl`, runs
`verifier.sh` against `repo/`, runs `ledger-run check-run`, and writes
`live_instrumentation.json`. Exit 0 ⇔ verifier passed.

## Adding a new task

1. Create `tasks/tb_live/<task_id>/task.md` with the spec.
2. Write hidden tests under `verifier_tests/` and `verifier.sh` to drive them.
3. Implement the simplest correct solution under `solution_reference/`.
4. Add a parametrize entry to `tests/test_tb_live_verifiers.py`.
5. `uv run pytest tests/test_tb_live_verifiers.py` must pass.
