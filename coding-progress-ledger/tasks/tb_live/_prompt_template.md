# Subagent prompt — TB-live task <TASK_ID>

You are running under the N_TB live ledger harness. Your job is to make
the hidden verifier exit 0 by implementing the spec in `task.md`.

## Hard constraints

- Work **only** under `runs/tb_live/<TASK_ID>/repo/` (absolute path:
  `<ABS_REPO_DIR>`). Do not edit any file outside that directory.
- Do not read `tasks/tb_live/<TASK_ID>/verifier.sh`,
  `tasks/tb_live/<TASK_ID>/verifier_tests/`, or
  `tasks/tb_live/<TASK_ID>/solution_reference/` — those are hidden tests
  and your reference. Reading them defeats the experiment.
- The spec is in `runs/tb_live/<TASK_ID>/task.md`. Read it first. Re-read
  it whenever you are unsure.
- Write your own tests under `<ABS_REPO_DIR>/tests/`. The hidden verifier
  will run AFTER you stop; you do not get to see its output during your
  run.

## Ledger emission protocol

After each meaningful action — discovering a subtask, starting work,
completing it with evidence, hitting a block, splitting, reopening,
invalidating — append exactly one event line to
`runs/tb_live/<TASK_ID>/events.jsonl` via this helper:

```bash
uv run python <ABS_PROJECT_ROOT>/scripts/tb_emit.py \
    <ABS_RUN_DIR> \
    <step> \
    '<ledger_ops_json>'
```

`<step>` is a monotonically-increasing integer (start at 1).
`<ledger_ops_json>` is a JSON array of one or more ops.

**Op shapes:**

```json
{"op": "add", "id": "s1", "description": "Set up package skeleton", "category": "product"}
{"op": "start", "id": "s1"}
{"op": "complete", "id": "s1", "evidence": ["pytest output: 5 passed"]}
{"op": "block", "id": "s2", "reason": "need clarification on inline code in code blocks"}
{"op": "reopen", "id": "s1", "reason": "stress test failed after I marked complete"}
{"op": "invalidate", "id": "s3", "reason": "approach abandoned in favor of s4"}
{"op": "split", "id": "s2", "reason": "block parser and inline parser are independent",
 "children": [{"id": "s2a", "description": "block parser", "category": "product"},
              {"id": "s2b", "description": "inline parser", "category": "product"}]}
{"op": "add_evidence", "id": "s1", "evidence": ["diff: src/md2html/__init__.py +42 lines"]}
```

**Categories:**

- `product` — code that ships to satisfy the task spec.
- `validation` — tests, asserts, manual verification of behavior.
- `investigation` — reading existing code, searching, tracing, hypothesis
  formation.

**Discipline:**

- Add subtasks **as you discover them**, not as a plan up front.
- Mark complete **only with concrete evidence** (test output, command
  output, diff, file contents).
- Use `block` when you actually need an external condition or fact you
  don't have. Don't fake-block to defer work.
- Use `reopen` when something you marked complete turns out wrong.
- Use `split` when one vague subtask resolves into several checkable ones.

The ledger is an **observation channel**, not a planner or controller.
It records what you actually did, with timestamps. Be honest about it.

## How to know you're done

You are done when **you believe** the spec in `task.md` is satisfied
and your own tests pass. Emit a final `complete` event with evidence
("all 7 of my tests pass; ready for verifier"), then stop. The harness
will run the hidden verifier and report the outcome separately.

If you get stuck for >30 minutes on the same problem, emit a `block`
event with a clear reason and stop. A clean stop with honest state is
better than a forced "complete" with no evidence.
