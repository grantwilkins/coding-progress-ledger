# Agent Usage Protocol

Use `LedgerSession` to record discovered work while coding. The ledger is an observation channel, not a planner or controller.

```python
from ledger_progress import LedgerSession

session = LedgerSession("Fix parse_offset to accept +0530")
s1 = session.add("Understand expected offset behavior", step=1)
session.complete(s1, "Issue says +0530 means UTC+05:30", step=2)
print(session.score())
```

Protocol:

1. Start one session per user-facing coding task.
2. Add subtasks when required work becomes known, not when it is convenient to report.
3. Mark a subtask complete only with concrete evidence from code, tests, logs, diffs, or docs.
4. Use `start` for active work and `block` when progress needs an external condition or missing fact.
5. Use `split` when a vague subtask becomes several checkable leaf subtasks.
6. Use `reopen` when completed work is shown incomplete.
7. Use `invalidate` when a subtask or approach should remain in history but stop counting as active work.
8. Export JSONL at the end of the run; it is the replayable source of truth.
9. Export the CSV curve when you need a compact progress trace.

Do not call an LLM from the ledger layer. Do not ingest traces automatically. Do not treat progress as elapsed time or force it to be monotonic.

Coding progress (PRODUCT + VALIDATION + INVESTIGATION leaves) can be 1.0 while overall progress is lower if artifact or documentation leaves remain. Both are valid observations, not scoring errors.
