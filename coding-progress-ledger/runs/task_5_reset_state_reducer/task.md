# TASK 5: Frontend Reset-State Bug

Create a tiny reducer-only reproduction of a frontend reset bug. The initial
implementation resets the visible count but leaves stale internal submitted or
derived state. The final implementation must reset the whole state, including
validation errors.

Constraints:

- Keep all writes inside `runs/task_5_reset_state_reducer/`.
- Use `LedgerSession` and export `ledger.jsonl` plus `progress.csv`.
- Keep tests deterministic and lightweight.
- Avoid changing ledger scoring or adding LLM calls.
