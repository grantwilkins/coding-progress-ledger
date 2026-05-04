# tasks/tb_live

Live benchmark tasks for coding-run instrumentation.

## Per-task structure

- `task.md`: prompt/spec given to the coding agent.
- `verifier.sh`: deterministic checker used to score correctness.
- `verifier_tests/` (optional): unit/integration tests for verifier behavior.
- `solution_reference/` (optional): minimal reference implementation.

## Included tasks

Directory names encode the objective (for example `lru-cache-threadsafe`, `markdown-to-html-cli`, `xss-filter-bypass-then-fix`).

Use these tasks as stable inputs for comparing live ledger instrumentation behavior across runs.
