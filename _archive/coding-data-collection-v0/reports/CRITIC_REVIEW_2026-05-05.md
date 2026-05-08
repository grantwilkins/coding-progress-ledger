# Critic Review - 2026-05-05

Three small critic agents reviewed the initial `coding-data-collection`
implementation.

## Main Findings

- `prepare_run.py` copied directories with only top-level name filtering,
  so nested hidden files and symlinks could enter `agent_workspace`.
- Artifact completeness treated all non-terminal statuses as partial,
  which could allow `agent_timeout` or verifier failures into analysis
  without transcript/ledger artifacts.
- Leakage scanning was too narrow and name-only.
- `TASKS.md` overstated maturity by marking scaffolds as in progress
  without distinguishing real executable collection work.
- Existing tests did not sufficiently pin sparse-step verifier timing,
  visibility flags, status-specific artifact requirements, nested leakage,
  task-scoring arithmetic, or ledger wire event shape.

## Fixes Applied

- Recursive hidden-path exclusion and symlink skipping in `prepare_run.py`.
- `tests/` and content-marker leakage detection.
- Status-specific artifact requirements in `protocol.py`.
- `finalize_run.py` records missing verifier result as
  `infrastructure_failure`, not `completed_failure`.
- Added `tests/test_semantic_contracts.py` using the research-test-creator
  standard: hand-checkable and boundary tests for the dangerous contracts.
- Rewrote `TASKS.md` to distinguish shipped scaffold from real blocked work.

## Verification

```bash
env UV_PROJECT_ENVIRONMENT=.venv312 uv run pytest tests
# 21 passed
```

