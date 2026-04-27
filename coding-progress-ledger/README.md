# Coding Progress Ledger

Deterministic, append-only event log for coding tasks. Computes:

```text
completed active leaf work / total active discovered leaf work
```

Progress decreases when new work is discovered, completed work is reopened, or work is invalidated. The event log is the replayable source of truth. No LLM, no agent, no forced monotonicity.

## Install

```sh
pip install -e .
```

## Quickstart

`LedgerSession` is the primary API:

```python
from ledger_progress import LedgerSession

session = LedgerSession("Fix timezone parser")
s1 = session.add("Locate parser code", step=1)
s2 = session.add("Patch offset regex", step=1)

session.start(s1, step=2)
session.complete(s1, "Found lib/parser.py:88", step=3)
print(session.score().progress)  # 0.5

session.complete(s2, "pytest -q: 4 passed", step=4)
print(session.score().progress)  # 1.0

session.export_jsonl("ledger.jsonl")   # replayable source of truth
session.export_curve_csv("progress.csv")
```

## Progress Is Not Success

Progress is the fraction of active, discovered leaf work with completion evidence. It does not indicate whether the solution is correct, whether tests are sufficient, or whether undiscovered requirements are covered.

A failed run can have high progress if most known leaves were complete before one defect remained. A successful run can have progress < 1.0 if artifact or documentation leaves are still active. Treat final success, test status, and evidence quality as separate metadata.

## Event Types

| Event | When to use |
| --- | --- |
| `ADD_SUBTASK` | discover required work |
| `UPDATE_STATUS` | mark in_progress, blocked, or complete |
| `ADD_EVIDENCE` | attach proof without changing status |
| `SPLIT_SUBTASK` | break a vague task into checkable leaves |
| `REOPEN_SUBTASK` | completed work shown to be incomplete |
| `INVALIDATE_SUBTASK` | wrong approach; stays in history, stops counting |
| `DELETE_SUBTASK` | duplicate or unnecessary; stays in history |

## Scoring

Only leaf subtasks count (nodes with no active children). Filter by category to measure coding progress separately from housekeeping work:

```python
from ledger_progress import score, CODING_CATEGORIES  # PRODUCT, VALIDATION, INVESTIGATION

obs = score(ledger, categories=CODING_CATEGORIES)
print(obs.progress, obs.complete_leaf_count, obs.active_leaf_count)
```

## CLI

```sh
ledger-run init-run <run_dir>                   # scaffold run directory
ledger-run export-run <run_dir>                 # regenerate CSVs from ledger.jsonl
ledger-run capture-tests <run_dir> -- <cmd>     # run tests and save output
ledger-run capture-diff <run_dir>               # git diff → final_diff.patch
ledger-run check-run <run_dir>                  # verify all artifacts present
ledger-run summarize-run <run_dir>              # print run summary
```

## Runs

`runs/` contains eight toy benchmark runs and two negative controls with full ledger artifacts. See [runs/README.md](runs/README.md).
