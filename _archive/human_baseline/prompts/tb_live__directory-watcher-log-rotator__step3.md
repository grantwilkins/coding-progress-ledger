# Human baseline prompt — tb_live/directory-watcher-log-rotator

**Midpoint step:** 3
**Events visible:** 6 of 10 total

## Task

# directory-watcher-log-rotator

Build a minimal log rotation library as a Python package.

## What you must produce

A package importable as `logrotator` exposing exactly one public function:

```python
def rotate(path: str, max_bytes: int) -> list[str]
```

## Behavior

Given a single log file at `path`:

- If the file's size in bytes is **less than or equal to** `max_bytes`: do nothing and return `[]`. The original file must remain untouched.
- If the file's size exceeds `max_bytes`: split the file into chunks of at most `max_bytes` bytes each. Write the chunks as `path.0001`, `path.0002`, `path.0003`, … in lexical order. Remove the original file. Return the list of part paths created, in order.

Chunking is **byte-based** — do not respect line boundaries. Read the file in binary mode and slice into chunks of exactly `max_bytes` bytes (the last chunk may be smaller).

Concatenating all part file contents in order must recover the original file byte-for-byte.

## Naming convention

Parts use a 4-digit zero-padded suffix appended directly to the original path:

```
/tmp/app.log      →  /tmp/app.log.0001
                     /tmp/app.log.0002
                     /tmp/app.log.0003
                     ...
```

## Error handling

If `path` does not exist, `rotate` must raise `FileNotFoundError`. Do not swallow or wrap this error.

## What is NOT required

You do not need to support: directory watching, inotify/fsevents, background threads, compression, gzip output, time-based rotation, signal handling, or any configuration beyond `path` and `max_bytes`. The verifier will not exercise these.

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    logrotator/
      __init__.py     # exports `rotate`
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest` against hidden test files. You may add a `pyproject.toml`, a `tests/` directory, a `README.md`, or anything else you want; only the `src/logrotator/` contract is load-bearing.

## Quick self-check

Before declaring done, verify these cases locally:

1. A file smaller than the threshold → returns `[]`, original file still exists.
2. An empty file (0 bytes) → returns `[]`, original file still exists.
3. A file whose size equals `max_bytes` exactly → returns `[]` (uses `≤` comparison).
4. A 100 KB file with `max_bytes=30*1024` → returns 4 part paths; original removed; parts are named `.0001`–`.0004`; concatenating them equals the original bytes.
5. A path that does not exist → raises `FileNotFoundError`.

## How to track progress

You are running under the N_TB live ledger harness. After each meaningful action (subtask added, started, completed, blocked, etc.), emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/directory-watcher-log-rotator \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md` for the protocol. Use `product` for code-that-ships, `validation` for tests / asserts / manual checks, `investigation` for reading / search / trace work. Add subtasks as you discover them, not as a plan up front. Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The verifier is hidden — you cannot read it. Your fastest path to done is to write your own tests for each case in the spec above, run them, and only declare a leaf complete when the test passes.


## Ledger events visible (prefix only)

```jsonl
{"event_type": "EventType.INIT", "payload": {"root_task": "directory-watcher-log-rotator"}, "reason": null, "step": 0, "subtask_id": null, "timestamp": "2026-05-04T07:54:03.651342Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "investigation", "description": "read task spec", "parent_id": null, "weight": 1.0}, "reason": null, "step": 1, "subtask_id": "inv1", "timestamp": "2026-05-04T07:54:03.651342Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 1, "subtask_id": "inv1", "timestamp": "2026-05-04T07:54:03.651342Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["task spec read: rotate fn, size<=max no-op, chunking byte-based, 4-digit zero-padded naming, FileNotFoundError on missing path"], "status": "complete"}, "reason": null, "step": 2, "subtask_id": "inv1", "timestamp": "2026-05-04T07:54:10.390249Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "product", "description": "implement logrotator package with rotate() function", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:17.547028Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:17.547028Z"}
```

## Predict

Given only the prefix above, fill in `human_predictions.csv` with one row per target:

```csv
run_id,target,p_success
directory-watcher-log-rotator,y_success_eventual,<your probability in [0, 1]>
directory-watcher-log-rotator,y_future_progress_drop_h5,<your probability in [0, 1]>
```
