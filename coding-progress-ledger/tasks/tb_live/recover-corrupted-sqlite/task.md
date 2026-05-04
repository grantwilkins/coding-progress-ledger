# recover-corrupted-sqlite

Implement a best-effort SQLite recovery function that returns as many rows as possible from a truncated database file.

## Background

SQLite databases can be corrupted in various ways. One common scenario is truncation — the file is cut short, losing the tail pages. SQLite's storage engine stores rows in B-tree pages; a truncated file still has its early pages intact and SQLite can often read them without error. Your job is to exploit that tolerance and surface whatever rows are readable.

## What you must produce

A package importable as `sqlite_recover` exposing exactly one public function:

```python
def recover(path: str) -> list[dict]
```

`recover` takes a filesystem path to a SQLite database file and returns a list of row dictionaries. Each dictionary has exactly these keys:

```python
{"id": int, "name": str, "value": int}
```

The rows must be returned in ascending `id` order.

### Behaviour contract

| Situation | Expected behaviour |
|---|---|
| Normal intact database | Return all rows, sorted by `id` |
| Truncated database | Return however many rows SQLite can read, sorted by `id`. Do NOT raise. |
| File does not exist | Raise `FileNotFoundError` |
| File is empty (0 bytes) | Return `[]` |

## The table schema

The database this function must read always has exactly one table:

```sql
CREATE TABLE records (
    id    INTEGER PRIMARY KEY,
    name  TEXT,
    value INTEGER
);
```

`recover` must return all rows from this table, in `id` order, as a list of dicts. If the file is truncated and SQLite raises `sqlite3.DatabaseError` mid-query, swallow the error and return whatever rows were collected before the exception.

## Implementation sketch

A minimal working implementation:

1. Check the path exists — raise `FileNotFoundError` if not.
2. Check the file is non-empty — return `[]` immediately if size is 0.
3. Open with `sqlite3.connect(path)`.
4. Execute `SELECT id, name, value FROM records ORDER BY id`.
5. Collect rows one at a time with `cursor.fetchone()` inside a `try/except sqlite3.DatabaseError` loop, appending each to a results list.
6. Return the list of dicts.

You do **not** need to repair the database, reconstruct missing pages, handle WAL files, or recover deleted rows. Best-effort means: return what SQLite gives you before it complains.

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    sqlite_recover/
      __init__.py     # exports `recover`
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest` against hidden test files. You may add a `pyproject.toml`, a `tests/` directory, or anything else you like; only `src/sqlite_recover/__init__.py` is load-bearing.

## What is NOT required

You do not need to: handle WAL or journal files, repair B-tree pages, reconstruct rows from raw page bytes, support schemas other than `records(id, name, value)`, or handle concurrency. The verifier exercises only the four scenarios in the behaviour table above.

## How to track progress

You are running under the N_TB live ledger harness. After each meaningful action (subtask added, started, completed, blocked, etc.), emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/recover-corrupted-sqlite \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md` for the protocol. Use `product` for code-that-ships, `validation` for tests / asserts / manual checks, `investigation` for reading / search / trace work. Add subtasks as you discover them, not as a plan up front. Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The verifier is hidden — you cannot read it. Your fastest path to done is to write your own tests for each scenario in the spec above, run them, and only declare a leaf complete when the test passes.
