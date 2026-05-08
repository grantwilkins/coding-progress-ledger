# Human baseline prompt — tb_live/b-tree-on-disk

**Midpoint step:** 3
**Events visible:** 9 of 16 total

## Task

# btree

Build an in-memory ordered map as a Python package called `btree`.

## What you must produce

A package importable as `btree` exposing exactly one public class:

```python
class BTree:
    def __init__(self, order: int = 4): ...
```

The `order` parameter is the maximum number of children per internal node
(the canonical B-tree branching factor). It is part of the constructor
signature, but the hidden verifier tests correctness only — you do not need
to implement a true B-tree internally. A sorted dict or bisect-based
structure that satisfies the contract below will pass.

## Public API

Every method listed here is exercised by the hidden verifier.

```python
def insert(self, key: int, value) -> None
```

Insert or update. If `key` already exists, replace its associated value.
This operation must be O(log n) in expectation for reasonable implementations.

```python
def get(self, key: int) -> value | None
```

Return the value associated with `key`, or `None` if the key is not present.

```python
def __contains__(self, key: int) -> bool
```

Return `True` if `key` is present, `False` otherwise. Enables `key in tree`.

```python
def __len__(self) -> int
```

Return the number of distinct keys currently stored.

```python
def range(self, lo: int, hi: int) -> list[tuple[int, value]]
```

Return all `(key, value)` pairs where `lo <= key < hi` (half-open interval),
sorted ascending by key. Return an empty list if no keys fall in the range,
or if `lo >= hi`.

```python
def keys(self) -> list[int]
```

Return all keys in sorted ascending order as a plain Python list.

## Behavior details

- `insert` with a duplicate key must update, not add. After
  `insert(5, "a"); insert(5, "b")`, `get(5)` returns `"b"` and `len` is 1.
- `range(lo, hi)` is a half-open interval: `lo` is inclusive, `hi` is
  exclusive. `range(2, 5)` includes keys 2, 3, 4 — not 5.
- `range(lo, hi)` where `lo >= hi` returns `[]`.
- `range` with bounds that fall entirely outside the stored key range still
  returns the correct subset (possibly empty).
- Values may be any Python object (strings, integers, dicts, etc.).
- Keys are always integers.

## Repository layout

Use the standard `src/` layout:

```
<agent_repo>/
  src/
    btree/
      __init__.py     # exports BTree
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, local tests, or
anything else; only the `src/btree/__init__.py` contract is load-bearing.

## Suggested self-test before submitting

Before declaring yourself done, run at minimum these checks locally:

```python
from btree import BTree

t = BTree()
assert len(t) == 0
assert t.get(1) is None
assert t.range(0, 10) == []

t.insert(5, "hello")
assert t.get(5) == "hello"
assert 5 in t
assert len(t) == 1

t.insert(5, "world")
assert t.get(5) == "world"
assert len(t) == 1

for i in range(10):
    t.insert(i, i * 2)
assert t.range(2, 5) == [(2, 4), (3, 6), (4, 8)]
assert t.range(5, 2) == []
assert t.keys() == list(range(10))
```

## How to track progress

You are running under the N_TB live ledger harness. After each meaningful
action (subtask added, started, completed, blocked, etc.), emit one
wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/b-tree-on-disk \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md`
for the protocol. Use `product` for code-that-ships, `validation` for
tests / asserts / manual checks, `investigation` for reading / search /
trace work. Add subtasks as you discover them, not as a plan up front.
Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The verifier is
hidden — you cannot read it. Your fastest path to done is to write your own
tests for each behavior in the spec above, run them, and only declare a leaf
complete when the test passes.


## Ledger events visible (prefix only)

```jsonl
{"event_type": "EventType.INIT", "payload": {"root_task": "b-tree-on-disk"}, "reason": null, "step": 0, "subtask_id": null, "timestamp": "2026-05-04T07:54:13.653714Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "investigation", "description": "read task spec", "parent_id": null, "weight": 1.0}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:54:13.653714Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:54:13.653714Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["task.md read: need btree package with BTree class, insert/get/__contains__/__len__/range/keys methods, src/ layout"], "status": "complete"}, "reason": null, "step": 2, "subtask_id": "i1", "timestamp": "2026-05-04T07:54:22.985837Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "product", "description": "create src/btree/__init__.py with BTree implementation", "parent_id": null, "weight": 1.0}, "reason": null, "step": 2, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:22.985837Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 2, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:22.985837Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["src/btree/__init__.py created with bisect-based BTree: insert/get/__contains__/__len__/keys/range"], "status": "complete"}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:39.041102Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "validation", "description": "write pytest tests covering all API behaviors", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "v1", "timestamp": "2026-05-04T07:54:39.041102Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 3, "subtask_id": "v1", "timestamp": "2026-05-04T07:54:39.041102Z"}
```

## Predict

Given only the prefix above, fill in `human_predictions.csv` with one row per target:

```csv
run_id,target,p_success
b-tree-on-disk,y_success_eventual,<your probability in [0, 1]>
b-tree-on-disk,y_future_progress_drop_h5,<your probability in [0, 1]>
```
