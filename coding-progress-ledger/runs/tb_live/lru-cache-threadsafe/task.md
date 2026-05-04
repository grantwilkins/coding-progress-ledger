# lru-cache-threadsafe

Build a thread-safe LRU cache as a Python package.

## What you must produce

A package importable as `lru_cache_ts` with a single public class:

```python
from lru_cache_ts import LRUCache
```

The package lives under `src/lru_cache_ts/__init__.py` (standard `src` layout).

## Public API

### `LRUCache.__init__(self, maxsize: int)`

Create an LRU cache that holds at most `maxsize` entries. `maxsize` is
guaranteed to be ≥ 1.

### `LRUCache.get(self, key) -> value | None`

Return the value stored under `key`, or `None` if `key` is not present.
A successful hit makes `key` the most-recently-used entry.

### `LRUCache.put(self, key, value) -> None`

Insert or update `key` with `value`. After the call the key is the
most-recently-used entry. If the cache is already at `maxsize` before a
new (absent) key is inserted, evict the least-recently-used entry first.
Updating an existing key does not trigger eviction.

### `LRUCache.__len__(self) -> int`

Return the number of entries currently stored (0 ≤ len ≤ maxsize).

### `LRUCache.__contains__(self, key) -> bool`

Return whether `key` is present. This must NOT change the recency order
of any entry.

## Behavior details

**Eviction order.** The eviction policy is strict LRU: the entry that
was least recently *used* (read via `get` or written via `put`) is
removed when capacity is exceeded. `__contains__` is explicitly excluded
from "use" — it is a non-mutating membership test.

**Update recency.** Calling `put` on a key that already exists updates
both the stored value and the key's recency (it becomes most-recently-used).

**Miss returns None.** `get` on an absent key returns `None` and does not
alter cache state.

## Thread safety

All four operations (`get`, `put`, `__len__`, `__contains__`) must be
safe to call concurrently from multiple threads with no external
synchronisation. Under concurrent load:

- No exception may propagate out of any operation.
- `len(cache)` must never exceed `maxsize`.
- A value retrieved by `get` must equal the last value passed to `put`
  for that key at the time of the read (no torn writes).

You may use `threading.Lock` or any other stdlib primitive. The lock
need not be reentrant unless your design requires it.

## Repository layout

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. Place your implementation here:

```
<agent_repo>/
  src/
    lru_cache_ts/
      __init__.py    # must export LRUCache
```

You may add `pyproject.toml`, a `tests/` directory, or anything else;
only `src/lru_cache_ts/__init__.py` is load-bearing.

## What is NOT required

You do not need a CLI, `__main__.py`, TTL expiry, per-key callbacks,
cache statistics, or thread-local storage. The verifier will not
exercise these.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/lru-cache-threadsafe \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md`
for the protocol. Use `product` for code-that-ships, `validation` for
tests / asserts / manual checks, `investigation` for reading / search /
trace work. Add subtasks as you discover them, not as a plan up front.
Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The
verifier is hidden — you cannot read it. Your fastest path to done is
to write your own tests for each behaviour in the spec above, run them,
and only declare a leaf complete when the test passes.
