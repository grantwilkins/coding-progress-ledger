# sliding-window-rate-limiter

Build a sliding-window rate limiter as a Python package.

## What you must produce

A package importable as `ratelim` exposing exactly one public class:

```python
class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float, time_fn=time.time): ...
    def try_acquire(self, key: str = "default") -> bool: ...
    def current_count(self, key: str = "default") -> int: ...
```

## API contract

### `__init__(self, max_requests: int, window_seconds: float, time_fn=time.time)`

Creates a rate limiter that allows at most `max_requests` requests in any
rolling `window_seconds`-wide window. The `time_fn` argument is a callable
returning the current time as a float (defaults to `time.time`). Injecting
a custom `time_fn` is required for deterministic testing; your implementation
must use `time_fn()` for every timestamp operation — no direct calls to
`time.time` or `time.sleep`.

### `try_acquire(self, key: str = "default") -> bool`

Attempts to record a new request for `key`. Returns `True` if the request
is within the limit and has been recorded; returns `False` if the limit has
been reached and the request is denied. Each call that returns `True` must
record the current timestamp so that it is visible to `current_count` and
counts against subsequent `try_acquire` calls. Per-key buckets are fully
independent: denying a request on key `"a"` has no effect on key `"b"`.

### `current_count(self, key: str = "default") -> int`

Returns the number of requests recorded for `key` that fall within the
rolling window ending at `time_fn()` right now. Requests older than
`time_fn() - window_seconds` are outside the window and must not be counted.

## Implementation hint

Keep a `collections.deque` of timestamps per key. On every `try_acquire`
or `current_count` call, pop timestamps from the left of the deque while
they are older than `time_fn() - window_seconds`. After pruning, the length
of the deque is the current count. For `try_acquire`, allow the request iff
the count is strictly less than `max_requests`, then append `time_fn()` to
the deque before returning `True`.

## What is NOT required

You do not need persistence, distributed coordination, async support, thread
safety, or any storage backend. An in-memory Python dict of deques is
sufficient.

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    ratelim/
      __init__.py     # exports SlidingWindowRateLimiter
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, a `tests/`
directory, or anything else you want; only the `src/ratelim/` contract
is load-bearing.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/sliding-window-rate-limiter \
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
