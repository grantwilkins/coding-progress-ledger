# decouple-state-from-controller

Build a pure data-pipeline function as a Python package.

## What you must produce

A package importable as `pipeline` exposing exactly one public function:

```python
def process(items: list[dict], config: dict) -> list[dict]
```

`process` takes a list of dicts and a configuration dict, runs the items
through a fixed sequence of pure stages, and returns a new list of dicts.
It must never mutate the input list or any input dict.

## Repository layout

Use the standard `src/` layout:

```
<agent_repo>/
  src/
    pipeline/
      __init__.py     # exports `process`
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, a `tests/`
directory, or anything else you want; only the `src/pipeline/` contract
is load-bearing.

## Pipeline stages

Run the following stages in order. Every stage is pure (no I/O, no
global state). The same input always produces the same output.

### 1. Filter

Keep only items where `item["status"] == config["accept_status"]`.

If `config` does not contain `"accept_status"`, default to `"ok"`.

### 2. Project

Keep only the keys listed in `config["fields"]` on each surviving item.

If `config` does not contain `"fields"`, keep all keys on each item.

### 3. Normalize

For every value in each item that is a `str`, strip leading/trailing
whitespace and lowercase the result.

### 4. Tag

Add the key `"_pipeline_version"` to each item. Its value is
`config.get("version", "v1")`.

### 5. Sort

If `config` contains `"sort_key"`, sort the list ascending by that key
(stable sort). Otherwise preserve the order coming out of the previous
stage.

### 6. Limit

Take only the first `config.get("limit", len(items))` items from the
list, where `len(items)` is the length of the original input list (i.e.
no limit by default).

## Exact behaviour notes

- **Purity**: `process` must return a fresh list of fresh dicts. After
  the call, the caller's original `items` list must be unchanged, and
  none of the original dicts inside it may have been mutated.
- **Stage order matters**: Filter first, then Project, then Normalize,
  then Tag, then Sort, then Limit.
- **Projection does not add keys**: if a key in `config["fields"]` does
  not exist on an item, it is silently omitted from that item's output
  dict (do not raise, do not insert a `None`).
- **Normalize is shallow**: only top-level string values are normalized;
  nested structures are left as-is.
- **Tag always runs**: `"_pipeline_version"` is added to every item that
  survives the filter, even if it was already in the original dict.
- **Sort is stable**: items with equal values for `sort_key` preserve
  their relative order from the input.
- **Limit uses original `len(items)`**: pass `len(items)` as the default
  for `config.get("limit", ...)`, not the count after filtering.

## Example

```python
from pipeline import process

items = [
    {"status": "ok",  "name": "  Alice ", "score": 3},
    {"status": "err", "name": "Bob",      "score": 1},
    {"status": "ok",  "name": " Carol",   "score": 1},
]
config = {
    "accept_status": "ok",
    "fields": ["name", "score"],
    "version": "v2",
    "sort_key": "score",
    "limit": 2,
}
result = process(items, config)
# result == [
#     {"name": "alice", "score": 1, "_pipeline_version": "v2"},
#     {"name": "alice", "score": 3, "_pipeline_version": "v2"},
# ]
```

Wait — Carol normalizes to `"carol"`, not `"alice"`. Corrected:

```python
# result == [
#     {"name": "carol", "score": 1, "_pipeline_version": "v2"},
#     {"name": "alice", "score": 3, "_pipeline_version": "v2"},
# ]
```

## What is NOT required

You do not need: CLI entry point, async support, nested-dict
normalization, error handling for malformed config keys, or any I/O.
The verifier will not exercise these.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/decouple-state-from-controller \
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
to write your own tests for each stage described above, run them, and
only declare a leaf complete when the test passes.
