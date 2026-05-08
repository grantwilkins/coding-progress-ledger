# Human baseline prompt — tb_live/fix-broken-pyproject-build

**Midpoint step:** 3
**Events visible:** 9 of 13 total

## Task

# fix-broken-pyproject-build

Build a minimal installable Python package called `smolpkg` using modern
PEP 621 `pyproject.toml` syntax and a `src/` layout.

## What you must produce

A directory that `pip install -e .` (or `uv pip install -e .`) can install
cleanly, exposing exactly one public function:

```python
import smolpkg
smolpkg.entry()   # returns "hello from smolpkg v0.1.0"
```

The verifier will:

1. Copy your repo into a fresh temp directory.
2. Create a new virtual environment with `uv venv`.
3. Run `uv pip install -e .` inside it.
4. Execute `python -c 'import smolpkg; print(smolpkg.entry())'`.
5. Assert the output is exactly `hello from smolpkg v0.1.0`.

## Required files

### `pyproject.toml`

Use PEP 621 metadata (the `[project]` table). A minimal working example
has these sections:

```
[build-system]      — specifies the build backend
[project]           — name, version, requires-python
[tool.<backend>.build.targets.wheel]
                    — tells the backend where your package lives
```

The package name must be `smolpkg` and the version must be `0.1.0`.
`requires-python` must be `>=3.9` or broader.

Common pitfalls:

- Forgetting `[build-system]` entirely — the installer cannot determine
  how to build the wheel.
- Omitting the backend's package-discovery hint, so the backend picks up
  nothing (or the wrong directory) when using a `src/` layout.
- Putting `__init__.py` directly at the repo root instead of inside
  `src/smolpkg/`.
- Mismatching the `name` in `[project]` with the actual directory name
  under `src/`.

### `src/smolpkg/__init__.py`

Must define:

```python
def entry() -> str:
    return "hello from smolpkg v0.1.0"
```

The function name, return type, and return value must match exactly
(including capitalisation and spacing). The verifier uses a string
equality check.

## Repository layout

```
<your_repo>/
  pyproject.toml
  src/
    smolpkg/
      __init__.py
```

You may add other files (`README.md`, `tests/`, etc.) but they are not
required and the verifier ignores them.

## Build backend

Use `hatchling`. Your `[build-system]` block should be:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

And add a wheel target so hatchling finds your `src/` package:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/smolpkg"]
```

Without the `packages` list, hatchling will not know to include
`src/smolpkg` in the wheel.

## Checking your work locally

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -c 'import smolpkg; print(smolpkg.entry())'
```

Expected output:

```
hello from smolpkg v0.1.0
```

If the import fails, check:

- Is `src/smolpkg/__init__.py` present?
- Does `pyproject.toml` have a valid `[build-system]` table?
- Does the `[tool.hatch.build.targets.wheel]` section point at
  `src/smolpkg`?

If the output is wrong, check that `entry()` returns the exact string
`"hello from smolpkg v0.1.0"` with no trailing whitespace or newline
beyond what `print` adds.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/fix-broken-pyproject-build \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md`
for the protocol. Use `product` for code-that-ships, `validation` for
tests / asserts / manual checks, `investigation` for reading / search /
trace work. Add subtasks as you discover them, not as a plan up front.
Mark complete only with concrete evidence.

## Done condition

You are done when the verifier exits 0 against your repo. The verifier is
hidden — you cannot read it. Your fastest path to done is to run the
install-and-import check above locally and confirm the output matches
before declaring any subtask complete.


## Ledger events visible (prefix only)

```jsonl
{"event_type": "EventType.INIT", "payload": {"root_task": "fix-broken-pyproject-build"}, "reason": null, "step": 0, "subtask_id": null, "timestamp": "2026-05-04T07:52:07.104119Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "investigation", "description": "read task spec", "parent_id": null, "weight": 1.0}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:52:07.104119Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:52:07.104119Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["task.md read: need pyproject.toml with hatchling backend, src/smolpkg/__init__.py returning \"hello from smolpkg v0.1.0\""], "status": "complete"}, "reason": null, "step": 2, "subtask_id": "i1", "timestamp": "2026-05-04T07:52:16.921399Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "product", "description": "write pyproject.toml with PEP 621 + hatchling backend", "parent_id": null, "weight": 1.0}, "reason": null, "step": 2, "subtask_id": "p1", "timestamp": "2026-05-04T07:52:16.921399Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 2, "subtask_id": "p1", "timestamp": "2026-05-04T07:52:16.921399Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["pyproject.toml written with hatchling backend and src/smolpkg package hint"], "status": "complete"}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:52:24.714727Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "product", "description": "write src/smolpkg/__init__.py defining entry()", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "p2", "timestamp": "2026-05-04T07:52:24.714727Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 3, "subtask_id": "p2", "timestamp": "2026-05-04T07:52:24.714727Z"}
```

## Predict

Given only the prefix above, fill in `human_predictions.csv` with one row per target:

```csv
run_id,target,p_success
fix-broken-pyproject-build,y_success_eventual,<your probability in [0, 1]>
fix-broken-pyproject-build,y_future_progress_drop_h5,<your probability in [0, 1]>
```
