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
