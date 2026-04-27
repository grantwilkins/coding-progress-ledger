# Run Notes

## Progress Changes

The run began with five concrete subtasks: create the fixture repo, write tests,
fix imports, verify commands, and export artifacts. While working, the broad
test subtask was split into separate command-surface checks for module
execution, direct test invocation, and package import.

## New Work Discovery

Fixing `python -m widget_runner.module` with a relative import made the old
script-from-inside-package style no longer part of the supported contract. That
compatibility concern was reopened after verification, then invalidated as
out-of-scope because the task asks for package/module execution from repo root,
direct test invocation, and package import.

The first verification attempt also discovered that `pytest` is not installed in
this environment. The test harness was adjusted to stdlib `unittest`, which
keeps the commands deterministic and dependency-free.

## Evidence-Backed Completions

The final evidence is in `test_output.txt`: module execution prints the expected
message, direct test invocation runs two tests successfully, and package import
prints the same expected message.

## Ledger Notes

The ledger was useful for representing that apparent progress dropped when
completed import work was reopened after the compatibility question. The
awkward part is that environment-level command naming (`python` unavailable,
`python3` available) is not really product progress, so it is captured in notes
rather than modeled as a separate active coding task.
