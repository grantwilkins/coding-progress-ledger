# Run Notes

## Progress Changes

The run started with seven concrete subtasks: scaffold repo, write tests, fix
reset behavior, capture artifacts, export ledger artifacts, document the run,
and add TypeScript tooling if plain JS proved insufficient. The TypeScript
tooling branch was invalidated after `node:test` met the lightweight reducer
requirement without dependency setup.

Progress was intentionally non-monotonic. The broad reset fix was first marked
complete after submitted and derived state were cleared. A follow-up test run
showed validation error still persisted, so that subtask was reopened and split
into two checkable reset leaks. This produced a largest progress drop of
`0.16666666666666669` in `progress.csv`.

## New Work Discovery

The first failing test run showed both stale submitted state and stale error
state. The intermediate fix addressed submitted and derived values only. The
remaining failing assertion discovered that validation state was a separate
reset concern, so the reset work was split.

## Evidence-Backed Completions

- Repo scaffold completion: `repo/package.json`, `repo/src/reducer.js`, and
  `repo/test/reducer.test.js` exist and the buggy baseline was committed.
- Test completion: initial `npm test` failed on stale submitted state and stale
  validation error, matching the intended bug.
- Fix completion: final `npm test` passed all four reducer tests.
- Artifact completion: `test_output.txt`, `final_diff.patch`, `ledger.jsonl`,
  `progress.csv`, and `summary.json` exist at run root.

## Ledger Notes

The ledger was useful for representing the real regression-discovery shape:
after one apparent reset fix, the remaining validation-state failure forced a
reopen and split. The awkward part is that narrative/documentation artifacts
are easiest to describe after the run, so the ledger records their completion
as an observed artifact milestone rather than as every keystroke.
