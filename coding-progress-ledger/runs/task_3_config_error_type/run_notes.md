# Run Notes

## Progress Changes

The run began with five concrete active subtasks. Progress rose as the toy repo
and tests were completed, then dropped when the broad fix subtask was reopened
and split after evidence showed only one of two ValueError raise sites had been
converted. The largest observed progress drop was 0.100000.

## New Work Discovery

The second ValueError site was discovered after the first narrow fix and a
pytest rerun. That changed the work model from one broad implementation task to
two branch-specific checks: missing required keys and wrong timeout type.

## Evidence-Backed Completions

The initial repo completion is backed by a git commit containing two wrong
ValueError raise sites. The test completion is backed by failing pytest output
on the baseline. The final verification completion is backed by the passing
`../../../.venv/bin/python -m pytest` output in `test_output.txt`.

## Ledger Notes

The ledger was useful for making the non-monotonic discovery explicit instead
of treating the first partial fix as steady progress. The awkward part is that
artifact export itself is real work but the ledger files are only available
after the final ledger event, so the export completion evidence has to describe
the pending final side effect.
