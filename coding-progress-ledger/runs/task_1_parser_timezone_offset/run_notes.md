# Run Notes

Progress increased when the repo scaffold, tests, baseline failure confirmation,
parser child tasks, compatibility check, and artifact export were completed.

Progress decreased at step 8 after discovering explicit work to preserve the
already-working colon form while adding compact syntax. It decreased again at
step 15 when artifact export was reopened because `summary.json` still needed
computed progress-drop metrics.

New work was discovered because a compact-offset fix could plausibly regress
the existing `+05:30` behavior, and because the final summary required metrics
derived from the ledger rather than hand-waved values.

Completion evidence was attached to each completed subtask: file creation,
test contents, observed baseline failures, parser behavior covered by tests,
final pytest success, and artifact generation.

The ledger felt useful for making the late compatibility check visible and for
recording non-monotonic progress. It was slightly awkward for such a tiny patch
because the bookkeeping is larger than the implementation, but that is useful
for an empirical progress-ledger fixture.
