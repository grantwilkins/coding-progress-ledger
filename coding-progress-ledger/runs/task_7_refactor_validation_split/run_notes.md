# Run Notes

## Progress Changes

The ledger started with five active subtasks. Progress rose as the repo,
baseline implementation, refactor, and first validation pass completed. It then
dropped when the broad validation task was split into three concrete leaves:
targeted unit tests, broader regression tests, and API compatibility checks.
The largest observed progress drop was 0.228571.

## Validation Split

The validation task was deliberately vague at first. After the helper extraction
passed the targeted tests, the work model changed to expose the missing evidence:
broader regression tests for accounting invariants and API checks for callers
that import and invoke `summarize_invoice`.

## Evidence-Backed Completions

The initial implementation completion is backed by passing baseline pytest
output and the initial git commit. The refactor completion is backed by targeted
tests passing after extracting `_normalize_item` and `_line_totals`. The final
validation leaves are backed by full pytest output captured in `test_output.txt`.

## Ledger Notes

The ledger was useful because the split made incomplete validation visible as a
non-monotonic progress event instead of allowing one broad checkbox to hide the
remaining test work. The awkward part is that final artifact export is itself a
task, but `ledger.jsonl` and `progress.csv` can only be exported after the last
ledger event, so the artifact completion evidence describes that pending final
side effect.
