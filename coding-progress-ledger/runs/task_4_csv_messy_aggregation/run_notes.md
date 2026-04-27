# Run Notes

Progress increased when the clean repeated-user regression passed and the basic
aggregation fix was in place.

Progress decreased when clean-row success proved too narrow. The messy input
case introduced newly active work for whitespace normalization, missing amounts,
and row-order-independent output. The validation subtask was then split into
three checkable leaves: clean rows, messy rows, and deterministic ordering.

Completed subtasks were backed by concrete evidence: diffs to
`aggregator.py`, the added tests in `tests/test_aggregator.py`, the sample data
files, and the final pytest output in `test_output.txt`.

The ledger was useful for separating the first apparent success from the later
messy-input requirements. It was slightly awkward to describe sample-data
inspection as work because it is not a code change, but it mattered for why the
denominator expanded.
