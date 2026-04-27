# Task 4: CSV Aggregation with Messy Input

The repository contains a small Python CSV aggregation script. It should read
rows with `user_id,amount` and write per-user totals.

Fix the implementation so it:

- trims whitespace around user IDs,
- treats missing numeric amounts as zero,
- aggregates repeated users correctly,
- emits deterministic output even when rows are out of order.

Preserve the clean-row behavior and update tests using hand-checkable CSV
examples.
