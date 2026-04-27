# TASK 8: Package import failure

Create a tiny Python package with an intentionally buggy internal import, then
fix it so package/module execution and package import work from the repo root.

Required evidence:

- `python -m widget_runner.module`
- direct test invocation with `python tests/test_imports.py`
- package import with `python -c "from widget_runner import build_message; ..."`

The simulated coding run is recorded with `LedgerSession` and exported as
`ledger.jsonl` and `progress.csv`.
