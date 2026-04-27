# Negative Control: Incomplete Budget-Limited Run

Run tests from `repo/`:

```sh
../../../.venv/bin/python -m pytest -q
```

This run is intentionally incomplete. `test_output.txt` records failing tests,
and `summary.json` should report `test_status: failed` and final progress below
1.00.
