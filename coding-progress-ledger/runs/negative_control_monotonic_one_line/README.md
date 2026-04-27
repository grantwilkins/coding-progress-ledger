# Negative Control: Monotonic One-Line Fix

Run tests from `repo/`:

```sh
../../../.venv/bin/python -m pytest -q
```

This run is a control case: progress should increase monotonically and reach
1.00 only after the validation test passes.
