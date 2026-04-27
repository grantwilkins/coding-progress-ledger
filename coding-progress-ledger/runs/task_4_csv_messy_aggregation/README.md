# Task 4 Run

Toy repo: `repo/`

Run the tests from `repo/`:

```sh
../../../.venv/bin/python -m pytest -q
```

The intentionally buggy baseline is the initial git commit inside `repo/`.
`final_diff.patch` is the diff from that baseline to the solved version.

Inspect the progress ledger:

```sh
python -m json.tool ../ledger.jsonl
cat ../progress.csv
```
