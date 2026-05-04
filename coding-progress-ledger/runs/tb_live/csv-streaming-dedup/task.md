# csv-streaming-dedup

Build a CSV deduplicator as a Python package and CLI.

## What you must produce

A package importable as `csvdedup` and runnable as a module:

```bash
python -m csvdedup path/to/input.csv   # reads file, writes deduplicated CSV to stdout
python -m csvdedup                      # reads from stdin if no path argument
```

The package only needs to support the CLI interface above. No public Python API beyond what is needed to make the module runnable is required.

## Behavior

- The first row is always the header. It is written to stdout verbatim and is never treated as a duplicate.
- Every subsequent row is compared against all previously seen rows as a full-row tuple (all fields, in order). The first occurrence of each distinct row is kept; all later occurrences of the same row are dropped.
- Rows differing in any field — including case or whitespace — are considered distinct.
- Output rows are written in the order they first appear.
- Output is written to stdout.
- Use Python's stdlib `csv` module. No third-party dependencies are allowed.
- Streaming is the intended implementation style: process rows one at a time rather than reading the entire file into memory before writing any output. The verifier does not enforce a memory cap, but the design goal is that large files should work with constant memory (excluding the seen-set).

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    csvdedup/
      __init__.py
      __main__.py
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest` against hidden test files. You may add a `pyproject.toml`, a `tests/` directory, or anything else you want; only the `src/csvdedup/` contract is load-bearing.

## Output format

The verifier compares `result.stdout == expected` (exact string equality). To match:

- Use `\n` line endings (not `\r\n`). When constructing a `csv.writer`, pass `lineterminator="\n"`.
- When reading an input file, open it with `newline=""` so Python does not translate line endings before the csv module sees them.
- Fields that contain commas must be quoted in the output (the csv module does this automatically when you write the row back).
- The final output row must be followed by a `\n` (the csv writer adds this automatically).

## Example

Input (`input.csv`):

```
name,age,city
Alice,30,NYC
Bob,25,LA
Alice,30,NYC
Charlie,35,Chicago
Bob,25,LA
```

Running `python -m csvdedup input.csv` must produce:

```
name,age,city
Alice,30,NYC
Bob,25,LA
Charlie,35,Chicago
```

## What the hidden verifier exercises

The hidden tests run your CLI via `subprocess` against fixture CSV files and compare stdout to expected output files. The tests cover:

- A file with several duplicate rows interleaved with unique rows.
- A file with no duplicates (all rows pass through unchanged).
- A file that contains only a header row and no data rows (output is just the header).
- A file with quoted fields containing commas (the csv module handles quoting; your deduplication must compare the parsed field values, not the raw text).
- A file where every data row is identical (output is header plus one row).
- A file with five columns where duplicates match on all five fields.
- Stdin mode: the same input piped via stdin instead of a file path.

## How to track progress

You are running under the N_TB live ledger harness. After each meaningful action (subtask added, started, completed, blocked, etc.), emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/csv-streaming-dedup \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md` for the protocol. Use `product` for code-that-ships, `validation` for tests / asserts / manual checks, `investigation` for reading / search / trace work. Add subtasks as you discover them, not as a plan up front. Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The verifier is hidden — you cannot read it. Your fastest path to done is to write your own tests for each feature in the spec above, run them, and only declare a leaf complete when the test passes.
