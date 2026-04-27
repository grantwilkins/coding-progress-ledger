# Coding Progress Ledger Toy Benchmark Runs

This directory contains eight self-contained toy coding runs plus two negative
controls. Each benchmark run lives in `task_*/`; negative controls live in
`negative_control_*/`. Each run includes:

- `task.md`: user-facing task description,
- `README.md`: task-specific reproduction notes,
- `agent_transcript.md`: compact run transcript,
- `ledger.jsonl`: append-only `LedgerSession` event log,
- `progress.csv`: per-step progress curve,
- `final_diff.patch`: solved repo diff from the intentionally buggy baseline,
- `test_output.txt`: captured final test output,
- `run_notes.md`: progress interpretation,
- `summary.json`: machine-readable metrics.

## Run Tests

Run each command from the listed toy repo directory.

| Run | Directory | Test command |
| --- | --- | --- |
| Parser timezone offset | `task_1_parser_timezone_offset/repo` | `../../../.venv/bin/python -m pytest -q` |
| CLI output flag | `task_2_cli_output_flag/repo` | `../../../.venv/bin/python -m pytest -q` |
| Config exception type | `task_3_config_error_type/repo` | `../../../.venv/bin/python -m pytest -q` |
| CSV messy aggregation | `task_4_csv_messy_aggregation/repo` | `../../../.venv/bin/python -m pytest -q` |
| Reset-state reducer | `task_5_reset_state_reducer/repo` | `npm test` |
| Async stale result | `task_6_async_stale_result/repo` | `../../../.venv/bin/python -m pytest -q` |
| Refactor validation split | `task_7_refactor_validation_split/repo` | `../../../.venv/bin/python -m pytest -q` |
| Package import failure | `task_8_package_import_failure/repo` | `python3 -m widget_runner.module && python3 tests/test_imports.py && python3 -c "from widget_runner import build_message; print(build_message(' ada   lovelace '))"` |
| Negative control: monotonic one-line fix | `negative_control_monotonic_one_line/repo` | `../../../.venv/bin/python -m pytest -q` |
| Negative control: incomplete budget-limited run | `negative_control_incomplete_budget_limited/repo` | `../../../.venv/bin/python -m pytest -q` expects failure |

## Inspect Ledgers

Each `ledger.jsonl` is newline-delimited JSON. Inspect the raw replayable event
source:

```sh
sed -n '1,20p' task_1_parser_timezone_offset/ledger.jsonl
```

Inspect the compact progress curve:

```sh
cat task_1_parser_timezone_offset/progress.csv
```

The progress value is:

```text
completed active discovered leaf work / total active discovered leaf work
```

Drops are expected when new active work is discovered, a completed subtask is
reopened, a vague subtask is split into concrete children, or prior work is
invalidated.

## Regenerate Per-Run Summaries

Some runs include helper scripts used to build their ledger artifacts:

```sh
../../.venv/bin/python build_run.py
```

Run that command from a task directory when `build_run.py` exists. Task 1 uses
`generate_ledger.py` similarly. For runs without a helper script, `summary.json`
is derived directly from `ledger.jsonl` and `progress.csv`; inspect or recompute
the same fields with `ledger_progress.from_jsonl` and the CSV drop calculation.

## Suite Summary

See `SUITE_SUMMARY.md` for aggregate metrics, largest progress drops, and notes
on where the ledger was useful or awkward.

See `LEDGER_AUDIT.md` for the evidence-quality audit and negative-control
interpretation.
