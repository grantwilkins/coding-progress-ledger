# Task 8 Package Import Failure

The toy repository is in `repo/`.

From `repo/`, run:

```bash
python -m widget_runner.module
python tests/test_imports.py
python -c "from widget_runner import build_message; print(build_message(' ada   lovelace '))"
```

In this execution environment, `python` is unavailable, so `test_output.txt`
records the same commands run with `python3`.

Artifacts:

- `task.md`: task statement.
- `agent_transcript.md`: concise simulated coding transcript.
- `ledger.jsonl`: replayable `LedgerSession` event log.
- `progress.csv`: progress curve exported from the ledger.
- `final_diff.patch`: diff from the intentionally buggy initial repo to the final solved repo.
- `test_output.txt`: actual command output from the final repo.
- `run_notes.md`: notes on progress changes and ledger usefulness.
- `summary.json`: machine-readable run summary.
