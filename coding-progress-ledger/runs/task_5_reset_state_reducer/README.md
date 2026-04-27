# Task 5 Reset-State Reducer Run

This run creates a tiny reducer-only frontend state bug reproduction in
`repo/`. The committed baseline intentionally resets only the visible count.
The final reducer resets to `initialState`, clearing visible, submitted,
derived, and validation-error state.

Run the verification from the repo directory:

```sh
cd repo
npm test
```

Primary artifacts:

- `ledger.jsonl`: replayable LedgerSession event log.
- `progress.csv`: progress curve exported by LedgerSession.
- `final_diff.patch`: diff from the intentionally buggy baseline commit to the final fix.
- `test_output.txt`: captured output from the passing `npm test` run.
