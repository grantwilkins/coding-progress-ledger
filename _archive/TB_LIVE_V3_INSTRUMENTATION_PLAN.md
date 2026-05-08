# TB Live V3 Instrumentation Plan

## Goal

Preserve validation, error, oracle-read, path-mismatch, and
verifier-disagreement structure without changing upstream ledger scoring
or progress semantics.

## Emission plan

1. Keep `transcript.jsonl` as the raw agent trace.
2. Keep `events.jsonl -> ledger.jsonl` as the existing sidecar path.
3. Add `observation_events.jsonl` during runner finalization.
4. Backfill the same schema for frozen `tb_live_v2` so evaluation can
   compare `time_only`, `ledger_basic`, and `observation_basic` on the
   same corpus.

## Mapping

- Transcript shell validation/test/check commands -> `validation_attempt`
- Transcript nonzero shell failures / error snippets -> `error_observed`
- Repeated normalized error signatures -> `error_repeated`
- `read_file` on `solution.sh` -> `solution_oracle_read`
- `write_file` / `edit_file` -> `product_file_written`
- `done` line -> `agent_claims_done`
- Final verifier outcome from `run_manifest.json` / `verifier_output.txt`
  -> `verifier_pass` or `verifier_fail`
- Done claim before verifier failure -> `verifier_disagreement`
- Done claim with no expected-path writes -> `expected_file_missing`

## Guardrails

- No mutation of ledger semantics.
- No use of verifier terminal events at preterminal checkpoints.
- No use of `arm` / `model_name` in the headline estimator.
- No new same-schema live collection until this observation channel is
  live and evaluated.
