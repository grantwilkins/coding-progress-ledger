# Terminal-Bench V10 OpenAI M1 Evaluation

Evaluated: 2026-05-07

## Verdict

M1 is a useful scale-path validation run, but it is not a passing M gate.

The adaptive collector worked: it attempted the full 36-run overcomplete plan, rejected provider/setup failures, and produced 23 eligible trajectories. The remaining blocker is data mix, not infrastructure: the corpus does not satisfy the visible validation-failure coverage gate.

## Gate Result

Final gate: failed.

Only failed gate after verifier determinism refresh:

- `validation_fail_observed_coverage`

Passing gates:

- artifact hardening
- estimator artifact build
- estimator prefix safety and alignment
- high-progress failures or verifier disagreements
- median observation events
- median transcript steps
- prefix provenance
- progress-drop coverage
- real agent pilot runs present
- shell exit-code coverage
- shell stdout/stderr snippet coverage
- terminal failure rate
- validation-attempt coverage
- verifier determinism
- zero leakage incidents

Key gate inputs:

- attempted runs: 36
- accepted eligible runs: 23
- rejected runs: 13
- target eligible runs: 18
- terminal failure rate: 0.435
- high-progress failure or disagreement count: 17
- validation attempt run fraction: 0.609
- validation failure observed run fraction: 0.043
- progress drop run fraction: 1.000
- median transcript steps: 20
- median observation events per run: 12
- leakage incidents: 0
- verifier determinism: passed after one 2-trial sample

## Accepted / Rejected

Accepted eligible trajectories by arm:

| arm | eligible | success | terminal failure | runs with validation attempt | runs with observed validation failure | verifier disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt53codex` | 8 | 5 | 3 | 6 | 0 | 3 |
| `gpt54` | 7 | 4 | 3 | 3 | 0 | 1 |
| `gpt54mini` | 8 | 4 | 4 | 5 | 1 | 3 |

Rejected trajectories:

- 13 rejected as replaceable preflight/setup failures.
- 12 were expected weak/reserve task setup failures from `classifier-debug`, `adaptive-rejection-sampler`, `blind-maze-explorer-algorithm`, and `nginx-request-logging`.
- 1 was a transient `gpt-5.4` provider route preflight failure on `broken-python`, with OpenAI HTTP 500. This was correctly rejected and should not count as model/task behavior.

## Task Outcomes

Fully eligible tasks:

- `grid-pattern-transform`: 3 eligible, 2 successes
- `extract-safely`: 3 eligible, 3 successes
- `attention-mil`: 3 eligible, 1 success
- `csv-to-parquet`: 3 eligible, 0 successes
- `fix-permissions`: 3 eligible, 3 successes
- `aimo-airline-departures`: 3 eligible, 2 successes
- `count-dataset-tokens`: 3 eligible, 0 successes

Partially eligible:

- `broken-python`: 2 eligible, 2 successes; `gpt54` rejected due to provider HTTP 500 during route preflight

No eligible runs:

- `classifier-debug`
- `adaptive-rejection-sampler`
- `blind-maze-explorer-algorithm`
- `nginx-request-logging`

## Cost And Usage

Estimated token cost using current standard OpenAI text prices:

- total accepted eligible model cost: about `$5.59`
- `gpt-5.3-codex`: 8 runs, about `$2.47`, average `$0.309`
- `gpt-5.4`: 7 runs, about `$1.80`, average `$0.257`
- `gpt-5.4-mini`: 8 runs, about `$1.32`, average `$0.165`

Token usage across accepted eligible runs:

- input tokens: 1,937,975
- output tokens: 296,778

Largest cost drivers:

- `aimo-airline-departures__gpt53codex`: about `$1.01`
- `count-dataset-tokens__gpt54mini`: about `$0.78`
- `csv-to-parquet__gpt53codex`: about `$0.52`

The cost was higher than the L1 extrapolation because several M1 trajectories were much longer, especially `gpt-5.3-codex` on `aimo-airline-departures` and `gpt-5.4-mini` on `count-dataset-tokens`.

## Artifact Status

Estimator artifacts were built for 23 runs:

- `datasets/terminal_bench_v10_openai_m1_estimator/checkpoints.parquet`
- `datasets/terminal_bench_v10_openai_m1_estimator/labels.parquet`
- `datasets/terminal_bench_v10_openai_m1_estimator/estimator_predictions.parquet`
- `datasets/terminal_bench_v10_openai_m1_estimator/checkpoint_feature_manifest.json`

Artifact report:

- checkpoint rows: 814
- prefix provenance complete: true
- estimator alignment: passed
- artifact hardening: passed
- leakage incidents: 0

## Interpretation

M1 validates the OpenAI scale path:

- direct OpenAI provider route works across the three arms;
- fallbacks remained zero;
- setup/provider failures were rejected rather than included;
- estimator artifacts can be produced from the accepted corpus;
- artifact and leakage hard gates pass.

M1 does not validate this task mix as the next production collection recipe:

- visible validation failures are too sparse: only 1 of 23 eligible runs had `validation_fail_observed`;
- several reserve tasks consumed attempts but produced only setup failures;
- some tasks produce terminal failures mainly through hidden verifier disagreement or final verifier failure, not visible validation-loop evidence.

## Recommended Next Action

Do not promote M1 as a passing M gate.

Use M1 as evidence that the adaptive OpenAI path is operational, then prepare M1b with a stricter task plan:

- keep `gpt-5.4`, `gpt-5.3-codex`, and `gpt-5.4-mini`;
- exclude the four tasks that produced no eligible runs;
- add or prioritize tasks with visible failing test loops;
- keep the adaptive rejection behavior;
- require verifier determinism sampling as part of the run finalization path, not as a manual after-step.

For the current M1 artifacts, the accepted 23 trajectories are usable for infrastructure and exploratory estimator checks, but they should be marked as gate-failed for production training or final Workstream M evidence.
