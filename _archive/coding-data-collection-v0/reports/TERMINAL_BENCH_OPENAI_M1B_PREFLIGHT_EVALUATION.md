# Terminal-Bench OpenAI M1b Preflight Evaluation

Evaluated: 2026-05-07

## Verdict

M1b fixed the M1 task-mix problem but did not pass the final gate because verifier reproducibility failed on `broken-python__gpt54`.

The preflight target did what it was designed to do: it increased visible validation-loop signal while keeping the same direct OpenAI adaptive collection path.

## Gate Result

Final gate: failed.

Only failed gate:

- `verifier_outcomes_reproducible`

Important passing gates:

- `validation_fail_observed_coverage`
- `validation_disagreement_coverage`
- `validation_attempt_coverage`
- `terminal_failure_rate`
- `progress_drop_coverage`
- artifact hardening
- estimator artifacts
- estimator prefix safety and alignment
- zero leakage

Key inputs:

- planned runs: 12
- attempted runs: 12
- accepted eligible runs: 12
- rejected runs: 0
- terminal failure rate: 0.667
- validation attempt run fraction: 0.833
- validation fail observed run fraction: 0.333
- validation disagreement run fraction: 0.583
- progress drop run fraction: 1.000
- median transcript steps: 28.5
- median observation events per run: 16.0
- leakage incidents: 0

## Task Outcomes

| task_id | eligible | success | terminal failure | runs with validation failure | validation disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `broken-python` | 3 | 3 | 0 | 3 | 0 |
| `attention-mil` | 3 | 0 | 3 | 1 | 3 |
| `grid-pattern-transform` | 3 | 1 | 2 | 0 | 2 |
| `csv-to-parquet` | 3 | 0 | 3 | 0 | 2 |

Interpretation:

- `broken-python` is excellent for visible validation-loop signal, but problematic for verifier reproducibility because the task repairs system-level Python/pip state that is not fully represented by the saved agent workspace.
- `attention-mil` is a strong M1b target: terminal failures plus one visible validation failure and three validation disagreements.
- `grid-pattern-transform` and `csv-to-parquet` mostly produce Type B disagreement/final-verifier failures rather than visible validation failures, but still help disagreement coverage.

## Model Outcomes

| arm | eligible | success | terminal failure | runs with validation failure | validation disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt53codex` | 4 | 1 | 3 | 1 | 3 |
| `gpt54` | 4 | 2 | 2 | 2 | 2 |
| `gpt54mini` | 4 | 1 | 3 | 1 | 2 |

The three-arm ladder remains useful: all arms contributed failures and at least one visible validation-failure run.

## Verifier Determinism

Two verifier determinism reports now exist:

- `broken-python__gpt54`: failed
- `grid-pattern-transform__gpt54`: passed

The `broken-python__gpt54` reruns failed with:

```text
ModuleNotFoundError: No module named 'pip'
/task/run-tests.sh: line 5: pytest: command not found
```

This is not normal hidden-verifier nondeterminism. It indicates the verifier rerun workspace does not capture the global/system Python repair performed by the agent. For this task, the successful terminal verifier depends on environment mutation outside the persisted `agent_workspace`.

Claim boundary:

```text
M1b validates visible validation-loop task selection.
M1b does not validate verifier reproducibility for system-level environment-repair tasks.
```

## Cost And Usage

Estimated accepted-run model cost:

- total: about `$5.85`
- `gpt-5.4`: 4 runs, about `$4.11`, average `$1.03`
- `gpt-5.3-codex`: 4 runs, about `$1.31`, average `$0.33`
- `gpt-5.4-mini`: 4 runs, about `$0.43`, average `$0.11`

Token usage:

- input tokens: 1,566,175
- output tokens: 212,707

Largest outlier:

- `grid-pattern-transform__gpt54`: about `$2.80`, with 933,692 input tokens

The cost guard proposed after M1 is justified. M1b hit both the per-run warning and hard-stop thresholds on `grid-pattern-transform__gpt54`.

## Recommended Next Action

Do not scale from M1b as-is.

Prepare an M1c or revised M1b with the following changes:

1. Keep the three OpenAI arms.
2. Keep `attention-mil` as a validation-loop target.
3. Keep `broken-python` only if the artifact protocol can capture system-level Python/pip mutations or if verifier determinism sampling excludes it with an explicit rationale.
4. Replace or downweight `grid-pattern-transform__gpt54`-like loops with a per-run cost guard.
5. Add at least one more task that can produce visible failed checks without requiring global environment mutation.
6. Make verifier determinism sampling part of finalization and sample at least one file-output task plus one dependency/environment task.

M1b should be treated as a successful task-selection preflight for visible validation signal, but still gate-failed for production collection because verifier reproducibility is not clean.
