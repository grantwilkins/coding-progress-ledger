# Terminal-Bench V10 OpenAI M1 Plan

Created: 2026-05-07

## Frozen L1 Baseline

`terminal_bench_v10_openai_adaptive_l1` is the canonical passing L1 artifact. Do not rerun or broaden it unless a concrete defect is found.

Canonical L1 artifacts:

- `manifests/pilots/terminal_bench_v10_openai_adaptive_l1_plan.json`
- `runs/terminal_bench_v10_openai_adaptive_l1/`
- `reports/terminal_bench_v10_openai_adaptive_l1_execution.json`
- `reports/terminal_bench_v10_openai_adaptive_l1_accepted.json`
- `reports/terminal_bench_v10_openai_adaptive_l1_rejected.json`
- `reports/TERMINAL_BENCH_V10_OPENAI_ADAPTIVE_L1_GATE_REPORT.json`
- `datasets/terminal_bench_v10_openai_adaptive_l1_estimator/`

Parent repo status note: the `coding-data-collection` subtree is still untracked from the parent repository.

## M1 Intent

M1 is a scale-path validation run, not estimator-training data production signoff. Its purpose is to validate direct-OpenAI adaptive collection at modest scale, including cost, artifact volume, setup-failure rejection, and downstream gate behavior. If M1 passes cleanly, the next run can be declared production data collection with a larger target.

## M1 Gate

Target:

- `target_eligible_runs`: 12
- `target_eligible_runs`: 18
- `max_attempts`: 36
- planned queue: 36 runs from 12 tasks x 3 model arms

Allowed model arms:

- `gpt54`: direct OpenAI `gpt-5.4`
- `gpt53codex`: direct OpenAI `gpt-5.3-codex`
- `gpt54mini`: direct OpenAI `gpt-5.4-mini`

Disallowed:

- OpenRouter
- fallback model routing
- protocol-smoke or scripted arms as M1 gate inputs

Hard gates:

- zero leakage incidents
- artifact completeness
- provider metadata present, including resolved model and fallback count
- prefix provenance present and aligned
- verifier determinism sampled and passing
- provider/setup failures rejected rather than counted as eligible trajectories
- estimator artifacts build successfully from eligible runs

## Task Mix

Plan: `manifests/pilots/terminal_bench_v10_openai_m1_plan.json`

Preflight result: passed, 36 planned runs, 0 blockers, 0 warnings.

Sources:

- `terminal_bench_hf`: 12 tasks

Setup status:

- `v9_compatible`: 8 tasks
- `exclude_prebuild_required`: 2 tasks
- `exclude_hidden_artifact_risk`: 2 tasks

Categories:

- `data-processing`: 2
- `data-transform`: 1
- `debugging`: 1
- `mathematics`: 1
- `model-training`: 1
- `python`: 1
- `scientific-computing`: 1
- `security`: 1
- `software-engineering`: 1
- `systems`: 1
- `text-processing`: 1

Planned tasks:

| task_id | category | difficulty | setup_status | priority |
| --- | --- | --- | --- | ---: |
| grid-pattern-transform | data-transform | medium | v9_compatible | 37 |
| broken-python | python | easy | v9_compatible | 36 |
| count-dataset-tokens | data-processing | medium | v9_compatible | 34 |
| classifier-debug | debugging | medium | exclude_prebuild_required | 33 |
| extract-safely | security | medium | v9_compatible | 32 |
| attention-mil | model-training | medium | v9_compatible | 31 |
| csv-to-parquet | data-processing | medium | v9_compatible | 31 |
| fix-permissions | systems | easy | v9_compatible | 31 |
| aimo-airline-departures | mathematics | hard | v9_compatible | 29 |
| adaptive-rejection-sampler | scientific-computing | medium | exclude_hidden_artifact_risk | 29 |
| blind-maze-explorer-algorithm | software-engineering | medium | exclude_hidden_artifact_risk | 29 |
| nginx-request-logging | text-processing | medium | exclude_prebuild_required | 28 |

The four non-`v9_compatible` tasks are intentionally retained only because M1 is adaptive and overcomplete. If they fail provider/setup/readiness, they should be rejected and replaced by later plan entries, not counted as eligible trajectories.

## Expected Cost

The current OpenAI standard text token prices used for this estimate are:

- `gpt-5.4`: $2.50 / 1M input tokens, $15.00 / 1M output tokens
- `gpt-5.3-codex`: $1.75 / 1M input tokens, $14.00 / 1M output tokens
- `gpt-5.4-mini`: $0.75 / 1M input tokens, $4.50 / 1M output tokens

Source: https://openai.com/api/pricing/

L1 observed usage across 9 accepted runs:

- input tokens: 260,248
- output tokens: 75,030
- estimated model cost: $1.1963
- average accepted run: $0.1329
- average `gpt-5.4` run: $0.1896
- average `gpt-5.4-mini` run: $0.0621

M1 rough estimate from L1 usage:

- 18 eligible runs balanced across `gpt-5.4`, `gpt-5.3-codex`, and `gpt-5.4-mini`: about $2.51
- 12 eligible runs balanced across the three arms: about $1.68
- all 36 planned attempts if fully consumed at L1-like token lengths: about $5.03

This estimate excludes verifier/container infrastructure costs and assumes standard processing, no cached-input discount, and similar trajectory lengths.

## Adaptive Command

Do not launch until the task mix and expected cost above are accepted.

```bash
uv run python scripts/run_adaptive_l1.py \
  manifests/pilots/terminal_bench_v10_openai_m1_plan.json \
  --out reports/terminal_bench_v10_openai_m1_execution.json \
  --accepted-out reports/terminal_bench_v10_openai_m1_accepted.json \
  --rejected-out reports/terminal_bench_v10_openai_m1_rejected.json \
  --gate-out reports/TERMINAL_BENCH_V10_OPENAI_M1_GATE_REPORT.json \
  --failure-out reports/TERMINAL_BENCH_V10_OPENAI_M1_FAILURE_ANALYSIS.md \
  --artifact-dir datasets/terminal_bench_v10_openai_m1_estimator \
  --corpus-id terminal_bench_v10_openai_m1_estimator \
  --target-eligible-runs 18 \
  --max-attempts 36
```
