# Terminal-Bench OpenAI M Curated Dataset

Created: 2026-05-07

## Verdict

Built a `coding-estimator`-consumable exploratory dataset from existing M1/M1b runs only. No new provider calls were made.

Dataset:

```text
datasets/terminal_bench_openai_m_curated_estimator/
```

Use this for:

- exploratory estimator checks
- task/model outcome profiling
- hidden-disagreement analysis
- pipeline validation

Do not use this as production training data. After pruning, the corpus is clean enough to consume structurally, but it still fails the visible validation-failure coverage gate.

## Selection

Selected 15 runs:

- M1b `attention-mil`: 3 runs
- M1b `grid-pattern-transform`: 3 runs
- M1b `csv-to-parquet`: 3 runs
- M1 `extract-safely`: 3 runs
- M1 `fix-permissions`: 3 runs

Excluded 20 accepted runs:

- 5 `broken-python` runs: environment mutation is not reproducible from saved `agent_workspace`
- 9 M1 runs superseded by M1b reruns of the same task/arm family
- 6 hidden-only or cost-risk runs from `count-dataset-tokens` and `aimo-airline-departures`

Setup/provider failures were not included.

## Estimator Artifacts

Artifact validation passed:

- `checkpoints.parquet`
- `labels.parquet`
- `estimator_predictions.parquet`
- `checkpoint_feature_manifest.json`
- `estimator_source_manifest.json`

Summary:

- selected runs: 15
- checkpoint rows: 551
- prefix provenance complete: true
- artifact validation issues: none

The source manifest records selected and excluded runs with curation reasons.

## Gate Sanity Check

The curated run set still fails one corpus gate:

- `validation_fail_observed_coverage`

Key inputs:

- eligible runs: 15
- terminal failure rate: 0.533
- validation attempt run fraction: 0.867
- validation fail observed run fraction: 0.067
- validation disagreement run fraction: 0.467
- progress drop run fraction: 1.000
- median transcript steps: 21
- median observation events: 12
- leakage incidents: 0
- verifier determinism passed: true

Interpretation:

Pruning removed the non-reproducible `broken-python` validation-failure signal. The remaining high-quality runs are structurally safe and estimator-consumable, but they do not contain enough Type A visible validation-failure trajectories for production training.

## Coding-Estimator Consumption

The artifacts were built through the normal collection artifact path. During the build, `../coding-estimator/runs/terminal_bench_pilot` was restaged to the curated selected runs.

The dataset can be consumed directly from:

```text
coding-data-collection/datasets/terminal_bench_openai_m_curated_estimator/
```

or via the staged `terminal_bench_pilot` source in `coding-estimator` for exploratory runs.

## Next No-Spend Direction

Do not run more provider-backed collection until local/no-spend task screening identifies visible-validation tasks that are also verifier-reproducible from captured artifacts.

Use the new no-spend local collection plan in `TASKS.md` to collect Docker-backed local traces from Terminal-Bench/HF tasks without API calls.
