# observation-channel

Deterministic progress annotation for coding-agent traces.

The package converts cached source traces into canonical turn JSONL, feeds those turns through the same online-compatible `Annotator`, and writes per-turn CSV rows plus per-trace summaries.

## Commands

```sh
uv run --project observation-channel observation-channel cache --source all
uv run --project observation-channel observation-channel preprocess --source swe-agent --limit 10 --local-files-only
uv run --project observation-channel observation-channel annotate-corpus data/turns/swe-agent
uv run --project observation-channel observation-channel cached-annotator-diagnostic
uv run --project observation-channel observation-channel empirical-bayes-eval
uv run --project observation-channel observation-channel empirical-bayes-diagnostics
uv run pytest observation-channel
```

`data/` is ignored. It holds downloaded Hugging Face cache files, canonical turns, and generated outputs.
The empirical-Bayes evaluator reads cached diagnostic `turns.csv` and `traces.csv`, writes review tables and plots under `reports/empirical_bayes_v1/`, and stores the regenerated lookup bundle under ignored `data/estimators/`. Bootstrap bands use trace-level preaggregation with default `B=400`. V1.6 lookup features require regenerating cached annotations from canonical turn JSONL so `turns.csv` includes `recent_error_bucket`, `touched_source`, and `investigation_ratio_bucket`; old CSVs still load with neutral defaults but are not meaningful v1.6 training inputs. The diagnostics command adds grid-offset reliability, SWE-Agent category bias, current-step bias, exact-prefix cohort width checks, turn-bucket support counts, and rate-bucket conditional histograms from generated heldout predictions.
