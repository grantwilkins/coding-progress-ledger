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
uv run --project observation-channel observation-channel gbm-trial-train
uv run --project observation-channel observation-channel gbm-trial-eval --bootstrap-resamples 1000
uv run --project observation-channel observation-channel belief-tracker-eval
uv run pytest observation-channel
```

`data/` is ignored. It holds downloaded Hugging Face cache files, canonical turns, and generated outputs.
The empirical-Bayes evaluator reads cached diagnostic `turns.csv` and `traces.csv`, writes review tables and plots under `reports/empirical_bayes_v1/`, and stores the regenerated lookup bundle under ignored `data/estimators/`. Bootstrap bands use trace-level preaggregation with default `B=400`. V1.6 lookup features require regenerating cached annotations from canonical turn JSONL so `turns.csv` includes `recent_error_bucket`, `recent_error_rate`, `touched_source`, `investigation_ratio_bucket`, and `investigation_ratio`; old CSVs still load with neutral defaults but are not meaningful v1.6 training inputs. The diagnostics command adds grid-offset reliability, SWE-Agent category bias, current-step bias, exact-prefix cohort width checks, turn-bucket support counts, and rate-bucket conditional histograms from generated heldout predictions.

The GBM trial trains LightGBM quantile regressors from those raw continuous features, saves models under ignored `data/estimators/gbm_trial/`, and evaluates through the same held-out split and v1.6 support gate into `reports/gbm_trial/`. The report includes GBM progress-tracking examples and the three strongest uncertainty-shrinkage examples when the v1.5 reference examples and saved GBM models are available. The large GBM `heldout_predictions.csv` and `prefix_predictions.csv` files remain ignored; summary tables, plots, and `REPORT.md` are intended review artifacts.

The belief-tracker evaluator replays held-out prefixes with empirical-Bayes direct, empirical-Bayes filtered, GBM direct, and empirical-Bayes plus GBM filtered final-work beliefs. It requires the saved GBM model and writes `reports/belief_tracker/progress_beliefs.csv`, claim calibration pairs, summary tables, trace plots, and `REPORT.md`.
