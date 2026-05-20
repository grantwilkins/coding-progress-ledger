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
uv run --project observation-channel observation-channel belief-filter-calibration
uv run --project observation-channel observation-channel progress-label-audit
TOGETHER_API_KEY=... uv run --project observation-channel observation-channel together-replay-ensemble --raw-index 349 --cache-dir observation-channel/data/raw/hf_cache --report-dir observation-channel/reports/together_replay_time
uv run pytest observation-channel
```

`data/` is ignored. It holds downloaded Hugging Face cache files, canonical turns, and generated outputs.
The empirical-Bayes evaluator reads cached diagnostic `turns.csv` and `traces.csv`, writes review tables and plots under `reports/empirical_bayes_v1/`, and stores the regenerated lookup bundle under ignored `data/estimators/`. Bootstrap bands use trace-level preaggregation with default `B=400`. V1.6 lookup features require regenerating cached annotations from canonical turn JSONL so `turns.csv` includes `recent_error_bucket`, `recent_error_rate`, `touched_source`, `investigation_ratio_bucket`, and `investigation_ratio`; old CSVs still load with neutral defaults but are not meaningful v1.6 training inputs. The diagnostics command adds grid-offset reliability, SWE-Agent category bias, current-step bias, exact-prefix cohort width checks, turn-bucket support counts, and rate-bucket conditional histograms from generated heldout predictions.

The GBM trial trains LightGBM quantile regressors from those raw continuous features, saves models under ignored `data/estimators/gbm_trial/`, and evaluates through the same held-out split and v1.6 support gate into `reports/gbm_trial/`. The report includes GBM progress-tracking examples and the three strongest uncertainty-shrinkage examples when the v1.5 reference examples and saved GBM models are available. The large GBM `heldout_predictions.csv` and `prefix_predictions.csv` files remain ignored; summary tables, plots, and `REPORT.md` are intended review artifacts.

The belief-tracker evaluator replays held-out prefixes with empirical-Bayes direct, empirical-Bayes filtered, GBM direct, and empirical-Bayes plus GBM filtered final-work beliefs. It requires the saved GBM model and empirical-Bayes v1.6 lookup, then writes `reports/belief_tracker/progress_beliefs.csv`, binned claim calibration rows, summary tables, trace plots, and `REPORT.md`.

The belief-filter calibration evaluator runs EB-only alpha and event-gated filter sweeps against the same held-out split. It reuses the saved empirical-Bayes v1.6 lookup and writes the same artifact names under `reports/belief_filter_calibration/`; the large `progress_beliefs.csv` remains local and ignored.

The progress-label audit is read-only. It checks whether opened-unit progress reaches 100% while a large fraction of the trace remains, compares opened-unit, closed-unit, and step progress on selected worst traces, and writes evidence under `reports/progress_label_audit/`.

The Together replay runs several observer models over a selected solved SWE-Agent trace turn by turn and writes seconds-left estimates plus a remaining-time plot with inverse-confidence error bars under `reports/together_replay_time/`. Each observer call receives the original prompt and only the observed work prefix available at that turn. Invalid model responses are retried. It requires `TOGETHER_API_KEY` and uses Together's OpenAI-compatible chat endpoint.
