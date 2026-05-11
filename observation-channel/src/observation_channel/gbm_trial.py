from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .empirical_bayes import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    TURN_GRID,
    EmpiricalBayesLookup,
    PrefixRow,
    TraceMeta,
    _bootstrap_bands,
    _coverage_rows,
    _feature_distribution_rows,
    _heldout_diagnostics,
    _interval_width_by_trace_position_rows,
    _length_terciles,
    _plot_category_bias,
    _plot_grid_offset_reliability,
    _plot_interval_width_by_trace_position,
    _plot_prefix_cohort_distribution,
    _plot_rate_histograms,
    _plot_reliability,
    _plot_step_bias,
    _prediction_keys,
    _prefix_followup_diagnostics,
    _prefix_state_key,
    _progress,
    _rate_bucket,
    _rate_bucket_conditional_histograms,
    _reliability_rows,
    _sharpness_rows,
    _skipped_censored_rows,
    _stable_hash,
    _strata,
    _turn_categories,
    _write_csv,
    eligible_prefixes,
    read_prefixes_csv,
    read_traces_csv,
    source_stratified_split,
)


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
FEATURES = (
    "source",
    "current_total",
    "current_category",
    "current_step",
    "current_unit_age",
    "had_stuck_episode",
    "recent_error_rate",
    "touched_source",
    "investigation_ratio",
)
CATEGORICAL_FEATURES = ("source", "current_category", "had_stuck_episode", "touched_source")
RAW_FEATURE_COLUMNS = ("recent_error_rate", "investigation_ratio")
DEFAULT_MODEL_DIR = Path("data/estimators/gbm_trial")
DEFAULT_REPORT_DIR = Path("reports/gbm_trial")


@dataclass(frozen=True)
class GbmQuantilePrediction:
    raw_quantiles: tuple[float, ...]
    quantiles: tuple[float, ...]

    @classmethod
    def from_raw(cls, raw_quantiles: Iterable[float], current_total: int) -> "GbmQuantilePrediction":
        raw = tuple(float(value) for value in raw_quantiles)
        clamped = tuple(max(float(current_total), value) for value in raw)
        return cls(raw, tuple(sorted(clamped)))

    @property
    def crossed(self) -> bool:
        return any(self.raw_quantiles[index] > self.raw_quantiles[index + 1] for index in range(len(self.raw_quantiles) - 1))

    @property
    def reordering_magnitude(self) -> float:
        monotone = sorted(self.raw_quantiles)
        return max((abs(raw - adjusted) for raw, adjusted in zip(self.raw_quantiles, monotone)), default=0.0)

    def cdf(self, threshold: int) -> float:
        points: dict[float, float] = {}
        for value, probability in zip(self.quantiles, QUANTILES):
            points[value] = max(probability, points.get(value, 0.0))
        xs = sorted(points)
        if not xs:
            raise ValueError("GBM prediction has no quantiles")
        threshold_f = float(threshold)
        if threshold_f < xs[0]:
            return 0.0
        if threshold_f > xs[-1]:
            return 1.0
        if threshold_f in points:
            return points[threshold_f]
        high_index = next(index for index, value in enumerate(xs) if value > threshold_f)
        low = xs[high_index - 1]
        high = xs[high_index]
        low_p = points[low]
        high_p = points[high]
        return low_p + (high_p - low_p) * ((threshold_f - low) / (high - low))


class GbmTrialBundle:
    def __init__(self, boosters: dict[float, Any], manifest: dict[str, Any]) -> None:
        self.boosters = boosters
        self.manifest = manifest

    @classmethod
    def load(cls, model_dir: Path) -> "GbmTrialBundle":
        import lightgbm as lgb

        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        boosters = {float(q): lgb.Booster(model_file=str(model_dir / _model_name(float(q)))) for q in manifest["quantiles"]}
        return cls(boosters, manifest)

    def predict(self, rows: list[PrefixRow]) -> list[GbmQuantilePrediction]:
        frame = _feature_frame(rows, self.manifest["categories"])
        predictions = []
        for quantile in QUANTILES:
            booster = self.boosters[quantile]
            best_iteration = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
            predictions.append(booster.predict(frame, num_iteration=best_iteration))
        return [
            GbmQuantilePrediction.from_raw((predictions[q_i][row_i] for q_i in range(len(QUANTILES))), row.total)
            for row_i, row in enumerate(rows)
        ]


def train_gbm_trial(
    turns_csv: Path,
    traces_csv: Path,
    model_dir: Path,
    *,
    seed: int = 1729,
    min_support: int = 25,
) -> dict[str, Any]:
    import lightgbm as lgb

    started = time.monotonic()
    progress = lambda message: _progress(message, started)
    progress("loading trace metadata")
    traces = read_traces_csv(traces_csv)
    progress("loading raw-feature prefix rows")
    prefixes = read_gbm_prefixes_csv(turns_csv, traces, progress)
    progress("splitting traces")
    train_keys, eval_keys, split_rows = source_stratified_split(traces.values())
    eligible = eligible_prefixes(prefixes, traces)
    outer_train_keys = {row.trace_key for row in eligible if row.trace_key in train_keys}
    inner_train_keys, inner_valid_keys = trace_level_validation_split(outer_train_keys)
    train_rows = [row for row in eligible if row.trace_key in inner_train_keys]
    valid_rows = [row for row in eligible if row.trace_key in inner_valid_keys]
    if not train_rows or not valid_rows:
        raise ValueError("GBM trial needs non-empty train and validation prefix rows")

    categories = _categories_for([row for row in eligible if row.trace_key in outer_train_keys])
    train_frame = _feature_frame(train_rows, categories)
    valid_frame = _feature_frame(valid_rows, categories)
    train_targets = [traces[row.trace_key].final_total for row in train_rows]
    valid_targets = [traces[row.trace_key].final_total for row in valid_rows]
    train_data = lgb.Dataset(train_frame, label=train_targets, categorical_feature=list(CATEGORICAL_FEATURES), free_raw_data=False)
    valid_data = lgb.Dataset(valid_frame, label=valid_targets, categorical_feature=list(CATEGORICAL_FEATURES), free_raw_data=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "min_support": min_support,
        "quantiles": list(QUANTILES),
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories": categories,
        "hyperparameters": {
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
        },
        "split_summary": split_rows,
        "inner_validation": {
            "train_traces": len(inner_train_keys),
            "validation_traces": len(inner_valid_keys),
            "train_prefixes": len(train_rows),
            "validation_prefixes": len(valid_rows),
            "heldout_eval_traces": len(eval_keys),
        },
        "turns_csv": str(turns_csv),
        "traces_csv": str(traces_csv),
    }
    for quantile in QUANTILES:
        progress(f"training LightGBM quantile {quantile}")
        params = {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "verbose": -1,
        }
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        booster.save_model(str(model_dir / _model_name(quantile)))
        manifest[f"best_iteration_{quantile}"] = booster.best_iteration
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress("done")
    return {"model_dir": str(model_dir), "train_prefixes": len(train_rows), "validation_prefixes": len(valid_rows)}


def evaluate_gbm_trial(
    turns_csv: Path,
    traces_csv: Path,
    report_dir: Path,
    model_dir: Path,
    conditional_cohorts_csv: Path,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 1729,
    min_support: int = 25,
) -> dict[str, Any]:
    started = time.monotonic()
    progress = lambda message: _progress(message, started)
    progress("loading trace metadata")
    traces = read_traces_csv(traces_csv)
    progress("loading raw-feature prefix rows")
    prefixes = read_gbm_prefixes_csv(turns_csv, traces, progress)
    progress("loading GBM model bundle")
    bundle = GbmTrialBundle.load(model_dir)
    progress("splitting traces")
    train_keys, eval_keys, split_rows = source_stratified_split(traces.values())
    eligible = eligible_prefixes(prefixes, traces)
    train_prefixes = [row for row in eligible if row.trace_key in train_keys]
    eval_prefixes = [row for row in eligible if row.trace_key in eval_keys]
    train_traces = {key: trace for key, trace in traces.items() if key in train_keys and not trace.parse_error and not trace.censored_right_tail}
    progress("building v1.6 support gate lookup")
    support_lookup = EmpiricalBayesLookup.build(train_prefixes, train_traces, min_support=min_support)
    length_terciles = _length_terciles([traces[key] for key in eval_keys if not traces[key].censored_right_tail])
    progress(f"generating GBM predictions from {len(eval_prefixes):,} eval prefixes")
    pairs, prefix_predictions, crossing = gbm_prediction_rows(eval_prefixes, traces, support_lookup, bundle, length_terciles, progress)
    progress(f"generated {len(prefix_predictions):,} prefix predictions and {len(pairs):,} calibration pairs")

    report_dir.mkdir(parents=True, exist_ok=True)
    reliability = _reliability_rows(pairs)
    progress(f"bootstrapping reliability bands with B={bootstrap_resamples:,}, seed={seed}")
    bands = _bootstrap_bands(pairs, bootstrap_resamples, seed)
    sharpness = _sharpness_rows(prefix_predictions)
    coverage = _coverage_rows(prefix_predictions)
    skipped = _skipped_censored_rows(prefixes, traces, eval_keys, support_lookup)

    _write_csv(report_dir / "heldout_predictions.csv", pairs)
    _write_csv(report_dir / "prefix_predictions.csv", prefix_predictions)
    _write_csv(report_dir / "reliability.csv", reliability)
    _write_csv(report_dir / "bootstrap_bands.csv", bands)
    _write_csv(report_dir / "sharpness_summary.csv", sharpness)
    _write_csv(report_dir / "coverage_summary.csv", coverage)
    _write_csv(report_dir / "split_summary.csv", split_rows)
    _write_csv(report_dir / "censored_skipped_summary.csv", skipped)
    _write_csv(report_dir / "quantile_crossing_summary.csv", crossing)
    _write_gbm_report(report_dir / "REPORT.md", split_rows, skipped, crossing, bootstrap_resamples, seed)
    _plot_reliability(report_dir / "reliability.png", reliability, bands)
    _write_gbm_diagnostics(report_dir, turns_csv, traces_csv, conditional_cohorts_csv)
    progress("done")
    return {
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
        "prediction_pairs": len(pairs),
        "prefix_predictions": len(prefix_predictions),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }


def read_gbm_prefixes_csv(
    path: Path,
    traces: dict[str, TraceMeta],
    progress: Any | None = None,
) -> list[PrefixRow]:
    require_raw_feature_columns(path)
    return read_prefixes_csv(path, traces, progress)


def require_raw_feature_columns(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    missing = [field for field in RAW_FEATURE_COLUMNS if field not in fieldnames]
    if missing:
        raise ValueError(f"GBM trial requires raw feature columns in {path}: {', '.join(missing)}")


def trace_level_validation_split(trace_keys: Iterable[str], validation_fraction: float = 0.1) -> tuple[set[str], set[str]]:
    ordered = sorted(set(trace_keys), key=lambda key: (_stable_hash(key), key))
    if len(ordered) < 2:
        raise ValueError("trace-level validation split needs at least two traces")
    cutoff = max(1, min(len(ordered) - 1, math.floor(len(ordered) * (1 - validation_fraction))))
    return set(ordered[:cutoff]), set(ordered[cutoff:])


def gbm_prediction_rows(
    prefixes: list[PrefixRow],
    traces: dict[str, TraceMeta],
    support_lookup: EmpiricalBayesLookup,
    bundle: Any,
    length_terciles: dict[str, str],
    progress: Any | None = None,
    *,
    chunk_size: int = 200_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    prefix_predictions = []
    supported: list[tuple[PrefixRow, Any]] = []
    prediction_cache: dict[tuple[Any, ...], Any | None] = {}
    for index, row in enumerate(prefixes, start=1):
        if progress and index % 100_000 == 0:
            progress(f"support-gated {index:,} eval prefixes; {len(supported):,} supported")
        cache_key = _prefix_state_key(row)
        if cache_key not in prediction_cache:
            try:
                prediction_cache[cache_key] = support_lookup.predict(row)
            except ValueError:
                prediction_cache[cache_key] = None
        support = prediction_cache[cache_key]
        if support is not None:
            supported.append((row, support))

    crossing_inputs: dict[str, list[GbmQuantilePrediction]] = {"pooled": []}
    for start in range(0, len(supported), chunk_size):
        chunk = supported[start : start + chunk_size]
        gbm_predictions = bundle.predict([row for row, _ in chunk])
        for (row, support), gbm_prediction in zip(chunk, gbm_predictions):
            trace = traces[row.trace_key]
            strata = _strata(row, length_terciles)
            crossing_inputs.setdefault(row.source, []).append(gbm_prediction)
            crossing_inputs["pooled"].append(gbm_prediction)
            p10 = gbm_prediction.quantiles[0]
            p90 = gbm_prediction.quantiles[-1]
            prefix_predictions.append(
                {
                    "trace_key": row.trace_key,
                    "source": row.source,
                    "step": row.step,
                    "current_total": row.total,
                    "recent_error_bucket": row.recent_error_bucket,
                    "touched_source": row.touched_source,
                    "investigation_ratio_bucket": row.investigation_ratio_bucket,
                    "fallback_depth": support.fallback_depth,
                    "support_count": support.support_count,
                    "retained_fields": ";".join(support.retained_fields),
                    "low_confidence_flags": ";".join(support.low_confidence_reasons),
                    "length_tercile": strata["length_tercile"],
                    "p10": p10,
                    "p90": p90,
                    "interval80_width": p90 - p10,
                }
            )
            max_supported = support.values[-1]
            for offset in TURN_GRID:
                threshold = row.total + offset
                if max_supported < threshold:
                    continue
                pairs.append(
                    {
                        "trace_key": row.trace_key,
                        "source": row.source,
                        "step": row.step,
                        "current_total": row.total,
                        "current_category": row.current_category,
                        "recent_error_bucket": row.recent_error_bucket,
                        "touched_source": row.touched_source,
                        "investigation_ratio_bucket": row.investigation_ratio_bucket,
                        "threshold": threshold,
                        "grid_offset": offset,
                        "current_rate": row.total / row.step,
                        "rate_bucket": _rate_bucket(row.total / row.step),
                        "predicted_p": gbm_prediction.cdf(threshold),
                        "outcome": int(trace.final_total <= threshold),
                        "fallback_depth": support.fallback_depth,
                        "support_count": support.support_count,
                        "retained_fields": ";".join(support.retained_fields),
                        "low_confidence_flags": ";".join(support.low_confidence_reasons),
                        "length_tercile": strata["length_tercile"],
                        "source_length_tercile": strata["source_length_tercile"],
                    }
                )
    return pairs, prefix_predictions, quantile_crossing_summary_rows(crossing_inputs)


def quantile_crossing_summary_rows(predictions_by_source: dict[str, list[GbmQuantilePrediction]]) -> list[dict[str, Any]]:
    rows = []
    for source, predictions in sorted(predictions_by_source.items()):
        crossed = [prediction.reordering_magnitude for prediction in predictions if prediction.crossed]
        rows.append(
            {
                "source": source,
                "n": len(predictions),
                "crossing_count": len(crossed),
                "crossing_rate": len(crossed) / len(predictions) if predictions else "",
                "mean_reordering_magnitude_when_crossed": sum(crossed) / len(crossed) if crossed else "",
                "p95_reordering_magnitude_when_crossed": _nearest_percentile(crossed, 0.95),
            }
        )
    return rows


def _write_gbm_diagnostics(report_dir: Path, turns_csv: Path, traces_csv: Path, conditional_cohorts_csv: Path) -> None:
    categories = _turn_categories(turns_csv, _prediction_keys(report_dir / "heldout_predictions.csv"))
    reliability, category_bias, step_bias = _heldout_diagnostics(report_dir / "heldout_predictions.csv", categories)
    rate_hist = _rate_bucket_conditional_histograms(conditional_cohorts_csv)
    non_near_cap_rate_hist = _rate_bucket_conditional_histograms(conditional_cohorts_csv, non_near_cap=True)
    traces = read_traces_csv(traces_csv)
    train_keys, _, _ = source_stratified_split(traces.values())
    cohort_hist, cohort_summary, support_summary = _prefix_followup_diagnostics(
        turns_csv,
        traces,
        train_keys,
        step=10,
        total=1,
    )
    interval_width_rows = _interval_width_by_trace_position_rows(report_dir / "prefix_predictions.csv", traces)
    feature_distribution_rows = _feature_distribution_rows(turns_csv, traces, train_keys)
    _write_csv(report_dir / "reliability_by_grid_offset.csv", reliability)
    _write_csv(report_dir / "category_stratum_bias.csv", category_bias)
    _write_csv(report_dir / "current_step_bias.csv", step_bias)
    _write_csv(report_dir / "rate_bucket_conditional_histograms.csv", rate_hist)
    _write_csv(report_dir / "rate_bucket_conditional_histograms_non_near_cap.csv", non_near_cap_rate_hist)
    _write_csv(report_dir / "prefix_cohort_distribution.csv", cohort_hist)
    _write_csv(report_dir / "prefix_cohort_distribution_summary.csv", cohort_summary)
    _write_csv(report_dir / "fine_turn_bucket_support_summary.csv", support_summary)
    _write_csv(report_dir / "interval_width_by_trace_position.csv", interval_width_rows)
    _write_csv(report_dir / "failure_feature_distributions.csv", feature_distribution_rows)
    _plot_grid_offset_reliability(report_dir / "reliability_by_grid_offset.png", reliability)
    _plot_category_bias(report_dir / "category_stratum_bias.png", category_bias)
    _plot_step_bias(report_dir / "current_step_bias.png", step_bias)
    _plot_rate_histograms(report_dir / "rate_bucket_conditional_histograms.png", rate_hist)
    _plot_rate_histograms(report_dir / "rate_bucket_conditional_histograms_non_near_cap.png", non_near_cap_rate_hist)
    _plot_prefix_cohort_distribution(report_dir / "prefix_cohort_distribution.png", cohort_hist, cohort_summary[0])
    _plot_interval_width_by_trace_position(report_dir / "interval_width_by_trace_position.png", interval_width_rows)
    examples_csv = report_dir.parent / "empirical_bayes_v1.5" / "progress_tracking_examples.csv"
    model_dir = report_dir.parents[1] / "data" / "estimators" / "gbm_trial"
    if examples_csv.exists() and model_dir.exists():
        write_progress_tracking_examples(
            examples_csv,
            turns_csv,
            traces_csv,
            model_dir,
            report_dir / "progress_tracking_examples.csv",
            report_dir / "progress_tracking_examples.png",
        )
        write_best_progress_tracking_examples(
            report_dir / "prefix_predictions.csv",
            turns_csv,
            traces_csv,
            model_dir,
            report_dir / "best_progress_tracking_examples.csv",
            report_dir / "best_progress_tracking_examples.png",
        )
    _append_gbm_diagnostics_report(report_dir / "REPORT.md")


def _write_gbm_report(
    path: Path,
    split_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    crossing_rows: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    lines = [
        "# GBM Quantile Trial",
        "",
        "LightGBM quantile regressors evaluated through the v1.6 support gate.",
        f"Trace-level bootstrap uses B={resamples} and seed={seed}.",
        "",
        "## Split Summary",
        "",
        "| source | train_traces | eval_traces | total_traces |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in split_rows:
        lines.append(f"| {row['source']} | {row['train_traces']} | {row['eval_traces']} | {row['total_traces']} |")
    lines.extend(["", "## Censored Eval Prefixes", "", "| source | skipped_prefixes | skipped_with_prediction | long_tail_rate |", "| --- | ---: | ---: | ---: |"])
    for row in skipped:
        lines.append(f"| {row['source']} | {row['skipped_prefixes']} | {row['skipped_with_prediction']} | {row['long_tail_rate']} |")
    lines.extend(["", "## Quantile Crossing", "", "| source | n | crossing_rate | mean_reordering_magnitude | p95_reordering_magnitude |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in crossing_rows:
        lines.append(
            f"| {row['source']} | {row['n']} | {row['crossing_rate']} | "
            f"{row['mean_reordering_magnitude_when_crossed']} | {row['p95_reordering_magnitude_when_crossed']} |"
        )
    lines.extend(["", "[Quantile crossing summary](quantile_crossing_summary.csv)", "", "![Reliability](reliability.png)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_gbm_diagnostics_report(path: Path) -> None:
    marker = "## Follow-up Diagnostics"
    addition = [
        marker,
        "",
        "### Model-dependent diagnostics",
        "",
        "![Reliability by grid offset](reliability_by_grid_offset.png)",
        "",
        "![SWE-Agent category bias](category_stratum_bias.png)",
        "",
        "![Current-step bias](current_step_bias.png)",
        "",
        "![Progress tracking examples](progress_tracking_examples.png)",
        "",
        "![Best progress tracking examples](best_progress_tracking_examples.png)",
        "",
        "![Interval width by trace position](interval_width_by_trace_position.png)",
        "",
        "### Shared v1.6 context diagnostics",
        "",
        "These plots use the same cached corpus, split, conditional cohorts, and support diagnostics as v1.6; they are expected to match unless the input corpus changes.",
        "",
        "![Rate-bucket conditional histograms](rate_bucket_conditional_histograms.png)",
        "",
        "![Non-near-cap rate-bucket conditional histograms](rate_bucket_conditional_histograms_non_near_cap.png)",
        "",
        "![Exact prefix D_T distribution](prefix_cohort_distribution.png)",
        "",
        "[Failure feature distributions](failure_feature_distributions.csv)",
        "",
    ]
    text = path.read_text(encoding="utf-8")
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n"
    path.write_text(text.rstrip() + "\n\n" + "\n".join(addition), encoding="utf-8")


def write_progress_tracking_examples(
    reference_csv: Path,
    turns_csv: Path,
    traces_csv: Path,
    model_dir: Path,
    out_csv: Path,
    out_png: Path,
) -> list[dict[str, Any]]:
    traces = read_traces_csv(traces_csv)
    reference_rows = _progress_reference_rows(reference_csv)
    selected_steps = {(row["trace_key"], int(row["step"])) for row in reference_rows}
    prefixes = _selected_progress_prefixes(turns_csv, selected_steps)
    bundle = GbmTrialBundle.load(model_dir)
    ordered_prefixes = [prefixes[(row["trace_key"], int(row["step"]))] for row in reference_rows]
    rows = _progress_tracking_rows(reference_rows, ordered_prefixes, traces, bundle.predict(ordered_prefixes))
    _write_csv(out_csv, rows)
    _plot_progress_tracking_examples(out_png, rows)
    return rows


def write_best_progress_tracking_examples(
    prefix_predictions_csv: Path,
    turns_csv: Path,
    traces_csv: Path,
    model_dir: Path,
    out_csv: Path,
    out_png: Path,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    traces = read_traces_csv(traces_csv)
    reference_rows = _best_progress_reference_rows(_progress_reference_rows(prefix_predictions_csv), limit)
    selected_steps = {(row["trace_key"], int(row["step"])) for row in reference_rows}
    prefixes = _selected_progress_prefixes(turns_csv, selected_steps)
    bundle = GbmTrialBundle.load(model_dir)
    ordered_prefixes = [prefixes[(row["trace_key"], int(row["step"]))] for row in reference_rows]
    rows = _progress_tracking_rows(reference_rows, ordered_prefixes, traces, bundle.predict(ordered_prefixes))
    _write_csv(out_csv, rows)
    _plot_progress_tracking_examples(out_png, rows)
    return rows


def _best_progress_reference_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trace_key"], []).append(row)
    scored = []
    for trace_key, items in grouped.items():
        ordered = sorted(items, key=lambda row: int(row["step"]))
        nonzero = [row for row in ordered if int(row["current_total"]) > 0]
        if len(nonzero) < 5:
            continue
        early = nonzero[int(0.25 * (len(nonzero) - 1))]
        late = ordered[-1]
        early_width = _progress_band_width(early)
        late_width = _progress_band_width(late)
        shrink = early_width - late_width
        if shrink > 0:
            scored.append((shrink, -late_width, trace_key))
    selected = [trace_key for _, _, trace_key in sorted(scored, reverse=True)[:limit]]
    rows_by_trace = {trace_key: [] for trace_key in selected}
    for row in rows:
        if row["trace_key"] in rows_by_trace:
            rows_by_trace[row["trace_key"]].append(row)
    return [row for trace_key in selected for row in rows_by_trace[trace_key]]


def _progress_tracking_rows(
    reference_rows: list[dict[str, Any]],
    prefixes: list[PrefixRow],
    traces: dict[str, TraceMeta],
    predictions: list[GbmQuantilePrediction],
) -> list[dict[str, Any]]:
    rows = []
    for reference, prefix, prediction in zip(reference_rows, prefixes, predictions):
        final_total = traces[prefix.trace_key].final_total
        p10, _, p50, _, p90 = prediction.quantiles
        rows.append(
            {
                "trace_key": prefix.trace_key,
                "source": prefix.source,
                "length_tercile": reference["length_tercile"],
                "step": prefix.step,
                "current_total": prefix.total,
                "final_total": final_total,
                "rho_actual": prefix.total / final_total if final_total else 0.0,
                "rho_p10": _progress_fraction(prefix.total, p90),
                "rho_p50": _progress_fraction(prefix.total, p50),
                "rho_p90": _progress_fraction(prefix.total, p10),
                "p10_final_total": p10,
                "p50_final_total": p50,
                "p90_final_total": p90,
            }
        )
    return rows


def _progress_reference_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_progress_prefixes(path: Path, selected_steps: set[tuple[str, int]]) -> dict[tuple[str, int], PrefixRow]:
    prefixes = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["trace_key"], int(row["step"]))
            if key not in selected_steps:
                continue
            prefixes[key] = PrefixRow(
                trace_key=row["trace_key"],
                source=row["source"],
                step=int(row["step"]),
                total=int(row["total"]),
                current_category=row.get("current_category") or "NONE",
                current_unit_age=int(row["current_unit_age"]),
                had_stuck_episode=_csv_bool(row.get("had_stuck_episode", "")),
                recent_error_bucket=row.get("recent_error_bucket") or "clean",
                recent_error_rate=float(row.get("recent_error_rate") or 0.0),
                touched_source=_csv_bool(row.get("touched_source", "")),
                investigation_ratio_bucket=row.get("investigation_ratio_bucket") or "moderate",
                investigation_ratio=float(row.get("investigation_ratio") or 0.0),
            )
    missing = selected_steps - set(prefixes)
    if missing:
        raise ValueError(f"missing progress example prefixes for {sorted(missing)}")
    return prefixes


def _plot_progress_tracking_examples(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trace.setdefault(str(row["trace_key"]), []).append(row)
    traces = list(by_trace)
    cols = 3
    panel_rows = max(1, math.ceil(len(traces) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(cols * 4.8, panel_rows * 3.0), squeeze=False, sharey=True)
    handles = None
    for axis, trace_key in zip(axes.flat, traces):
        items = sorted(by_trace[trace_key], key=lambda row: int(row["step"]))
        xs = [int(row["step"]) for row in items]
        fill = axis.fill_between(
            xs,
            [float(row["rho_p10"]) for row in items],
            [float(row["rho_p90"]) for row in items],
            alpha=0.18,
            color="tab:blue",
            label="predicted 80% interval",
        )
        median_line = axis.plot(xs, [float(row["rho_p50"]) for row in items], color="tab:blue", linewidth=1.5, label="predicted median")[0]
        actual_line = axis.plot(xs, [float(row["rho_actual"]) for row in items], color="0.2", linestyle="--", linewidth=1.3, label="actual N_t / D_T")[0]
        handles = (fill, median_line, actual_line)
        first = items[0]
        axis.set_title(f"{first['source']} {first['length_tercile']}\n{trace_key}", fontsize=9)
        axis.set_xlabel("current_step")
        axis.set_ylim(-0.03, 1.03)
    for axis in axes.flat[len(traces) :]:
        axis.axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("progress fraction rho_t")
    if handles is not None:
        fig.legend(handles, [handle.get_label() for handle in handles], loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _progress_fraction(current_total: int, predicted_final_total: float) -> float:
    return current_total / max(1.0, predicted_final_total)


def _progress_band_width(row: dict[str, Any]) -> float:
    current_total = int(row["current_total"])
    return _progress_fraction(current_total, float(row["p10"])) - _progress_fraction(current_total, float(row["p90"]))


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _feature_frame(rows: list[PrefixRow], categories: dict[str, list[str]]) -> Any:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "source": [row.source for row in rows],
            "current_total": [row.total for row in rows],
            "current_category": [row.current_category or "NONE" for row in rows],
            "current_step": [row.step for row in rows],
            "current_unit_age": [row.current_unit_age for row in rows],
            "had_stuck_episode": [str(row.had_stuck_episode) for row in rows],
            "recent_error_rate": [row.recent_error_rate for row in rows],
            "touched_source": [str(row.touched_source) for row in rows],
            "investigation_ratio": [row.investigation_ratio for row in rows],
        }
    )
    for field in CATEGORICAL_FEATURES:
        frame[field] = pd.Categorical(frame[field], categories=categories[field])
    return frame[list(FEATURES)]


def _categories_for(rows: list[PrefixRow]) -> dict[str, list[str]]:
    return {
        "source": sorted({row.source for row in rows}),
        "current_category": sorted({row.current_category or "NONE" for row in rows}),
        "had_stuck_episode": ["False", "True"],
        "touched_source": ["False", "True"],
    }


def _nearest_percentile(values: list[float], probability: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _model_name(quantile: float) -> str:
    return f"quantile_{int(quantile * 100):02d}.txt"
