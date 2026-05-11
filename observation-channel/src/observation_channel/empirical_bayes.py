from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable


TURN_GRID = (1, 2, 4, 8, 16, 32, 64)
KEY_FIELDS = (
    "source",
    "recent_error_bucket",
    "investigation_ratio_bucket",
    "touched_source",
    "total",
    "current_category",
    "turn_bucket",
    "age_bucket",
    "stuck",
)
FALLBACK_FIELDS = (
    "recent_error_bucket",
    "investigation_ratio_bucket",
    "touched_source",
    "stuck",
    "age_bucket",
    "turn_bucket",
    "current_category",
    "total",
)
COMFORTABLE_SUPPORT = 50
RATE_BUCKETS = (0.1, 0.2, 0.3, 0.5)
DIAGNOSTIC_COHORT_STEP = 10
DIAGNOSTIC_COHORT_TOTAL = 1
NEAR_CAP_FINAL_TOTAL = 94
DEFAULT_BOOTSTRAP_RESAMPLES = 400
TRACE_POSITION_BINS = 20
ERROR_BUCKETS = ("clean", "mild", "moderate", "heavy")
INVESTIGATION_RATIO_BUCKETS = ("low", "moderate", "high", "dominant")
TRACE_POSITION_THIRDS = ("early", "middle", "late")
FIVE_READ_TRACES = (
    {
        "trace_key": "swe-agent:001020:pydantic__pydantic-1989",
        "requested_step": 30,
        "human_read": "fake reproduction file despite active code-writing",
    },
    {
        "trace_key": "swe-agent:011335:dwavesystems__dwave-cloud-client-338",
        "requested_step": 30,
        "human_read": "spent about 28 of 30 turns investigating",
    },
    {
        "trace_key": "swe-agent:067060:pydantic__pydantic-4354",
        "requested_step": 30,
        "human_read": "repeated edit failures with identical indentation errors",
    },
    {
        "trace_key": "swe-agent:068615:stephantul__reach-23",
        "requested_step": 30,
        "human_read": "repeated failed edits while operating on a fake reproduction script",
    },
    {
        "trace_key": "swe-agent:064153:qiboteam__qibo-953",
        "requested_step": 30,
        "human_read": "repeated wrong-symbol searches; negative search results may not count as failures",
    },
)


@dataclass(frozen=True)
class TraceMeta:
    trace_key: str
    source: str
    final_total: int
    total_turns: int
    first_stuck_step: int | None = None
    censored_right_tail: bool = False
    parse_error: str = ""


@dataclass(frozen=True)
class PrefixRow:
    trace_key: str
    source: str
    step: int
    total: int
    current_category: str
    current_unit_age: int
    had_stuck_episode: bool
    recent_error_bucket: str = "clean"
    recent_error_rate: float = 0.0
    touched_source: bool = False
    investigation_ratio_bucket: str = "moderate"
    investigation_ratio: float = 0.0


@dataclass(frozen=True)
class Prediction:
    values: tuple[int, ...]
    support_count: int
    retained_fields: tuple[str, ...]
    fallback_depth: int
    source_retained: bool
    low_confidence_reasons: tuple[str, ...]

    def cdf(self, threshold: int) -> float:
        if not self.values:
            raise ValueError("prediction has no support")
        return bisect_right(self.values, threshold) / len(self.values)

    def quantile(self, probability: float) -> int:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if not self.values:
            raise ValueError("prediction has no support")
        index = max(0, math.ceil(probability * len(self.values)) - 1)
        return self.values[index]

    def progress_interval(self, current_total: int, mass: float = 0.8) -> tuple[float, float]:
        if not 0 < mass < 1:
            raise ValueError("mass must be in (0, 1)")
        low_tail = (1 - mass) / 2
        high_tail = 1 - low_tail
        high_final = max(1, self.quantile(high_tail))
        low_final = max(1, self.quantile(low_tail))
        return current_total / high_final, current_total / low_final

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {
            "values": list(self.values),
            "retained_fields": list(self.retained_fields),
            "low_confidence_reasons": list(self.low_confidence_reasons),
        }


class EmpiricalBayesLookup:
    def __init__(
        self,
        cells: dict[tuple[Any, ...], tuple[int, ...]],
        source_p90: dict[str, int],
        *,
        min_support: int = 25,
        comfortable_support: int = COMFORTABLE_SUPPORT,
    ) -> None:
        self.cells = cells
        self.source_p90 = source_p90
        self.min_support = min_support
        self.comfortable_support = comfortable_support

    @classmethod
    def build(
        cls,
        prefixes: Iterable[PrefixRow],
        traces: dict[str, TraceMeta],
        *,
        min_support: int = 25,
        comfortable_support: int = COMFORTABLE_SUPPORT,
    ) -> "EmpiricalBayesLookup":
        cell_traces: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
        for row in prefixes:
            final_total = traces[row.trace_key].final_total
            for depth in range(len(FALLBACK_FIELDS) + 1):
                cell_traces[_fallback_key(row, depth)][row.trace_key] = final_total
        cells = {key: tuple(sorted(values.values())) for key, values in cell_traces.items()}
        return cls(cells, source_p90_thresholds(traces.values()), min_support=min_support, comfortable_support=comfortable_support)

    def predict(self, row: PrefixRow) -> Prediction:
        for depth in range(len(FALLBACK_FIELDS) + 1):
            key = _fallback_key(row, depth)
            cell_values = self.cells.get(key, ())
            values = cell_values[bisect_left(cell_values, row.total) :]
            if len(values) >= self.min_support:
                reasons = []
                if depth:
                    reasons.append("fallback_depth")
                if len(values) < self.comfortable_support:
                    reasons.append("thin_support")
                if row.source in self.source_p90 and row.total >= self.source_p90[row.source]:
                    reasons.append("long_tail")
                return Prediction(
                    values=values,
                    support_count=len(values),
                    retained_fields=_retained_fields(depth),
                    fallback_depth=depth,
                    source_retained=True,
                    low_confidence_reasons=tuple(reasons),
                )
        raise ValueError("no supported empirical-Bayes bin for prefix")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_support": self.min_support,
            "comfortable_support": self.comfortable_support,
            "source_p90": self.source_p90,
            "cells": [{"key": _encode_key(key), "values": list(values)} for key, values in self.cells.items()],
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmpiricalBayesLookup":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = {tuple(item["key"]): tuple(int(value) for value in item["values"]) for item in payload["cells"]}
        return cls(
            cells,
            {str(key): int(value) for key, value in payload["source_p90"].items()},
            min_support=int(payload["min_support"]),
            comfortable_support=int(payload["comfortable_support"]),
        )


def turn_bucket(step: int) -> str:
    return _fine_turn_bucket(step)


def age_bucket(age: int) -> str:
    if age <= 0:
        return "0"
    if age == 1:
        return "1"
    if age <= 4:
        return "2-4"
    if age <= 9:
        return "5-9"
    if age <= 19:
        return "10-19"
    return "20+"


def read_traces_csv(path: Path) -> dict[str, TraceMeta]:
    traces = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            final_total = row.get("final_total", "")
            total_turns = row.get("total_turns", "")
            traces[row["trace_key"]] = TraceMeta(
                trace_key=row["trace_key"],
                source=row["source"],
                final_total=int(final_total) if final_total else 0,
                total_turns=int(total_turns) if total_turns else 0,
                first_stuck_step=int(row["first_stuck_step"]) if row.get("first_stuck_step") else None,
                censored_right_tail=_bool(row.get("censored_right_tail", "")),
                parse_error=row.get("parse_error", ""),
            )
    return traces


def read_prefixes_csv(
    path: Path,
    traces: dict[str, TraceMeta],
    progress: Callable[[str], None] | None = None,
) -> list[PrefixRow]:
    prefixes = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        has_stuck = "had_stuck_episode" in (reader.fieldnames or ())
        for index, row in enumerate(reader, start=1):
            trace = traces[row["trace_key"]]
            step = int(row["step"])
            if has_stuck:
                stuck = _bool(row["had_stuck_episode"])
            else:
                stuck = trace.first_stuck_step is not None and step >= trace.first_stuck_step
            prefixes.append(
                PrefixRow(
                    trace_key=row["trace_key"],
                    source=row["source"],
                    step=step,
                    total=int(row["total"]),
                    current_category=row.get("current_category") or "NONE",
                    current_unit_age=int(row["current_unit_age"]),
                    had_stuck_episode=stuck,
                    recent_error_bucket=row.get("recent_error_bucket") or "clean",
                    recent_error_rate=float(row.get("recent_error_rate") or 0.0),
                    touched_source=_bool(row.get("touched_source", "")),
                    investigation_ratio_bucket=row.get("investigation_ratio_bucket") or "moderate",
                    investigation_ratio=float(row.get("investigation_ratio") or 0.0),
                )
            )
            if progress and index % 1_000_000 == 0:
                progress(f"loaded {index:,} prefix rows")
    return prefixes


def eligible_prefixes(prefixes: Iterable[PrefixRow], traces: dict[str, TraceMeta]) -> list[PrefixRow]:
    return [
        row
        for row in prefixes
        if not traces[row.trace_key].parse_error
        and not traces[row.trace_key].censored_right_tail
        and row.step < traces[row.trace_key].total_turns
    ]


def source_stratified_split(traces: Iterable[TraceMeta], train_fraction: float = 0.8) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    by_source: dict[str, list[TraceMeta]] = defaultdict(list)
    for trace in traces:
        if not trace.parse_error:
            by_source[trace.source].append(trace)

    train, eval_ = set(), set()
    summary = []
    for source, items in sorted(by_source.items()):
        ordered = sorted(items, key=lambda trace: (_stable_hash(trace.trace_key), trace.trace_key))
        cutoff = math.floor(len(ordered) * train_fraction)
        train_keys = {trace.trace_key for trace in ordered[:cutoff]}
        eval_keys = {trace.trace_key for trace in ordered[cutoff:]}
        train.update(train_keys)
        eval_.update(eval_keys)
        summary.append(
            {
                "source": source,
                "train_traces": len(train_keys),
                "eval_traces": len(eval_keys),
                "total_traces": len(ordered),
            }
        )
    return train, eval_, summary


def evaluate(
    turns_csv: Path,
    traces_csv: Path,
    report_dir: Path,
    bundle_path: Path,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 1729,
    min_support: int = 25,
) -> dict[str, Any]:
    started = time.monotonic()
    progress = lambda message: _progress(message, started)
    progress("loading trace metadata")
    traces = read_traces_csv(traces_csv)
    progress(f"loaded {len(traces):,} traces")
    progress("loading prefix rows")
    prefixes = read_prefixes_csv(turns_csv, traces, progress)
    progress(f"loaded {len(prefixes):,} prefix rows")
    progress("splitting traces")
    train_keys, eval_keys, split_rows = source_stratified_split(traces.values())
    progress("filtering train/eval prefixes")
    eligible = eligible_prefixes(prefixes, traces)
    train_prefixes = [row for row in eligible if row.trace_key in train_keys]
    eval_prefixes = [row for row in eligible if row.trace_key in eval_keys]
    train_traces = {key: trace for key, trace in traces.items() if key in train_keys and not trace.parse_error and not trace.censored_right_tail}
    progress(f"building lookup from {len(train_prefixes):,} train prefixes")
    lookup = EmpiricalBayesLookup.build(train_prefixes, train_traces, min_support=min_support)

    report_dir.mkdir(parents=True, exist_ok=True)
    progress(f"saving lookup with {len(lookup.cells):,} cells")
    lookup.save(bundle_path)

    progress("assigning held-out trace length terciles")
    length_terciles = _length_terciles([traces[key] for key in eval_keys if not traces[key].censored_right_tail])
    progress(f"generating held-out predictions from {len(eval_prefixes):,} eval prefixes")
    pairs, prefix_predictions = _prediction_rows(eval_prefixes, traces, lookup, length_terciles, progress)
    progress(f"generated {len(prefix_predictions):,} prefix predictions and {len(pairs):,} calibration pairs")
    progress("computing reliability tables")
    reliability = _reliability_rows(pairs)
    progress(f"bootstrapping reliability bands with B={bootstrap_resamples:,}, seed={seed}")
    bands = _bootstrap_bands(pairs, bootstrap_resamples, seed)
    progress("computing sharpness, coverage, and censored summaries")
    sharpness = _sharpness_rows(prefix_predictions)
    coverage = _coverage_rows(prefix_predictions)
    skipped = _skipped_censored_rows(prefixes, traces, eval_keys, lookup)

    progress("writing CSV and markdown artifacts")
    _write_csv(report_dir / "heldout_predictions.csv", pairs)
    _write_csv(report_dir / "prefix_predictions.csv", prefix_predictions)
    _write_csv(report_dir / "reliability.csv", reliability)
    _write_csv(report_dir / "bootstrap_bands.csv", bands)
    _write_csv(report_dir / "sharpness_summary.csv", sharpness)
    _write_csv(report_dir / "coverage_summary.csv", coverage)
    _write_csv(report_dir / "split_summary.csv", split_rows)
    _write_csv(report_dir / "censored_skipped_summary.csv", skipped)
    _write_report(report_dir / "REPORT.md", split_rows, skipped, bootstrap_resamples, seed)
    progress("plotting reliability diagram")
    _plot_reliability(report_dir / "reliability.png", reliability, bands)
    progress("done")
    return {
        "bundle_path": str(bundle_path),
        "report_dir": str(report_dir),
        "prediction_pairs": len(pairs),
        "prefix_predictions": len(prefix_predictions),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }


def query_json(
    lookup: EmpiricalBayesLookup,
    *,
    source: str,
    total: int,
    current_category: str,
    step: int,
    current_unit_age: int,
    had_stuck_episode: bool,
) -> dict[str, Any]:
    row = PrefixRow(
        trace_key="live",
        source=source,
        step=step,
        total=total,
        current_category=current_category or "NONE",
        current_unit_age=current_unit_age,
        had_stuck_episode=had_stuck_episode,
    )
    prediction = lookup.predict(row)
    return prediction.to_json() | {
        "p50_final_total": prediction.quantile(0.5),
        "p80_progress_interval": prediction.progress_interval(total),
    }


def write_diagnostics(
    heldout_csv: Path,
    turns_csv: Path,
    traces_csv: Path,
    conditional_cohorts_csv: Path,
    report_dir: Path,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    categories = _turn_categories(turns_csv, _prediction_keys(heldout_csv))
    reliability, category_bias, step_bias = _heldout_diagnostics(heldout_csv, categories)
    rate_hist = _rate_bucket_conditional_histograms(conditional_cohorts_csv)
    non_near_cap_rate_hist = _rate_bucket_conditional_histograms(conditional_cohorts_csv, non_near_cap=True)
    traces = read_traces_csv(traces_csv)
    train_keys, _, _ = source_stratified_split(traces.values())
    cohort_hist, cohort_summary, support_summary = _prefix_followup_diagnostics(
        turns_csv,
        traces,
        train_keys,
        step=DIAGNOSTIC_COHORT_STEP,
        total=DIAGNOSTIC_COHORT_TOTAL,
    )
    _write_csv(report_dir / "reliability_by_grid_offset.csv", reliability)
    _write_csv(report_dir / "category_stratum_bias.csv", category_bias)
    _write_csv(report_dir / "current_step_bias.csv", step_bias)
    _write_csv(report_dir / "rate_bucket_conditional_histograms.csv", rate_hist)
    _write_csv(report_dir / "rate_bucket_conditional_histograms_non_near_cap.csv", non_near_cap_rate_hist)
    _write_csv(report_dir / "prefix_cohort_distribution.csv", cohort_hist)
    _write_csv(report_dir / "prefix_cohort_distribution_summary.csv", cohort_summary)
    _write_csv(report_dir / "fine_turn_bucket_support_summary.csv", support_summary)
    interval_width_rows = _interval_width_by_trace_position_rows(report_dir / "prefix_predictions.csv", traces)
    _write_csv(report_dir / "interval_width_by_trace_position.csv", interval_width_rows)
    feature_distribution_rows = _feature_distribution_rows(turns_csv, traces, train_keys)
    _write_csv(report_dir / "failure_feature_distributions.csv", feature_distribution_rows)
    five_read_rows = []
    if bundle_path is not None:
        five_read_rows = _five_read_trace_feature_rows(
            report_dir / "prefix_predictions.csv",
            turns_csv,
            EmpiricalBayesLookup.load(bundle_path),
        )
        _write_csv(report_dir / "failure_five_read_trace_features.csv", five_read_rows)
    _plot_grid_offset_reliability(report_dir / "reliability_by_grid_offset.png", reliability)
    _plot_category_bias(report_dir / "category_stratum_bias.png", category_bias)
    _plot_step_bias(report_dir / "current_step_bias.png", step_bias)
    _plot_rate_histograms(report_dir / "rate_bucket_conditional_histograms.png", rate_hist)
    _plot_rate_histograms(report_dir / "rate_bucket_conditional_histograms_non_near_cap.png", non_near_cap_rate_hist)
    _plot_prefix_cohort_distribution(report_dir / "prefix_cohort_distribution.png", cohort_hist, cohort_summary[0])
    _plot_interval_width_by_trace_position(report_dir / "interval_width_by_trace_position.png", interval_width_rows)
    _append_diagnostics_report(report_dir / "REPORT.md")
    return {
        "current_categories": len(categories),
        "reliability_rows": len(reliability),
        "category_bias_rows": len(category_bias),
        "step_bias_rows": len(step_bias),
        "rate_histogram_rows": len(rate_hist),
        "non_near_cap_rate_histogram_rows": len(non_near_cap_rate_hist),
        "prefix_cohort_histogram_rows": len(cohort_hist),
        "fine_turn_bucket_support_rows": len(support_summary),
        "interval_width_rows": len(interval_width_rows),
        "feature_distribution_rows": len(feature_distribution_rows),
        "five_read_trace_rows": len(five_read_rows),
    }


def source_p90_thresholds(traces: Iterable[TraceMeta]) -> dict[str, int]:
    values: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        if not trace.parse_error and not trace.censored_right_tail:
            values[trace.source].append(trace.final_total)
    return {source: _quantile(sorted(finals), 0.9) for source, finals in values.items() if finals}


def _prediction_rows(
    prefixes: list[PrefixRow],
    traces: dict[str, TraceMeta],
    lookup: EmpiricalBayesLookup,
    length_terciles: dict[str, str],
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    prefix_predictions = []
    prediction_cache: dict[tuple[Any, ...], Prediction | None] = {}
    for index, row in enumerate(prefixes, start=1):
        if progress and index % 100_000 == 0:
            progress(
                f"processed {index:,} eval prefixes; "
                f"{len(prefix_predictions):,} predictions; {len(pairs):,} calibration pairs; "
                f"{len(prediction_cache):,} cached states"
            )
        trace = traces[row.trace_key]
        if trace.censored_right_tail:
            continue
        cache_key = _prefix_state_key(row)
        if cache_key not in prediction_cache:
            try:
                prediction_cache[cache_key] = lookup.predict(row)
            except ValueError:
                prediction_cache[cache_key] = None
        prediction = prediction_cache[cache_key]
        if prediction is None:
            continue
        strata = _strata(row, length_terciles)
        prefix_predictions.append(
            {
                "trace_key": row.trace_key,
                "source": row.source,
                "step": row.step,
                "current_total": row.total,
                "recent_error_bucket": row.recent_error_bucket,
                "touched_source": row.touched_source,
                "investigation_ratio_bucket": row.investigation_ratio_bucket,
                "fallback_depth": prediction.fallback_depth,
                "support_count": prediction.support_count,
                "retained_fields": ";".join(prediction.retained_fields),
                "low_confidence_flags": ";".join(prediction.low_confidence_reasons),
                "length_tercile": strata["length_tercile"],
                "p10": prediction.quantile(0.1),
                "p90": prediction.quantile(0.9),
                "interval80_width": prediction.quantile(0.9) - prediction.quantile(0.1),
            }
        )
        max_supported = prediction.values[-1]
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
                    "predicted_p": prediction.cdf(threshold),
                    "outcome": int(trace.final_total <= threshold),
                    "fallback_depth": prediction.fallback_depth,
                    "support_count": prediction.support_count,
                    "retained_fields": ";".join(prediction.retained_fields),
                    "low_confidence_flags": ";".join(prediction.low_confidence_reasons),
                    "length_tercile": strata["length_tercile"],
                    "source_length_tercile": strata["source_length_tercile"],
                }
            )
    return pairs, prefix_predictions


def _reliability_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stratum_name, stratum_pairs in _all_strata(pairs).items():
        rows.extend(_reliability_for_pairs(stratum_name, stratum_pairs))
    return rows


def _reliability_for_pairs(stratum: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        bins[min(9, int(float(row["predicted_p"]) * 10))].append(row)
    rows = []
    for bin_index in range(10):
        items = bins.get(bin_index, [])
        rows.append(
            {
                "stratum": stratum,
                "bin": bin_index,
                "n": len(items),
                "mean_predicted_p": _mean(float(item["predicted_p"]) for item in items),
                "observed_rate": _mean(int(item["outcome"]) for item in items),
            }
        )
    return rows


def _bootstrap_bands(
    pairs: list[dict[str, Any]],
    resamples: int,
    seed: int,
    *,
    chunk_size: int = 128,
) -> list[dict[str, Any]]:
    """
    Trace-cluster bootstrap using preaggregated trace x reliability-bin counts.

    This preserves the existing statistical object: sampled traces contribute
    all of their calibration rows. The NumPy path uses NumPy's seeded generator,
    so bands are reproducible for a fixed seed but not bit-identical to the old
    random.Random row-list implementation.
    """
    try:
        import numpy as np
    except ImportError:
        return _bootstrap_bands_preaggregated_python(pairs, resamples, seed)

    rng = np.random.default_rng(seed)
    bands = []
    for stratum, trace_bins in _bootstrap_trace_bins(pairs).items():
        n_traces = len(trace_bins)
        counts = np.asarray([item[0] for item in trace_bins], dtype=np.int64)
        outcomes = np.asarray([item[1] for item in trace_bins], dtype=np.int64)
        rates = np.full((resamples, 10), np.nan, dtype=np.float64)
        probs = np.full(n_traces, 1.0 / n_traces, dtype=np.float64)
        for start in range(0, resamples, chunk_size):
            stop = min(start + chunk_size, resamples)
            sample_weights = rng.multinomial(n_traces, probs, size=stop - start)
            boot_counts = sample_weights @ counts
            boot_outcomes = sample_weights @ outcomes
            rates[start:stop, :] = np.divide(
                boot_outcomes,
                boot_counts,
                out=np.full((stop - start, 10), np.nan, dtype=np.float64),
                where=boot_counts > 0,
            )
        for bin_index in range(10):
            values = rates[:, bin_index]
            values = values[~np.isnan(values)]
            bands.append(
                {
                    "stratum": stratum,
                    "bin": bin_index,
                    "bootstrap_resamples": resamples,
                    "seed": seed,
                    "observed_low": _nearest_np_percentile(values, 0.025),
                    "observed_high": _nearest_np_percentile(values, 0.975),
                }
            )
    return bands


def _bootstrap_bands_preaggregated_python(pairs: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    bands = []
    for stratum, trace_bins in _bootstrap_trace_bins(pairs).items():
        count_rows = [counts for counts, _ in trace_bins]
        outcome_rows = [outcomes for _, outcomes in trace_bins]
        samples: list[list[float]] = [[] for _ in range(10)]
        for _ in range(resamples):
            boot_counts = [0] * 10
            boot_outcomes = [0] * 10
            for trace_i in rng.choices(range(len(trace_bins)), k=len(trace_bins)):
                for bin_i in range(10):
                    boot_counts[bin_i] += count_rows[trace_i][bin_i]
                    boot_outcomes[bin_i] += outcome_rows[trace_i][bin_i]
            for bin_i in range(10):
                if boot_counts[bin_i]:
                    samples[bin_i].append(boot_outcomes[bin_i] / boot_counts[bin_i])
        for bin_index, values in enumerate(samples):
            bands.append(
                {
                    "stratum": stratum,
                    "bin": bin_index,
                    "bootstrap_resamples": resamples,
                    "seed": seed,
                    "observed_low": _percentile(sorted(values), 0.025),
                    "observed_high": _percentile(sorted(values), 0.975),
                }
            )
    return bands


def _bootstrap_trace_bins(pairs: list[dict[str, Any]]) -> dict[str, list[tuple[list[int], list[int]]]]:
    by_trace: dict[str, tuple[str, str, list[int], list[int]]] = {}
    for row in pairs:
        trace_key = str(row["trace_key"])
        source = str(row["source"])
        source_length = str(row["source_length_tercile"])
        trace = by_trace.get(trace_key)
        if trace is None:
            trace = (source, source_length, [0] * 10, [0] * 10)
            by_trace[trace_key] = trace
        elif trace[0] != source or trace[1] != source_length:
            raise ValueError(f"inconsistent bootstrap strata for trace {trace_key}")
        bin_i = min(9, int(float(row["predicted_p"]) * 10))
        trace[2][bin_i] += 1
        trace[3][bin_i] += int(row["outcome"])

    strata: dict[str, list[tuple[list[int], list[int]]]] = defaultdict(list)
    for source, source_length, counts, outcomes in by_trace.values():
        trace_bins = (counts, outcomes)
        strata["pooled"].append(trace_bins)
        strata[source].append(trace_bins)
        strata[source_length].append(trace_bins)
    return {key: strata[key] for key in sorted(strata, key=_stratum_sort_key)}


def _sharpness_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stratum, items in _prefix_strata(prefix_rows).items():
        widths = sorted(float(item["interval80_width"]) for item in items)
        rows.append({"stratum": stratum, "n": len(items), "median_interval80_width": median(widths) if widths else ""})
    return rows


def _coverage_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stratum, items in _prefix_strata(prefix_rows).items():
        if not items:
            rows.append({"stratum": stratum, "n": 0, "full_key_rate": "", "low_confidence_rate": ""})
            continue
        rows.append(
            {
                "stratum": stratum,
                "n": len(items),
                "full_key_rate": _mean(int(int(item["fallback_depth"]) == 0) for item in items),
                "low_confidence_rate": _mean(int(bool(item["low_confidence_flags"])) for item in items),
            }
        )
    return rows


def _skipped_censored_rows(
    prefixes: list[PrefixRow],
    traces: dict[str, TraceMeta],
    eval_keys: set[str],
    lookup: EmpiricalBayesLookup,
) -> list[dict[str, Any]]:
    rows = []
    by_source: dict[str, list[PrefixRow]] = defaultdict(list)
    for row in prefixes:
        trace = traces[row.trace_key]
        if row.trace_key in eval_keys and trace.censored_right_tail and row.step < trace.total_turns:
            by_source[row.source].append(row)
    for source, items in sorted(by_source.items()):
        long_tail = 0
        predicted = 0
        for row in items:
            try:
                prediction = lookup.predict(row)
            except ValueError:
                continue
            predicted += 1
            long_tail += int("long_tail" in prediction.low_confidence_reasons)
        rows.append(
            {
                "source": source,
                "skipped_prefixes": len(items),
                "skipped_with_prediction": predicted,
                "long_tail_rate": long_tail / predicted if predicted else "",
            }
        )
    return rows


def _plot_reliability(path: Path, reliability: list[dict[str, Any]], bands: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_stratum = defaultdict(list)
    band_by_key = {(row["stratum"], int(row["bin"])): row for row in bands}
    for row in reliability:
        if row["n"]:
            by_stratum[row["stratum"]].append(row)
    strata = [name for name in sorted(by_stratum) if name == "pooled" or " x " not in name][:6]
    if "pooled" not in strata and "pooled" in by_stratum:
        strata.insert(0, "pooled")
    cols = 2
    rows = max(1, math.ceil(len(strata) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for axis, stratum in zip(axes.flat, strata):
        xs, ys, lows, highs = [], [], [], []
        for row in by_stratum[stratum]:
            bin_index = int(row["bin"])
            xs.append(float(row["mean_predicted_p"]))
            ys.append(float(row["observed_rate"]))
            band = band_by_key.get((stratum, bin_index), {})
            lows.append(float(band.get("observed_low") or row["observed_rate"]))
            highs.append(float(band.get("observed_high") or row["observed_rate"]))
        axis.plot([0, 1], [0, 1], color="0.7", linewidth=1)
        axis.plot(xs, ys, marker="o")
        axis.fill_between(xs, lows, highs, alpha=0.2)
        axis.set_title(stratum)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    for axis in axes.flat[len(strata) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_grid_offset_reliability(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    strata = sorted({str(row["source_length_tercile"]) for row in rows}, key=_stratum_sort_key)
    offsets = [offset for offset in TURN_GRID if any(int(row["grid_offset"]) == offset for row in rows)]
    for row in rows:
        by_key[(int(row["grid_offset"]), str(row["source_length_tercile"]))].append(row)
    fig, axes = plt.subplots(len(offsets), len(strata), figsize=(2.4 * len(strata), 1.9 * len(offsets)), squeeze=False)
    for row_index, offset in enumerate(offsets):
        for col_index, stratum in enumerate(strata):
            axis = axes[row_index][col_index]
            points = sorted(by_key.get((offset, stratum), []), key=lambda item: int(item["bin"]))
            axis.plot([0, 1], [0, 1], color="0.75", linewidth=0.8)
            axis.plot([row["mean_predicted_p"] for row in points], [row["observed_rate"] for row in points], marker=".", linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            if row_index == 0:
                axis.set_title(stratum, fontsize=8)
            if col_index == 0:
                axis.set_ylabel(f"+{offset}", fontsize=8)
            axis.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_category_bias(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = sorted({str(row["source_length_tercile"]) for row in rows}, key=_stratum_sort_key)
    fig, axes = plt.subplots(1, len(strata), figsize=(5 * max(1, len(strata)), 3), squeeze=False)
    for axis, stratum in zip(axes.flat, strata):
        items = sorted((row for row in rows if row["source_length_tercile"] == stratum), key=lambda row: abs(float(row["mean_bias"])), reverse=True)
        axis.bar([row["current_category"] for row in items], [row["mean_bias"] for row in items])
        axis.axhline(0, color="0.3", linewidth=0.8)
        axis.set_title(stratum)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.set_ylabel("predicted - observed")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_step_bias(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted({str(row["source"]) for row in rows})
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * max(1, len(sources)), 3), squeeze=False)
    for axis, source in zip(axes.flat, sources):
        for length in ("short", "medium", "long"):
            items = sorted(
                (row for row in rows if row["source"] == source and row["length_tercile"] == length),
                key=lambda row: int(row["current_step"]),
            )
            if items:
                axis.plot([row["current_step"] for row in items], [row["mean_bias"] for row in items], label=length, linewidth=1)
        axis.axhline(0, color="0.3", linewidth=0.8)
        axis.set_title(source)
        axis.set_xlabel("current_step")
        axis.set_ylabel("predicted - observed")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_rate_histograms(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted({str(row["condition_id"]) for row in rows}, key=int)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), squeeze=False)
    for axis, condition in zip(axes.flat, conditions):
        items = [row for row in rows if row["condition_id"] == condition]
        for bucket in sorted({row["rate_bucket"] for row in items}):
            xs = [int(row["final_total"]) for row in items if row["rate_bucket"] == bucket]
            weights = [int(row["n"]) for row in items if row["rate_bucket"] == bucket]
            axis.hist(xs, bins=30, weights=weights, alpha=0.6, label=bucket)
        first = items[0]
        axis.set_title(f"{condition}: step {first['selected_step']}, total {first['selected_total']}, {first['current_category']}")
        axis.set_xlabel("final_total")
        axis.set_ylabel("traces")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(conditions) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_prefix_cohort_distribution(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    for group in ("pooled_step", "exact_prefix"):
        items = [row for row in rows if row["group"] == group]
        axis.hist(
            [int(row["final_total"]) for row in items],
            bins=30,
            weights=[int(row["n"]) for row in items],
            alpha=0.55,
            label=group,
        )
    axis.axvline(int(summary["current_total"]), color="0.25", linewidth=1, linestyle="--")
    axis.set_title(f"step {summary['current_step']}, current_total {summary['current_total']}")
    axis.set_xlabel("D_T")
    axis.set_ylabel("prefixes")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_interval_width_by_trace_position(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    for source in sorted({str(row["source"]) for row in rows}, key=_stratum_sort_key):
        items = [row for row in rows if row["source"] == source]
        xs = [float(row["position_midpoint"]) for row in items]
        axis.plot(xs, [float(row["mean_interval80_width"]) for row in items], marker=".", linewidth=1.5, label=source)
        axis.fill_between(
            xs,
            [float(row["p25_interval80_width"]) for row in items],
            [float(row["p75_interval80_width"]) for row in items],
            alpha=0.15,
        )
    axis.set_xlabel("trace position")
    axis.set_ylabel("80% interval width")
    axis.set_xlim(0, 1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(path: Path, split_rows: list[dict[str, Any]], skipped: list[dict[str, Any]], resamples: int, seed: int) -> None:
    lines = [
        "# Empirical Bayes v1",
        "",
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
        lines.append(
            f"| {row['source']} | {row['skipped_prefixes']} | {row['skipped_with_prediction']} | {row['long_tail_rate']} |"
        )
    lines.extend(["", "![Reliability](reliability.png)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_diagnostics_report(path: Path) -> None:
    marker = "## Follow-up Diagnostics"
    addition = [
        marker,
        "",
        "![Rate-bucket conditional histograms](rate_bucket_conditional_histograms.png)",
        "",
        "![Reliability by grid offset](reliability_by_grid_offset.png)",
        "",
        "![SWE-Agent category bias](category_stratum_bias.png)",
        "",
        "![Current-step bias](current_step_bias.png)",
        "",
        "![Exact prefix D_T distribution](prefix_cohort_distribution.png)",
        "",
        "![Interval width by trace position](interval_width_by_trace_position.png)",
        "",
        "[Failure feature distributions](failure_feature_distributions.csv)",
        "",
        "[Five-read trace features](failure_five_read_trace_features.csv)",
        "",
        "![Non-near-cap rate-bucket conditional histograms](rate_bucket_conditional_histograms_non_near_cap.png)",
        "",
    ]
    text = path.read_text(encoding="utf-8") if path.exists() else "# Empirical Bayes v1\n\n"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n"
    path.write_text(text.rstrip() + "\n\n" + "\n".join(addition), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_keys(path: Path) -> set[tuple[str, int]]:
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add((row["trace_key"], int(row["step"])))
    return keys


def _turn_categories(path: Path, keys: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    categories = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["trace_key"], int(row["step"]))
            if key in keys:
                categories[key] = row.get("current_category") or "NONE"
    missing = keys - set(categories)
    if missing:
        raise ValueError(f"missing current_category for {len(missing)} prediction prefixes")
    return categories


def _heldout_diagnostics(
    path: Path,
    categories: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reliability: dict[tuple[str, int, int], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    category_bias: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    step_bias: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            predicted = float(row["predicted_p"])
            outcome = int(row["outcome"])
            offset = int(row.get("grid_offset") or int(row["threshold"]) - int(row["current_total"]))
            source = row["source"]
            length = row["length_tercile"]
            stratum = row["source_length_tercile"]
            bin_index = min(9, int(predicted * 10))
            _add_diag(reliability[(stratum, offset, bin_index)], predicted, outcome)
            _add_diag(step_bias[(source, length, step)], predicted, outcome)
            if source == "swe-agent" and length in {"short", "long"}:
                category = row.get("current_category") or categories[(row["trace_key"], step)]
                _add_diag(category_bias[(stratum, category)], predicted, outcome)
    return (
        [
            _diag_row(
                {"source_length_tercile": key[0], "grid_offset": key[1], "bin": key[2]},
                values,
            )
            for key, values in sorted(reliability.items())
        ],
        [
            _diag_row({"source_length_tercile": key[0], "current_category": key[1]}, values)
            for key, values in sorted(category_bias.items())
        ],
        [
            _diag_row({"source": key[0], "length_tercile": key[1], "current_step": key[2]}, values)
            for key, values in sorted(step_bias.items())
        ],
    )


def _prefix_cohort_distribution(
    prefixes: Iterable[PrefixRow],
    traces: dict[str, TraceMeta],
    *,
    step: int,
    total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled: list[int] = []
    cohort: list[int] = []
    for row in prefixes:
        if row.step != step:
            continue
        final_total = traces[row.trace_key].final_total
        pooled.append(final_total)
        if row.total == total:
            cohort.append(final_total)
    summary = _prefix_cohort_summary(step, total, pooled, cohort)
    return _histogram_rows("pooled_step", pooled) + _histogram_rows("exact_prefix", cohort), [summary]


def _finer_turn_bucket_support_rows(prefixes: list[PrefixRow], min_support: int = 25) -> list[dict[str, Any]]:
    cell_traces: dict[tuple[int, tuple[Any, ...]], set[str]] = defaultdict(set)
    cell_prefixes: dict[tuple[int, tuple[Any, ...]], int] = defaultdict(int)
    for row in prefixes:
        for depth in range(len(FALLBACK_FIELDS) + 1):
            key = (depth, _fine_fallback_key(row, depth))
            cell_traces[key].add(row.trace_key)
            cell_prefixes[key] += 1
    return _support_summary_rows(cell_traces, cell_prefixes, min_support)


def _prefix_followup_diagnostics(
    turns_csv: Path,
    traces: dict[str, TraceMeta],
    train_keys: set[str],
    *,
    step: int,
    total: int,
    min_support: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pooled: list[int] = []
    cohort: list[int] = []
    cell_traces: dict[tuple[int, tuple[Any, ...]], set[str]] = defaultdict(set)
    cell_prefixes: dict[tuple[int, tuple[Any, ...]], int] = defaultdict(int)
    with turns_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trace = traces[row["trace_key"]]
            current_step = int(row["step"])
            if trace.parse_error or trace.censored_right_tail or current_step >= trace.total_turns:
                continue
            current_total = int(row["total"])
            if current_step == step:
                pooled.append(trace.final_total)
                if current_total == total:
                    cohort.append(trace.final_total)
            if row["trace_key"] not in train_keys:
                continue
            stuck = trace.first_stuck_step is not None and current_step >= trace.first_stuck_step
            for depth in range(len(FALLBACK_FIELDS) + 1):
                key = (
                    depth,
                    _fine_fallback_key_values(
                        row["source"],
                        current_total,
                        row.get("current_category") or "NONE",
                        current_step,
                        int(row["current_unit_age"]),
                        stuck,
                        row.get("recent_error_bucket") or "clean",
                        _bool(row.get("touched_source", "")),
                        row.get("investigation_ratio_bucket") or "moderate",
                        depth,
                    ),
                )
                cell_traces[key].add(row["trace_key"])
                cell_prefixes[key] += 1
    return (
        _histogram_rows("pooled_step", pooled) + _histogram_rows("exact_prefix", cohort),
        [_prefix_cohort_summary(step, total, pooled, cohort)],
        _support_summary_rows(cell_traces, cell_prefixes, min_support),
    )


def _support_summary_rows(
    cell_traces: dict[tuple[int, tuple[Any, ...]], set[str]],
    cell_prefixes: dict[tuple[int, tuple[Any, ...]], int],
    min_support: int,
) -> list[dict[str, Any]]:
    supports = {key: len(trace_keys) for key, trace_keys in cell_traces.items()}
    rows = []
    for depth in range(len(FALLBACK_FIELDS) + 1):
        keys = [key for key in supports if key[0] == depth]
        values = sorted(supports[key] for key in keys)
        total_prefixes = sum(cell_prefixes[key] for key in keys)
        supported_prefixes = sum(cell_prefixes[key] for key in keys if supports[key] >= min_support)
        rows.append(
            {
                "fallback_depth": depth,
                "retained_fields": ";".join(_retained_fields(depth)).replace("turn_bucket", "fine_turn_bucket"),
                "min_support": min_support,
                "cells": len(values),
                "supported_cells": sum(1 for value in values if value >= min_support),
                "supported_cell_rate": sum(1 for value in values if value >= min_support) / len(values) if values else "",
                "prefixes": total_prefixes,
                "supported_prefixes": supported_prefixes,
                "supported_prefix_rate": supported_prefixes / total_prefixes if total_prefixes else "",
                "p10_support": _percentile_int(values, 0.1),
                "median_support": _percentile_int(values, 0.5),
                "p90_support": _percentile_int(values, 0.9),
                "max_support": values[-1] if values else "",
            }
        )
    return rows


def _rate_bucket_conditional_histograms(path: Path, *, non_near_cap: bool = False) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, float, int, str], int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if _bool(row["censored_right_tail"]) or (non_near_cap and _near_cap(row)):
                continue
            step = int(row["selected_step"])
            total = int(row["selected_total"])
            key = (
                row["condition_id"],
                row["selected_category"],
                str(step),
                str(total),
                total / step,
                int(row["final_total"]),
                _rate_bucket(total / step),
            )
            counts[key] += 1
    return [
        {
            "condition_id": key[0],
            "current_category": key[1],
            "selected_step": key[2],
            "selected_total": key[3],
            "current_rate": key[4],
            "final_total": key[5],
            "rate_bucket": key[6],
            "n": count,
        }
        for key, count in sorted(counts.items())
    ]


def _interval_width_by_trace_position_rows(
    prefix_predictions_csv: Path,
    traces: dict[str, TraceMeta],
    bins: int = TRACE_POSITION_BINS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    with prefix_predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trace = traces[row["trace_key"]]
            if trace.total_turns <= 0:
                raise ValueError(f"trace {trace.trace_key} has non-positive total_turns")
            position = min(1.0, max(0.0, int(row["step"]) / trace.total_turns))
            bin_index = min(bins - 1, int(position * bins))
            grouped[(row["source"], bin_index)].append(float(row["interval80_width"]))

    rows = []
    for source in sorted({source for source, _ in grouped}, key=_stratum_sort_key):
        for bin_index in range(bins):
            values = sorted(grouped.get((source, bin_index), []))
            if not values:
                continue
            low = bin_index / bins
            high = (bin_index + 1) / bins
            rows.append(
                {
                    "source": source,
                    "position_bin": bin_index,
                    "position_low": low,
                    "position_high": high,
                    "position_midpoint": (low + high) / 2,
                    "n": len(values),
                    "mean_interval80_width": _mean(values),
                    "p25_interval80_width": _quantile(values, 0.25),
                    "p75_interval80_width": _quantile(values, 0.75),
                }
            )
    return rows


def _feature_distribution_rows(turns_csv: Path, traces: dict[str, TraceMeta], train_keys: set[str]) -> list[dict[str, Any]]:
    recent_counts: dict[tuple[str, str], int] = defaultdict(int)
    touched_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    investigation_bucket_counts: dict[tuple[str, str], int] = defaultdict(int)
    recent_totals: dict[str, int] = defaultdict(int)
    touched_totals: dict[tuple[str, str], int] = defaultdict(int)
    investigation_bucket_totals: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    investigation_action_counts: dict[str, int] = defaultdict(int)
    corr_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    with turns_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trace_key = row["trace_key"]
            if row.get("kind") == "action":
                action_counts[trace_key] += 1
                investigation_action_counts[trace_key] += int(row.get("current_category") == "INVESTIGATION")

            trace = traces[trace_key]
            step = int(row["step"])
            if trace_key not in train_keys or trace.parse_error or trace.censored_right_tail or step >= trace.total_turns:
                continue

            source = row["source"]
            recent_bucket = row.get("recent_error_bucket") or "clean"
            touched_source = str(_bool(row.get("touched_source", "")))
            investigation_bucket = row.get("investigation_ratio_bucket") or "moderate"
            position_third = _trace_position_third(step, trace.total_turns)
            raw_ratio = investigation_action_counts[trace_key] / action_counts[trace_key] if action_counts[trace_key] else 0.0

            recent_counts[(source, recent_bucket)] += 1
            recent_totals[source] += 1
            touched_counts[(source, position_third, touched_source)] += 1
            touched_totals[(source, position_third)] += 1
            investigation_bucket_counts[(source, investigation_bucket)] += 1
            investigation_bucket_totals[source] += 1
            _add_corr(corr_sums[source], raw_ratio, int(row["total"]))

    rows = []
    sources = sorted(set(recent_totals) | set(investigation_bucket_totals) | {source for source, _ in touched_totals}, key=_stratum_sort_key)
    for source in sources:
        for bucket in ERROR_BUCKETS:
            rows.append(_distribution_row("recent_error_bucket", source, "", bucket, recent_counts[(source, bucket)], recent_totals[source]))
        for third in TRACE_POSITION_THIRDS:
            for value in ("False", "True"):
                rows.append(
                    _distribution_row(
                        "touched_source_by_trace_third",
                        source,
                        third,
                        value,
                        touched_counts[(source, third, value)],
                        touched_totals[(source, third)],
                    )
                )
        for bucket in INVESTIGATION_RATIO_BUCKETS:
            rows.append(
                _distribution_row(
                    "investigation_ratio_bucket",
                    source,
                    "",
                    bucket,
                    investigation_bucket_counts[(source, bucket)],
                    investigation_bucket_totals[source],
                )
            )
        rows.append(
            {
                "diagnostic": "investigation_ratio_current_total_correlation",
                "source": source,
                "position_third": "",
                "value": "",
                "n": int(corr_sums[source][0]),
                "total": int(corr_sums[source][0]),
                "fraction": "",
                "pearson_current_total": _pearson_from_sums(corr_sums[source]),
            }
        )
    return rows


def _five_read_trace_feature_rows(
    prefix_predictions_csv: Path,
    turns_csv: Path,
    lookup: EmpiricalBayesLookup,
    targets: tuple[dict[str, Any], ...] = FIVE_READ_TRACES,
) -> list[dict[str, Any]]:
    target_by_key = {str(target["trace_key"]): target for target in targets}
    selected_predictions: dict[str, dict[str, Any]] = {}
    with prefix_predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = target_by_key.get(row["trace_key"])
            if target is None:
                continue
            current = selected_predictions.get(row["trace_key"])
            score = (abs(int(row["step"]) - int(target["requested_step"])), int(row["step"]))
            if current is None or score < current["_score"]:
                selected_predictions[row["trace_key"]] = row | {"_score": score}

    missing = set(target_by_key) - set(selected_predictions)
    if missing:
        raise ValueError(f"missing five-read prediction rows for {sorted(missing)}")

    selected_steps = {(trace_key, int(row["step"])) for trace_key, row in selected_predictions.items()}
    selected_prefixes, raw_ratios = _selected_turn_prefixes_and_ratios(turns_csv, selected_steps)
    rows = []
    for target in targets:
        trace_key = str(target["trace_key"])
        prediction_row = selected_predictions[trace_key]
        step = int(prediction_row["step"])
        prefix = selected_prefixes[(trace_key, step)]
        prediction = lookup.predict(prefix)
        rows.append(
            {
                "trace_key": trace_key,
                "instance_id": trace_key.rsplit(":", 1)[-1],
                "requested_step": int(target["requested_step"]),
                "selected_step": step,
                "step_distance": abs(step - int(target["requested_step"])),
                "human_read": target["human_read"],
                "recent_error_bucket": prediction_row["recent_error_bucket"],
                "touched_source": prediction_row["touched_source"],
                "investigation_ratio_bucket": prediction_row["investigation_ratio_bucket"],
                "investigation_ratio_raw": raw_ratios[(trace_key, step)],
                "current_total": int(prediction_row["current_total"]),
                "p10": prediction.quantile(0.1),
                "p50": prediction.quantile(0.5),
                "p90": prediction.quantile(0.9),
                "fallback_depth": int(prediction_row["fallback_depth"]),
                "support_count": int(prediction_row["support_count"]),
                "low_confidence_flags": prediction_row["low_confidence_flags"],
            }
        )
    return rows


def _selected_turn_prefixes_and_ratios(
    turns_csv: Path,
    selected_steps: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], PrefixRow], dict[tuple[str, int], float]]:
    action_counts: dict[str, int] = defaultdict(int)
    investigation_action_counts: dict[str, int] = defaultdict(int)
    prefixes = {}
    raw_ratios = {}
    with turns_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trace_key = row["trace_key"]
            if row.get("kind") == "action":
                action_counts[trace_key] += 1
                investigation_action_counts[trace_key] += int(row.get("current_category") == "INVESTIGATION")
            key = (trace_key, int(row["step"]))
            if key in selected_steps:
                prefixes[key] = PrefixRow(
                    trace_key=trace_key,
                    source=row["source"],
                    step=int(row["step"]),
                    total=int(row["total"]),
                    current_category=row.get("current_category") or "NONE",
                    current_unit_age=int(row["current_unit_age"]),
                    had_stuck_episode=_bool(row.get("had_stuck_episode", "")),
                    recent_error_bucket=row.get("recent_error_bucket") or "clean",
                    touched_source=_bool(row.get("touched_source", "")),
                    investigation_ratio_bucket=row.get("investigation_ratio_bucket") or "moderate",
                )
                raw_ratios[key] = investigation_action_counts[trace_key] / action_counts[trace_key] if action_counts[trace_key] else 0.0

    missing = selected_steps - set(prefixes)
    if missing:
        raise ValueError(f"missing turn rows for {sorted(missing)}")
    return prefixes, raw_ratios


def _distribution_row(diagnostic: str, source: str, position_third: str, value: str, n: int, total: int) -> dict[str, Any]:
    return {
        "diagnostic": diagnostic,
        "source": source,
        "position_third": position_third,
        "value": value,
        "n": n,
        "total": total,
        "fraction": n / total if total else "",
        "pearson_current_total": "",
    }


def _trace_position_third(step: int, total_turns: int) -> str:
    position = min(1.0, max(0.0, step / total_turns))
    if position < 1 / 3:
        return "early"
    if position < 2 / 3:
        return "middle"
    return "late"


def _add_corr(sums: list[float], x: float, y: float) -> None:
    sums[0] += 1
    sums[1] += x
    sums[2] += y
    sums[3] += x * x
    sums[4] += y * y
    sums[5] += x * y


def _pearson_from_sums(sums: list[float]) -> float | str:
    n, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
    return numerator / denominator if denominator else ""


def _histogram_rows(group: str, values: list[int]) -> list[dict[str, Any]]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return [{"group": group, "final_total": value, "n": count} for value, count in sorted(counts.items())]


def _prefix_cohort_summary(step: int, total: int, pooled: list[int], cohort: list[int]) -> dict[str, Any]:
    pooled_width = _iqr(pooled)
    cohort_width = _iqr(cohort)
    return {
        "current_step": step,
        "current_total": total,
        "pooled_step_n": len(pooled),
        "exact_prefix_n": len(cohort),
        "pooled_step_iqr": pooled_width,
        "exact_prefix_iqr": cohort_width,
        "pooled_step_p90_minus_p10": _quantile_width(pooled, 0.1, 0.9),
        "exact_prefix_p90_minus_p10": _quantile_width(cohort, 0.1, 0.9),
        "conditional_iqr_narrower": bool(cohort and pooled and cohort_width < pooled_width),
    }


def _iqr(values: list[int]) -> int | str:
    return _quantile_width(values, 0.25, 0.75)


def _quantile_width(values: list[int], low: float, high: float) -> int | str:
    ordered = sorted(values)
    return _quantile(ordered, high) - _quantile(ordered, low) if ordered else ""


def _fine_turn_bucket(step: int) -> str:
    for upper in (2, 4, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191):
        if step <= upper:
            lower = 1 if upper == 2 else {4: 3, 7: 5, 11: 8, 15: 12, 23: 16, 31: 24, 47: 32, 63: 48, 95: 64, 127: 96, 191: 128}[upper]
            return f"{lower}-{upper}"
    return "192+"


def _fine_fallback_key(row: PrefixRow, depth: int) -> tuple[Any, ...]:
    return _fine_fallback_key_values(
        row.source,
        row.total,
        row.current_category or "NONE",
        row.step,
        row.current_unit_age,
        row.had_stuck_episode,
        row.recent_error_bucket,
        row.touched_source,
        row.investigation_ratio_bucket,
        depth,
    )


def _fine_fallback_key_values(
    source: str,
    total: int,
    current_category: str,
    step: int,
    current_unit_age: int,
    had_stuck_episode: bool,
    recent_error_bucket: str,
    touched_source: bool,
    investigation_ratio_bucket: str,
    depth: int,
) -> tuple[Any, ...]:
    values = {
        "source": source,
        "recent_error_bucket": recent_error_bucket,
        "investigation_ratio_bucket": investigation_ratio_bucket,
        "touched_source": touched_source,
        "total": total,
        "current_category": current_category or "NONE",
        "turn_bucket": _fine_turn_bucket(step),
        "age_bucket": age_bucket(current_unit_age),
        "stuck": had_stuck_episode,
    }
    for field in FALLBACK_FIELDS[:depth]:
        values[field] = None
    return tuple(values[field] for field in KEY_FIELDS)


def _percentile_int(values: list[int], probability: float) -> int | str:
    return _quantile(values, probability) if values else ""


def _near_cap(row: dict[str, Any]) -> bool:
    return row["source"] == "swe-agent" and int(row["final_total"]) >= NEAR_CAP_FINAL_TOTAL


def _add_diag(values: list[float], predicted: float, outcome: int) -> None:
    values[0] += 1
    values[1] += predicted
    values[2] += outcome


def _diag_row(prefix: dict[str, Any], values: list[float]) -> dict[str, Any]:
    n, predicted, outcome = values
    mean_predicted = predicted / n
    observed = outcome / n
    return prefix | {
        "n": int(n),
        "mean_predicted_p": mean_predicted,
        "observed_rate": observed,
        "mean_bias": mean_predicted - observed,
    }


def _progress(message: str, started: float) -> None:
    elapsed = time.monotonic() - started
    print(f"[empirical-bayes {elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


def _fallback_key(row: PrefixRow, depth: int) -> tuple[Any, ...]:
    key = _prefix_state_key(row)
    values = dict(zip(KEY_FIELDS, key))
    for field in FALLBACK_FIELDS[:depth]:
        values[field] = None
    return tuple(values[field] for field in KEY_FIELDS)


def _prefix_state_key(row: PrefixRow) -> tuple[Any, ...]:
    return (
        row.source,
        row.recent_error_bucket,
        row.investigation_ratio_bucket,
        row.touched_source,
        row.total,
        row.current_category or "NONE",
        turn_bucket(row.step),
        age_bucket(row.current_unit_age),
        row.had_stuck_episode,
    )


def _retained_fields(depth: int) -> tuple[str, ...]:
    dropped = set(FALLBACK_FIELDS[:depth])
    return tuple(field for field in KEY_FIELDS if field not in dropped)


def _encode_key(key: tuple[Any, ...]) -> list[Any]:
    return list(key)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quantile(values: list[int], probability: float) -> int:
    return values[max(0, math.ceil(probability * len(values)) - 1)]


def _percentile(values: list[float], probability: float) -> float | str:
    if not values:
        return ""
    return values[max(0, min(len(values) - 1, math.ceil(probability * len(values)) - 1))]


def _nearest_np_percentile(values: Any, probability: float) -> float | str:
    if len(values) == 0:
        return ""
    ordered = values.copy()
    ordered.sort()
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _rate_bucket(rate: float) -> str:
    lower = 0.0
    for upper in RATE_BUCKETS:
        if rate <= upper:
            return f"{lower:.1f}-{upper:.1f}"
        lower = upper
    return f"{RATE_BUCKETS[-1]:.1f}+"


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float | str:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else ""


def _strata(row: PrefixRow, length_terciles: dict[str, str]) -> dict[str, str]:
    length = length_terciles.get(row.trace_key, "unknown")
    return {"length_tercile": length, "source_length_tercile": f"{row.source} x {length}"}


def _all_strata(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        strata["pooled"].append(row)
        strata[str(row["source"])].append(row)
        strata[str(row["source_length_tercile"])].append(row)
    return strata


def _stratum_sort_key(value: str) -> tuple[int, int, str]:
    if value == "pooled":
        return (-1, -1, value)
    sources = {"hermes": 0, "swe-agent": 1, "terminalbench": 2}
    lengths = {"short": 0, "medium": 1, "long": 2}
    if " x " not in value:
        return (sources.get(value, 99), 99, value)
    source, length = value.split(" x ", 1)
    return (sources.get(source, 99), lengths.get(length, 99), value)


def _prefix_strata(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata["pooled"].append(row)
        strata[str(row["source"])].append(row)
        strata[f"{row['source']} x {row['length_tercile']}"].append(row)
    return strata


def _length_terciles(traces: list[TraceMeta]) -> dict[str, str]:
    by_source: dict[str, list[TraceMeta]] = defaultdict(list)
    for trace in traces:
        if not trace.parse_error:
            by_source[trace.source].append(trace)
    result = {}
    for source, items in by_source.items():
        ordered = sorted(items, key=lambda trace: (trace.total_turns, trace.trace_key))
        for index, trace in enumerate(ordered):
            tercile = "short" if index < len(ordered) / 3 else "medium" if index < 2 * len(ordered) / 3 else "long"
            result[trace.trace_key] = tercile
    return result
