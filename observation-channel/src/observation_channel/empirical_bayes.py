from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TURN_GRID = (1, 2, 4, 8, 16, 32, 64)
FALLBACK_FIELDS = ("stuck", "age_bucket", "turn_bucket", "current_category", "total")
COMFORTABLE_SUPPORT = 50


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
        count = 0
        for value in self.values:
            if value <= threshold:
                count += 1
            else:
                break
        return count / len(self.values)

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
            values = tuple(value for value in self.cells.get(key, ()) if value >= row.total)
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
    if step <= 4:
        return "0-4"
    if step <= 9:
        return "5-9"
    if step <= 19:
        return "10-19"
    if step <= 39:
        return "20-39"
    if step <= 79:
        return "40-79"
    if step <= 159:
        return "80-159"
    return "160+"


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


def read_prefixes_csv(path: Path, traces: dict[str, TraceMeta]) -> list[PrefixRow]:
    prefixes = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        has_stuck = "had_stuck_episode" in (reader.fieldnames or ())
        for row in reader:
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
                )
            )
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
    bootstrap_resamples: int = 1000,
    seed: int = 1729,
    min_support: int = 25,
) -> dict[str, Any]:
    traces = read_traces_csv(traces_csv)
    prefixes = read_prefixes_csv(turns_csv, traces)
    train_keys, eval_keys, split_rows = source_stratified_split(traces.values())
    train_prefixes = [row for row in eligible_prefixes(prefixes, traces) if row.trace_key in train_keys]
    eval_prefixes = [row for row in eligible_prefixes(prefixes, traces) if row.trace_key in eval_keys]
    train_traces = {key: trace for key, trace in traces.items() if key in train_keys and not trace.parse_error and not trace.censored_right_tail}
    lookup = EmpiricalBayesLookup.build(train_prefixes, train_traces, min_support=min_support)

    report_dir.mkdir(parents=True, exist_ok=True)
    lookup.save(bundle_path)

    length_terciles = _length_terciles([traces[key] for key in eval_keys if not traces[key].censored_right_tail])
    pairs, prefix_predictions = _prediction_rows(eval_prefixes, traces, lookup, length_terciles)
    reliability = _reliability_rows(pairs)
    bands = _bootstrap_bands(pairs, bootstrap_resamples, seed)
    sharpness = _sharpness_rows(prefix_predictions)
    coverage = _coverage_rows(prefix_predictions)
    skipped = _skipped_censored_rows(prefixes, traces, eval_keys, lookup)

    _write_csv(report_dir / "heldout_predictions.csv", pairs)
    _write_csv(report_dir / "prefix_predictions.csv", prefix_predictions)
    _write_csv(report_dir / "reliability.csv", reliability)
    _write_csv(report_dir / "bootstrap_bands.csv", bands)
    _write_csv(report_dir / "sharpness_summary.csv", sharpness)
    _write_csv(report_dir / "coverage_summary.csv", coverage)
    _write_csv(report_dir / "split_summary.csv", split_rows)
    _write_csv(report_dir / "censored_skipped_summary.csv", skipped)
    _write_report(report_dir / "REPORT.md", split_rows, skipped, bootstrap_resamples, seed)
    _plot_reliability(report_dir / "reliability.png", reliability, bands)
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    prefix_predictions = []
    for row in prefixes:
        trace = traces[row.trace_key]
        if trace.censored_right_tail:
            continue
        try:
            prediction = lookup.predict(row)
        except ValueError:
            continue
        strata = _strata(row, length_terciles)
        prefix_predictions.append(
            {
                "trace_key": row.trace_key,
                "source": row.source,
                "step": row.step,
                "current_total": row.total,
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
        for offset in TURN_GRID:
            threshold = row.total + offset
            if max(prediction.values) < threshold:
                continue
            pairs.append(
                {
                    "trace_key": row.trace_key,
                    "source": row.source,
                    "step": row.step,
                    "current_total": row.total,
                    "threshold": threshold,
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


def _bootstrap_bands(pairs: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    bands = []
    for stratum, stratum_pairs in _all_strata(pairs).items():
        by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in stratum_pairs:
            by_trace[str(row["trace_key"])].append(row)
        trace_keys = sorted(by_trace)
        samples: dict[int, list[float]] = defaultdict(list)
        if not trace_keys:
            continue
        for _ in range(resamples):
            resampled = []
            for trace_key in rng.choices(trace_keys, k=len(trace_keys)):
                resampled.extend(by_trace[trace_key])
            for row in _reliability_for_pairs(stratum, resampled):
                if row["n"]:
                    samples[int(row["bin"])].append(float(row["observed_rate"]))
        for bin_index in range(10):
            values = sorted(samples.get(bin_index, []))
            bands.append(
                {
                    "stratum": stratum,
                    "bin": bin_index,
                    "bootstrap_resamples": resamples,
                    "seed": seed,
                    "observed_low": _percentile(values, 0.025),
                    "observed_high": _percentile(values, 0.975),
                }
            )
    return bands


def _sharpness_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stratum, items in _prefix_strata(prefix_rows).items():
        widths = sorted(int(item["interval80_width"]) for item in items)
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fallback_key(row: PrefixRow, depth: int) -> tuple[Any, ...]:
    values = {
        "source": row.source,
        "total": row.total,
        "current_category": row.current_category or "NONE",
        "turn_bucket": turn_bucket(row.step),
        "age_bucket": age_bucket(row.current_unit_age),
        "stuck": row.had_stuck_episode,
    }
    for field in FALLBACK_FIELDS[:depth]:
        values[field] = None
    return (
        values["source"],
        values["total"],
        values["current_category"],
        values["turn_bucket"],
        values["age_bucket"],
        values["stuck"],
    )


def _retained_fields(depth: int) -> tuple[str, ...]:
    dropped = set(FALLBACK_FIELDS[:depth])
    return tuple(field for field in ("source", "total", "current_category", "turn_bucket", "age_bucket", "stuck") if field not in dropped)


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
