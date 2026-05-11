from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .empirical_bayes import (
    EmpiricalBayesLookup,
    Prediction,
    _length_terciles,
    _mean,
    _progress,
    _write_csv,
    eligible_prefixes,
    read_traces_csv,
    source_stratified_split,
)
from .gbm_trial import GbmQuantilePrediction, GbmTrialBundle, read_gbm_prefixes_csv


VARIANTS = ("eb_direct", "eb_filter", "gbm_direct", "eb_gbm_mixed_filter")
REMAINING_FRACTION_CLAIMS = (("remaining_le_25pct", 0.25), ("remaining_le_50pct", 0.50))
FINISH_WITHIN_OFFSETS = (8, 16, 32)
EPSILON = 1e-12


@dataclass(frozen=True)
class BeliefTrackerConfig:
    filter_alpha: float = 0.35
    gbm_weight: float = 0.15
    gbm_crossing_tolerance: float = 1.0


@dataclass(frozen=True)
class FinalWorkBelief:
    masses: dict[int, float]

    @classmethod
    def from_prediction(cls, prediction: Prediction, current_total: int) -> "FinalWorkBelief":
        counts: dict[int, int] = defaultdict(int)
        for value in prediction.values:
            if value >= current_total:
                counts[int(value)] += 1
        total = sum(counts.values())
        if not total:
            raise ValueError("final-work belief has no feasible support")
        return cls({value: count / total for value, count in sorted(counts.items())})

    @classmethod
    def from_gbm(cls, prediction: GbmQuantilePrediction, support: Iterable[int]) -> "FinalWorkBelief":
        masses = {}
        previous = None
        for value in sorted(set(int(item) for item in support)):
            low = prediction.cdf(previous) if previous is not None else 0.0
            masses[value] = max(0.0, prediction.cdf(value) - low)
            previous = value
        return cls._normalize(masses)

    @classmethod
    def _normalize(cls, masses: dict[int, float]) -> "FinalWorkBelief":
        positive = {value: weight for value, weight in masses.items() if weight > 0 and math.isfinite(weight)}
        total = sum(positive.values())
        if total <= 0:
            raise ValueError("final-work belief has no positive mass")
        return cls({value: weight / total for value, weight in sorted(positive.items())})

    def with_minimum(self, current_total: int) -> "FinalWorkBelief":
        return self._normalize({value: weight for value, weight in self.masses.items() if value >= current_total})

    def cdf(self, threshold: int) -> float:
        return sum(weight for value, weight in self.masses.items() if value <= threshold)

    def quantile(self, probability: float) -> int:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        cumulative = 0.0
        for value, weight in sorted(self.masses.items()):
            cumulative += weight
            if cumulative + EPSILON >= probability:
                return value
        return max(self.masses)

    def remaining_fraction_quantile(self, probability: float, current_total: int) -> float:
        transformed = {
            value_i: (weight, _remaining_fraction(current_total, final_work))
            for value_i, (final_work, weight) in enumerate(sorted(self.masses.items()))
        }
        cumulative = 0.0
        for _, (weight, fraction) in sorted(transformed.items(), key=lambda item: item[1][1]):
            cumulative += weight
            if cumulative + EPSILON >= probability:
                return fraction
        return max(fraction for _, fraction in transformed.values())

    def entropy(self) -> float:
        return -sum(weight * math.log(weight) for weight in self.masses.values() if weight > 0)


@dataclass
class _TraceState:
    eb_filter: FinalWorkBelief | None = None
    eb_filter_median: int | None = None
    mixed_filter: FinalWorkBelief | None = None
    mixed_filter_median: int | None = None


class BeliefTracker:
    def __init__(self, lookup: EmpiricalBayesLookup, config: BeliefTrackerConfig | None = None) -> None:
        self.lookup = lookup
        self.config = config or BeliefTrackerConfig()
        self._states: dict[str, _TraceState] = {}

    def update(self, row: Any, gbm_prediction: GbmQuantilePrediction | None = None) -> dict[str, Any]:
        state = self._states.setdefault(row.trace_key, _TraceState())
        output: dict[str, Any] = {}
        try:
            eb_prediction = self.lookup.predict(row)
        except ValueError:
            eb_prediction = None

        if gbm_prediction is not None:
            output.update(_gbm_columns("gbm_direct", row.total, gbm_prediction))
        else:
            output.update(_blank_columns("gbm_direct"))

        if eb_prediction is None:
            output.update(_blank_columns("eb_direct"))
            output.update(_blank_columns("eb_filter"))
            output.update(_blank_columns("eb_gbm_mixed_filter"))
            output.update(_blank_mixed_diagnostics())
            output["confidence_flags"] = "no_empirical_bayes_support"
            return output

        eb_belief = FinalWorkBelief.from_prediction(eb_prediction, row.total)
        output.update(_belief_columns("eb_direct", row.total, eb_belief, eb_prediction.low_confidence_reasons))

        eb_filter, eb_jump = _filter_update(state.eb_filter, eb_belief, row.total, self.config.filter_alpha, state.eb_filter_median)
        state.eb_filter = eb_filter
        state.eb_filter_median = eb_filter.quantile(0.5)
        output.update(_belief_columns("eb_filter", row.total, eb_filter, eb_prediction.low_confidence_reasons))
        output["eb_filter_posterior_entropy"] = eb_filter.entropy()
        output["eb_filter_posterior_median_jump"] = eb_jump

        mixed_observation = eb_belief
        gbm_used = False
        gbm_reason = ""
        gbm_outside = ""
        gbm_reordering = ""
        if gbm_prediction is None:
            gbm_reason = "no_gbm_prediction"
        else:
            gbm_reordering = gbm_prediction.reordering_magnitude
            gbm_outside = _gbm_median_outside_eb_band(gbm_prediction, eb_belief)
            gbm_reason = _gbm_rejection_reason(gbm_prediction, eb_belief, self.config)
            if not gbm_reason:
                gbm_belief = FinalWorkBelief.from_gbm(gbm_prediction, eb_belief.masses)
                mixed_observation = _mix_beliefs(eb_belief, gbm_belief, self.config.gbm_weight)
                gbm_used = True

        mixed_filter, mixed_jump = _filter_update(
            state.mixed_filter,
            mixed_observation,
            row.total,
            self.config.filter_alpha,
            state.mixed_filter_median,
        )
        state.mixed_filter = mixed_filter
        state.mixed_filter_median = mixed_filter.quantile(0.5)
        mixed_flags = tuple(reason for reason in (*eb_prediction.low_confidence_reasons, gbm_reason) if reason)
        output.update(_belief_columns("eb_gbm_mixed_filter", row.total, mixed_filter, mixed_flags))
        output["eb_gbm_mixed_filter_gbm_used"] = gbm_used
        output["eb_gbm_mixed_filter_gbm_rejected_reason"] = gbm_reason
        output["eb_gbm_mixed_filter_gbm_reordering_magnitude"] = gbm_reordering
        output["eb_gbm_mixed_filter_gbm_median_outside_eb_band"] = gbm_outside
        output["eb_gbm_mixed_filter_posterior_entropy"] = mixed_filter.entropy()
        output["eb_gbm_mixed_filter_posterior_median_jump"] = mixed_jump
        output["confidence_flags"] = ";".join(sorted(set((*eb_prediction.low_confidence_reasons, gbm_reason)) - {""}))
        return output


def evaluate_belief_tracker(
    turns_csv: Path,
    traces_csv: Path,
    report_dir: Path,
    model_dir: Path,
    *,
    bundle_path: Path | None = None,
    min_support: int = 25,
    filter_alpha: float = 0.35,
    gbm_weight: float = 0.15,
    bundle: Any | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    progress = lambda message: _progress(message, started)
    progress("loading trace metadata")
    traces = read_traces_csv(traces_csv)
    progress("loading raw-feature prefix rows")
    prefixes = read_gbm_prefixes_csv(turns_csv, traces, progress)
    progress("loading GBM model bundle")
    gbm_bundle = bundle if bundle is not None else GbmTrialBundle.load(model_dir)
    progress("splitting traces")
    train_keys, eval_keys, split_rows = source_stratified_split(traces.values())
    eligible = eligible_prefixes(prefixes, traces)
    eval_prefixes = [row for row in eligible if row.trace_key in eval_keys]
    if bundle_path is not None:
        progress("loading empirical-Bayes lookup")
        lookup = EmpiricalBayesLookup.load(bundle_path)
    else:
        train_prefixes = [row for row in eligible if row.trace_key in train_keys]
        train_traces = {key: trace for key, trace in traces.items() if key in train_keys and not trace.parse_error and not trace.censored_right_tail}
        progress("building empirical-Bayes lookup")
        lookup = EmpiricalBayesLookup.build(train_prefixes, train_traces, min_support=min_support)
    tracker = BeliefTracker(lookup, BeliefTrackerConfig(filter_alpha=filter_alpha, gbm_weight=gbm_weight))
    length_terciles = _length_terciles([traces[key] for key in eval_keys if not traces[key].censored_right_tail])

    rows = []
    ordered_eval = sorted(eval_prefixes, key=lambda row: (row.trace_key, row.step))
    progress(f"replaying {len(ordered_eval):,} held-out prefixes")
    chunk_size = 200_000
    for start in range(0, len(ordered_eval), chunk_size):
        chunk = ordered_eval[start : start + chunk_size]
        gbm_predictions = gbm_bundle.predict(chunk)
        for row, gbm_prediction in zip(chunk, gbm_predictions):
            trace = traces[row.trace_key]
            result = _prefix_columns(row, trace, length_terciles.get(row.trace_key, "unknown"))
            result.update(tracker.update(row, gbm_prediction))
            rows.append(result)
        progress(f"replayed {len(rows):,} held-out prefixes")

    report_dir.mkdir(parents=True, exist_ok=True)
    claim_rows = _claim_calibration_rows(rows)
    summary = _belief_summary(rows, claim_rows)
    _write_csv(report_dir / "progress_beliefs.csv", rows)
    _write_csv(report_dir / "belief_threshold_pairs.csv", claim_rows)
    _write_csv(report_dir / "belief_summary.csv", summary)
    _write_csv(report_dir / "split_summary.csv", split_rows)
    _plot_trace_examples(report_dir / "trace_belief_examples.png", rows)
    _write_report(report_dir / "REPORT.md", split_rows, summary)
    progress("done")
    return {
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
        "prefix_rows": len(rows),
        "threshold_pairs": len(claim_rows),
    }


def _filter_update(
    previous: FinalWorkBelief | None,
    observation: FinalWorkBelief,
    current_total: int,
    alpha: float,
    previous_median: int | None,
) -> tuple[FinalWorkBelief, int | str]:
    if previous is None:
        belief = observation.with_minimum(current_total)
    else:
        belief = _log_pool(previous.with_minimum(current_total), observation.with_minimum(current_total), alpha)
    median = belief.quantile(0.5)
    return belief, abs(median - previous_median) if previous_median is not None else ""


def _log_pool(previous: FinalWorkBelief, observation: FinalWorkBelief, alpha: float) -> FinalWorkBelief:
    values = sorted(set(previous.masses) | set(observation.masses))
    logs = {
        value: (1 - alpha) * math.log(previous.masses.get(value, EPSILON))
        + alpha * math.log(observation.masses.get(value, EPSILON))
        for value in values
    }
    peak = max(logs.values())
    return FinalWorkBelief._normalize({value: math.exp(log_weight - peak) for value, log_weight in logs.items()})


def _mix_beliefs(left: FinalWorkBelief, right: FinalWorkBelief, weight: float) -> FinalWorkBelief:
    values = sorted(set(left.masses) | set(right.masses))
    return FinalWorkBelief._normalize(
        {value: (1 - weight) * left.masses.get(value, 0.0) + weight * right.masses.get(value, 0.0) for value in values}
    )


def _gbm_rejection_reason(prediction: GbmQuantilePrediction, eb_belief: FinalWorkBelief, config: BeliefTrackerConfig) -> str:
    if prediction.crossed and prediction.reordering_magnitude > config.gbm_crossing_tolerance:
        return "gbm_quantile_crossing"
    if _gbm_median_outside_eb_band(prediction, eb_belief):
        return "gbm_median_outside_eb_band"
    return ""


def _gbm_median_outside_eb_band(prediction: GbmQuantilePrediction, eb_belief: FinalWorkBelief) -> bool:
    median = prediction.quantiles[2]
    return median < eb_belief.quantile(0.1) or median > eb_belief.quantile(0.9)


def _belief_columns(prefix: str, current_total: int, belief: FinalWorkBelief, flags: Iterable[str]) -> dict[str, Any]:
    columns = {
        f"{prefix}_estimated_final_work_p10": belief.quantile(0.1),
        f"{prefix}_estimated_final_work_p50": belief.quantile(0.5),
        f"{prefix}_estimated_final_work_p90": belief.quantile(0.9),
        f"{prefix}_remaining_work_fraction_p10": belief.remaining_fraction_quantile(0.1, current_total),
        f"{prefix}_remaining_work_fraction_p50": belief.remaining_fraction_quantile(0.5, current_total),
        f"{prefix}_remaining_work_fraction_p90": belief.remaining_fraction_quantile(0.9, current_total),
        f"{prefix}_prob_remaining_work_le_25pct": belief.cdf(_remaining_fraction_threshold(current_total, 0.25)),
        f"{prefix}_prob_remaining_work_le_50pct": belief.cdf(_remaining_fraction_threshold(current_total, 0.50)),
        f"{prefix}_confidence_flags": ";".join(reason for reason in flags if reason),
    }
    for offset in FINISH_WITHIN_OFFSETS:
        columns[f"{prefix}_prob_finish_within_{offset}_work_units"] = belief.cdf(current_total + offset)
    return columns


def _gbm_columns(prefix: str, current_total: int, prediction: GbmQuantilePrediction) -> dict[str, Any]:
    p10, _, p50, _, p90 = prediction.quantiles
    columns = {
        f"{prefix}_estimated_final_work_p10": p10,
        f"{prefix}_estimated_final_work_p50": p50,
        f"{prefix}_estimated_final_work_p90": p90,
        f"{prefix}_remaining_work_fraction_p10": _remaining_fraction(current_total, p10),
        f"{prefix}_remaining_work_fraction_p50": _remaining_fraction(current_total, p50),
        f"{prefix}_remaining_work_fraction_p90": _remaining_fraction(current_total, p90),
        f"{prefix}_prob_remaining_work_le_25pct": prediction.cdf(_remaining_fraction_threshold(current_total, 0.25)),
        f"{prefix}_prob_remaining_work_le_50pct": prediction.cdf(_remaining_fraction_threshold(current_total, 0.50)),
        f"{prefix}_confidence_flags": "quantile_crossing" if prediction.crossed else "",
    }
    for offset in FINISH_WITHIN_OFFSETS:
        columns[f"{prefix}_prob_finish_within_{offset}_work_units"] = prediction.cdf(current_total + offset)
    return columns


def _blank_columns(prefix: str) -> dict[str, str]:
    columns = {
        f"{prefix}_estimated_final_work_p10": "",
        f"{prefix}_estimated_final_work_p50": "",
        f"{prefix}_estimated_final_work_p90": "",
        f"{prefix}_remaining_work_fraction_p10": "",
        f"{prefix}_remaining_work_fraction_p50": "",
        f"{prefix}_remaining_work_fraction_p90": "",
        f"{prefix}_prob_remaining_work_le_25pct": "",
        f"{prefix}_prob_remaining_work_le_50pct": "",
        f"{prefix}_confidence_flags": "",
    }
    for offset in FINISH_WITHIN_OFFSETS:
        columns[f"{prefix}_prob_finish_within_{offset}_work_units"] = ""
    return columns


def _blank_mixed_diagnostics() -> dict[str, str]:
    return {
        "eb_filter_posterior_entropy": "",
        "eb_filter_posterior_median_jump": "",
        "eb_gbm_mixed_filter_gbm_used": "",
        "eb_gbm_mixed_filter_gbm_rejected_reason": "",
        "eb_gbm_mixed_filter_gbm_reordering_magnitude": "",
        "eb_gbm_mixed_filter_gbm_median_outside_eb_band": "",
        "eb_gbm_mixed_filter_posterior_entropy": "",
        "eb_gbm_mixed_filter_posterior_median_jump": "",
    }


def _prefix_columns(row: Any, trace: Any, length_tercile: str) -> dict[str, Any]:
    return {
        "trace_key": row.trace_key,
        "source": row.source,
        "length_tercile": length_tercile,
        "step": row.step,
        "current_total": row.total,
        "current_category": row.current_category,
        "current_unit_age": row.current_unit_age,
        "had_stuck_episode": row.had_stuck_episode,
        "recent_error_bucket": row.recent_error_bucket,
        "recent_error_rate": row.recent_error_rate,
        "touched_source": row.touched_source,
        "investigation_ratio_bucket": row.investigation_ratio_bucket,
        "investigation_ratio": row.investigation_ratio,
        "actual_final_work": trace.final_total,
        "actual_remaining_work_fraction": _remaining_fraction(row.total, trace.final_total),
    }


def _claim_calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for row in rows:
        current_total = int(row["current_total"])
        actual_final_work = int(row["actual_final_work"])
        for variant in VARIANTS:
            for claim, fraction in REMAINING_FRACTION_CLAIMS:
                predicted = row.get(f"{variant}_prob_remaining_work_le_{int(fraction * 100)}pct")
                if predicted == "":
                    continue
                _add_claim(bins, variant, claim, float(predicted), _remaining_fraction(current_total, actual_final_work) <= fraction)
            for offset in FINISH_WITHIN_OFFSETS:
                predicted = row.get(f"{variant}_prob_finish_within_{offset}_work_units")
                if predicted == "":
                    continue
                _add_claim(bins, variant, f"finish_within_{offset}_work_units", float(predicted), actual_final_work <= current_total + offset)
    return [
        {
            "variant": variant,
            "claim": claim,
            "bin": bin_index,
            "n": int(values[0]),
            "mean_predicted_p": values[1] / values[0],
            "observed_rate": values[2] / values[0],
            "mean_bias": (values[1] - values[2]) / values[0],
        }
        for (variant, claim, bin_index), values in sorted(bins.items())
        if values[0]
    ]


def _add_claim(
    bins: dict[tuple[str, str, int], list[float]],
    variant: str,
    claim: str,
    predicted: float,
    outcome: bool,
) -> None:
    values = bins[(variant, claim, min(9, int(predicted * 10)))]
    values[0] += 1
    values[1] += predicted
    values[2] += int(outcome)


def _belief_summary(rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in claim_rows:
        by_key[(row["variant"], row["claim"])].append(row)
    for variant, claim in sorted(by_key):
        items = by_key[(variant, claim)]
        n = sum(int(item["n"]) for item in items)
        summary.append(
            {
                "summary_type": "claim_calibration",
                "variant": variant,
                "claim": claim,
                "n": n,
                "mean_predicted_p": sum(int(item["n"]) * float(item["mean_predicted_p"]) for item in items) / n,
                "observed_rate": sum(int(item["n"]) * float(item["observed_rate"]) for item in items) / n,
                "mean_bias": sum(int(item["n"]) * float(item["mean_bias"]) for item in items) / n,
                "ece": _ece(items),
                "median_absolute_final_work_error": "",
                "median_interval80_width": "",
            }
        )
    for variant in VARIANTS:
        medians = []
        widths = []
        for row in rows:
            p10 = row.get(f"{variant}_estimated_final_work_p10")
            p50 = row.get(f"{variant}_estimated_final_work_p50")
            p90 = row.get(f"{variant}_estimated_final_work_p90")
            if p50 == "":
                continue
            medians.append(abs(float(p50) - int(row["actual_final_work"])))
            widths.append(float(p90) - float(p10))
        summary.append(
            {
                "summary_type": "final_work",
                "variant": variant,
                "claim": "",
                "n": len(medians),
                "mean_predicted_p": "",
                "observed_rate": "",
                "mean_bias": "",
                "ece": "",
                "median_absolute_final_work_error": _median(medians),
                "median_interval80_width": _median(widths),
            }
        )
    return summary


def _ece(items: list[dict[str, Any]]) -> float | str:
    if not items:
        return ""
    n = sum(int(item["n"]) for item in items)
    return sum(
        int(item["n"]) / n * abs(float(item["mean_predicted_p"]) - float(item["observed_rate"]))
        for item in items
    )


def _plot_trace_examples(path: Path, rows: list[dict[str, Any]], limit: int = 6) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trace[str(row["trace_key"])].append(row)
    selected = sorted(by_trace, key=lambda key: (-len(by_trace[key]), key))[:limit]
    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 2.5 * len(selected)), squeeze=False, sharex=False)
    for axis, trace_key in zip(axes.flat, selected):
        items = sorted(by_trace[trace_key], key=lambda row: int(row["step"]))
        xs = [int(row["step"]) for row in items]
        actual = [_progress_fraction(int(row["current_total"]), int(row["actual_final_work"])) for row in items]
        axis.plot(xs, actual, color="0.2", linestyle="--", linewidth=1.2, label="actual")
        for variant in VARIANTS:
            ys = [
                _progress_fraction(int(row["current_total"]), float(row[f"{variant}_estimated_final_work_p50"]))
                if row.get(f"{variant}_estimated_final_work_p50") != ""
                else math.nan
                for row in items
            ]
            axis.plot(xs, ys, linewidth=1, label=variant)
        axis.set_title(trace_key, fontsize=8)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("observed / estimated final")
    axes.flat[0].legend(fontsize=7, ncol=3)
    axes.flat[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(path: Path, split_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Progress-Belief Tracker",
        "",
        "Replayable live estimator over final work using empirical Bayes, GBM, and filtered combinations.",
        "",
        "## Split Summary",
        "",
        "| source | train_traces | eval_traces | total_traces |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in split_rows:
        lines.append(f"| {row['source']} | {row['train_traces']} | {row['eval_traces']} | {row['total_traces']} |")
    lines.extend(["", "## Final-Work Summary", "", "| variant | n | median_abs_error | median_interval80_width |", "| --- | ---: | ---: | ---: |"])
    for row in summary:
        if row["summary_type"] == "final_work":
            lines.append(
                f"| {row['variant']} | {row['n']} | {row['median_absolute_final_work_error']} | {row['median_interval80_width']} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- [Progress beliefs](progress_beliefs.csv)",
            "- [Claim calibration pairs](belief_threshold_pairs.csv)",
            "- [Belief summary](belief_summary.csv)",
            "",
            "![Trace belief examples](trace_belief_examples.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _remaining_fraction_threshold(current_total: int, fraction: float) -> int:
    return math.floor(current_total / (1.0 - fraction))


def _remaining_fraction(current_total: int, final_work: float) -> float:
    return max(0.0, (float(final_work) - current_total) / max(1.0, float(final_work)))


def _progress_fraction(current_total: int, final_work: float) -> float:
    return current_total / max(1.0, float(final_work))


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
