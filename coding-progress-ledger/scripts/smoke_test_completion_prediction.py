from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_FEATURES = (
    "step",
    "event_index",
    "coding_progress",
    "overall_progress",
    "active_coding_weight",
    "completed_coding_weight",
    "active_overall_weight",
    "completed_overall_weight",
    "active_coding_leaves",
    "completed_coding_leaves",
    "active_overall_leaves",
    "completed_overall_leaves",
    "num_splits_so_far",
    "num_reopens_so_far",
    "num_invalidations_so_far",
    "delta_coding_progress",
    "delta_overall_progress",
    "category_resolution_mode",
    "category_overrides_applied",
)

MODEL_FEATURES = {
    "progress_only": ("coding_progress",),
    "ledger_basic": (
        "coding_progress",
        "overall_progress",
        "active_coding_weight",
        "completed_coding_weight",
        "active_coding_leaves",
        "completed_coding_leaves",
        "num_splits_so_far",
        "num_reopens_so_far",
        "num_invalidations_so_far",
        "delta_coding_progress",
    ),
    "elapsed_only": ("step", "event_index"),
}

REQUIRED_COLUMNS = frozenset(("run_id", "final_success", *ALLOWED_FEATURES))
PREDICTION_COLUMNS = (
    "run_id",
    "step",
    "event_index",
    "final_success",
    "model_name",
    "predicted_success_probability",
    "coding_progress",
    "overall_progress",
)
FORBIDDEN_EXACT_FEATURES = frozenset(
    (
        "run_id",
        "final_success",
        "final_success_source",
        "event_type",
        "subtask_id",
        "coding_drop_source",
        "overall_drop_source",
        "native_coding_drop_source",
        "native_overall_drop_source",
    )
)
FORBIDDEN_PREFIXES = ("native_", "summary_by_category", "test_result")
DISCLAIMER = (
    "This smoke test verifies the completion-prediction plumbing on a tiny curated dataset. "
    "It is not evidence of general predictive performance. The next scientific test requires "
    "retrospective SWE-agent or Terminal-Bench trajectories with many more natural successes and failures."
)
HIGH_PROGRESS_FAILURE_THRESHOLD = 0.8
EPSILON = 1e-12


@dataclass(frozen=True)
class SmokeResult:
    rows: list[dict[str, str]]
    predictions: list[dict[str, str]]
    metrics_by_model: dict[str, dict[str, float | None]]
    estimator_name: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run completion-prediction plumbing smoke test.")
    parser.add_argument("--input-csv", default="datasets/ledger_observations_v0_step.csv")
    parser.add_argument("--predictions-csv", default="datasets/completion_prediction_smoke_predictions.csv")
    parser.add_argument("--report-md", default="datasets/completion_prediction_smoke_report.md")
    args = parser.parse_args(argv)

    run_smoke(Path(args.input_csv), Path(args.predictions_csv), Path(args.report_md))
    return 0


def run_smoke(input_csv: Path, predictions_csv: Path, report_md: Path) -> SmokeResult:
    rows = read_rows(input_csv)
    run_labels = validate_rows(rows, MODEL_FEATURES)
    predictions, estimator_name = predict_leave_one_run_out(rows, run_labels, MODEL_FEATURES)
    metrics_by_model = {
        model_name: metrics_for_predictions(model_predictions)
        for model_name, model_predictions in _predictions_by_model(predictions).items()
    }
    write_predictions_csv(predictions_csv, predictions)
    write_report(report_md, rows, predictions, metrics_by_model, estimator_name)
    return SmokeResult(rows, predictions, metrics_by_model, estimator_name)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no checkpoint rows")
    return rows


def validate_rows(
    rows: list[dict[str, str]],
    feature_sets: dict[str, tuple[str, ...]],
) -> dict[str, bool]:
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    validate_feature_sets(feature_sets)

    run_labels: dict[str, bool] = {}
    for index, row in enumerate(rows, start=2):
        run_id = row["run_id"]
        if not run_id:
            raise ValueError(f"row {index}: run_id is required")
        label = parse_final_success(row["final_success"], row_number=index)
        previous = run_labels.setdefault(run_id, label)
        if previous != label:
            raise ValueError(f"{run_id}: final_success changes across checkpoint rows")

        for feature in _used_numeric_features(feature_sets):
            parse_float(row[feature], feature, index)

    labels = set(run_labels.values())
    if labels != {False, True}:
        raise ValueError("expected at least one success run and one failure run")

    validate_leave_one_run_out_splits(leave_one_run_out_splits(run_labels))
    return run_labels


def validate_feature_sets(feature_sets: dict[str, tuple[str, ...]]) -> None:
    for model_name, features in feature_sets.items():
        if not features:
            raise ValueError(f"{model_name}: feature set is empty")
        for feature in features:
            if is_forbidden_feature(feature):
                raise ValueError(f"{model_name}: forbidden feature used: {feature}")
            if feature not in ALLOWED_FEATURES:
                raise ValueError(f"{model_name}: feature is not in allowed list: {feature}")


def is_forbidden_feature(feature: str) -> bool:
    return feature in FORBIDDEN_EXACT_FEATURES or any(feature.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def leave_one_run_out_splits(run_labels: dict[str, bool]) -> list[tuple[set[str], set[str]]]:
    run_ids = sorted(run_labels)
    return [(set(run_id for run_id in run_ids if run_id != held_out), {held_out}) for held_out in run_ids]


def validate_leave_one_run_out_splits(splits: list[tuple[set[str], set[str]]]) -> None:
    for train_run_ids, test_run_ids in splits:
        overlap = train_run_ids & test_run_ids
        if overlap:
            raise ValueError(f"train/test run_id leakage: {', '.join(sorted(overlap))}")
        if len(test_run_ids) != 1:
            raise ValueError("leave-one-run-out split must hold out exactly one run")


def predict_leave_one_run_out(
    rows: list[dict[str, str]],
    run_labels: dict[str, bool],
    feature_sets: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, str]], str]:
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    estimator_name = "sklearn LogisticRegression" if sklearn_available else "deterministic binned success-rate baseline"
    by_run = _rows_by_run(rows)
    predictions: list[dict[str, str]] = []

    for model_name, features in feature_sets.items():
        for train_run_ids, test_run_ids in leave_one_run_out_splits(run_labels):
            train_rows = [row for run_id in sorted(train_run_ids) for row in by_run[run_id]]
            test_rows = [row for run_id in sorted(test_run_ids) for row in by_run[run_id]]
            predict_probability = (
                _fit_sklearn_logistic(train_rows, features)
                if sklearn_available
                else _fit_binned_success_rate(train_rows, features)
            )
            for row in test_rows:
                probability = _clip_probability(predict_probability(row))
                predictions.append(
                    {
                        "run_id": row["run_id"],
                        "step": row["step"],
                        "event_index": row["event_index"],
                        "final_success": _canonical_label(row["final_success"]),
                        "model_name": model_name,
                        "predicted_success_probability": f"{probability:.6f}",
                        "coding_progress": row["coding_progress"],
                        "overall_progress": row["overall_progress"],
                    }
                )
    return predictions, estimator_name


def _fit_sklearn_logistic(
    train_rows: list[dict[str, str]],
    features: tuple[str, ...],
) -> Callable[[dict[str, str]], float]:
    labels = [int(parse_final_success(row["final_success"])) for row in train_rows]
    if len(set(labels)) < 2:
        rate = sum(labels) / len(labels)
        return lambda row: rate

    from sklearn.linear_model import LogisticRegression

    categories = _category_levels(train_rows, features)
    x_train = [_feature_vector(row, features, categories) for row in train_rows]
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(x_train, labels)

    def predict(row: dict[str, str]) -> float:
        return float(model.predict_proba([_feature_vector(row, features, categories)])[0][1])

    return predict


def _fit_binned_success_rate(
    train_rows: list[dict[str, str]],
    features: tuple[str, ...],
    *,
    bins: int = 5,
) -> Callable[[dict[str, str]], float]:
    labels = [int(parse_final_success(row["final_success"])) for row in train_rows]
    global_rate = sum(labels) / len(labels)
    ranges = _feature_ranges(train_rows, features)
    rates_by_bin: dict[int, float] = {}
    labels_by_bin: dict[int, list[int]] = {}
    for row, label in zip(train_rows, labels):
        labels_by_bin.setdefault(_bin_index(_binned_score(row, features, ranges), bins), []).append(label)
    for bin_index, bin_labels in labels_by_bin.items():
        rates_by_bin[bin_index] = sum(bin_labels) / len(bin_labels)

    def predict(row: dict[str, str]) -> float:
        return rates_by_bin.get(_bin_index(_binned_score(row, features, ranges), bins), global_rate)

    return predict


def metrics_for_predictions(predictions: list[dict[str, str]]) -> dict[str, float | None]:
    labels = [1 if row["final_success"] == "true" else 0 for row in predictions]
    probabilities = [float(row["predicted_success_probability"]) for row in predictions]
    return {
        "auroc": auroc(labels, probabilities),
        "brier": sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels),
        "log_loss": log_loss(labels, probabilities),
    }


def write_predictions_csv(path: Path, predictions: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    predictions: list[dict[str, str]],
    metrics_by_model: dict[str, dict[str, float | None]],
    estimator_name: str,
) -> None:
    run_labels = _run_labels(rows)
    success_runs = sorted(run_id for run_id, label in run_labels.items() if label)
    failure_runs = sorted(run_id for run_id, label in run_labels.items() if not label)
    grouped_predictions = _predictions_by_model(predictions)
    lines = [
        "# Completion Prediction Smoke Report",
        "",
        DISCLAIMER,
        "",
        "## Dataset",
        "",
        f"- Number of runs: {len(run_labels)}",
        f"- Number of checkpoint rows: {len(rows)}",
        f"- Success runs: {len(success_runs)} ({_inline_list(success_runs)})",
        f"- Failure runs: {len(failure_runs)} ({_inline_list(failure_runs)})",
        "",
        "## Evaluation",
        "",
        "- Method: leave-one-run-out by run_id",
        "- Train/test run_id overlap: none, validated before fitting",
        f"- Estimator: {estimator_name}",
        "",
        "## Feature Sets Used",
        "",
    ]
    for model_name, features in MODEL_FEATURES.items():
        lines.append(f"- `{model_name}`: " + ", ".join(f"`{feature}`" for feature in features))

    lines.extend(
        [
            "",
            "## Leakage Exclusions",
            "",
            "- `run_id` is used only for leave-one-run-out grouping, never as a model feature.",
            "- `final_success` is used only as the label.",
            "- `final_success_source`, `event_type`, `subtask_id`, all `native_*` fields, drop-source fields, test-result fields, final-row aggregates copied backward, and summary_by_category final metrics are excluded from model features.",
            "",
            "## Metrics",
            "",
            "| model | AUROC | Brier score | log loss |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model_name, metrics in metrics_by_model.items():
        lines.append(
            "| {model} | {auroc} | {brier} | {log_loss} |".format(
                model=model_name,
                auroc=_format_optional(metrics["auroc"]),
                brier=_format_float(metrics["brier"]),
                log_loss=_format_float(metrics["log_loss"]),
            )
        )

    lines.extend(
        [
            "",
            "## Mean Predicted Probability",
            "",
            f"High-progress failures are failure rows with `coding_progress >= {HIGH_PROGRESS_FAILURE_THRESHOLD}`.",
            "",
            "| model | successes | failures | high-progress failures | monotonic incomplete failures |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name, model_predictions in grouped_predictions.items():
        lines.append(
            "| {model} | {successes} | {failures} | {high_progress_failures} | {monotonic_incomplete_failures} |".format(
                model=model_name,
                successes=_format_optional(_mean_probability(model_predictions, _is_success)),
                failures=_format_optional(_mean_probability(model_predictions, _is_failure)),
                high_progress_failures=_format_optional(_mean_probability(model_predictions, _is_high_progress_failure)),
                monotonic_incomplete_failures=_format_optional(
                    _mean_probability(model_predictions, _is_monotonic_incomplete_failure)
                ),
            )
        )

    lines.extend(["", "## Case Notes", ""])
    row_groups = _rows_by_run(rows)
    for run_id in (
        "control_high_progress_wrong_solution",
        "control_monotonic_incomplete_failure",
        "control_coding_complete_artifacts_incomplete",
    ):
        if run_id not in row_groups:
            lines.append(f"- `{run_id}`: not present in input dataset.")
            continue
        final_row = row_groups[run_id][-1]
        run_predictions = [prediction for prediction in predictions if prediction["run_id"] == run_id]
        model_means = _model_mean_text(run_predictions)
        lines.append(
            "- `{run_id}`: final_success={label}, checkpoint_rows={rows}, final coding_progress={coding}, "
            "final overall_progress={overall}, model mean predicted probabilities: {means}.".format(
                run_id=run_id,
                label=_canonical_label(final_row["final_success"]),
                rows=len(row_groups[run_id]),
                coding=_format_float(float(final_row["coding_progress"])),
                overall=_format_float(float(final_row["overall_progress"])),
                means=model_means,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_final_success(value: str, row_number: int | None = None) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    location = f"row {row_number}: " if row_number is not None else ""
    raise ValueError(f"{location}final_success must be known true/false, got {value!r}")


def parse_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field} must be finite, got {value!r}")
    return parsed


def auroc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][0] == ranked[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:next_index])
        index = next_index
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def log_loss(labels: list[int], probabilities: list[float], *, clip: float = 1e-6) -> float:
    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = min(max(probability, clip), 1 - clip)
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / len(labels)


def _feature_ranges(rows: list[dict[str, str]], features: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for feature in features:
        if feature == "category_resolution_mode":
            continue
        values = [float(row[feature]) for row in rows]
        ranges[feature] = (min(values), max(values))
    return ranges


def _binned_score(
    row: dict[str, str],
    features: tuple[str, ...],
    ranges: dict[str, tuple[float, float]],
) -> float:
    numeric_features = [feature for feature in features if feature != "category_resolution_mode"]
    if not numeric_features:
        return 0.5
    normalized_values = []
    for feature in numeric_features:
        low, high = ranges[feature]
        if abs(high - low) < EPSILON:
            normalized_values.append(0.5)
        else:
            normalized_values.append(min(max((float(row[feature]) - low) / (high - low), 0.0), 1.0))
    return sum(normalized_values) / len(normalized_values)


def _bin_index(score: float, bins: int) -> int:
    return min(bins - 1, max(0, int(score * bins)))


def _category_levels(rows: list[dict[str, str]], features: tuple[str, ...]) -> list[str]:
    if "category_resolution_mode" not in features:
        return []
    return sorted({row["category_resolution_mode"] for row in rows})


def _feature_vector(
    row: dict[str, str],
    features: tuple[str, ...],
    category_levels: list[str],
) -> list[float]:
    vector: list[float] = []
    for feature in features:
        if feature == "category_resolution_mode":
            vector.extend(1.0 if row[feature] == level else 0.0 for level in category_levels)
        else:
            vector.append(float(row[feature]))
    return vector


def _used_numeric_features(feature_sets: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        feature
        for features in feature_sets.values()
        for feature in features
        if feature != "category_resolution_mode"
    }


def _rows_by_run(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(row)
    return grouped


def _run_labels(rows: list[dict[str, str]]) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for row in rows:
        labels.setdefault(row["run_id"], parse_final_success(row["final_success"]))
    return labels


def _predictions_by_model(predictions: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction["model_name"], []).append(prediction)
    return grouped


def _clip_probability(probability: float) -> float:
    return min(max(probability, 0.001), 0.999)


def _canonical_label(value: str) -> str:
    return "true" if parse_final_success(value) else "false"


def _is_success(row: dict[str, str]) -> bool:
    return row["final_success"] == "true"


def _is_failure(row: dict[str, str]) -> bool:
    return row["final_success"] == "false"


def _is_high_progress_failure(row: dict[str, str]) -> bool:
    return _is_failure(row) and float(row["coding_progress"]) >= HIGH_PROGRESS_FAILURE_THRESHOLD


def _is_monotonic_incomplete_failure(row: dict[str, str]) -> bool:
    return _is_failure(row) and "monotonic_incomplete" in row["run_id"]


def _mean_probability(
    predictions: list[dict[str, str]],
    predicate: Callable[[dict[str, str]], bool],
) -> float | None:
    values = [float(row["predicted_success_probability"]) for row in predictions if predicate(row)]
    if not values:
        return None
    return sum(values) / len(values)


def _model_mean_text(predictions: list[dict[str, str]]) -> str:
    grouped = _predictions_by_model(predictions)
    return ", ".join(
        f"{model_name}={_format_float(_mean_probability(model_predictions, lambda row: True))}"
        for model_name, model_predictions in grouped.items()
    )


def _format_optional(value: float | None) -> str:
    if value is None:
        return "not computable"
    return _format_float(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return "not computable"
    return f"{value:.6f}"


def _inline_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"


if __name__ == "__main__":
    raise SystemExit(main())
