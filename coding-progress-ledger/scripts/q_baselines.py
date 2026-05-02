"""Q3 + Q4 — Baseline evaluation for the Q1 channel-native targets.

Joins the W3 estimator checkpoint table to the Q1 label table, fits
four baselines per target, and evaluates each via leave-one-run-out
by `run_id`. Emits per-row predictions and a metrics summary.

Models:
  always_mean        train-set base rate, no features
  elapsed_only       single feature: step
  progress_only      single feature: coding_progress
  checkpoint_table   all numeric W3 features

Estimator: scikit-learn LogisticRegression if importable, else a
deterministic 5-bin base-rate baseline.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_COLUMNS = (
    "future_progress_drop",
    "product_reopened_after_completion",
    "validation_exposes_new_work",
    "stuck_loop_next_window",
    "submit_without_validation_state",
)

CHECKPOINT_FEATURES = (
    "step",
    "active_leaf_count", "active_coding_leaf_count", "active_validation_leaf_count",
    "completed_leaf_count", "coding_progress", "validation_progress",
    "num_reopens_so_far", "num_invalidations_so_far", "largest_progress_drop_so_far",
    "num_splits_so_far", "steps_since_new_subtask", "denominator_growth_so_far",
    "steps_since_completion", "blocked_leaf_count", "repeated_observation_loop_flag",
    "validation_started", "validation_complete", "validation_failed",
    "submit_without_validation",
    "strong_completion_count", "manual_only_completion_count",
    "weak_product_completion_count",
)

MODEL_FEATURES = {
    "always_mean": (),
    "elapsed_only": ("step",),
    "progress_only": ("coding_progress",),
    "checkpoint_table": CHECKPOINT_FEATURES,
}

PREDICTION_COLUMNS = (
    "run_id", "step", "target", "model_name", "label", "predicted_probability",
)
PROB_CLIP = (0.001, 0.999)
EPSILON = 1e-12


# ── data loading ────────────────────────────────────────────────────────────


def _to_float(v: str) -> float:
    s = v.strip().lower()
    if s == "true":
        return 1.0
    if s == "false":
        return 0.0
    return float(v)


def _label_int(value) -> int:
    if isinstance(value, bool):
        return int(value)
    return 1 if str(value).strip().lower() == "true" else 0


def join_rows(checkpoint_csv: Path, labels_csv: Path) -> list[dict]:
    with checkpoint_csv.open() as fh:
        ckpt = {(r["run_id"], r["step"]): r for r in csv.DictReader(fh)}
    rows = []
    with labels_csv.open() as fh:
        for lab in csv.DictReader(fh):
            key = (lab["run_id"], lab["step"])
            if key not in ckpt:
                raise ValueError(f"label row {key} has no W3 checkpoint")
            merged = {**ckpt[key]}
            for col in TARGET_COLUMNS:
                merged[f"label_target_{col}"] = lab[col]
            rows.append(merged)
    return rows


def _feature_vector(row: dict, features: tuple[str, ...]) -> list[float]:
    return [_to_float(row[f]) for f in features]


# ── model fitting ───────────────────────────────────────────────────────────


def _train_labels(train_rows: list[dict], target: str) -> list[int]:
    return [_label_int(r[f"label_target_{target}"]) for r in train_rows]


def _fit_always_mean(train_labels: list[int]) -> float:
    return sum(train_labels) / len(train_labels) if train_labels else 0.5


def _fit_sklearn(
    train_rows: list[dict], train_labels: list[int], features: tuple[str, ...]
):
    from sklearn.linear_model import LogisticRegression
    x_train = [_feature_vector(r, features) for r in train_rows]
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(x_train, train_labels)

    def predict(row: dict) -> float:
        return float(model.predict_proba([_feature_vector(row, features)])[0][1])

    return predict


def _fit_binned(
    train_rows: list[dict],
    train_labels: list[int],
    features: tuple[str, ...],
    bins: int = 5,
):
    base_rate = sum(train_labels) / len(train_labels)
    ranges = {f: (
        min(_to_float(r[f]) for r in train_rows),
        max(_to_float(r[f]) for r in train_rows),
    ) for f in features}

    def _normalized_score(row: dict) -> float:
        norms = []
        for f in features:
            lo, hi = ranges[f]
            v = _to_float(row[f])
            norms.append(0.5 if abs(hi - lo) < EPSILON
                         else min(1.0, max(0.0, (v - lo) / (hi - lo))))
        return sum(norms) / len(norms)

    def _bin(row: dict) -> int:
        return min(bins - 1, max(0, int(_normalized_score(row) * bins)))

    rates: dict[int, float] = {}
    buckets: dict[int, list[int]] = {}
    for r, y in zip(train_rows, train_labels):
        buckets.setdefault(_bin(r), []).append(y)
    for b, ys in buckets.items():
        rates[b] = sum(ys) / len(ys)

    return lambda row: rates.get(_bin(row), base_rate)


_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def fit_predictor(
    train_rows: list[dict], features: tuple[str, ...], target: str
):
    train_labels = _train_labels(train_rows, target)
    base_rate = _fit_always_mean(train_labels)
    if not features or len(set(train_labels)) < 2:
        return lambda row: base_rate
    if _HAS_SKLEARN:
        return _fit_sklearn(train_rows, train_labels, features)
    return _fit_binned(train_rows, train_labels, features)


# ── splits and metrics ──────────────────────────────────────────────────────


def loro_splits(run_ids: list[str]) -> list[tuple[set[str], set[str]]]:
    unique = sorted(set(run_ids))
    return [({rid for rid in unique if rid != held}, {held}) for held in unique]


def auroc(labels: list[int], probs: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(zip(probs, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(y for _, y in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def brier(labels: list[int], probs: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probs)) / len(labels)


def log_loss(labels: list[int], probs: list[float], clip: float = 1e-6) -> float:
    total = 0.0
    for y, p in zip(labels, probs):
        p = min(max(p, clip), 1 - clip)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(labels)


def _clip(p: float) -> float:
    lo, hi = PROB_CLIP
    return min(max(p, lo), hi)


# Compatibility aliases for existing tests.
_loro_splits = loro_splits
_auroc = auroc
_brier = brier
_log_loss = log_loss


def _fit_predict(
    train_rows: list[dict],
    test_rows: list[dict],
    features: tuple[str, ...],
    target: str,
) -> list[float]:
    predictor = fit_predictor(train_rows, features, target)
    return [predictor(row) for row in test_rows]


# ── evaluation loop ─────────────────────────────────────────────────────────


def _rows_by_run(rows: list[dict]) -> dict[str, list[dict]]:
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    return by_run


def evaluate(rows: list[dict]) -> tuple[list[dict], dict]:
    by_run = _rows_by_run(rows)
    run_ids = sorted(by_run)
    predictions: list[dict] = []
    metrics: dict = {target: {} for target in TARGET_COLUMNS}

    for target in TARGET_COLUMNS:
        for model_name, features in MODEL_FEATURES.items():
            all_labels: list[int] = []
            all_probs: list[float] = []
            for train_ids, test_ids in loro_splits(run_ids):
                train = [r for rid in sorted(train_ids) for r in by_run[rid]]
                test = [r for rid in sorted(test_ids) for r in by_run[rid]]
                predictor = fit_predictor(train, features, target)
                for r in test:
                    p = _clip(predictor(r))
                    y = _label_int(r[f"label_target_{target}"])
                    all_labels.append(y)
                    all_probs.append(p)
                    predictions.append({
                        "run_id": r["run_id"],
                        "step": r["step"],
                        "target": target,
                        "model_name": model_name,
                        "label": "true" if y else "false",
                        "predicted_probability": f"{p:.6f}",
                    })
            metrics[target][model_name] = {
                "n": len(all_labels),
                "positive_rate": sum(all_labels) / len(all_labels) if all_labels else None,
                "auroc": auroc(all_labels, all_probs),
                "brier": brier(all_labels, all_probs),
                "log_loss": log_loss(all_labels, all_probs),
            }
    return predictions, metrics


# ── output ──────────────────────────────────────────────────────────────────


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)


def _format_optional_float(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def write_summary(path: Path, metrics: dict, sources: dict) -> None:
    estimator = "sklearn LogisticRegression" if _HAS_SKLEARN else "binned base-rate baseline"
    lines = [
        "# Q3 / Q4 — Baseline evaluation summary",
        "",
        "Leave-one-run-out by `run_id` over the W3 checkpoint table joined to",
        "the Q1 channel-native targets. **No predictive performance claim is",
        "made; see `datasets/RESULTS_DISCLAIMERS.md`.**",
        "",
        f"- Source W3 features: `{sources['checkpoint_csv']}`",
        f"- Source Q1 labels: `{sources['labels_csv']}`",
        f"- Estimator: {estimator}",
        f"- Splits: leave-one-run-out (N={sources['n_runs']})",
        "",
        "## Targets and label base rates",
        "",
        "| target | n rows | positive rate |",
        "|---|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        m = metrics[target]["always_mean"]
        lines.append(f"| `{target}` | {m['n']} | {_format_optional_float(m['positive_rate'])} |")
    lines += ["", "## LORO metrics by target × model", ""]
    for target in TARGET_COLUMNS:
        lines += [
            f"### `{target}`",
            "",
            "| model | AUROC | Brier | log loss |",
            "|---|---:|---:|---:|",
        ]
        for model_name in MODEL_FEATURES:
            m = metrics[target][model_name]
            lines.append(
                f"| `{model_name}` | {_format_optional_float(m['auroc'])} | "
                f"{m['brier']:.3f} | {m['log_loss']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Reading these numbers",
        "",
        "- AUROC \"n/a\" means the held-out fold had only one class; common at",
        "  N=20 LORO for skewed targets.",
        "- `always_mean` is the trivial base-rate baseline. A model below it on",
        "  Brier or log loss is *worse* than predicting the train-set mean.",
        "- The `submit_without_validation_state` target is constant per run, so",
        "  any model that uses run-level state at step S will look near-perfect",
        "  on rows where the agent has already committed to no-validation.",
        "  This is a property of the data, not predictive skill.",
        "- `progress_only` and `elapsed_only` mirror the `completion_prediction_smoke`",
        "  feature sets; they are informational baselines, not target models.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()
    rows = join_rows(args.checkpoint_csv, args.labels_csv)
    n_runs = len({r["run_id"] for r in rows})
    predictions, metrics = evaluate(rows)
    write_predictions(args.predictions_csv, predictions)
    write_summary(args.summary_md, metrics, {
        "checkpoint_csv": args.checkpoint_csv.as_posix(),
        "labels_csv": args.labels_csv.as_posix(),
        "n_runs": n_runs,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
