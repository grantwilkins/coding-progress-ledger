"""Q3 + Q4 — Baselines for the five Q1 targets, leave-one-run-out.

For each target column in `datasets/swe_agent_q_labels.csv`, fit four
baselines (always-mean, elapsed-only, progress-only, checkpoint-table)
on W3 features and evaluate via leave-one-run-out by `run_id`.
Writes per-row predictions and a metrics summary.
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

MODEL_FEATURES = {
    "always_mean": (),
    "elapsed_only": ("step",),
    "progress_only": ("coding_progress",),
    "checkpoint_table": (
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
    ),
}

PREDICTION_COLUMNS = (
    "run_id", "step", "target", "model_name", "label", "predicted_probability",
)
EPSILON = 1e-12


def _to_float(v: str) -> float:
    s = v.strip().lower()
    if s == "true":
        return 1.0
    if s == "false":
        return 0.0
    return float(v)


def _label_int(v) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    return 1 if str(v).strip().lower() == "true" else 0


def _join_rows(checkpoint_csv: Path, labels_csv: Path) -> list[dict]:
    with checkpoint_csv.open() as fh:
        ckpt = {(r["run_id"], r["step"]): r for r in csv.DictReader(fh)}
    rows = []
    with labels_csv.open() as fh:
        for lab in csv.DictReader(fh):
            key = (lab["run_id"], lab["step"])
            if key not in ckpt:
                raise ValueError(f"label row {key} has no W3 checkpoint")
            merged = {**ckpt[key], **{f"label_target_{c}": lab[c] for c in TARGET_COLUMNS}}
            rows.append(merged)
    return rows


def _feature_vector(row: dict, features: tuple[str, ...]) -> list[float]:
    return [_to_float(row[f]) for f in features]


def _fit_predict(
    train_rows: list[dict],
    test_rows: list[dict],
    features: tuple[str, ...],
    target: str,
) -> list[float]:
    target_col = f"label_target_{target}"
    labels = [_label_int(r[target_col]) for r in train_rows]
    base_rate = sum(labels) / len(labels) if labels else 0.5
    if not features:
        return [base_rate] * len(test_rows)
    if len(set(labels)) < 2:
        return [base_rate] * len(test_rows)
    sklearn_spec = importlib.util.find_spec("sklearn")
    if sklearn_spec is not None:
        from sklearn.linear_model import LogisticRegression
        x_train = [_feature_vector(r, features) for r in train_rows]
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(x_train, labels)
        return [
            float(model.predict_proba([_feature_vector(r, features)])[0][1])
            for r in test_rows
        ]
    return _binned_baseline(train_rows, test_rows, features, labels, base_rate)


def _binned_baseline(
    train_rows: list[dict],
    test_rows: list[dict],
    features: tuple[str, ...],
    labels: list[int],
    base_rate: float,
    bins: int = 5,
) -> list[float]:
    ranges = {}
    for f in features:
        vals = [_to_float(r[f]) for r in train_rows]
        ranges[f] = (min(vals), max(vals))

    def _score(row: dict) -> float:
        norm = []
        for f in features:
            lo, hi = ranges[f]
            v = _to_float(row[f])
            norm.append(0.5 if abs(hi - lo) < EPSILON else min(1.0, max(0.0, (v - lo) / (hi - lo))))
        return sum(norm) / len(norm)

    rates: dict[int, float] = {}
    buckets: dict[int, list[int]] = {}
    for r, lab in zip(train_rows, labels):
        idx = min(bins - 1, max(0, int(_score(r) * bins)))
        buckets.setdefault(idx, []).append(lab)
    for idx, b in buckets.items():
        rates[idx] = sum(b) / len(b)

    out = []
    for r in test_rows:
        idx = min(bins - 1, max(0, int(_score(r) * bins)))
        out.append(rates.get(idx, base_rate))
    return out


def _loro_splits(run_ids: list[str]) -> list[tuple[set[str], set[str]]]:
    return [({rid for rid in run_ids if rid != held}, {held}) for held in sorted(set(run_ids))]


def _auroc(labels: list[int], probs: list[float]) -> float | None:
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
        rank_sum += avg_rank * sum(lab for _, lab in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _brier(labels: list[int], probs: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probs)) / len(labels)


def _log_loss(labels: list[int], probs: list[float], clip: float = 1e-6) -> float:
    total = 0.0
    for y, p in zip(labels, probs):
        p = min(max(p, clip), 1 - clip)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(labels)


def evaluate(rows: list[dict]) -> tuple[list[dict], dict]:
    run_ids = sorted({r["run_id"] for r in rows})
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    predictions: list[dict] = []
    metrics: dict = {}
    for target in TARGET_COLUMNS:
        metrics[target] = {}
        for model_name, feats in MODEL_FEATURES.items():
            all_labels: list[int] = []
            all_probs: list[float] = []
            for train_ids, test_ids in _loro_splits(run_ids):
                train = [r for rid in sorted(train_ids) for r in by_run[rid]]
                test = [r for rid in sorted(test_ids) for r in by_run[rid]]
                probs = _fit_predict(train, test, feats, target)
                for r, p in zip(test, probs):
                    p = min(max(p, 0.001), 0.999)
                    label = _label_int(r[f"label_target_{target}"])
                    all_labels.append(label)
                    all_probs.append(p)
                    predictions.append({
                        "run_id": r["run_id"],
                        "step": r["step"],
                        "target": target,
                        "model_name": model_name,
                        "label": "true" if label else "false",
                        "predicted_probability": f"{p:.6f}",
                    })
            metrics[target][model_name] = {
                "n": len(all_labels),
                "positive_rate": sum(all_labels) / len(all_labels) if all_labels else None,
                "auroc": _auroc(all_labels, all_probs),
                "brier": _brier(all_labels, all_probs),
                "log_loss": _log_loss(all_labels, all_probs),
            }
    return predictions, metrics


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)


def write_summary(path: Path, metrics: dict, sources: dict) -> None:
    estimator = "sklearn LogisticRegression" if importlib.util.find_spec("sklearn") else "binned base-rate baseline"
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
        rate = "n/a" if m["positive_rate"] is None else f"{m['positive_rate']:.3f}"
        lines.append(f"| `{target}` | {m['n']} | {rate} |")
    lines += ["", "## LORO metrics by target × model", ""]
    for target in TARGET_COLUMNS:
        lines.append(f"### `{target}`")
        lines.append("")
        lines.append("| model | AUROC | Brier | log loss |")
        lines.append("|---|---:|---:|---:|")
        for model_name in MODEL_FEATURES:
            m = metrics[target][model_name]
            auroc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
            lines.append(f"| `{model_name}` | {auroc} | {m['brier']:.3f} | {m['log_loss']:.3f} |")
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
    rows = _join_rows(args.checkpoint_csv, args.labels_csv)
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
