"""Fit the serial migration model and evaluate it on a held-out repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_REPEATS = (0, 1)
VALIDATION_REPEAT = 2
PRIOR_REPLAY_POINTS = {
    "central": [[23224, 4935.46], [25607, 4565.42], [30800, 3949.87], [31562, 3875.10]],
    "faster": [[23224, 5675.78], [25607, 5250.23], [30800, 4542.35], [31562, 4456.37]],
    "slower": [[23224, 4195.14], [25607, 3880.61], [30800, 3357.39], [31562, 3293.84]],
}


def serial(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = rows[(rows.method == method) & (rows.concurrency == 1)
                    & (rows.activity == "none")].copy()
    if not {0, 1, 2} <= set(selected.repeat):
        raise ValueError(f"{method} needs repeats 0, 1, and 2")
    return selected


def fit_replay(rows: pd.DataFrame) -> tuple[list[list[float]], float]:
    selected = serial(rows, "replay")
    train = selected[selected.repeat.isin(TRAIN_REPEATS)].copy()
    train["rate"] = train.measured_processed_tokens / train.initial_time_to_first_response_s
    curve = train.groupby("measured_prompt_tokens").rate.median().sort_index()
    if len(curve) < 2 or (curve <= 0).any():
        raise ValueError("replay fit needs two positive measured prompt sizes")
    held = selected[selected.repeat == VALIDATION_REPEAT]
    predicted = held.measured_processed_tokens / np.interp(
        held.measured_prompt_tokens, curve.index, curve
    )
    error = np.quantile(abs(predicted / held.initial_time_to_first_response_s - 1), .9)
    return [[float(tokens), float(rate)] for tokens, rate in curve.items()], float(error)


def fit_kv(rows: pd.DataFrame) -> tuple[float, float]:
    selected = serial(rows, "kv_transfer")
    train = selected[selected.repeat.isin(TRAIN_REPEATS)]
    train = train[train.bandwidth_mbps == train.bandwidth_mbps.max()]
    size = train.measured_kv_bytes.to_numpy(float)
    elapsed = train.initial_time_to_first_response_s.to_numpy(float)
    rate = float(size @ size / (size @ elapsed))
    held = selected[selected.repeat == VALIDATION_REPEAT]
    network = held.measured_kv_bytes / (held.bandwidth_mbps * 1e6 / 8)
    predicted = np.maximum(network, held.measured_kv_bytes / rate)
    error = np.quantile(abs(predicted / held.initial_time_to_first_response_s - 1), .9)
    return rate, float(error)


def total_action_power(rows: pd.DataFrame, method: str, serial_only: bool) -> dict:
    selected = rows[(rows.kind == "migration") & (rows.method == method)
                    & (rows.activity == "none") & rows.repeat.isin(TRAIN_REPEATS)]
    if serial_only:
        selected = selected[selected.concurrency == 1]
    medians = selected.groupby("concurrency")[[
        "source_added_power_w", "destination_added_power_w"
    ]].median().clip(lower=0).cummax()
    if medians.empty:
        raise ValueError(f"no {method} action power measurements")
    return {str(int(width)): [float(source), float(destination)]
            for width, (source, destination) in medians.iterrows()}


def evaluation(rows: pd.DataFrame, replay_curve: list[list[float]],
               destination_rate: float) -> pd.DataFrame:
    out = []
    for method in ("replay", "kv_transfer"):
        held = serial(rows, method)
        held = held[held.repeat == VALIDATION_REPEAT]
        if method == "replay":
            predicted = held.measured_processed_tokens / np.interp(
                held.measured_prompt_tokens, *np.asarray(replay_curve).T
            )
        else:
            predicted = np.maximum(
                held.measured_kv_bytes / (held.bandwidth_mbps * 1e6 / 8),
                held.measured_kv_bytes / destination_rate,
            )
        for (_, row), value in zip(held.iterrows(), predicted):
            measured = row.initial_time_to_first_response_s
            out.append({
                "method": method, "session_id": row.get("session_id", ""),
                "bandwidth_mbps": row.bandwidth_mbps,
                "measured_prompt_tokens": row.measured_prompt_tokens,
                "measured_processed_tokens": row.get("measured_processed_tokens", 0),
                "measured_kv_bytes": row.get("measured_kv_bytes", 0),
                "measured_time_s": measured, "predicted_time_s": value,
                "predicted_over_measured": value / measured,
                "absolute_relative_error": abs(value / measured - 1),
            })
    return pd.DataFrame(out)


def plot_evaluation(rows: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, method in zip(axes, ("kv_transfer", "replay")):
        selected = rows[rows.method == method]
        for bandwidth, group in selected.groupby("bandwidth_mbps"):
            ax.scatter(group.measured_time_s, group.predicted_time_s, alpha=.65,
                       label=f"{bandwidth:g} Mb/s")
        limit = max(selected.measured_time_s.max(), selected.predicted_time_s.max())
        ax.plot([0, limit], [0, limit], "k--", label="equal")
        ax.set(title=method.replace("_", " "), xlabel="Measured time (s)",
               ylabel="Predicted time (s)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def fit_profile(run_root: Path, profile_path: Path) -> None:
    migrations = pd.read_csv(run_root / "migrations.csv")
    scenarios = pd.read_csv(run_root / "scenarios.csv")
    replay_curve, replay_error = fit_replay(migrations)
    destination_rate, kv_error = fit_kv(migrations)
    replay_power = total_action_power(scenarios, "replay", False)
    kv_power = total_action_power(scenarios, "kv_transfer", True)
    switches = serial(migrations, "kv_transfer")
    switches = switches[switches.repeat.isin(TRAIN_REPEATS)].route_switch_s
    raw = json.loads(profile_path.read_text())
    raw["schema"] = "queue-haul-model-profile-v2"
    raw["profile_id"] = "gpt-oss-20b-a100-tp1-20260716"
    raw["sources"]["replay"] = {
        "kind": "measured", "reference": (
            f"{run_root}/migrations.csv repeats 0-1 through {replay_curve[-1][0]:g} tokens; "
            "outputs/power_drain_live_20260714/live_profile_*.json above that range"
        ),
        "valid_range": [replay_curve[0][0], PRIOR_REPLAY_POINTS["central"][-1][0]],
        "relative_error": max(replay_error, .3),
    }
    kv_rows = serial(migrations, "kv_transfer")
    kv_train = kv_rows[kv_rows.repeat.isin(TRAIN_REPEATS)]
    raw["sources"]["kv_transfer"] = {
        "kind": "measured", "reference": f"{run_root}/migrations.csv repeats 0-1",
        "valid_range": [float(kv_train.measured_kv_bytes.min()),
                        float(kv_train.measured_kv_bytes.max())],
        "relative_error": kv_error,
    }
    raw["sources"]["transitions"]["reference"] = (
        f"{run_root}/migrations.csv and scenarios.csv for route switch and action power; "
        "TODO measure catch-up, sleep, and shutdown"
    )
    scales = {
        "central": (1, .5), "faster": (1 / (1 - replay_error), .25),
        "slower": (1 / (1 + replay_error), .75),
    }
    for name, (replay_scale, switch_quantile) in scales.items():
        case = raw["cases"][name]
        case["replay_tps"] = {
            "1": [[x, y * replay_scale] for x, y in replay_curve]
            + PRIOR_REPLAY_POINTS[name]
        }
        old = case["kv_transfer"]
        kv_scale = 1 if name == "central" else (
            1 / (1 - kv_error) if name == "faster" else 1 / (1 + kv_error)
        )
        case["kv_transfer"] = {
            "block_tokens": old["block_tokens"], "block_bytes": old["block_bytes"],
            "setup_s": 0, "destination_bytes_per_s": destination_rate * kv_scale,
            "sync_s": 0,
        }
        case["switch_s"] = float(switches.quantile(switch_quantile))
        case["action_power_w"] = {
            "replay": replay_power, "kv_transfer": kv_power,
            "replay_on_request": {"1": replay_power["1"]},
            "catch_up": {"1": [0, 0]}, "sleep": {"1": [0, 0]},
            "off": {"1": [0, 0]},
        }
    profile_path.write_text(json.dumps(raw, indent=2) + "\n")
    held = evaluation(migrations, replay_curve, destination_rate)
    held.to_csv(run_root / "profile_evaluation.csv", index=False)
    plot_evaluation(held, run_root / "profile_evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    fit_profile(args.run_root, args.profile)


if __name__ == "__main__":
    main()
