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

from profiles import ModelProfile, PROFILE_SCHEMA


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


def completion_s(rows: pd.DataFrame, method: str) -> float:
    selected = serial(rows, method)
    selected = selected[selected.repeat.isin(TRAIN_REPEATS)]
    return float((selected.initial_response_s + selected.initial_validation_s).median())


def fit_catch_up(rows: pd.DataFrame, block_tokens: int,
                 block_bytes: int) -> tuple[float, float]:
    selected = rows[
        (rows.method == "kv_transfer") & rows.repeat.isin(TRAIN_REPEATS)
    ].copy()
    if selected.empty or (selected.catch_up_prompt_tokens <= 0).any():
        raise ValueError("catch-up fit needs successful measured catch-up rows")
    initial_blocks = selected.measured_prompt_tokens // block_tokens
    body_blocks = selected.catch_up_prompt_tokens // block_tokens - initial_blocks
    network_s = body_blocks * block_bytes / (selected.bandwidth_mbps * 1e6 / 8)
    tail = selected.catch_up_prompt_tokens % block_tokens
    tail_s = selected.catch_up_time_to_first_response_s - network_s
    valid = (tail > 0) & (tail_s > 0)
    if not valid.any():
        raise ValueError("catch-up fit needs positive partial-tail work")
    tail_tps = float(np.median(tail[valid] / tail_s[valid]))
    fixed = float((selected.catch_up_response_s + selected.catch_up_validation_s).median())
    return tail_tps, fixed


def fit_sleep(root: Path) -> tuple[tuple[float, float, float],
                                   tuple[float, float, float]]:
    summary = pd.read_csv(root / "power_states/summary.csv")
    source = summary[summary.device == "source"].pivot(
        index="cycle", columns="state", values="mean_power_w"
    )
    if set(source) != {"awake", "sleep"} or len(source) < 2:
        raise ValueError("sleep fit needs paired source awake/sleep cycles")
    deltas = sorted(source.sleep - source.awake)
    result = json.loads((root / "power_states/result.json").read_text())
    durations = sorted(
        (row["sleep_transition_ns"][1] - row["sleep_transition_ns"][0]) / 1e9
        for row in result["cycles"]
    )
    return (
        (float(deltas[0]), float(np.median(deltas)), float(deltas[-1])),
        (durations[0], float(np.median(durations)), durations[-1]),
    )


def validate_run(root: Path) -> None:
    metadata = json.loads((root / "run_metadata.json").read_text())
    if metadata.get("dirty") or metadata.get("schema") != "queue-haul-migration-run-v2":
        raise ValueError(f"invalid run metadata: {root}")
    scenarios = pd.read_csv(root / "scenarios.csv")
    if set(scenarios.status) != {"complete"}:
        raise ValueError(f"incomplete scenarios: {root}")
    migrations = pd.read_csv(root / "migrations.csv")
    if "success" in migrations and not migrations.success.all():
        raise ValueError(f"failed migrations: {root}")


def merge_points(measured: list[list[float]], prior: list[list[float]]) -> list[list[float]]:
    points = {float(x): float(y) for x, y in prior}
    points.update((float(x), float(y)) for x, y in measured)
    return [[x, points[x]] for x in sorted(points)]


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
               destination_rate: float, replay_completion: float = 0,
               kv_completion: float = 0) -> pd.DataFrame:
    out = []
    for method in ("replay", "kv_transfer"):
        held = serial(rows, method)
        held = held[held.repeat == VALIDATION_REPEAT]
        if method == "replay":
            predicted = held.measured_processed_tokens / np.interp(
                held.measured_prompt_tokens, *np.asarray(replay_curve).T
            ) + replay_completion
        else:
            predicted = np.maximum(
                held.measured_kv_bytes / (held.bandwidth_mbps * 1e6 / 8),
                held.measured_kv_bytes / destination_rate,
            ) + kv_completion
        for (_, row), value in zip(held.iterrows(), predicted):
            measured = row.initial_request_s + row.initial_validation_s
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


def fit_profile(serial_root: Path, catch_up_root: Path, profile_path: Path,
                out_profile: Path, parallel_root: Path | None = None,
                evaluation_path: Path | None = None) -> None:
    validate_run(serial_root)
    validate_run(catch_up_root)
    migrations = pd.read_csv(serial_root / "migrations.csv")
    scenarios = pd.read_csv(serial_root / "scenarios.csv")
    catch_up = pd.read_csv(catch_up_root / "migrations.csv")
    replay_curve, replay_error = fit_replay(migrations)
    destination_rate, kv_error = fit_kv(migrations)
    replay_completion = completion_s(migrations, "replay")
    kv_completion = completion_s(migrations, "kv_transfer")
    replay_power = total_action_power(scenarios, "replay", False)
    kv_power = total_action_power(scenarios, "kv_transfer", True)
    switches = serial(migrations, "kv_transfer")
    switches = switches[switches.repeat.isin(TRAIN_REPEATS)].route_switch_s
    raw = json.loads(profile_path.read_text())
    old_kv = raw["cases"]["central"]["kv_transfer"]
    tail_tps, catch_up_fixed = fit_catch_up(
        catch_up, old_kv["block_tokens"], old_kv["block_bytes"]
    )
    sleep_deltas, sleep_times = fit_sleep(serial_root)
    combined_replay = merge_points(
        replay_curve, raw["cases"]["central"]["replay_tps"]["1"]
    )
    replay_uncertainty = max(replay_error, raw["sources"]["replay"]["relative_error"])
    raw["schema"] = PROFILE_SCHEMA
    sha = json.loads((serial_root / "run_metadata.json").read_text())["git_sha"][:8]
    raw["profile_id"] = f"gpt-oss-20b-a100-tp1-v3-{sha}"
    raw["sources"]["replay"] = {
        "kind": "measured", "reference": (
            "outputs/coding-run/migrations.csv and "
            f"{serial_root}/migrations.csv repeats 0-1; "
            "outputs/power_drain_live_20260714/live_profile_*.json"
        ),
        "valid_range": [combined_replay[0][0], combined_replay[-1][0]],
        "relative_error": replay_uncertainty,
    }
    kv_rows = serial(migrations, "kv_transfer")
    kv_train = kv_rows[kv_rows.repeat.isin(TRAIN_REPEATS)]
    raw["sources"]["kv_transfer"] = {
        "kind": "measured", "reference": (
            "outputs/coding-run/migrations.csv and "
            f"{serial_root}/migrations.csv repeats 0-1; "
            f"{catch_up_root}/migrations.csv incremental blocks and tail"
        ),
        "valid_range": [min(raw["sources"]["kv_transfer"]["valid_range"][0],
                            float(kv_train.measured_kv_bytes.min())),
                        float(kv_train.measured_kv_bytes.max())],
        "relative_error": kv_error,
    }
    raw["sources"]["transitions"]["reference"] = (
        f"{serial_root}/migrations.csv, scenarios.csv, and power_states for "
        f"route switch, action power, and GPU sleep; {catch_up_root}/migrations.csv "
        "for catch-up; shutdown remains sensitivity-only"
    )
    if max(replay_error, kv_error) >= 1:
        raise ValueError("relative timing error must be below one")
    scales = {
        "central": (1, .5, 1, 1, 1),
        "faster": (1 / (1 - replay_uncertainty), .25, .9, 1.2, .9),
        "slower": (1 / (1 + replay_uncertainty), .75, 1.1, 1 / 1.2, 1.1),
    }
    sleep_index = {"faster": 0, "central": 1, "slower": 2}
    for name, (replay_scale, switch_quantile, completion_scale,
               tail_scale, fixed_scale) in scales.items():
        case = raw["cases"][name]
        case["replay_tps"] = {
            "1": [[x, y * replay_scale] for x, y in combined_replay]
        }
        case["replay_completion_s"] = replay_completion * completion_scale
        old = case["kv_transfer"]
        kv_scale = 1 if name == "central" else (
            1 / (1 - kv_error) if name == "faster" else 1 / (1 + kv_error)
        )
        case["kv_transfer"] = {
            "block_tokens": old["block_tokens"], "block_bytes": old["block_bytes"],
            "setup_s": 0, "destination_bytes_per_s": destination_rate * kv_scale,
            "initial_completion_s": kv_completion * completion_scale,
            "catch_up_fixed_s": catch_up_fixed * fixed_scale,
            "tail_replay_tps": tail_tps * tail_scale,
        }
        case["switch_s"] = float(switches.quantile(switch_quantile))
        index = sleep_index[name]
        case["sleep_power_delta_w"] = sleep_deltas[index]
        case["sleep_s"] = sleep_times[index]
        case["shutdown_s"] = None
        case["action_power_w"] = {
            "replay": replay_power, "kv_transfer": kv_power,
            "replay_on_request": {"1": replay_power["1"]},
            "catch_up": {"1": kv_power["1"]},
            "sleep": {"1": [0, 0]}, "off": {"1": [0, 0]},
        }
    concurrency = 1
    if parallel_root:
        validate_run(parallel_root)
        gate = pd.read_csv(parallel_root / "parallel_gate.csv")
        if gate.empty or not gate.passed.all():
            raise ValueError("parallel evidence did not pass")
        concurrency = int(gate.concurrency.max())
    raw["max_source_streams"] = concurrency
    raw["max_destination_kv_streams"] = concurrency
    raw["max_destination_replays"] = 1
    out_profile.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_profile.with_suffix(out_profile.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2) + "\n")
    ModelProfile.load(temporary)
    temporary.replace(out_profile)
    held = evaluation(
        migrations, replay_curve, destination_rate,
        replay_completion, kv_completion,
    )
    evaluation_path = evaluation_path or out_profile.with_name(
        out_profile.stem + "_evaluation.csv"
    )
    held.to_csv(evaluation_path, index=False)
    plot_evaluation(held, evaluation_path.with_suffix(""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-root", type=Path, required=True)
    parser.add_argument("--catch-up-root", type=Path, required=True)
    parser.add_argument("--parallel-root", type=Path)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--out-profile", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    fit_profile(
        args.serial_root, args.catch_up_root, args.base_profile,
        args.out_profile, args.parallel_root, args.evaluation,
    )


if __name__ == "__main__":
    main()
