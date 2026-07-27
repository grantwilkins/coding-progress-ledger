"""Plot a measured two-A100 KV handoff timeline from tidy event tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SCENARIO = "m-70aec4041b98c310"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def extract(root: Path, scenario_id: str) -> tuple[list[dict], list[dict]]:
    run = root / "scenarios" / scenario_id
    scenario = json.loads((run / "scenario.json").read_text())
    if scenario["method"] != "kv_transfer" or scenario["concurrency"] != 4:
        raise ValueError("timeline requires a KV-transfer scenario with concurrency 4")
    migrations = [
        row for row in _read_csv(root / "migrations.csv")
        if row["scenario_id"] == scenario_id
    ]
    if len(migrations) != 4 or any(row["success"] != "True" for row in migrations):
        raise ValueError("timeline requires four successful measured migrations")
    result = json.loads((run / "result.json").read_text())
    continuations = {row["session_id"]: row for row in result["continuations"]}
    if continuations.keys() != {row["session_id"] for row in migrations}:
        raise ValueError("every migration must have one matching continuation")

    base = min(int(row["initial_start_ns"]) for row in migrations)

    def seconds(value: str | int) -> float:
        return (int(value) - base) / 1e9

    timeline = []
    for index, row in enumerate(sorted(migrations, key=lambda item: int(item["order"])), 1):
        continuation = continuations[row["session_id"]]
        bulk_start, bulk_finish = seconds(row["initial_start_ns"]), seconds(row["initial_end_ns"])
        quiesce, catch_start = seconds(row["pause_start_ns"]), seconds(row["catch_up_start_ns"])
        catch_finish = seconds(row["catch_up_end_ns"])
        switch_start, commit = seconds(row["switch_start_ns"]), seconds(row["switch_end_ns"])
        continuation_start = seconds(continuation["start_ns"])
        first_token = seconds(continuation["first_byte_ns"])
        timeline.append({
            "scenario_id": scenario_id,
            "session": f"S{index}",
            "session_id": row["session_id"],
            "method": row["method"],
            "kv_write_concurrency": int(row["concurrency"]),
            "bandwidth_gbps": float(row["bandwidth_mbps"]) / 1000,
            "activity": row["activity"],
            "bulk_start_s": bulk_start,
            "bulk_finish_s": bulk_finish,
            "bulk_s": bulk_finish - bulk_start,
            "quiesce_s": quiesce,
            "request_boundary_wait_s": catch_start - quiesce,
            "catch_up_start_s": catch_start,
            "catch_up_finish_s": catch_finish,
            "catch_up_s": catch_finish - catch_start,
            "switch_start_s": switch_start,
            "commit_s": commit,
            "route_switch_s": commit - switch_start,
            "continuation_start_s": continuation_start,
            "first_token_s": first_token,
            "commit_to_first_token_s": first_token - commit,
            "continuation_ttft_s": first_token - continuation_start,
            "continuation_finish_s": seconds(continuation["end_ns"]),
            "evidence_status": "measured",
            "provenance": f"{run}/result.json|{root}/migrations.csv",
        })

    power = _read_csv(run / "power.csv")
    gpus = sorted({int(row["gpu"]) for row in power})
    if len(gpus) != 2:
        raise ValueError("timeline requires power measurements from exactly two GPUs")
    source = gpus[0]
    power_rows = [{
        "scenario_id": scenario_id,
        "time_s": seconds(row["monotonic_ns"]),
        "source_power_w": float(row["power_w"]),
        "source_gpu": source,
        "evidence_status": "measured",
        "provenance": str(run / "power.csv"),
    } for row in power if int(row["gpu"]) == source and row["valid"] == "1"]
    return timeline, power_rows


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(timeline_path: Path, power_path: Path, out: Path) -> None:
    timeline, power = _read_csv(timeline_path), _read_csv(power_path)
    colors = {
        "bulk": "#4C78A8",
        "quiesce": "#ECA82C",
        "catch": "#72B7B2",
        "switch": "#E45756",
        "token": "#2A9D5B",
        "power": "#5F4B8B",
    }
    fig, (gantt, watts) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": .12},
    )
    phases = (
        ("bulk_start_s", "bulk_finish_s", "KV write + ingest", "bulk"),
        ("quiesce_s", "catch_up_start_s", "Quiesce", "quiesce"),
        ("catch_up_start_s", "catch_up_finish_s", "Catch-up", "catch"),
        ("switch_start_s", "commit_s", "Route switch", "switch"),
    )
    for y, row in enumerate(timeline):
        commit = float(row["commit_s"])
        gantt.barh(y, commit, height=.66, color="#ECEFF1", zorder=0)
        for start_name, end_name, label, color in phases:
            start, end = float(row[start_name]), float(row[end_name])
            gantt.barh(
                y, end - start, left=start, height=.52, color=colors[color],
                label=label if y == 0 else None, zorder=2,
            )
        gantt.scatter(
            commit, y, marker="D", s=45, color=colors["switch"],
            label="Commit" if y == 0 else None, zorder=4,
        )
        gantt.scatter(
            float(row["quiesce_s"]), y, marker="|", s=140,
            color=colors["quiesce"], label="Quiesce begins" if y == 0 else None,
            zorder=4,
        )
        first_token = float(row["first_token_s"])
        gantt.scatter(
            first_token, y, marker="*", s=95,
            color=colors["token"], label="First post-switch token" if y == 0 else None,
            zorder=4,
        )
        gantt.text(commit + .35, y, f"{commit:.1f}s", va="center", fontsize=8)
        gantt.text(first_token + .35, y, f"{first_token:.1f}s", va="center", fontsize=8)
    gantt.set_yticks(range(len(timeline)), [row["session"] for row in timeline])
    gantt.invert_yaxis()
    gantt.set_ylabel("Session")
    fig.suptitle(
        "Measured two-A100 KV handoff: four concurrent writes at "
        f"{float(timeline[0]['bandwidth_gbps']):g} Gbps", y=.98,
    )
    gantt.legend(
        frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.16),
    )
    gantt.grid(axis="x", alpha=.2)

    xs = [float(row["time_s"]) for row in power]
    ys = [float(row["source_power_w"]) for row in power]
    watts.plot(xs, ys, color=colors["power"], linewidth=1.2)
    for row in timeline:
        watts.axvline(float(row["commit_s"]), color=colors["switch"], alpha=.25)
    watts.set(xlabel="Time since first KV write (s)", ylabel="Source\npower (W)")
    watts.grid(axis="x", alpha=.2)
    end = max(float(row["first_token_s"]) for row in timeline) + 3
    gantt.set_xlim(0, end)
    fig.subplots_adjust(left=.1, right=.98, top=.79, bottom=.12, hspace=.12)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write(root: Path, scenario_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    timeline, power = extract(root, scenario_id)
    timeline_path = out / "kv_write_concurrency_4_timeline.csv"
    power_path = out / "kv_write_concurrency_4_power.csv"
    _write(timeline_path, timeline)
    _write(power_path, power)
    plot(timeline_path, power_path, out / "kv_write_concurrency_4_timeline")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/coding-run"))
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--out", type=Path, default=Path("outputs/mechanism-validation"))
    args = parser.parse_args(argv)
    write(args.root, args.scenario, args.out)


if __name__ == "__main__":
    main()
