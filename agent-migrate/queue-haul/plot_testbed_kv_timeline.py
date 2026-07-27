"""Plot a measured two-A100 KV handoff timeline from tidy event tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_SCENARIO = "m-0d41d4a3ced809ad"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def extract(root: Path, scenario_id: str) -> tuple[list[dict], list[dict]]:
    run = root / "scenarios" / scenario_id
    scenario = json.loads((run / "scenario.json").read_text())
    if scenario["method"] != "kv_transfer" or scenario["concurrency"] != 1:
        raise ValueError("timeline requires a KV-transfer scenario with concurrency 1")
    migrations = [
        row for row in _read_csv(root / "migrations.csv")
        if row["scenario_id"] == scenario_id
    ]
    if len(migrations) != 1 or migrations[0]["success"] != "True":
        raise ValueError("timeline requires one successful measured migration")
    result = json.loads((run / "result.json").read_text())
    continuations = {row["session_id"]: row for row in result["continuations"]}
    if continuations.keys() != {row["session_id"] for row in migrations}:
        raise ValueError("every migration must have one matching continuation")
    activities = sorted(result["activities"], key=lambda row: row["stage_index"])
    if not activities or any(
        row["session_id"] not in continuations or "first_byte_ns" not in row
        for row in activities
    ):
        raise ValueError("timeline requires measured inference phase boundaries")

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

    segments = []
    for index, activity in enumerate(activities):
        next_start = activities[index + 1]["start_ns"] \
            if index + 1 < len(activities) else None
        for phase, start, end in (
            ("Prefill", activity["start_ns"], activity["first_byte_ns"]),
            ("Decode", activity["first_byte_ns"], activity["end_ns"]),
            ("Tool Call", activity["end_ns"], next_start),
        ):
            if end and int(end) > int(start):
                segments.append({
                    "scenario_id": scenario_id,
                    "session_id": activity["session_id"],
                    "stage": activity["stage_index"],
                    "location": "source",
                    "phase": phase,
                    "start_s": seconds(start),
                    "finish_s": seconds(end),
                    "evidence_status": "measured" if phase != "Tool Call"
                        else "observed_application_gap",
                    "provenance": str(run / "result.json"),
                })
    for session_id, continuation in continuations.items():
        for phase, start, end in (
            ("Prefill", continuation["start_ns"], continuation["first_byte_ns"]),
            ("Decode", continuation["first_byte_ns"], continuation["end_ns"]),
        ):
            segments.append({
                "scenario_id": scenario_id,
                "session_id": session_id,
                "stage": "continuation",
                "location": "destination",
                "phase": phase,
                "start_s": seconds(start),
                "finish_s": seconds(end),
                "evidence_status": "measured",
                "provenance": str(run / "result.json"),
            })
    return timeline, segments


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(timeline_path: Path, inference_path: Path, out: Path) -> None:
    timeline, inference = _read_csv(timeline_path), _read_csv(inference_path)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "bulk": "#4298B5",
        "drain": "#DAD7CB",
        "catch": "#279989",
        "switch": "#8C1515",
        "token": "#175E54",
        "prefill": "#734675",
        "decode": "#E98300",
        "tool": "#7F7776",
        "text": "#2E2D29",
        "grid": "#DAD7CB",
    }
    fig, gantt = plt.subplots(figsize=(11, 3.8))
    phases = (
        ("bulk_start_s", "bulk_finish_s", "KV Initial Write", "bulk"),
        ("catch_up_start_s", "catch_up_finish_s", "Append final KV", "catch"),
    )
    gantt.axvspan(
        float(timeline[0]["quiesce_s"]),
        float(timeline[0]["catch_up_start_s"]),
        color=colors["drain"], alpha=.48,
        label="Background KV Transfer", zorder=0,
    )
    positions = [1.4 * index for index in range(len(timeline))]
    inference_labels = set()
    for y, row in zip(positions, timeline):
        source_y, migration_y, destination_y = y - .55, y, y + .55
        commit = float(row["commit_s"])
        for start_name, end_name, label, color in phases:
            start, end = float(row[start_name]), float(row[end_name])
            gantt.barh(
                migration_y, end - start, left=start, height=.34,
                color=colors[color],
                label=label if y == positions[0] else None, zorder=2,
            )
        session_segments = [
            segment for segment in inference
            if segment["session_id"] == row["session_id"]
        ]
        for segment in session_segments:
            phase = segment["phase"]
            start, end = float(segment["start_s"]), float(segment["finish_s"])
            color = {
                "Prefill": "prefill", "Decode": "decode", "Tool Call": "tool",
            }[phase]
            inference_y = source_y \
                if segment["location"] == "source" else destination_y
            gantt.barh(
                inference_y, end - start, left=start, height=.25,
                color=colors[color], hatch="//" if color == "tool" else None,
                edgecolor=colors["text"] if color == "tool" else "none",
                linewidth=.5,
                label=phase if phase not in inference_labels else None,
                zorder=3,
            )
            inference_labels.add(phase)
        gantt.scatter(
            commit, migration_y, marker="D", s=45, color=colors["switch"],
            label="Route Switch" if y == positions[0] else None, zorder=4,
        )
        first_token = float(row["first_token_s"])
        gantt.scatter(
            first_token, destination_y, marker="*", s=95,
            color=colors["token"],
            label="First Token at Destination" if y == positions[0] else None,
            zorder=4,
        )
    gantt.set_yticks(
        [positions[0] - .55, positions[0], positions[0] + .55],
        ["Inference at Source", "Migration", "Inference at Destination"],
    )
    gantt.invert_yaxis()
    legend = (
        Patch(facecolor=colors["prefill"], label="Prefill"),
        Patch(facecolor=colors["decode"], label="Decode"),
        Patch(
            facecolor=colors["tool"], edgecolor=colors["text"],
            hatch="//", label="Tool Call",
        ),
        Patch(facecolor=colors["bulk"], label="KV Initial Write"),
        Patch(facecolor=colors["drain"], alpha=.6, label="Background KV Transfer"),
        Patch(facecolor=colors["catch"], label="Append final KV"),
        Line2D(
            (), (), marker="D", linestyle="none", color=colors["switch"],
            markersize=8, label="Route Switch",
        ),
        Line2D(
            (), (), marker="*", linestyle="none", color=colors["token"],
            markersize=12, label="First Token at Destination",
        ),
    )
    gantt.legend(
        handles=legend, frameon=False, ncol=4, loc="upper center",
        bbox_to_anchor=(.5, 1.36),
    )
    gantt.grid(axis="x", color=colors["grid"], linewidth=.8, alpha=.7)
    gantt.grid(axis="y", visible=False)
    gantt.spines[["top", "right"]].set_visible(False)
    gantt.spines[["left", "bottom"]].set_color(colors["text"])
    gantt.tick_params(colors=colors["text"])
    gantt.set_facecolor("#FFFFFF")
    fig.set_facecolor("#FFFFFF")
    gantt.set_xlabel("Time since first KV write (s)")
    end = max(float(row["first_token_s"]) for row in timeline) + 3
    gantt.set_xlim(0, end)
    fig.subplots_adjust(left=.17, right=.98, top=.7, bottom=.17)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write(root: Path, scenario_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    timeline, inference = extract(root, scenario_id)
    timeline_path = out / "kv_write_concurrency_1_timeline.csv"
    inference_path = out / "kv_write_concurrency_1_inference.csv"
    _write(timeline_path, timeline)
    _write(inference_path, inference)
    plot(
        timeline_path, inference_path,
        out / "kv_write_concurrency_1_timeline",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/bounded-hardware-campaign-run"),
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--out", type=Path, default=Path("outputs/mechanism-validation"))
    args = parser.parse_args(argv)
    write(args.root, args.scenario, args.out)


if __name__ == "__main__":
    main()
