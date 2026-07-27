"""Plot measured two-A100 KV-transfer and replay handoff timelines."""

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

DEFAULT_SCENARIOS = {
    "kv_transfer": "m-0d41d4a3ced809ad",
    "replay": "m-b35dec3b9a228389",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def extract(root: Path, scenario_id: str) -> tuple[list[dict], list[dict]]:
    run = root / "scenarios" / scenario_id
    scenario = json.loads((run / "scenario.json").read_text())
    if scenario["method"] not in DEFAULT_SCENARIOS or scenario["concurrency"] != 1:
        raise ValueError("timeline requires replay or KV transfer with concurrency 1")
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

    replay_requests = {}
    if scenario["method"] == "replay":
        moves = {
            row["move"]["session_id"]: row for row in result["migrations"]
        }
        connections = _read_csv(run / "proxy_connections.csv")
        proxy_bytes = _read_csv(run / "proxy_bytes.csv")
        for session_id, move in moves.items():
            phases = {}
            for phase, request in (
                ("initial", move["initial"]), ("final", move["catch_up"]),
            ):
                connection = min(
                    (row for row in connections if row["route"] == "api"),
                    key=lambda row: abs(int(row["start_ns"]) - int(request["start_ns"])),
                )
                bins = [
                    row for row in proxy_bytes
                    if row["connection_id"] == connection["connection_id"]
                    and row["direction"] == "client_to_target"
                    and int(row["bytes"]) > 0
                ]
                if not bins:
                    raise ValueError("replay timeline requires context-transfer bins")
                send_finish = max(
                    int(row["monotonic_ns"]) + int(row["interval_ns"])
                    for row in bins
                )
                if send_finish >= int(request["first_byte_ns"]):
                    raise ValueError("context-transfer bound must precede replay output")
                phases[phase] = request | {"send_finish_ns": send_finish}
            replay_requests[session_id] = phases

    timeline = []
    for index, row in enumerate(sorted(migrations, key=lambda item: int(item["order"])), 1):
        continuation = continuations[row["session_id"]]
        bulk_start, bulk_finish = seconds(row["initial_start_ns"]), seconds(row["initial_end_ns"])
        quiesce, catch_start = seconds(row["pause_start_ns"]), seconds(row["catch_up_start_ns"])
        catch_finish = seconds(row["catch_up_end_ns"])
        switch_start, commit = seconds(row["switch_start_ns"]), seconds(row["switch_end_ns"])
        continuation_start = seconds(continuation["start_ns"])
        first_token = seconds(continuation["first_byte_ns"])
        replay = replay_requests.get(row["session_id"])
        timeline.append({
            "scenario_id": scenario_id,
            "session": f"S{index}",
            "session_id": row["session_id"],
            "method": row["method"],
            "migration_concurrency": int(row["concurrency"]),
            "bandwidth_gbps": float(row["bandwidth_mbps"]) / 1000,
            "activity": row["activity"],
            "bulk_start_s": bulk_start,
            "bulk_finish_s": bulk_finish,
            "bulk_s": bulk_finish - bulk_start,
            "bulk_send_finish_s": seconds(replay["initial"]["send_finish_ns"])
                if replay else None,
            "quiesce_s": quiesce,
            "request_boundary_wait_s": catch_start - quiesce,
            "catch_up_start_s": catch_start,
            "catch_up_finish_s": catch_finish,
            "catch_up_s": catch_finish - catch_start,
            "catch_up_send_finish_s": seconds(replay["final"]["send_finish_ns"])
                if replay else None,
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
    for session_id in continuations:
        first_start = min(
            int(row["start_ns"]) for row in activities
            if row["session_id"] == session_id
        )
        if first_start > base:
            segments.append({
                "scenario_id": scenario_id,
                "session_id": session_id,
                "stage": "pre_activity",
                "location": "source",
                "phase": "Tool Call",
                "start_s": 0,
                "finish_s": seconds(first_start),
                "evidence_status": "observed_application_gap",
                "provenance": str(run / "result.json"),
            })
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
    for session_id, phases in replay_requests.items():
        for stage, request in phases.items():
            for phase, start, end in (
                ("Prefill", request["send_finish_ns"], request["first_byte_ns"]),
                ("Decode", request["first_byte_ns"], request["end_ns"]),
            ):
                segments.append({
                    "scenario_id": scenario_id,
                    "session_id": session_id,
                    "stage": f"{stage}_replay",
                    "location": "destination",
                    "phase": phase,
                    "start_s": seconds(start),
                    "finish_s": seconds(end),
                    "evidence_status": "measured_250ms_send_bound"
                        if phase == "Prefill" else "measured",
                    "provenance": f"{run}/result.json|{run}/proxy_bytes.csv",
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
    method = timeline[0]["method"]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 11,
    })
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
    fig, gantt = plt.subplots(figsize=(10, 3.1))
    phase_labels = (
        "Bulk KV Transfer + Ingest", "Final KV Delta Transfer + Ingest",
    ) \
        if method == "kv_transfer" \
        else ("Initial Context Update", "Final Context Update")
    phases = (
        ("bulk_start_s", "bulk_finish_s", phase_labels[0], "bulk"),
        ("catch_up_start_s", "catch_up_finish_s", phase_labels[1], "catch"),
    )
    if method == "replay":
        phases = (
            ("bulk_start_s", "bulk_send_finish_s", "Context Transfer", "bulk"),
            ("catch_up_start_s", "catch_up_send_finish_s", "Context Transfer", "bulk"),
        )
    gantt.axvspan(
        float(timeline[0]["quiesce_s"]),
        float(timeline[0]["catch_up_start_s"]),
        color=colors["drain"], alpha=.48,
        label="Drain Active Request" if method == "kv_transfer"
            else "Pause (drain active request)",
        zorder=0,
    )
    positions = [1.4 * index for index in range(len(timeline))]
    inference_labels = set()
    for y, row in zip(positions, timeline):
        source_y, migration_y, destination_y = y - .32, y, y + .32
        commit = float(row["commit_s"])
        prior = (
            (-2.35, .65, "prefill"), (-1.7, .25, "decode"),
            (-1.45, .3, "tool"), (-1.15, .85, "prefill"),
            (-.3, .3, "decode"),
        )
        for left, width, phase in prior:
            gantt.barh(
                source_y, width, left=left, height=.32,
                color=colors[phase], hatch="//" if phase == "tool" else None,
                edgecolor=colors["text"] if phase == "tool" else "none",
                linewidth=0, zorder=3,
            )
        for start_name, end_name, label, color in phases:
            start, end = float(row[start_name]), float(row[end_name])
            gantt.barh(
                migration_y, end - start, left=start, height=.32,
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
                inference_y, end - start, left=start, height=.32,
                color=colors[color], hatch="//" if color == "tool" else None,
                edgecolor=colors["text"] if color == "tool" else "none",
                linewidth=0,
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
        [positions[0] - .32, positions[0], positions[0] + .32],
        ["Inference at Source", "Migration", "Inference at Destination"],
    )
    gantt.invert_yaxis()
    migration_handles = (
        Patch(facecolor=colors["bulk"], label=phase_labels[0]),
        Patch(
            facecolor=colors["drain"], alpha=.6,
            label="Drain Active Request",
        ),
        Patch(facecolor=colors["catch"], label=phase_labels[1]),
    ) if method == "kv_transfer" else (
        Patch(facecolor=colors["bulk"], label="Context Transfer"),
        Patch(
            facecolor=colors["drain"], alpha=.6,
            label="Pause (drain active request)",
        ),
    )
    legend = (
        Patch(facecolor=colors["prefill"], label="Prefill"),
        Patch(facecolor=colors["decode"], label="Decode"),
        Patch(
            facecolor=colors["tool"], edgecolor=colors["text"],
            hatch="//", linewidth=0, label="Tool Call",
        ),
        *migration_handles,
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
        bbox_to_anchor=(.5, 1.43),
    )
    gantt.grid(axis="x", color=colors["grid"], linewidth=.8, alpha=.7)
    gantt.grid(axis="y", visible=False)
    gantt.spines[["top", "right"]].set_visible(False)
    gantt.spines[["left", "bottom"]].set_color(colors["text"])
    gantt.tick_params(colors=colors["text"])
    gantt.set_facecolor("#FFFFFF")
    fig.set_facecolor("#FFFFFF")
    gantt.set_xlabel(
        f"Time since {'migration' if method == 'kv_transfer' else 'replay'} start (s)"
    )
    end = max(float(row["first_token_s"]) for row in timeline) + 3
    gantt.set_xlim(-2.5, end)
    gantt.set_xticks(range(0, int(end) + 1, 5))
    fig.tight_layout(pad=.25)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def write(root: Path, scenario_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    timeline, inference = extract(root, scenario_id)
    stem = "kv_write" if timeline[0]["method"] == "kv_transfer" else "replay"
    timeline_path = out / f"{stem}_concurrency_1_timeline.csv"
    inference_path = out / f"{stem}_concurrency_1_inference.csv"
    _write(timeline_path, timeline)
    _write(inference_path, inference)
    plot(
        timeline_path, inference_path,
        out / f"{stem}_concurrency_1_timeline",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/bounded-hardware-campaign-run"),
    )
    parser.add_argument("--method", choices=DEFAULT_SCENARIOS, default="kv_transfer")
    parser.add_argument("--scenario")
    parser.add_argument("--out", type=Path, default=Path("outputs/mechanism-validation"))
    args = parser.parse_args(argv)
    write(args.root, args.scenario or DEFAULT_SCENARIOS[args.method], args.out)


if __name__ == "__main__":
    main()
