"""Cache and plot canonical fixed-contract residual headroom."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from destination_bench import (
    CLASSES,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    KV_BLOCK_TOKENS,
    Pressure,
    WORKLOADS,
    architecture,
    log_bytes_per_token,
    pack_source,
    sample_sessions,
    scenario,
    trace_shapes,
)
from planner import source_power
from profiles import ModelProfile
from requirement_frontier import requirement_frontier


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "outputs/fixed-contract-residuals"
SESSIONS = 100_000
TARGETS = (.10, .25, .50, .75, .90, 1.0)
PRESSURE = Pressure(service=.8, bandwidth_gbps=5, migration_s=115)
FLEX = .10
DEBT = .10
SEED = 0
DATA_VERSION = 2
COLORS = (
    "#8C1515", "#007C92", "#53284F", "#E98300",
    "#175E54", "#B83A4B", "#4298B5", "#53565A",
)


def result_row(requested, achieved, resource, used, capacity, **fields):
    if capacity <= 0:
        raise ValueError("resource capacity must be positive")
    return {
        "requested_shed_w": requested,
        "achieved_shed_w": achieved,
        "unmet_shed_w": max(0, requested - achieved),
        "resource": resource,
        "used": used,
        "capacity": capacity,
        "normalized_slack": (capacity - used) / capacity,
        **fields,
    }


def _mixed_sessions(manifest, profile):
    counts = [SESSIONS // len(CLASSES)] * len(CLASSES)
    counts[0] += SESSIONS - sum(counts)
    sessions = []
    for job_class, count in zip(CLASSES, counts):
        sampled = sample_sessions(
            trace_shapes(manifest, job_class), count, SEED,
            log_bytes_per_token(WORKLOADS[job_class]),
        )
        sessions.extend(
            replace(session, session_id=f"{job_class}:{session.session_id}")
            for session in sampled
        )
    return pack_source(tuple(sessions), profile)


def generate(model_path, manifest_path):
    profile = ModelProfile.load(model_path)
    manifest = json.loads(manifest_path.read_text())
    packed, replicas = _mixed_sessions(manifest, profile)
    scenario_ = scenario(profile, packed, replicas, PRESSURE)
    architecture_ = architecture(profile, scenario_.sessions, replicas, PRESSURE)
    pool = replace(
        architecture_.pools[0],
        event_flex_fraction=FLEX,
        service_debt_fraction=DEBT,
    )
    architecture_ = replace(architecture_, pools=(pool,))
    q = architecture_.types[0]
    initial = source_power(scenario_, profile)
    minimum = source_power(
        scenario_, profile, (session.session_id for session in scenario_.sessions),
    )
    maximum = initial - minimum
    horizon = scenario_.deadline_s - profile.power_window_s
    bandwidth = PRESSURE.bandwidth_gbps * 125_000_000
    normal = np.asarray(q.normals[0])
    baseline = sum(
        (np.asarray(replica.baseline_work) for replica in pool.replicas),
        start=np.zeros(2),
    )
    baseline_service = float(normal @ baseline)
    stable = replicas * q.bounds["stable"][0]
    event = min(
        replicas * (
            q.bounds["normal"][0] + FLEX * q.bounds["stable"][0]
        ),
        stable,
    )
    baseline_blocks = sum(
        replica.baseline_kv_tokens // KV_BLOCK_TOKENS for replica in pool.replicas
    )
    capacities = {
        "Source stream time": horizon * profile.max_source_streams,
        "WAN transfer bytes": bandwidth * horizon,
        "Replay GPU time": replicas * horizon * profile.max_destination_replays,
        "KV-ingest GPU time": replicas * horizon * profile.max_destination_kv_streams,
        "Ongoing serving load": event - baseline_service,
        "Queued serving work": horizon * (
            stable - baseline_service + DEBT * stable
        ),
        "KV-cache blocks": replicas * (q.kv_capacity_tokens // q.kv_block_tokens)
        - baseline_blocks,
        "Migration makespan": horizon,
    }
    rows = []
    for fraction in TARGETS:
        requested = fraction * maximum
        requirement = requirement_frontier(
            scenario_, profile, q, requested, bandwidth, 0,
            profile.max_source_streams, solver_mode="greedy",
        )
        ongoing = float(normal @ requirement.destination_service_work)
        transition = float(normal @ requirement.destination_transition_work)
        kv_bytes = sum(
            action.route_bytes for action in requirement.actions
            if action.method == "kv_transfer"
        )
        uses = {
            "Source stream time": max(
                (seconds for _, seconds in requirement.source_stream_occupancy_s),
                default=0,
            ),
            "WAN transfer bytes": requirement.wan_bytes,
            "Replay GPU time": transition,
            "KV-ingest GPU time":
                kv_bytes / profile.case().kv_transfer.destination_bytes_per_s,
            "Ongoing serving load": ongoing,
            "Queued serving work": horizon * ongoing + transition,
            "KV-cache blocks": requirement.destination_kv_blocks,
            "Migration makespan": requirement.makespan_lower_bound_s,
        }
        mix = dict(requirement.method_mix)
        common = {
            "target_fraction": fraction,
            "maximum_modeled_shed_w": maximum,
            "sessions": SESSIONS,
            "source_replicas": replicas,
            "deadline_s": scenario_.deadline_s,
            "replay_moves": mix["replay"],
            "kv_moves": mix["kv_transfer"],
            "target_met": requirement.target_met,
            "solver_status": requirement.solver_status,
            "evidence_status": "sensitivity",
        }
        rows.extend(
            result_row(
                requested, requirement.achieved_source_power_reduction_w,
                resource, uses[resource], capacity, **common,
            )
            for resource, capacity in capacities.items()
        )
    return rows


def _fingerprint(model_path, manifest_path):
    paths = (
        model_path, manifest_path, *WORKLOADS.values(), ROOT / "destination.py",
        ROOT / "destination_bench.py", ROOT / "planner.py",
        ROOT / "pool_planner.py", ROOT / "requirement_frontier.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return {
        "data_version": DATA_VERSION,
        "input_sha256": digest.hexdigest(),
        "sessions": SESSIONS,
        "workloads": list(CLASSES),
        "seed": SEED,
        "deadline_s": 120,
        "bandwidth_gbps": PRESSURE.bandwidth_gbps,
        "baseline_service_fraction": PRESSURE.service,
        "event_flex_fraction": FLEX,
        "service_debt_fraction": DEBT,
        "target_fractions": list(TARGETS),
    }


def load_or_generate(out, model_path, manifest_path, refresh=False):
    data, metadata = out / "fixed_contract_residuals.csv", out / "metadata.json"
    expected = _fingerprint(model_path, manifest_path)
    if data.exists() and metadata.exists() and not refresh:
        if json.loads(metadata.read_text()) != expected:
            raise ValueError("cached inputs changed; rerun with --refresh")
        with data.open(newline="") as stream:
            return list(csv.DictReader(stream))
    rows = generate(model_path, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    with data.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata.write_text(json.dumps(expected, indent=2) + "\n")
    return rows


def plot(rows, out):
    for row in rows:
        for field in (
            "requested_shed_w", "normalized_slack", "target_fraction",
            "unmet_shed_w",
        ):
            row[field] = float(row[field])
    plt.style.use("seaborn-v0_8-whitegrid")
    resources = tuple(dict.fromkeys(row["resource"] for row in rows))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.3), sharex=True, sharey=True)
    for axis, resource, color in zip(axes.flat, resources, COLORS):
        selected = sorted(
            (row for row in rows if row["resource"] == resource),
            key=lambda row: row["requested_shed_w"],
        )
        x = np.asarray([row["requested_shed_w"] for row in selected]) / 1000
        y = np.asarray([row["normalized_slack"] for row in selected])
        shown = np.maximum(y, -1)
        axis.axhspan(-1, 0, color="#8C1515", alpha=.07)
        axis.axhline(0, color="#8C1515", linewidth=1.2)
        axis.plot(x, shown, "o-", color=color, linewidth=2.2, markersize=5)
        clipped = np.flatnonzero(y < -1)
        axis.scatter(x[clipped], shown[clipped], marker="v", s=55, color=color,
                     zorder=3)
        failed = [i for i, row in enumerate(selected)
                  if row["unmet_shed_w"] > 1e-7]
        axis.scatter(x[failed], shown[failed], marker="x", s=55, color="#8C1515",
                     linewidth=2, zorder=3)
        axis.set_title(resource, fontsize=13, fontweight="bold")
        axis.set_ylim(-1.08, 1.08)
        axis.grid(color="#D7D2CB", linewidth=.8, alpha=.75)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Unused / advertised capacity")
    axes[1, 0].set_ylabel("Unused / advertised capacity")
    for axis in axes[1]:
        axis.set_xlabel("Requested shed (kW)")
    fig.suptitle(
        "Which physical resource limits source-power shed?",
        fontsize=20, fontweight="bold", y=.985,
    )
    fig.text(
        .5, .94,
        "Canonical 100K-session mix · 120 s deadline · one A100 pool · "
        "5 Gbps route · assumed sensitivity",
        ha="center", fontsize=11, color="#53565A",
    )
    fig.text(
        .99, .01,
        "Below zero = requested work exceeds capacity   ▼ = clipped below −1   "
        "× = requested source shed unmet",
        ha="right", fontsize=9, color="#53565A",
    )
    fig.tight_layout(rect=(0, .055, 1, .84), h_pad=1.5)
    for extension in ("png", "pdf"):
        fig.savefig(
            out / f"fixed_contract_residuals.{extension}",
            dpi=180, facecolor="white", bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    rows = load_or_generate(
        args.out, args.model, args.manifest, args.refresh,
    )
    plot(rows, args.out)
    print(f"rows={len(rows)} output={args.out}")


if __name__ == "__main__":
    main()
