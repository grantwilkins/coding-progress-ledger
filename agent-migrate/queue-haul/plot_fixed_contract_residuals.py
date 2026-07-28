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
import seaborn as sns

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


def overrun_shares(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["requested_shed_w"]), []).append(row)
    result = []
    for selected in grouped.values():
        overrun = [max(0, -float(row["normalized_slack"])) for row in selected]
        total = sum(overrun)
        result.extend(
            {**row, "overrun_share": value / total if total else 0}
            for row, value in zip(selected, overrun)
        )
    return result


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
    rows = overrun_shares(rows)
    resources = tuple(
        resource for resource in dict.fromkeys(row["resource"] for row in rows)
        if any(row["resource"] == resource and row["overrun_share"] for row in rows)
    )
    targets = sorted({row["requested_shed_w"] for row in rows})
    x = np.asarray(targets) / 1000
    sns.set_theme()
    fig, axis = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(targets))
    for resource, color in zip(resources, sns.color_palette(n_colors=len(resources))):
        selected = sorted(
            (row for row in rows if row["resource"] == resource),
            key=lambda row: row["requested_shed_w"],
        )
        share = np.asarray([row["overrun_share"] for row in selected])
        axis.bar(x, share, width=6.5, bottom=bottom, label=resource, color=color)
        bottom += share
    representatives = {
        row["requested_shed_w"]: row for row in rows
        if row["resource"] == resources[0]
    }
    failed = [i for i, target in enumerate(targets)
              if float(representatives[target]["unmet_shed_w"]) > 1e-7]
    axis.scatter(x[failed], np.full(len(failed), 1.03), marker="x", color="black",
                 s=45, linewidth=1.5, label="Requested shed unmet", clip_on=False)
    axis.set(
        xlabel="Requested source-power shed (kW)",
        ylabel="Share of normalized capacity overrun",
        ylim=(0, 1.08),
    )
    axis.set_xticks(x, [f"{value:g}" for value in np.round(x)])
    handles, labels = axis.get_legend_handles_labels()
    order = [labels.index(resource) for resource in resources] \
        + [labels.index("Requested shed unmet")]
    axis.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        frameon=False, loc="center left", bbox_to_anchor=(1, .5),
    )
    sns.despine()
    fig.tight_layout()
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
